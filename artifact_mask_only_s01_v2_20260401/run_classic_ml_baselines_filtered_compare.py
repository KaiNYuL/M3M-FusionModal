import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int):
    np.random.seed(seed)


def load_deap_dat(path: Path, selected_channels_1based: list[int]) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    data = np.asarray(obj["data"], dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    data = np.clip(data, -1e6, 1e6)

    selected = [c - 1 for c in selected_channels_1based]
    x = data[:, selected, :]
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # [N, T, C]
    return x


def fit_signal_scaler(x: np.ndarray):
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_signal_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std


def random_mask_input(x: np.ndarray, ratio: float, rng: np.random.Generator):
    if ratio <= 0.0:
        return x
    keep = (rng.random(x.shape) > ratio).astype(x.dtype)
    return x * keep


def summarize_rows(rows: list[dict], key: str):
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def build_regressor(name: str, seed: int, n_comp_x: int, n_comp_y: int):
    key = name.strip().lower()
    if key == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if key == "pls":
        return PLSRegression(n_components=max(2, min(8, n_comp_x, n_comp_y)))
    if key == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classic model: {name}")


def fit_predict_one_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int):
    n_train = x_train.shape[0]
    n_comp_x = max(2, min(20, n_train - 1))
    n_comp_y = max(2, min(20, n_train - 1))

    x_pca = PCA(n_components=n_comp_x, svd_solver="randomized", random_state=seed)
    y_pca = PCA(n_components=n_comp_y, svd_solver="randomized", random_state=seed)

    xr_train = x_pca.fit_transform(x_train)
    xr_test = x_pca.transform(x_test)
    yr_train = y_pca.fit_transform(y_train)

    reg = build_regressor(model_name, seed=seed, n_comp_x=n_comp_x, n_comp_y=n_comp_y)
    reg.fit(xr_train, yr_train)
    yr_pred = reg.predict(xr_test)

    y_pred = y_pca.inverse_transform(yr_pred)
    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Run classic ML baselines on filtered subjects and compare with Mamba")
    parser.add_argument("--artifactRoot", type=str, default="artifact_mask_only_s01_v2_20260401")
    parser.add_argument("--dataDir", type=str, default="data")
    parser.add_argument("--config", type=str, default="config/deap_multimodal_mask_only_s01.yaml")
    parser.add_argument("--filteredCsv", type=str, default="results/summary/all_subjects_metrics_r2_ge0.csv")
    parser.add_argument("--outDir", type=str, default="results/baselines_classic/filtered_r2_ge0")
    parser.add_argument("--models", type=str, default="ridge,pls,random_forest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    artifact_root = Path(args.artifactRoot).resolve()
    data_dir = Path(args.dataDir).resolve()
    config_path = (artifact_root / args.config).resolve()
    filtered_csv = (artifact_root / args.filteredCsv).resolve()
    out_dir = (artifact_root / args.outDir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    selected_channels = [int(x) for x in cfg.get("selected_channels_1based", [])]
    if not selected_channels:
        raise ValueError("selected_channels_1based missing in config")

    test_size = float(cfg.get("test_size", 0.25))
    val_size = float(cfg.get("val_size", 0.25))
    train_mask_ratio = float(cfg.get("encoder_random_mask_ratio", 0.15))
    eval_mask_ratio = float(cfg.get("encoder_eval_mask_ratio", 0.0))
    mask_observed_residual = str(cfg.get("mask_observed_residual", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if eval_mask_ratio <= 0.0 and mask_observed_residual:
        eval_mask_ratio = max(0.05, train_mask_ratio)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    with open(filtered_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    subjects = [r["subject"] for r in rows]
    mamba_by_subject = {r["subject"]: r for r in rows}

    all_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for model_name in models:
        model_rows = []
        model_out_root = out_dir / model_name
        model_out_root.mkdir(parents=True, exist_ok=True)

        for sid in subjects:
            subject_file = data_dir / f"{sid}.dat"
            if not subject_file.exists():
                continue

            x = load_deap_dat(subject_file, selected_channels)

            x_train, x_test = train_test_split(x, test_size=test_size, random_state=args.seed)
            x_train_core, _x_val = train_test_split(x_train, test_size=val_size, random_state=args.seed)

            mean, std = fit_signal_scaler(x_train_core)
            x_train_core = apply_signal_scaler(x_train_core, mean, std).astype(np.float32)
            x_test = apply_signal_scaler(x_test, mean, std).astype(np.float32)

            x_train_in = random_mask_input(x_train_core, train_mask_ratio, rng)
            x_test_in = random_mask_input(x_test, eval_mask_ratio, rng)

            n_train = x_train_in.shape[0]
            n_test = x_test_in.shape[0]
            x_train_flat = x_train_in.reshape(n_train, -1)
            y_train_flat = x_train_core.reshape(n_train, -1)
            x_test_flat = x_test_in.reshape(n_test, -1)
            y_test_flat = x_test.reshape(n_test, -1)

            y_pred_flat = fit_predict_one_model(model_name, x_train_flat, y_train_flat, x_test_flat, seed=args.seed)

            mse = float(mean_squared_error(y_test_flat.reshape(-1), y_pred_flat.reshape(-1)))
            mae = float(mean_absolute_error(y_test_flat.reshape(-1), y_pred_flat.reshape(-1)))
            r2 = float(r2_score(y_test_flat.reshape(-1), y_pred_flat.reshape(-1)))

            run_dir = model_out_root / sid
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "metrics_report.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "subject": sid,
                        "model": model_name,
                        "metrics": {
                            "mse": mse,
                            "mae": mae,
                            "r2": r2,
                            "eval_mask_ratio": float(eval_mask_ratio),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            row = {
                "subject": sid,
                "model": model_name,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "eval_mask_ratio": float(eval_mask_ratio),
                "mamba_mse": float(mamba_by_subject[sid]["mse"]),
                "mamba_mae": float(mamba_by_subject[sid]["mae"]),
                "mamba_r2": float(mamba_by_subject[sid]["r2"]),
            }
            model_rows.append(row)
            all_rows.append(row)
            print(
                f"[{model_name}] {sid} | mse={mse:.6f} mae={mae:.6f} r2={r2:.6f} "
                f"| mamba_r2={row['mamba_r2']:.6f}"
            )

        if model_rows:
            agg = {
                "model": model_name,
                "n_subjects": len(model_rows),
                "mse": summarize_rows(model_rows, "mse"),
                "mae": summarize_rows(model_rows, "mae"),
                "r2": summarize_rows(model_rows, "r2"),
                "mamba_mse": summarize_rows(model_rows, "mamba_mse"),
                "mamba_mae": summarize_rows(model_rows, "mamba_mae"),
                "mamba_r2": summarize_rows(model_rows, "mamba_r2"),
            }
            with open(model_out_root / "aggregate_metrics.json", "w", encoding="utf-8") as f:
                json.dump(agg, f, ensure_ascii=False, indent=2)

    all_csv = out_dir / "classic_baselines_per_subject_vs_mamba.csv"
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "subject",
            "model",
            "mse",
            "mae",
            "r2",
            "eval_mask_ratio",
            "mamba_mse",
            "mamba_mae",
            "mamba_r2",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    comp_rows = []
    for model_name in models:
        part = [r for r in all_rows if r["model"] == model_name]
        if not part:
            continue
        comp_rows.append(
            {
                "model": model_name,
                "n_subjects": len(part),
                "mse_mean": summarize_rows(part, "mse")["mean"],
                "mae_mean": summarize_rows(part, "mae")["mean"],
                "r2_mean": summarize_rows(part, "r2")["mean"],
                "mamba_mse_mean": summarize_rows(part, "mamba_mse")["mean"],
                "mamba_mae_mean": summarize_rows(part, "mamba_mae")["mean"],
                "mamba_r2_mean": summarize_rows(part, "mamba_r2")["mean"],
            }
        )

    comp_csv = out_dir / "classic_baselines_comparison_table.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "n_subjects",
            "mse_mean",
            "mae_mean",
            "r2_mean",
            "mamba_mse_mean",
            "mamba_mae_mean",
            "mamba_r2_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in comp_rows:
            w.writerow(r)

    comp_json = out_dir / "classic_baselines_comparison_table.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(comp_rows, f, ensure_ascii=False, indent=2)

    print("saved:")
    print(all_csv)
    print(comp_csv)
    print(comp_json)


if __name__ == "__main__":
    main()
