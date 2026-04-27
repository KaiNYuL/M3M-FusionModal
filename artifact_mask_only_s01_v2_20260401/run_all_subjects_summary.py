import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_subject_files(data_dir: Path):
    files = sorted(data_dir.glob("s*.dat"))
    return [f for f in files if f.is_file()]


def read_metrics(report_path: Path):
    with open(report_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    m = obj.get("metrics", {})
    return {
        "mse": float(m.get("mse", np.nan)),
        "mae": float(m.get("mae", np.nan)),
        "r2": float(m.get("r2", np.nan)),
        "raw_mse": float(m.get("raw_mse", np.nan)),
        "raw_mae": float(m.get("raw_mae", np.nan)),
        "raw_r2": float(m.get("raw_r2", np.nan)),
    }


def run_subject(python_exe: str, train_script: Path, config_path: Path, data_file: Path, outdir: Path):
    cmd = [
        python_exe,
        str(train_script),
        "--config",
        str(config_path),
        "--data",
        str(data_file),
        "--outdir",
        str(outdir),
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_csv(rows, csv_path: Path):
    fieldnames = ["subject", "mse", "mae", "r2", "raw_mse", "raw_mae", "raw_r2", "report_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows):
    keys = ["mse", "mae", "r2", "raw_mse", "raw_mae", "raw_r2"]
    out = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def plot_subject_r2(rows, out_file: Path):
    subjects = [r["subject"] for r in rows]
    vals = [float(r["r2"]) for r in rows]
    x = np.arange(len(subjects))

    plt.figure(figsize=(12, 4.8))
    bars = plt.bar(x, vals, color="#2C7FB8")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(x, subjects, rotation=70, ha="right")
    plt.ylabel("R2")
    plt.title("Per-Subject R2 (standardized space)")
    plt.grid(axis="y", alpha=0.25)

    top_idx = np.argsort(vals)[-3:]
    for i in top_idx:
        plt.text(bars[i].get_x() + bars[i].get_width() / 2.0, vals[i], f"{vals[i]:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_metric_box(rows, out_file: Path):
    mse = [float(r["mse"]) for r in rows]
    mae = [float(r["mae"]) for r in rows]
    r2 = [float(r["r2"]) for r in rows]

    plt.figure(figsize=(8, 4.8))
    plt.boxplot([mse, mae, r2], labels=["mse", "mae", "r2"], patch_artist=True)
    plt.title("Metric Distribution Across Subjects")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_metric_hist(rows, out_file: Path):
    vals = np.array([float(r["r2"]) for r in rows], dtype=np.float64)

    plt.figure(figsize=(8, 4.8))
    plt.hist(vals, bins=10, color="#F28E2B", edgecolor="black", alpha=0.85)
    plt.axvline(float(np.mean(vals)), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(vals):.3f}")
    plt.axvline(float(np.median(vals)), color="green", linestyle="-.", linewidth=1.5, label=f"median={np.median(vals):.3f}")
    plt.xlabel("R2")
    plt.ylabel("Count")
    plt.title("R2 Histogram Across Subjects")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run all DEAP subjects and build summary/visualization.")
    parser.add_argument("--artifactRoot", type=str, default="artifact_mask_only_s01_v2_20260401")
    parser.add_argument("--dataDir", type=str, default="data")
    parser.add_argument("--trainScript", type=str, default="model/deap_mamba3_multimodal_decoder.py")
    parser.add_argument("--config", type=str, default="config/deap_multimodal_mask_only_s01.yaml")
    parser.add_argument("--resultsDir", type=str, default="results/runs/all_subjects")
    parser.add_argument("--summaryDir", type=str, default="results/summary")
    parser.add_argument("--pythonExe", type=str, default=sys.executable)
    parser.add_argument("--skipTrain", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifactRoot).resolve()
    data_dir = Path(args.dataDir).resolve()
    train_script = (artifact_root / args.trainScript).resolve()
    config_path = (artifact_root / args.config).resolve()
    results_dir = (artifact_root / args.resultsDir).resolve()
    summary_dir = (artifact_root / args.summaryDir).resolve()

    summary_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_files(data_dir)
    if not subjects:
        raise FileNotFoundError(f"No s*.dat found in {data_dir}")

    rows = []
    for sf in subjects:
        sid = sf.stem
        subj_out = results_dir / sid
        report_path = subj_out / "metrics_report.json"

        if not args.skipTrain:
            run_subject(args.pythonExe, train_script, config_path, sf, subj_out)

        if not report_path.exists():
            raise FileNotFoundError(f"Missing report for {sid}: {report_path}")

        m = read_metrics(report_path)
        row = {"subject": sid, **m, "report_path": str(report_path)}
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["subject"])

    csv_path = summary_dir / "all_subjects_metrics.csv"
    write_csv(rows, csv_path)

    agg = summarize(rows)
    agg_obj = {
        "n_subjects": len(rows),
        "config": str(config_path),
        "train_script": str(train_script),
        "metrics": agg,
    }
    agg_path = summary_dir / "aggregate_metrics.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_obj, f, ensure_ascii=False, indent=2)

    fig_dir = summary_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_subject_r2(rows, fig_dir / "r2_per_subject.png")
    plot_metric_box(rows, fig_dir / "metric_boxplot.png")
    plot_metric_hist(rows, fig_dir / "r2_histogram.png")

    print("Saved summary files:")
    print(csv_path)
    print(agg_path)
    print(fig_dir / "r2_per_subject.png")
    print(fig_dir / "metric_boxplot.png")
    print(fig_dir / "r2_histogram.png")


if __name__ == "__main__":
    main()
