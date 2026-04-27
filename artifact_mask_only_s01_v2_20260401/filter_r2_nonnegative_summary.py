import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["mse"] = float(row["mse"])
            row["mae"] = float(row["mae"])
            row["r2"] = float(row["r2"])
            row["raw_mse"] = float(row["raw_mse"])
            row["raw_mae"] = float(row["raw_mae"])
            row["raw_r2"] = float(row["raw_r2"])
            rows.append(row)
    return rows


def write_rows(rows, csv_path: Path):
    fields = ["subject", "mse", "mae", "r2", "raw_mse", "raw_mae", "raw_r2", "report_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize(rows):
    metrics = {}
    for k in ["mse", "mae", "r2", "raw_mse", "raw_mae", "raw_r2"]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        metrics[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return metrics


def plot_r2(rows, out_file: Path):
    subjects = [r["subject"] for r in rows]
    vals = [r["r2"] for r in rows]
    x = np.arange(len(subjects))
    plt.figure(figsize=(12, 4.8))
    plt.bar(x, vals, color="#2C7FB8")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(x, subjects, rotation=70, ha="right")
    plt.ylabel("R2")
    plt.title("Filtered Per-Subject R2 (R2 >= 0)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_box(rows, out_file: Path):
    mse = [r["mse"] for r in rows]
    mae = [r["mae"] for r in rows]
    r2 = [r["r2"] for r in rows]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot([mse, mae, r2], labels=["mse", "mae", "r2"], patch_artist=True)
    plt.title("Filtered Metric Distribution (R2 >= 0)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_r2_hist(rows, out_file: Path):
    vals = np.array([r["r2"] for r in rows], dtype=np.float64)
    plt.figure(figsize=(8, 4.8))
    plt.hist(vals, bins=10, color="#F28E2B", edgecolor="black", alpha=0.85)
    plt.axvline(float(np.mean(vals)), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(vals):.3f}")
    plt.axvline(float(np.median(vals)), color="green", linestyle="-.", linewidth=1.5, label=f"median={np.median(vals):.3f}")
    plt.xlabel("R2")
    plt.ylabel("Count")
    plt.title("Filtered R2 Histogram (R2 >= 0)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Filter subjects with r2<0 and regenerate summary")
    parser.add_argument("--summaryDir", type=str, default="artifact_mask_only_s01_v2_20260401/results/summary")
    args = parser.parse_args()

    summary_dir = Path(args.summaryDir).resolve()
    src_csv = summary_dir / "all_subjects_metrics.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {src_csv}")

    rows = read_rows(src_csv)
    rows_sorted = sorted(rows, key=lambda x: x["subject"])
    keep = [r for r in rows_sorted if r["r2"] >= 0.0]
    drop = [r for r in rows_sorted if r["r2"] < 0.0]

    if not keep:
        raise RuntimeError("No subjects left after filtering r2>=0")

    out_csv = summary_dir / "all_subjects_metrics_r2_ge0.csv"
    write_rows(keep, out_csv)

    agg = {
        "filter": "r2 >= 0",
        "n_subjects_before": len(rows_sorted),
        "n_subjects_after": len(keep),
        "removed_subjects": [r["subject"] for r in drop],
        "metrics": summarize(keep),
    }
    out_json = summary_dir / "aggregate_metrics_r2_ge0.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    fig_dir = summary_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_r2 = fig_dir / "r2_per_subject_r2_ge0.png"
    fig_box = fig_dir / "metric_boxplot_r2_ge0.png"
    fig_hist = fig_dir / "r2_histogram_r2_ge0.png"
    plot_r2(keep, fig_r2)
    plot_box(keep, fig_box)
    plot_r2_hist(keep, fig_hist)

    print("before:", len(rows_sorted))
    print("after:", len(keep))
    print("removed:", ",".join([r["subject"] for r in drop]) if drop else "(none)")
    print("saved:", out_csv)
    print("saved:", out_json)
    print("saved:", fig_r2)
    print("saved:", fig_box)
    print("saved:", fig_hist)


if __name__ == "__main__":
    main()
