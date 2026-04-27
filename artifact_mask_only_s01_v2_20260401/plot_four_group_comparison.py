from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("artifact_mask_only_s01_v2_20260401/results/baselines_deep/filtered_r2_ge0")
    table = root / "deep_baselines_comparison_table.csv"
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table)

    mamba = {
        "model": "mamba_mask_only",
        "mse_mean": float(df["mamba_mse_mean"].mean()),
        "mae_mean": float(df["mamba_mae_mean"].mean()),
        "r2_mean": float(df["mamba_r2_mean"].mean()),
    }

    keep = ["patch_transformer_ae", "masked_transformer_ae", "tcn_ae", "timesnet_ae"]
    part = df[df["model"].isin(keep)][["model", "mse_mean", "mae_mean", "r2_mean"]].copy()
    part = pd.concat([pd.DataFrame([mamba]), part], ignore_index=True)

    labels = part["model"].tolist()
    x = np.arange(len(labels))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(15, 4.8))

    plt.subplot(1, 3, 1)
    plt.bar(x, part["mse_mean"].values, color=colors)
    plt.yscale("log")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("MSE (mean, log scale)")
    plt.grid(axis="y", alpha=0.25)

    plt.subplot(1, 3, 2)
    plt.bar(x, part["mae_mean"].values, color=colors)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("MAE (mean)")
    plt.grid(axis="y", alpha=0.25)

    plt.subplot(1, 3, 3)
    plt.bar(x, part["r2_mean"].values, color=colors)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("R2 (mean)")
    plt.grid(axis="y", alpha=0.25)

    plt.suptitle("Four-Group Comparison (Filtered Subjects)")
    plt.tight_layout()

    out = fig_dir / "five_group_metric_comparison.png"
    plt.savefig(out, dpi=180)
    plt.close()

    # Baseline-only view for readability (exclude Mamba) with R2 focus.
    base = part[part["model"] != "mamba_mask_only"].copy().reset_index(drop=True)
    bx = np.arange(len(base))
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(bx, base["r2_mean"].values, color=["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    plt.xticks(bx, base["model"].tolist(), rotation=20, ha="right")
    plt.ylabel("R2(mean)")
    plt.title("Baseline R2 Mean Comparison")
    plt.grid(axis="y", alpha=0.25)
    out_r2 = fig_dir / "baseline_r2_mean_comparison.png"
    plt.tight_layout()
    plt.savefig(out_r2, dpi=180)
    plt.close()

    # Delta vs Mamba: positive means baseline is better (for R2), negative means better for errors.
    m_mse = float(mamba["mse_mean"])
    m_mae = float(mamba["mae_mean"])
    m_r2 = float(mamba["r2_mean"])
    delta = base.copy()
    delta["delta_mse"] = delta["mse_mean"] - m_mse
    delta["delta_mae"] = delta["mae_mean"] - m_mae
    delta["delta_r2"] = delta["r2_mean"] - m_r2

    plt.figure(figsize=(12, 4.8))
    plt.subplot(1, 3, 1)
    plt.bar(bx, delta["delta_mse"].values, color="#8c564b")
    plt.xticks(bx, delta["model"].tolist(), rotation=20, ha="right")
    plt.title("Delta MSE vs Mamba")
    plt.grid(axis="y", alpha=0.25)

    plt.subplot(1, 3, 2)
    plt.bar(bx, delta["delta_mae"].values, color="#e377c2")
    plt.xticks(bx, delta["model"].tolist(), rotation=20, ha="right")
    plt.title("Delta MAE vs Mamba")
    plt.grid(axis="y", alpha=0.25)

    plt.subplot(1, 3, 3)
    plt.bar(bx, delta["delta_r2"].values, color="#17becf")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(bx, delta["model"].tolist(), rotation=20, ha="right")
    plt.title("Delta R2 vs Mamba")
    plt.grid(axis="y", alpha=0.25)

    plt.suptitle("Baseline Gap Relative to Mamba")
    plt.tight_layout()
    out_delta = fig_dir / "baseline_delta_vs_mamba.png"
    plt.savefig(out_delta, dpi=180)
    plt.close()

    print(out)
    print(out_r2)
    print(out_delta)


if __name__ == "__main__":
    main()
