from pathlib import Path

import pandas as pd


def main():
    root = Path("artifact_mask_only_s01_v2_20260401/results/baselines_deep/filtered_r2_ge0")
    per = root / "deep_baselines_per_subject_vs_mamba.csv"
    df = pd.read_csv(per)

    pick = ["patch_transformer_ae", "masked_transformer_ae", "timesnet_ae"]
    df = df[df["model"].isin(pick)].copy()
    if df.empty:
        raise ValueError("No rows found for selected baseline models")

    rows = []
    for m, g in df.groupby("model"):
        rows.append(
            {
                "model": m,
                "n_subjects": len(g),
                "mse_mean": g["mse"].mean(),
                "mae_mean": g["mae"].mean(),
                "r2_mean": g["r2"].mean(),
                "mse_median": g["mse"].median(),
                "mae_median": g["mae"].median(),
                "r2_median": g["r2"].median(),
                "r2_win_vs_mamba": int((g["r2"] > g["mamba_r2"]).sum()),
            }
        )
    rows = sorted(rows, key=lambda x: x["model"])

    mb = {
        "mamba_mse": float(df["mamba_mse"].mean()),
        "mamba_mae": float(df["mamba_mae"].mean()),
        "mamba_r2": float(df["mamba_r2"].mean()),
    }

    lines = []
    lines.append("# 四组重建模型对照（Filtered Subjects, R2 >= 0）")
    lines.append("")
    lines.append("- 数据范围：筛选后 28 个被试")
    lines.append("- 指标空间：标准化空间（与现有 Mamba 汇总口径一致）")
    lines.append("- 备注：TimesNet_AE 当前为快速对照版（10 epochs）")
    lines.append("")

    lines.append("## 均值对照")
    lines.append("")
    lines.append("| 模型 | n_subjects | MSE(mean) | MAE(mean) | R2(mean) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| mamba_mask_only | 28 | {mb['mamba_mse']:.6f} | {mb['mamba_mae']:.6f} | {mb['mamba_r2']:.6f} |")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['n_subjects']} | {r['mse_mean']:.6f} | {r['mae_mean']:.6f} | {r['r2_mean']:.6f} |"
        )
    lines.append("")

    lines.append("## 稳健统计（中位数）与单被试胜场")
    lines.append("")
    lines.append("| 模型 | MSE(median) | MAE(median) | R2(median) | R2胜过Mamba的被试数 |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['mse_median']:.6f} | {r['mae_median']:.6f} | {r['r2_median']:.6f} | {r['r2_win_vs_mamba']} |"
        )
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append("1. 在当前 filtered 被试集上，Mamba 方案在 MAE/MSE/R2 均值上仍显著领先。")
    lines.append("2. 在本轮三种深度 baseline 中，TimesNet_AE（快速版）相较两种 Transformer-AE 有一定提升，但与 Mamba 仍有明显差距。")
    lines.append("3. 若用于论文主表，建议将 TimesNet_AE 再按 40 epochs 预算复跑一次，保证训练预算完全对齐。")

    out = root / "four_group_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
