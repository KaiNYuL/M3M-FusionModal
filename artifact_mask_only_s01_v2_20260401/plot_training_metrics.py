import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_report(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(history: dict, out_file: Path):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss and not val_loss:
        return False

    plt.figure(figsize=(8, 4.8))
    if train_loss:
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss", linewidth=1.8)
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return True


def plot_val_r2(history: dict, out_file: Path):
    val_r2 = history.get("val_r2", [])
    if not val_r2:
        return False

    plt.figure(figsize=(8, 4.8))
    plt.plot(range(1, len(val_r2) + 1), val_r2, color="#D94801", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title("Validation R2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return True


def plot_final_metrics(report: dict, out_file: Path):
    metrics = report.get("metrics", {})

    mse = metrics.get("mse")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")

    history = report.get("history", {})
    val_loss = history.get("val_loss", [])
    best_val_loss = min(val_loss) if val_loss else None

    names = []
    values = []

    if best_val_loss is not None:
        names.append("best_val_loss")
        values.append(float(best_val_loss))
    if mse is not None:
        names.append("test_mse")
        values.append(float(mse))
    if mae is not None:
        names.append("test_mae")
        values.append(float(mae))
    if r2 is not None:
        names.append("test_r2")
        values.append(float(r2))

    if not names:
        return False

    colors = ["#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"][: len(names)]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(names, values, color=colors)
    plt.title("Final Metrics Overview")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.25)

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return True


def plot_per_channel_mse(report: dict, out_file: Path):
    metrics = report.get("metrics", {})
    per_channel_mse = metrics.get("per_channel_mse", {})
    if not isinstance(per_channel_mse, dict) or not per_channel_mse:
        return False

    items = sorted(per_channel_mse.items(), key=lambda x: x[0])
    names = [k for k, _ in items]
    values = [float(v) for _, v in items]

    width = max(9.0, min(16.0, 0.45 * len(names)))
    plt.figure(figsize=(width, 4.8))
    bars = plt.bar(names, values, color="#4C78A8")
    plt.title("Per-Channel MSE")
    plt.xlabel("Channel")
    plt.ylabel("MSE")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")

    if len(values) <= 20:
        for bar, v in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot loss/mse/mae/r2 from metrics_report.json")
    parser.add_argument("--report", type=str, required=True, help="Path to metrics_report.json")
    parser.add_argument("--outdir", type=str, default="", help="Output folder for figures")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = load_report(report_path)

    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = report_path.parent / "figures"
    ensure_dir(out_dir)

    history = report.get("history", {})

    saved = []

    p1 = out_dir / "loss_curves.png"
    if plot_loss_curves(history, p1):
        saved.append(p1)

    p2 = out_dir / "val_r2_curve.png"
    if plot_val_r2(history, p2):
        saved.append(p2)

    p3 = out_dir / "metrics_bar.png"
    if plot_final_metrics(report, p3):
        saved.append(p3)

    p4 = out_dir / "per_channel_mse.png"
    if plot_per_channel_mse(report, p4):
        saved.append(p4)

    if not saved:
        print("No plottable fields found in the report.")
        return

    print("Saved figures:")
    for p in saved:
        print(str(p))


if __name__ == "__main__":
    main()
