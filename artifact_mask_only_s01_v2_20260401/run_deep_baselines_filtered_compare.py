import argparse
import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class SignalScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def load_deap_dat(path: Path, selected_channels_1based: list[int]) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    data = np.asarray(obj["data"], dtype=np.float64)  # [40, 40, 8064]
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    data = np.clip(data, -1e6, 1e6)

    selected = [c - 1 for c in selected_channels_1based]
    x = data[:, selected, :]
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # [trials, T, C]
    return x


def random_mask_input(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return x
    keep = (torch.rand_like(x) > mask_ratio).to(x.dtype)
    return x * keep


class PatchTransformerAE(nn.Module):
    def __init__(self, channels: int, seq_len: int, d_model: int = 64, patch_size: int = 96, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.patch_size = patch_size

        self.in_proj = nn.Conv1d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.n_tokens = int(np.ceil(seq_len / patch_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, channels * patch_size)

    def _fold(self, z: torch.Tensor) -> torch.Tensor:
        b, n, _ = z.shape
        y = self.head(z).view(b, n, self.patch_size, self.channels)
        y = y.reshape(b, n * self.patch_size, self.channels)
        return y[:, : self.seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos_emb[:, : h.shape[1], :]
        h = self.encoder(h)
        return self._fold(h)


class MaskedTransformerAE(nn.Module):
    def __init__(self, channels: int, seq_len: int, d_model: int = 64, patch_size: int = 96, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.patch_size = patch_size

        self.in_proj = nn.Conv1d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.n_tokens = int(np.ceil(seq_len / patch_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, channels * patch_size)

    def _fold(self, z: torch.Tensor) -> torch.Tensor:
        b, n, _ = z.shape
        y = self.head(z).view(b, n, self.patch_size, self.channels)
        y = y.reshape(b, n * self.patch_size, self.channels)
        return y[:, : self.seq_len, :]

    def forward(self, x: torch.Tensor, token_mask_ratio: float = 0.15, training: bool = False) -> torch.Tensor:
        h = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        if training and token_mask_ratio > 0.0:
            token_mask = (torch.rand(h.shape[0], h.shape[1], device=h.device) < token_mask_ratio).unsqueeze(-1)
            h = torch.where(token_mask, self.mask_token.expand_as(h), h)
        h = h + self.pos_emb[:, : h.shape[1], :]
        h = self.encoder(h)
        return self._fold(h)


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        return self.norm(F.gelu(h + x))


class TCNAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([DilatedBlock(hidden, dilation=2**i, dropout=dropout) for i in range(depth)])
        self.out_proj = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        y = self.out_proj(h)
        return y.transpose(1, 2)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.proj = nn.Conv1d(3 * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1)
        return self.proj(F.gelu(h))


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, top_k: int = 3, hidden_mult: int = 2, max_period: int = 512):
        super().__init__()
        self.top_k = int(max(1, top_k))
        self.max_period = int(max(8, max_period))
        hidden = int(d_model * hidden_mult)
        self.inception1 = InceptionBlock1D(d_model, hidden)
        self.inception2 = InceptionBlock1D(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _dominant_periods(self, x: torch.Tensor) -> list[int]:
        # x: [B, T, D]
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=(0, 2))
        amp[0] = 0.0
        top = torch.topk(amp, k=min(self.top_k, amp.numel())).indices
        periods = []
        t = x.shape[1]
        for i in top.tolist():
            p = int(round(t / max(1, i)))
            p = max(2, min(p, self.max_period, t))
            periods.append(p)
        return sorted(list(set(periods)))

    def _period_conv(self, x: torch.Tensor, period: int) -> torch.Tensor:
        # Treat each period segment as a local pattern and convolve along phase axis.
        b, t, d = x.shape
        pad_len = (period - (t % period)) % period
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)
        else:
            x_pad = x
        t2 = x_pad.shape[1]
        n_seg = t2 // period
        z = x_pad.view(b, n_seg, period, d).permute(0, 1, 3, 2).reshape(b * n_seg, d, period)
        z = self.inception1(z)
        z = self.inception2(z)
        z = z.reshape(b, n_seg, d, period).permute(0, 1, 3, 2).reshape(b, t2, d)
        return z[:, :t, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        periods = self._dominant_periods(x)
        outs = [self._period_conv(x, p) for p in periods]
        y = torch.stack(outs, dim=0).mean(dim=0)
        return self.norm(x + y)


class TimesNetAE(nn.Module):
    def __init__(self, channels: int, d_model: int = 64, n_blocks: int = 2, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(channels, d_model)
        self.blocks = nn.ModuleList([TimesBlock(d_model=d_model, top_k=top_k) for _ in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.drop(h)
        return self.out_proj(h)


def build_model(name: str, channels: int, seq_len: int, patch_size: int):
    key = name.strip().lower()
    if key == "patch_transformer_ae":
        return PatchTransformerAE(channels=channels, seq_len=seq_len, patch_size=patch_size)
    if key == "masked_transformer_ae":
        return MaskedTransformerAE(channels=channels, seq_len=seq_len, patch_size=patch_size)
    if key == "tcn_ae":
        return TCNAE(channels=channels)
    if key == "timesnet_ae":
        return TimesNetAE(channels=channels)
    raise ValueError(f"Unsupported model: {name}")


def load_existing_rows_if_any(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_model(model, loader, device, eval_mask_ratio: float, model_name: str):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            xin = random_mask_input(xb, eval_mask_ratio)
            if model_name == "masked_transformer_ae":
                yhat = model(xin, token_mask_ratio=0.0, training=False)
            else:
                yhat = model(xin)
            preds.append(yhat.cpu().numpy())
            trues.append(xb.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    mse = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
    r2 = float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))
    return {"mse": mse, "mae": mae, "r2": r2}


def train_one_subject(model_name: str, x_subject: np.ndarray, cfg: dict, device: torch.device):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    test_size = float(cfg.get("test_size", 0.25))
    val_size = float(cfg.get("val_size", 0.25))
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("baseline_epochs", 80))
    patience = int(cfg.get("baseline_patience", 10))
    lr = float(cfg.get("baseline_lr", cfg.get("lr", 1e-4)))
    weight_decay = float(cfg.get("baseline_weight_decay", cfg.get("weight_decay", 0.01)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    train_mask_ratio = float(cfg.get("encoder_random_mask_ratio", 0.15))

    eval_mask_ratio = float(cfg.get("encoder_eval_mask_ratio", 0.0))
    mask_observed_residual = to_bool(cfg.get("mask_observed_residual", True))
    if eval_mask_ratio <= 0.0 and mask_observed_residual:
        eval_mask_ratio = max(0.05, train_mask_ratio)

    x_train, x_test = train_test_split(x_subject, test_size=test_size, random_state=seed)
    x_train_core, x_val = train_test_split(x_train, test_size=val_size, random_state=seed)

    scaler = SignalScaler()
    scaler.fit(x_train_core)
    x_train_core = scaler.transform(x_train_core).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_loader = DataLoader(SeqDataset(x_train_core), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(x_test), batch_size=batch_size, shuffle=False)

    model = build_model(model_name, channels=x_subject.shape[2], seq_len=x_subject.shape[1], patch_size=int(cfg.get("patch_size", 96))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb in train_loader:
            xb = xb.to(device)
            xin = random_mask_input(xb, train_mask_ratio)
            optimizer.zero_grad()
            if model_name == "masked_transformer_ae":
                yhat = model(xin, token_mask_ratio=train_mask_ratio, training=True)
            else:
                yhat = model(xin)
            loss = loss_fn(yhat, xb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                xin = random_mask_input(xb, eval_mask_ratio)
                if model_name == "masked_transformer_ae":
                    yhat = model(xin, token_mask_ratio=0.0, training=False)
                else:
                    yhat = model(xin)
                val_losses.append(loss_fn(yhat, xb).item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = eval_model(model, test_loader, device, eval_mask_ratio=eval_mask_ratio, model_name=model_name)
    metrics["eval_mask_ratio"] = float(eval_mask_ratio)
    return model, metrics


def summarize_rows(rows: list[dict], key: str):
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run deep-learning baselines on filtered subjects and compare with Mamba model")
    parser.add_argument("--artifactRoot", type=str, default="artifact_mask_only_s01_v2_20260401")
    parser.add_argument("--dataDir", type=str, default="data")
    parser.add_argument("--config", type=str, default="config/deap_multimodal_mask_only_s01.yaml")
    parser.add_argument("--filteredCsv", type=str, default="results/summary/all_subjects_metrics_r2_ge0.csv")
    parser.add_argument("--outDir", type=str, default="results/baselines_deep/filtered_r2_ge0")
    parser.add_argument("--models", type=str, default="patch_transformer_ae,masked_transformer_ae,timesnet_ae")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline_epochs", type=int, default=80)
    parser.add_argument("--baseline_patience", type=int, default=10)
    parser.add_argument("--baseline_lr", type=float, default=1e-4)
    parser.add_argument("--baseline_weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    artifact_root = Path(args.artifactRoot).resolve()
    data_dir = Path(args.dataDir).resolve()
    config_path = (artifact_root / args.config).resolve()
    filtered_csv = (artifact_root / args.filteredCsv).resolve()
    out_dir = (artifact_root / args.outDir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["baseline_epochs"] = int(args.baseline_epochs)
    cfg["baseline_patience"] = int(args.baseline_patience)
    cfg["baseline_lr"] = float(args.baseline_lr)
    cfg["baseline_weight_decay"] = float(args.baseline_weight_decay)

    selected_channels = [int(x) for x in cfg.get("selected_channels_1based", [])]
    if not selected_channels:
        raise ValueError("selected_channels_1based missing in config")

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    with open(filtered_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    subjects = [r["subject"] for r in rows]
    mamba_by_subject = {r["subject"]: r for r in rows}

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    all_csv = out_dir / "deep_baselines_per_subject_vs_mamba.csv"
    existing_rows = load_existing_rows_if_any(all_csv)
    existing_rows = [r for r in existing_rows if r.get("model", "").strip().lower() not in set(models)]
    all_rows = []
    for model_name in models:
        model_out_root = out_dir / model_name
        model_out_root.mkdir(parents=True, exist_ok=True)

        model_rows = []
        for sid in subjects:
            subject_file = data_dir / f"{sid}.dat"
            if not subject_file.exists():
                print(f"[skip] missing subject file: {subject_file}")
                continue

            x = load_deap_dat(subject_file, selected_channels)
            model, met = train_one_subject(model_name=model_name, x_subject=x, cfg=cfg, device=device)

            run_dir = model_out_root / sid
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            with open(run_dir / "metrics_report.json", "w", encoding="utf-8") as f:
                json.dump({"subject": sid, "model": model_name, "metrics": met}, f, ensure_ascii=False, indent=2)

            row = {
                "subject": sid,
                "model": model_name,
                "mse": float(met["mse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "eval_mask_ratio": float(met["eval_mask_ratio"]),
                "mamba_mse": float(mamba_by_subject[sid]["mse"]),
                "mamba_mae": float(mamba_by_subject[sid]["mae"]),
                "mamba_r2": float(mamba_by_subject[sid]["r2"]),
            }
            model_rows.append(row)
            all_rows.append(row)
            print(
                f"[{model_name}] {sid} | mse={row['mse']:.6f} mae={row['mae']:.6f} r2={row['r2']:.6f} "
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

    all_rows = existing_rows + all_rows

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

    # Compact table: model-level means vs Mamba means on same subjects.
    comp_rows = []
    all_models = sorted(set(r["model"] for r in all_rows))
    for model_name in all_models:
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

    comp_csv = out_dir / "deep_baselines_comparison_table.csv"
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

    comp_json = out_dir / "deep_baselines_comparison_table.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(comp_rows, f, ensure_ascii=False, indent=2)

    print("saved:")
    print(all_csv)
    print(comp_csv)
    print(comp_json)


if __name__ == "__main__":
    main()
