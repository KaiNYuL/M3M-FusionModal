import argparse
import json
import pickle
import random
import sys
import typing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(typing, "TypeAlias"):
    from typing_extensions import TypeAlias as _TypeAlias

    typing.TypeAlias = _TypeAlias

from mamba3 import Mamba3, Mamba3Config, RMSNorm, get_device  # noqa: E402

# User-required DEAP channel list (1-based in request).
SELECTED_CHANNELS_1BASED = [8, 10, 15, 21, 22, 23, 24, 26, 27, 28, 31, 35, 36, 37, 38, 39, 40]
SELECTED_CHANNELS = [c - 1 for c in SELECTED_CHANNELS_1BASED]  # convert to 0-based


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, y_raw: np.ndarray = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if y_raw is None:
            y_raw = y
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_raw[idx]


class ZScoreScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def subject_wise_zscore(x: np.ndarray, subject_ids) -> np.ndarray:
    """Apply z-score normalization per subject over both sample and time axes."""
    x_norm = np.empty_like(x, dtype=np.float32)
    sids = np.asarray(subject_ids)

    for sid in np.unique(sids):
        idx = np.where(sids == sid)[0]
        block = x[idx]  # [Ns, T, C]
        mean = block.mean(axis=(0, 1), keepdims=True)
        std = block.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x_norm[idx] = ((block - mean) / std).astype(np.float32)

    return x_norm


class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y: np.ndarray):
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


class DEAPMambaRegressor(nn.Module):
    """
    Rebuilt model for raw DEAP time-series:
    Input: [batch, time, n_selected_channels]
    Core: bidirectional Mamba3 (forward + backward)
    Head: configurable Conv1d head or simplified linear head
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 96,
        d_state: int = 128,
        chunk_size: int = 63,
        dropout: float = 0.2,
        headdim: int = 32,
        head_type: str = "linear",
        device=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.drop = nn.Dropout(dropout)
        self.headdim = int(headdim)

        args = Mamba3Config(
            d_model=d_model,
            n_layer=1,
            d_state=d_state,
            expand=2,
            headdim=self.headdim,
            chunk_size=chunk_size,
            vocab_size=16,
        )

        # Two directional Mamba blocks: one forward, one backward.
        self.norm_fwd = RMSNorm(d_model, device=device)
        self.mamba_fwd = Mamba3(args, device=device)
        self.norm_bwd = RMSNorm(d_model, device=device)
        self.mamba_bwd = Mamba3(args, device=device)

        self.head_type = head_type
        if self.head_type == "conv":
            # Conv head keeps local temporal mixing but has higher capacity.
            self.conv1d = nn.Conv1d(2 * d_model, d_model, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.head_norm = nn.Identity()
            out_in_dim = d_model
        elif self.head_type == "linear":
            # Linear head reduces capacity to mitigate overfitting.
            self.conv1d = None
            self.act = nn.Identity()
            self.head_norm = nn.LayerNorm(2 * d_model)
            out_in_dim = 2 * d_model
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        self.out = nn.Linear(out_in_dim, 2)

    def encode(self, x: torch.Tensor):
        x = self.drop(self.input_proj(x))

        # Forward branch.
        y_fwd, _ = self.mamba_fwd(self.norm_fwd(x), None)
        h_fwd = x + y_fwd

        # Backward branch (reverse sequence, then reverse back).
        x_rev = torch.flip(x, dims=[1])
        y_bwd, _ = self.mamba_bwd(self.norm_bwd(x_rev), None)
        h_bwd = torch.flip(x_rev + y_bwd, dims=[1])

        # Merge bidirectional features.
        h = torch.cat([h_fwd, h_bwd], dim=-1)          # (b, t, 2*d_model)
        return h

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        if self.head_type == "conv":
            h = h.transpose(1, 2)                      # (b, 2*d_model, t)
            h = self.act(self.conv1d(h))               # (b, d_model, t)
            h = h.transpose(1, 2)                      # (b, t, d_model)
            pooled = h.mean(dim=1)
        else:
            pooled = self.head_norm(h.mean(dim=1))     # (b, 2*d_model)
        # For regression on standardized targets, keep output linear (no clipping).
        return self.out(pooled)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_headdim(d_model: int, requested_headdim: int) -> int:
    """Return a valid headdim so d_inner (=2*d_model) is divisible by headdim."""
    d_inner = 2 * int(d_model)
    req = max(1, int(requested_headdim))
    if d_inner % req == 0:
        return req

    for cand in (32, 16, 8, 4, 2, 1):
        if cand <= d_inner and d_inner % cand == 0:
            print(
                f"[warn] requested headdim={req} is invalid for d_inner={d_inner}; "
                f"auto-adjust to headdim={cand}."
            )
            return cand

    return 1


def build_stratify_bins(arousal: np.ndarray, n_bins: int = 5):
    # quantile bins for stable split on tiny 40-trial subject data.
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(arousal, q)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins = np.digitize(arousal, edges[1:-1], right=True)
    uniq, cnt = np.unique(bins, return_counts=True)
    if len(uniq) >= 2 and cnt.min() >= 2:
        return bins
    return None


def resolve_data_path(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    candidates = [Path.cwd() / p, PROJECT_ROOT / p, WORKSPACE_ROOT / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Data file not found: {data_arg}")


def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires pyyaml. Install with: pip install pyyaml") from exc

    p = Path(config_path)
    candidates = [p, Path.cwd() / p, PROJECT_ROOT / p, WORKSPACE_ROOT / p]
    target = None
    for c in candidates:
        if c.exists():
            target = c
            break

    if target is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(target, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def coerce_config_types(cfg: dict) -> dict:
    int_keys = {
        "epochs",
        "batch_size",
        "seed",
        "d_model",
        "d_state",
        "headdim",
        "chunk_size",
        "n_blocks",
        "mlp_hidden",
        "patience",
        "max_subjects",
        "window_size",
        "window_stride",
    }
    float_keys = {
        "dropout",
        "lr",
        "min_lr",
        "weight_decay",
        "noise_std",
        "cls_weight",
        "cls_threshold",
    }
    str_keys = {"config", "device", "data", "outdir", "seeds", "loss_mode", "head_type"}
    bool_keys = {"group_split", "subject_norm"}

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"Cannot parse boolean config value: {v}")

    out = dict(cfg)
    for k in list(out.keys()):
        v = out[k]
        if v is None:
            continue
        if k in int_keys:
            out[k] = int(v)
        elif k in float_keys:
            out[k] = float(v)
        elif k in str_keys:
            out[k] = str(v)
        elif k in bool_keys:
            out[k] = _to_bool(v)
    return out


def load_deap_dat(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if "data" not in obj or "labels" not in obj:
        raise ValueError("Invalid DEAP .dat file: expected keys 'data' and 'labels'.")

    # Load as float64 first to reduce cast overflow risk, then sanitize and cast.
    data = np.asarray(obj["data"], dtype=np.float64)  # (40, 40, 8064)
    labels = np.asarray(obj["labels"], dtype=np.float64)  # (40, 4)

    if data.ndim != 3 or data.shape[1] < 40 or data.shape[2] < 8064:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"Unexpected labels shape: {labels.shape}")

    # Replace NaN/Inf and clip extreme outliers before float32 cast.
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    data = np.clip(data, -1e6, 1e6)
    labels = np.nan_to_num(labels, nan=5.0, posinf=9.0, neginf=1.0)
    labels = np.clip(labels, 1.0, 9.0)

    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    # Channels requested by user.
    x = data[:, SELECTED_CHANNELS, :]
    # Only keep the latter half 4032 points.
    x = x[:, :, 4032:8064]

    # Rearrange to sequence-major for Mamba: [N, T, C]
    x = np.transpose(x, (0, 2, 1))

    # User requirement: arousal(second col), valence(first col)
    # DEAP labels default order: [valence, arousal, dominance, liking]
    y = labels[:, [1, 0]]

    return x, y


def load_deap_dataset(data_arg: str, max_subjects: int = 0):
    """Load one subject .dat or all subjects under a directory and concatenate.

    Returns
        x_all: (n_trials_total, 4032, n_channels)
        y_all: (n_trials_total, 2)
        subject_ids: list[str], one per trial in x_all/y_all
    """
    p = resolve_data_path(data_arg)

    if p.is_file():
        x, y = load_deap_dat(p)
        sid = p.stem
        subject_ids = [sid for _ in range(x.shape[0])]
        return x, y, subject_ids

    if p.is_dir():
        files = sorted(p.glob("s*.dat"))
        if not files:
            raise FileNotFoundError(f"No subject files like s*.dat found in directory: {p}")

        if max_subjects > 0:
            files = files[:max_subjects]

        x_list = []
        y_list = []
        subject_ids = []

        for f in files:
            x, y = load_deap_dat(f)
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Sample/label mismatch in {f}: {x.shape[0]} vs {y.shape[0]}")
            x_list.append(x)
            y_list.append(y)
            subject_ids.extend([f.stem for _ in range(x.shape[0])])

        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        if x_all.shape[0] != y_all.shape[0] or x_all.shape[0] != len(subject_ids):
            raise ValueError("Concatenated sample/label/subject length mismatch.")

        return x_all, y_all, subject_ids

    raise ValueError(f"Unsupported data path: {p}")


def first_difference(x: np.ndarray) -> np.ndarray:
    """Compute first-order temporal difference along time axis."""
    return x[:, 1:, :] - x[:, :-1, :]


def pad_time_to_chunk_multiple(x: np.ndarray, chunk_size: int) -> np.ndarray:
    """Right-pad time axis so length is divisible by chunk_size."""
    if chunk_size <= 1:
        return x
    t = x.shape[1]
    rem = t % chunk_size
    if rem == 0:
        return x
    pad_t = chunk_size - rem
    return np.pad(x, ((0, 0), (0, pad_t), (0, 0)), mode="constant", constant_values=0.0)


def sliding_window_augment(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids,
    window_size: int,
    window_stride: int,
):
    """Convert each trial into multiple overlapping windows to increase sample count."""
    if window_size <= 0 or window_stride <= 0:
        return x, y, subject_ids

    n, t, c = x.shape
    if window_size >= t:
        return x, y, subject_ids

    x_out = []
    y_out = []
    sid_out = []

    for i in range(n):
        starts = list(range(0, t - window_size + 1, window_stride))
        last_start = t - window_size
        if starts[-1] != last_start:
            starts.append(last_start)

        for s in starts:
            x_out.append(x[i, s : s + window_size, :])
            y_out.append(y[i])
            sid_out.append(subject_ids[i])

    return np.asarray(x_out, dtype=np.float32), np.asarray(y_out, dtype=np.float32), sid_out


def train_one_seed(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    min_lr,
    weight_decay,
    noise_std,
    patience,
    loss_mode,
    cls_weight,
    cls_threshold,
    y_mean,
    y_std,
):
    criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=min_lr,
    )

    model.to(device)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)
    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    def compute_loss(pred_scaled, y_scaled, y_raw):
        reg_loss = criterion(pred_scaled, y_scaled)
        if loss_mode == "hybrid_cls":
            pred_raw = pred_scaled * y_std_t + y_mean_t
            cls_logits = pred_raw - cls_threshold
            cls_target = (y_raw > cls_threshold).float()
            cls_loss = bce_criterion(cls_logits, cls_target)
            return reg_loss + cls_weight * cls_loss
        return reg_loss

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb, yb_raw in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yb_raw = yb_raw.to(device)
            xb = xb + noise_std * torch.randn_like(xb)

            optimizer.zero_grad()
            pred = model(xb)
            loss = compute_loss(pred, yb, yb_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb, yb_raw in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yb_raw = yb_raw.to(device)
                va_losses.append(compute_loss(model(xb), yb, yb_raw).item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        scheduler.step(va)
        print(f"Epoch {ep:03d} | train_loss={tr:.4f} | val_loss={va:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        if wait >= patience:
            print(f"Early stopping at epoch {ep} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate(model, x_test, y_test, device, y_scaler: TargetScaler, cls_threshold: float, eval_batch_size: int = 32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_test), eval_batch_size):
            xb = torch.tensor(x_test[i : i + eval_batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())

    pred_scaled = np.concatenate(preds, axis=0)
    pred = y_scaler.inverse_transform(pred_scaled)

    def binary_metrics_for_target(y_true: np.ndarray, y_pred: np.ndarray, thr: float, name: str) -> dict:
        y_true_bin = (y_true > thr).astype(np.int32)
        y_pred_bin = (y_pred > thr).astype(np.int32)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        out = {
            f"acc_bin_{name}": float(np.mean(y_true_bin == y_pred_bin)),
            f"precision_bin_{name}": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            f"recall_bin_{name}": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            f"f1_bin_{name}": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            f"specificity_bin_{name}": specificity,
            f"balanced_acc_bin_{name}": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
            f"mcc_bin_{name}": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
            f"tp_bin_{name}": tp,
            f"tn_bin_{name}": tn,
            f"fp_bin_{name}": fp,
            f"fn_bin_{name}": fn,
            f"support_pos_bin_{name}": int(np.sum(y_true_bin == 1)),
            f"support_neg_bin_{name}": int(np.sum(y_true_bin == 0)),
        }

        # AUC metrics require both classes in y_true.
        if len(np.unique(y_true_bin)) >= 2:
            out[f"roc_auc_bin_{name}"] = float(roc_auc_score(y_true_bin, y_pred))
            out[f"pr_auc_bin_{name}"] = float(average_precision_score(y_true_bin, y_pred))
        else:
            out[f"roc_auc_bin_{name}"] = None
            out[f"pr_auc_bin_{name}"] = None

        return out

    # Keep evaluation threshold consistent with training classification threshold.
    thr = float(cls_threshold)
    y_true_bin = y_test > thr
    y_pred_bin = pred > thr

    # Both dimensions must be correct for a sample to count as correct.
    acc_joint_bin = float(np.mean(np.all(y_true_bin == y_pred_bin, axis=1)))

    # Per-target classification metrics under threshold split.
    arousal_cls = binary_metrics_for_target(y_test[:, 0], pred[:, 0], thr, "arousal")
    valence_cls = binary_metrics_for_target(y_test[:, 1], pred[:, 1], thr, "valence")

    metrics = {
        "mse_arousal": float(mean_squared_error(y_test[:, 0], pred[:, 0])),
        "mse_valence": float(mean_squared_error(y_test[:, 1], pred[:, 1])),
        "mae_arousal": float(mean_absolute_error(y_test[:, 0], pred[:, 0])),
        "mae_valence": float(mean_absolute_error(y_test[:, 1], pred[:, 1])),
        "r2_arousal": float(r2_score(y_test[:, 0], pred[:, 0])),
        "r2_valence": float(r2_score(y_test[:, 1], pred[:, 1])),
        "bin_threshold": thr,
        "acc_bin_joint": acc_joint_bin,
        **arousal_cls,
        **valence_cls,
    }
    return pred, metrics


def main():
    parser = argparse.ArgumentParser(description="Rebuilt DEAP raw time-series emotion regression with Mamba3.")
    parser.add_argument("--config", type=str, default="", help="YAML config file path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--data", type=str, default="data/s01.dat")
    parser.add_argument("--max_subjects", type=int, default=0, help="Use first N subjects from data directory; 0 means all")
    parser.add_argument("--group_split", type=str, default="true", help="Use subject-wise group split (true/false)")
    parser.add_argument("--subject_norm", type=str, default="true", help="Use subject-wise z-score normalization (true/false)")
    parser.add_argument("--window_size", type=int, default=768, help="Sliding window size on time axis; <=0 disables")
    parser.add_argument("--window_stride", type=int, default=384, help="Sliding window stride on time axis; <=0 disables")
    parser.add_argument("--outdir", type=str, default="outputs/deap_raw_s01")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=63)
    parser.add_argument("--head_type", type=str, default="linear", choices=["conv", "linear"])
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss_mode", type=str, default="regression", choices=["regression", "hybrid_cls"])
    parser.add_argument("--cls_weight", type=float, default=0.6)
    parser.add_argument("--cls_threshold", type=float, default=4.0)
    args = parser.parse_args()

    if args.config:
        cfg = coerce_config_types(load_yaml_config(args.config))
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if isinstance(args.group_split, str):
        args.group_split = args.group_split.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        args.group_split = bool(args.group_split)

    if isinstance(args.subject_norm, str):
        args.subject_norm = args.subject_norm.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        args.subject_norm = bool(args.subject_norm)

    args.headdim = resolve_headdim(args.d_model, args.headdim)

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seed_list) == 0:
        seed_list = [args.seed]

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device=cuda is requested but CUDA is not available in current torch environment.")
    data_path = resolve_data_path(args.data)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, subject_ids = load_deap_dataset(args.data, max_subjects=args.max_subjects)
    x_all, y_all, subject_ids = sliding_window_augment(
        x_all,
        y_all,
        subject_ids,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )

    print("=" * 72)
    print("DEAP Raw Time-Series Mamba3 Regression (Rebuilt)")
    print("=" * 72)
    print(f"Data: {data_path}")
    print(f"Subjects loaded: {len(set(subject_ids))}")
    print(f"X shape: {x_all.shape}  (samples, time, channels={x_all.shape[2]})")
    print(f"Window augmentation: size={args.window_size}, stride={args.window_stride}")
    print(f"Group split by subject: {args.group_split}")
    print(f"Subject-wise normalization: {args.subject_norm}")
    print("Targets: [Arousal(label col 2), Valence(label col 1)]")
    print(f"Channels used (1-based): {SELECTED_CHANNELS_1BASED}")
    print(f"Device: {device}")
    print(f"Seeds: {seed_list}")

    all_seed_metrics = []
    best = None

    for sd in seed_list:
        print(f"\n--- Seed {sd} ---")
        set_seed(sd)

        groups_all = np.asarray(subject_ids)
        if args.group_split:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=sd)
            tr_idx, te_idx = next(gss_test.split(x_all, y_all, groups=groups_all))
            x_train, x_test = x_all[tr_idx], x_all[te_idx]
            y_train, y_test = y_all[tr_idx], y_all[te_idx]
            groups_train = groups_all[tr_idx]
            groups_test = groups_all[te_idx]

            gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=sd)
            tr2_idx, va_idx = next(gss_val.split(x_train, y_train, groups=groups_train))
            x_train_core, x_val = x_train[tr2_idx], x_train[va_idx]
            y_train_core, y_val = y_train[tr2_idx], y_train[va_idx]
            groups_train_core = groups_train[tr2_idx]
            groups_val = groups_train[va_idx]
        else:
            strat_bins = build_stratify_bins(y_all[:, 0], n_bins=5)
            x_train, x_test, y_train, y_test, groups_train, groups_test = train_test_split(
                x_all,
                y_all,
                groups_all,
                test_size=0.3,
                random_state=sd,
                stratify=strat_bins,
            )

            strat_train = build_stratify_bins(y_train[:, 0], n_bins=5)
            x_train_core, x_val, y_train_core, y_val, groups_train_core, groups_val = train_test_split(
                x_train,
                y_train,
                groups_train,
                test_size=0.2,
                random_state=sd,
                stratify=strat_train,
            )

        if args.subject_norm:
            x_train_core = subject_wise_zscore(x_train_core, groups_train_core)
            x_val = subject_wise_zscore(x_val, groups_val)
            x_test_norm = subject_wise_zscore(x_test, groups_test)
        else:
            # Fallback to global feature-wise z-score using train stats.
            x_scaler = ZScoreScaler()
            x_scaler.fit(x_train_core.reshape(-1, x_train_core.shape[-1]))
            x_train_core = x_scaler.transform(x_train_core.reshape(-1, x_train_core.shape[-1])).reshape(x_train_core.shape)
            x_val = x_scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
            x_test_norm = x_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

        # Use first-order difference signal as model input.
        x_train_core = first_difference(x_train_core)
        x_val = first_difference(x_val)
        x_test_norm = first_difference(x_test_norm)

        # Ensure SSD chunk divisibility while keeping larger chunk_size for memory efficiency.
        x_train_core = pad_time_to_chunk_multiple(x_train_core, args.chunk_size)
        x_val = pad_time_to_chunk_multiple(x_val, args.chunk_size)
        x_test_norm = pad_time_to_chunk_multiple(x_test_norm, args.chunk_size)

        y_scaler = TargetScaler()
        y_scaler.fit(y_train_core)
        y_train_scaled = y_scaler.transform(y_train_core)
        y_val_scaled = y_scaler.transform(y_val)

        train_ds = TimeSeriesDataset(x_train_core, y_train_scaled, y_train_core)
        val_ds = TimeSeriesDataset(x_val, y_val_scaled, y_val)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = DEAPMambaRegressor(
            in_channels=x_all.shape[2],
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            chunk_size=args.chunk_size,
            dropout=args.dropout,
            head_type=args.head_type,
            device=device,
        )

        print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(x_test_norm)}")
        model, history = train_one_seed(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            noise_std=args.noise_std,
            patience=args.patience,
            loss_mode=args.loss_mode,
            cls_weight=args.cls_weight,
            cls_threshold=args.cls_threshold,
            y_mean=y_scaler.mean,
            y_std=y_scaler.std,
        )

        _, metrics = evaluate(
            model,
            x_test_norm,
            y_test,
            device,
            y_scaler,
            cls_threshold=args.cls_threshold,
            eval_batch_size=args.batch_size,
        )
        all_seed_metrics.append({"seed": sd, **metrics})

        mean_mse = 0.5 * (metrics["mse_arousal"] + metrics["mse_valence"])
        if best is None or mean_mse < best["mean_mse"]:
            best = {
                "seed": sd,
                "mean_mse": mean_mse,
                "metrics": metrics,
                "history": history,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        print("Seed metrics:")
        for k, v in metrics.items():
            if v is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: {float(v):.4f}")

    metric_keys = [k for k in all_seed_metrics[0].keys() if k != "seed"]
    agg = {}
    for k in metric_keys:
        vals = np.array(
            [np.nan if m[k] is None else float(m[k]) for m in all_seed_metrics],
            dtype=np.float32,
        )
        if np.all(np.isnan(vals)):
            agg[k] = {"mean": None, "std": None}
        else:
            agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    report = {
        "data_path": str(data_path),
        "max_subjects": int(args.max_subjects),
        "n_subjects": int(len(set(subject_ids))),
        "subjects": sorted(set(subject_ids)),
        "x_shape": list(x_all.shape),
        "windowing": {
            "enabled": bool(args.window_size > 0 and args.window_stride > 0),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
        },
        "group_split": bool(args.group_split),
        "subject_norm": bool(args.subject_norm),
        "channels_used_1based": SELECTED_CHANNELS_1BASED,
        "time_range_used": [4032, 8064],
        "targets": ["Arousal(label_col_2)", "Valence(label_col_1)"],
        "seed_list": seed_list,
        "model_hyperparams": {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "headdim": args.headdim,
            "chunk_size": args.chunk_size,
            "dropout": args.dropout,
            "bidirectional_mamba": True,
            "head_type": args.head_type,
            "conv1d": {
                "in_channels": 2 * args.d_model,
                "out_channels": args.d_model,
                "kernel_size": 3,
            }
            if args.head_type == "conv"
            else None,
            "linear_head_in_dim": 2 * args.d_model if args.head_type == "linear" else args.d_model,
            "first_difference_input": True,
        },
        "optim_hyperparams": {
            "device": str(device),
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "noise_std": args.noise_std,
            "patience": args.patience,
            "loss_mode": args.loss_mode,
            "cls_weight": args.cls_weight,
            "cls_threshold": args.cls_threshold,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        "per_seed_metrics": all_seed_metrics,
        "aggregate_metrics": agg,
        "best_seed": int(best["seed"]),
        "best_seed_metrics": best["metrics"],
        "history_best_seed": best["history"],
    }

    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    torch.save(best["state"], out_dir / "best_model.pt")

    print("\nAggregate metrics across seeds:")
    for k, v in agg.items():
        if v["mean"] is None:
            print(f"  {k}: mean=None, std=None")
        else:
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    print("\nArtifacts saved:")
    print(f"  {out_dir / 'best_model.pt'}")
    print(f"  {out_dir / 'metrics_report.json'}")


if __name__ == "__main__":
    main()
