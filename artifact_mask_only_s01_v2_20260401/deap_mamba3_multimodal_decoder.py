import argparse
import json
import pickle
import random
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class LabelScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, y: np.ndarray):
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std


class ForecastDataset(Dataset):
    def __init__(self, x_prefix: np.ndarray, y_aux: np.ndarray, y_target: np.ndarray):
        self.x_prefix = torch.tensor(x_prefix, dtype=torch.float32)
        self.y_aux = torch.tensor(y_aux, dtype=torch.float32)
        self.y_target = torch.tensor(y_target, dtype=torch.float32)

    def __len__(self):
        return len(self.x_prefix)

    def __getitem__(self, idx):
        return self.x_prefix[idx], self.y_aux[idx], self.y_target[idx]


class KANLayer(nn.Module):
    """Lightweight KAN-style layer using Gaussian basis expansion per feature."""

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 8, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.log_bw = nn.Parameter(torch.zeros(in_dim, 1))

        self.base = nn.Linear(in_dim, out_dim)
        self.spline = nn.Linear(in_dim * grid_size, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        x_exp = x.unsqueeze(-1)  # [B, in_dim, 1]
        bw = torch.exp(self.log_bw).unsqueeze(0) + 1e-6  # [1, in_dim, 1]
        basis = torch.exp(-((x_exp - self.grid.view(1, 1, -1)) ** 2) / (2.0 * bw**2))
        basis_flat = basis.reshape(x.shape[0], self.in_dim * self.grid_size)
        return self.base(x) + self.spline(basis_flat)


class KANFusion(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, grid_size: int, dropout: float):
        super().__init__()
        self.kan1 = KANLayer(in_dim, hidden_dim, grid_size=grid_size)
        self.kan2 = KANLayer(hidden_dim, out_dim, grid_size=grid_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.kan1(x))
        x = self.drop(x)
        x = F.silu(self.kan2(x))
        return x


class MultiModalMambaKANDecoder(nn.Module):
    """Prefix-to-future decoder: (signal prefix + subjective ratings) -> future multi-channel signal."""

    def __init__(
        self,
        in_channels: int,
        aux_dim: int,
        out_channels: int,
        out_len: int,
        d_model: int,
        d_state: int,
        headdim: int,
        use_mimo: bool,
        mimo_rank: int,
        n_bi_layers: int,
        chunk_size: int,
        patch_size: int,
        dropout: float,
        kan_hidden: int,
        kan_grid_size: int,
        preconv_kernel: int,
        disable_preconv: bool,
        device,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_len = out_len
        self.patch_size = patch_size
        self.n_bi_layers = int(max(1, n_bi_layers))
        self.disable_preconv = bool(disable_preconv)

        pad = preconv_kernel // 2
        if self.disable_preconv:
            self.pre_conv = None
            self.input_proj = nn.Linear(in_channels, d_model)
        else:
            self.pre_conv = nn.Conv1d(in_channels, d_model, kernel_size=preconv_kernel, padding=pad)
            self.input_proj = None
        self.patch_embed = nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=patch_size)
        self.drop = nn.Dropout(dropout)

        m_args = Mamba3Config(
            d_model=d_model,
            n_layer=1,
            d_state=d_state,
            expand=2,
            headdim=headdim,
            use_mimo=use_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            vocab_size=16,
        )

        self.norms_fwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_fwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])
        self.norms_bwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_bwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])

        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.fusion = KANFusion(3 * d_model, kan_hidden, d_model, grid_size=kan_grid_size, dropout=dropout)
        self.shared = nn.Linear(d_model, d_model)

        # Multi-head output: one regression head per channel.
        self.channel_heads = nn.ModuleList([nn.Linear(d_model, out_len) for _ in range(out_channels)])

    def _pad_tokens_to_chunk(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        t = x.shape[1]
        rem = t % chunk_size
        if rem == 0:
            return x
        pad_t = chunk_size - rem
        return F.pad(x, (0, 0, 0, pad_t, 0, 0), mode="constant", value=0.0)

    def encode_prefix(self, x_prefix: torch.Tensor) -> torch.Tensor:
        # x_prefix: [B, T, C]
        if self.disable_preconv:
            x = F.silu(self.input_proj(x_prefix)).transpose(1, 2)  # [B, d_model, T]
        else:
            x = x_prefix.transpose(1, 2)  # [B, C, T]
            x = F.silu(self.pre_conv(x))
        x = self.patch_embed(x)  # [B, d_model, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, d_model]
        x = self.drop(x)

        x = self._pad_tokens_to_chunk(x, self.blocks_fwd[0].args.chunk_size)

        # Deep bidirectional stack: forward and backward streams are updated layer by layer.
        h_fwd = x
        h_bwd = torch.flip(x, dims=[1])
        for i in range(self.n_bi_layers):
            y_fwd, _ = self.blocks_fwd[i](self.norms_fwd[i](h_fwd), None)
            h_fwd = h_fwd + y_fwd

            y_bwd, _ = self.blocks_bwd[i](self.norms_bwd[i](h_bwd), None)
            h_bwd = h_bwd + y_bwd

        h_bwd = torch.flip(h_bwd, dims=[1])

        # Pool sequence into forward/backward global context.
        ctx_fwd = h_fwd.mean(dim=1)
        ctx_bwd = h_bwd.mean(dim=1)
        return torch.cat([ctx_fwd, ctx_bwd], dim=-1)

    def forward(self, x_prefix: torch.Tensor, y_aux: torch.Tensor, training: bool = False) -> torch.Tensor:
        seq_ctx = self.encode_prefix(x_prefix)
        aux_ctx = F.silu(self.aux_proj(y_aux))
        fused = torch.cat([seq_ctx, aux_ctx], dim=-1)
        fused = self.fusion(fused)
        fused = F.silu(self.shared(fused))

        per_channel = [head(fused).unsqueeze(-1) for head in self.channel_heads]
        y_hat = torch.cat(per_channel, dim=-1)  # [B, out_len, C]
        return y_hat


class MultiModalMambaKANEncoderMask(nn.Module):
    """Encoder-style masked patch predictor for future-signal reconstruction."""

    def __init__(
        self,
        in_channels: int,
        aux_dim: int,
        out_channels: int,
        out_len: int,
        d_model: int,
        d_state: int,
        headdim: int,
        use_mimo: bool,
        mimo_rank: int,
        n_bi_layers: int,
        chunk_size: int,
        patch_size: int,
        dropout: float,
        preconv_kernel: int,
        disable_preconv: bool,
        encoder_random_mask_ratio: float,
        device,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_len = out_len
        self.patch_size = patch_size
        self.n_bi_layers = int(max(1, n_bi_layers))
        self.disable_preconv = bool(disable_preconv)
        self.encoder_random_mask_ratio = float(max(0.0, min(1.0, encoder_random_mask_ratio)))

        pad = preconv_kernel // 2
        if self.disable_preconv:
            self.pre_conv = None
            self.input_proj = nn.Linear(in_channels, d_model)
        else:
            self.pre_conv = nn.Conv1d(in_channels, d_model, kernel_size=preconv_kernel, padding=pad)
            self.input_proj = None
        self.patch_embed = nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=patch_size)
        self.drop = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        m_args = Mamba3Config(
            d_model=d_model,
            n_layer=1,
            d_state=d_state,
            expand=2,
            headdim=headdim,
            use_mimo=use_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            vocab_size=16,
        )

        self.norms_fwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_fwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])
        self.norms_bwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_bwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])

        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.patch_out = nn.Linear(2 * d_model, out_channels * patch_size)

    def _pad_tokens_to_chunk(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        t = x.shape[1]
        rem = t % chunk_size
        if rem == 0:
            return x
        pad_t = chunk_size - rem
        return F.pad(x, (0, 0, 0, pad_t, 0, 0), mode="constant", value=0.0)

    def _future_patch_mask(self, n_tokens: int, prefix_len: int, total_len: int, device) -> torch.Tensor:
        starts = torch.arange(n_tokens, device=device) * self.patch_size
        ends = torch.minimum(starts + self.patch_size, torch.tensor(total_len, device=device))
        return starts >= int(prefix_len)

    def _build_mask(self, n_tokens: int, prefix_len: int, total_len: int, batch: int, device, training: bool) -> torch.Tensor:
        base = self._future_patch_mask(n_tokens, prefix_len, total_len, device=device).unsqueeze(0).repeat(batch, 1)
        if (not training) or self.encoder_random_mask_ratio <= 0.0:
            return base
        random_mask = torch.rand(batch, n_tokens, device=device) < self.encoder_random_mask_ratio
        return base | random_mask

    def _fold_patches(self, patch_pred: torch.Tensor, total_len: int) -> torch.Tensor:
        # patch_pred: [B, n_tokens, C*P]
        b, n, _ = patch_pred.shape
        y = patch_pred.view(b, n, self.out_channels, self.patch_size)
        y = y.reshape(b, n * self.patch_size, self.out_channels)
        return y[:, :total_len, :]

    def forward(self, x_prefix: torch.Tensor, y_aux: torch.Tensor, training: bool = False) -> torch.Tensor:
        # Build full timeline with unknown future slots as zeros; future patches are masked.
        b, t_prefix, c = x_prefix.shape
        t_total = t_prefix + self.out_len
        x_full = torch.zeros(b, t_total, c, device=x_prefix.device, dtype=x_prefix.dtype)
        x_full[:, :t_prefix, :] = x_prefix

        if self.disable_preconv:
            x = F.silu(self.input_proj(x_full)).transpose(1, 2)
        else:
            x = x_full.transpose(1, 2)
            x = F.silu(self.pre_conv(x))
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.drop(x)

        n_tokens = x.shape[1]
        mask = self._build_mask(n_tokens, t_prefix, t_total, b, x.device, training=training)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)

        aux_bias = F.silu(self.aux_proj(y_aux)).unsqueeze(1)
        x = x + aux_bias

        x = self._pad_tokens_to_chunk(x, self.blocks_fwd[0].args.chunk_size)

        h_fwd = x
        h_bwd = torch.flip(x, dims=[1])
        for i in range(self.n_bi_layers):
            y_fwd, _ = self.blocks_fwd[i](self.norms_fwd[i](h_fwd), None)
            h_fwd = h_fwd + y_fwd

            y_bwd, _ = self.blocks_bwd[i](self.norms_bwd[i](h_bwd), None)
            h_bwd = h_bwd + y_bwd

        h_bwd = torch.flip(h_bwd, dims=[1])
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = h[:, :n_tokens, :]

        patch_pred = self.patch_out(h)
        y_full = self._fold_patches(patch_pred, t_total)
        return y_full[:, t_prefix:, :]


class MultiModalMambaKANEncoderMaskOnly(nn.Module):
    """Pure mask modeling: random patch masking and full-sequence reconstruction."""

    def __init__(
        self,
        in_channels: int,
        aux_dim: int,
        out_channels: int,
        seq_len: int,
        d_model: int,
        d_state: int,
        headdim: int,
        use_mimo: bool,
        mimo_rank: int,
        n_bi_layers: int,
        chunk_size: int,
        patch_size: int,
        dropout: float,
        preconv_kernel: int,
        disable_preconv: bool,
        encoder_random_mask_ratio: float,
        encoder_eval_mask_ratio: float,
        mask_observed_residual: bool,
        device,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_bi_layers = int(max(1, n_bi_layers))
        self.disable_preconv = bool(disable_preconv)
        self.encoder_random_mask_ratio = float(max(0.0, min(1.0, encoder_random_mask_ratio)))
        self.encoder_eval_mask_ratio = float(max(0.0, min(1.0, encoder_eval_mask_ratio)))
        self.mask_observed_residual = bool(mask_observed_residual)
        self.last_time_mask = None

        pad = preconv_kernel // 2
        if self.disable_preconv:
            self.pre_conv = None
            self.input_proj = nn.Linear(in_channels, d_model)
        else:
            self.pre_conv = nn.Conv1d(in_channels, d_model, kernel_size=preconv_kernel, padding=pad)
            self.input_proj = None
        self.patch_embed = nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=patch_size)
        self.drop = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        m_args = Mamba3Config(
            d_model=d_model,
            n_layer=1,
            d_state=d_state,
            expand=2,
            headdim=headdim,
            use_mimo=use_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            vocab_size=16,
        )

        self.norms_fwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_fwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])
        self.norms_bwd = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(self.n_bi_layers)])
        self.blocks_bwd = nn.ModuleList([Mamba3(m_args, device=device) for _ in range(self.n_bi_layers)])

        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.patch_out = nn.Linear(2 * d_model, out_channels * patch_size)

    def _pad_tokens_to_chunk(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        t = x.shape[1]
        rem = t % chunk_size
        if rem == 0:
            return x
        pad_t = chunk_size - rem
        return F.pad(x, (0, 0, 0, pad_t, 0, 0), mode="constant", value=0.0)

    def _build_mask(self, batch: int, n_tokens: int, device, training: bool) -> torch.Tensor:
        ratio = self.encoder_random_mask_ratio if training else self.encoder_eval_mask_ratio
        if ratio <= 0.0:
            return torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)
        return torch.rand(batch, n_tokens, device=device) < ratio

    def _fold_patches(self, patch_pred: torch.Tensor, total_len: int) -> torch.Tensor:
        b, n, _ = patch_pred.shape
        y = patch_pred.view(b, n, self.out_channels, self.patch_size)
        y = y.reshape(b, n * self.patch_size, self.out_channels)
        return y[:, :total_len, :]

    def forward(self, x_seq: torch.Tensor, y_aux: torch.Tensor, training: bool = False) -> torch.Tensor:
        b, t, _ = x_seq.shape
        if self.disable_preconv:
            x = F.silu(self.input_proj(x_seq)).transpose(1, 2)
        else:
            x = x_seq.transpose(1, 2)
            x = F.silu(self.pre_conv(x))
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.drop(x)

        n_tokens = x.shape[1]
        mask = self._build_mask(b, n_tokens, x.device, training=training)
        time_mask = mask.unsqueeze(-1).repeat_interleave(self.patch_size, dim=1)[:, :t, :]
        self.last_time_mask = time_mask
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)

        aux_bias = F.silu(self.aux_proj(y_aux)).unsqueeze(1)
        x = x + aux_bias

        x = self._pad_tokens_to_chunk(x, self.blocks_fwd[0].args.chunk_size)

        h_fwd = x
        h_bwd = torch.flip(x, dims=[1])
        for i in range(self.n_bi_layers):
            y_fwd, _ = self.blocks_fwd[i](self.norms_fwd[i](h_fwd), None)
            h_fwd = h_fwd + y_fwd

            y_bwd, _ = self.blocks_bwd[i](self.norms_bwd[i](h_bwd), None)
            h_bwd = h_bwd + y_bwd

        h_bwd = torch.flip(h_bwd, dims=[1])
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = h[:, :n_tokens, :]

        patch_pred = self.patch_out(h)
        y_full = self._fold_patches(patch_pred, t)
        if self.mask_observed_residual and self.last_time_mask is not None:
            m = self.last_time_mask.to(dtype=y_full.dtype)
            y_full = y_full * m + x_seq * (1.0 - m)
        return y_full


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_headdim(d_model: int, requested_headdim: int) -> int:
    d_inner = 2 * int(d_model)
    req = max(1, int(requested_headdim))
    if d_inner % req == 0:
        return req
    for cand in (32, 16, 8, 4, 2, 1):
        if cand <= d_inner and d_inner % cand == 0:
            print(f"[warn] requested headdim={req} invalid for d_inner={d_inner}; auto-adjust to {cand}")
            return cand
    return 1


def resolve_mimo_mode(device: torch.device, use_mimo_cfg: str, min_vram_gb_for_mimo: float) -> tuple[bool, float | None]:
    mode = str(use_mimo_cfg).strip().lower()
    if mode not in {"auto", "true", "false"}:
        raise ValueError(f"use_mimo must be one of auto/true/false, got: {use_mimo_cfg}")

    gpu_total_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu_total_gb = props.total_memory / (1024**3)

    if mode == "true":
        if device.type != "cuda":
            print("[warn] use_mimo=true requested but CUDA is not active; fallback to use_mimo=false")
            return False, gpu_total_gb
        return True, gpu_total_gb

    if mode == "false":
        return False, gpu_total_gb

    # auto: enable only when CUDA VRAM is sufficient
    if gpu_total_gb is None:
        return False, gpu_total_gb
    return bool(gpu_total_gb >= float(min_vram_gb_for_mimo)), gpu_total_gb


def resolve_data_path(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    candidates = [Path.cwd() / p, PROJECT_ROOT / p, WORKSPACE_ROOT / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Data not found: {data_arg}")


def resolve_existing_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute() and p.exists():
        return p
    candidates = [Path.cwd() / p, PROJECT_ROOT / p, WORKSPACE_ROOT / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"File not found: {path_arg}")


def warm_start_from_checkpoint(model: nn.Module, ckpt_path: str, load_mode: str = "backbone") -> dict:
    path = resolve_existing_path(ckpt_path)
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob.keys()):
        state = blob
    elif isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        state = blob["state_dict"]
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    allow_prefix = (
        "pre_conv.",
        "patch_embed.",
        "norms_fwd.",
        "blocks_fwd.",
        "norms_bwd.",
        "blocks_bwd.",
        "aux_proj.",
        "mask_token",
    )

    curr = model.state_dict()
    loaded = 0
    skipped_shape = 0
    skipped_mode = 0

    for k, v in state.items():
        if k not in curr:
            continue
        if load_mode == "backbone" and not any(k.startswith(p) for p in allow_prefix):
            skipped_mode += 1
            continue
        if curr[k].shape != v.shape:
            skipped_shape += 1
            continue
        curr[k] = v
        loaded += 1

    model.load_state_dict(curr, strict=False)
    info = {
        "checkpoint": str(path),
        "load_mode": str(load_mode),
        "loaded_params": int(loaded),
        "skipped_by_shape": int(skipped_shape),
        "skipped_by_mode": int(skipped_mode),
    }
    print(f"[warm-start] {info}")
    return info


def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires pyyaml") from exc

    p = Path(config_path)
    candidates = [p, Path.cwd() / p, PROJECT_ROOT / p, WORKSPACE_ROOT / p]
    target = None
    for c in candidates:
        if c.exists():
            target = c
            break
    if target is None:
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(target, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def coerce_config_types(cfg: dict) -> dict:
    int_keys = {
        "max_subjects",
        "seed",
        "epochs",
        "batch_size",
        "d_model",
        "d_state",
        "headdim",
        "n_bi_layers",
        "mimo_rank",
        "chunk_size",
        "patch_size",
        "kan_hidden",
        "kan_grid_size",
        "preconv_kernel",
        "n_missing_random",
        "robustness_repeats",
    }
    float_keys = {
        "input_ratio",
        "dropout",
        "lr",
        "min_lr",
        "weight_decay",
        "noise_std",
        "grad_clip",
        "test_size",
        "val_size",
        "min_vram_gb_for_mimo",
        "huber_delta",
        "channel_weight_min",
        "channel_weight_max",
        "encoder_random_mask_ratio",
        "encoder_eval_mask_ratio",
        "mask_visible_loss_weight",
    }
    str_keys = {
        "config",
        "device",
        "data",
        "outdir",
        "selected_channels_1based",
        "use_mimo",
        "loss_type",
        "selection_metric",
        "prediction_mode",
        "init_from_checkpoint",
        "init_load_mode",
    }
    bool_keys = {"group_split", "subject_norm", "enforce_single_subject", "use_channel_weight"}
    bool_keys.add("use_last_step_residual")
    bool_keys.add("mask_loss_on_masked_only")
    bool_keys.add("mask_observed_residual")
    bool_keys.add("disable_preconv")

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
        raise ValueError(f"Cannot parse bool: {v}")

    out = dict(cfg)
    for k, v in list(out.items()):
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
        elif k == "random_missing_list":
            out[k] = [int(x) for x in v]
    return out


def parse_channels(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        ch = [int(x) for x in value]
    else:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            ch = [int(x.strip()) for x in s[1:-1].split(",") if x.strip()]
        else:
            ch = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not ch:
        raise ValueError("selected_channels_1based is empty")
    return ch


def load_deap_dat_forecast(path: Path, selected_channels_1based: list[int]):
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    data = np.asarray(obj["data"], dtype=np.float64)  # [40, 40, 8064]
    labels = np.asarray(obj["labels"], dtype=np.float64)  # [40, 4]

    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    data = np.clip(data, -1e6, 1e6)
    labels = np.nan_to_num(labels, nan=5.0, posinf=9.0, neginf=1.0)
    labels = np.clip(labels, 1.0, 9.0)

    selected = [c - 1 for c in selected_channels_1based]
    x = data[:, selected, :]  # [trials, C, 8064]
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # [trials, 8064, C]
    y_aux = labels.astype(np.float32)  # all subjective ratings [trials, 4]
    return x, y_aux


def load_deap_dataset(data_arg: str, selected_channels_1based: list[int], max_subjects: int = 0):
    p = resolve_data_path(data_arg)

    if p.is_file():
        x, y_aux = load_deap_dat_forecast(p, selected_channels_1based)
        sid = p.stem
        subject_ids = [sid for _ in range(x.shape[0])]
        return x, y_aux, subject_ids

    if p.is_dir():
        files = sorted(p.glob("s*.dat"))
        if max_subjects > 0:
            files = files[:max_subjects]
        if not files:
            raise FileNotFoundError(f"No s*.dat files under {p}")

        x_list = []
        y_list = []
        sids = []
        for f in files:
            x, y_aux = load_deap_dat_forecast(f, selected_channels_1based)
            x_list.append(x)
            y_list.append(y_aux)
            sids.extend([f.stem for _ in range(x.shape[0])])

        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0), sids

    raise ValueError(f"Unsupported data path: {p}")


def split_prefix_suffix(x_full: np.ndarray, input_ratio: float):
    t = x_full.shape[1]
    split_idx = int(t * input_ratio)
    split_idx = max(1, min(split_idx, t - 1))
    x_prefix = x_full[:, :split_idx, :]
    y_future = x_full[:, split_idx:, :]
    return x_prefix, y_future, split_idx


def apply_subject_wise_norm(x: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for sid in np.unique(subject_ids):
        idx = np.where(subject_ids == sid)[0]
        part = x[idx]
        mean = part.mean(axis=(0, 1), keepdims=True)
        std = part.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        out[idx] = ((part - mean) / std).astype(np.float32)
    return out


def run_epoch(model, loader, optimizer, device, noise_std: float, grad_clip: float, use_last_step_residual: bool):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    for xb, yb_aux, yb in loader:
        xb = xb.to(device)
        yb_aux = yb_aux.to(device)
        yb = yb.to(device)

        if noise_std > 0:
            xb = xb + noise_std * torch.randn_like(xb)

        optimizer.zero_grad()
        pred = decode_with_last_step_residual(model, xb, yb_aux, use_last_step_residual, training=True)
        loss = loss_fn(pred, yb)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def get_model_time_mask(model, pred: torch.Tensor) -> torch.Tensor | None:
    tm = getattr(model, "last_time_mask", None)
    if tm is None:
        return None
    if tm.ndim == 2:
        tm = tm.unsqueeze(-1)
    if tm.shape[0] != pred.shape[0] or tm.shape[1] != pred.shape[1]:
        return None
    return tm.to(device=pred.device)


def reduce_weighted_loss(
    loss_raw: torch.Tensor,
    channel_weight: torch.Tensor | None,
    time_mask: torch.Tensor | None,
    mask_loss_on_masked_only: bool,
    mask_visible_loss_weight: float,
) -> torch.Tensor:
    weight = torch.ones_like(loss_raw)
    if channel_weight is not None:
        weight = weight * channel_weight

    if mask_loss_on_masked_only and time_mask is not None:
        tm = time_mask.to(dtype=loss_raw.dtype)
        tm = tm + float(mask_visible_loss_weight) * (1.0 - tm)
        weight = weight * tm

    denom = weight.sum().clamp_min(1e-6)
    return (loss_raw * weight).sum() / denom


def decode_with_last_step_residual(
    model,
    xb: torch.Tensor,
    yb_aux: torch.Tensor,
    use_last_step_residual: bool,
    training: bool = False,
) -> torch.Tensor:
    pred = model(xb, yb_aux, training=training)
    if use_last_step_residual:
        base = xb[:, -1:, :].expand(-1, pred.shape[1], -1)
        pred = pred + base
    return pred


def build_loss_function(loss_type: str, huber_delta: float):
    lt = str(loss_type).strip().lower()
    if lt == "mse":
        return nn.MSELoss(reduction="none")
    if lt == "huber":
        return nn.HuberLoss(delta=float(huber_delta), reduction="none")
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def run_epoch_weighted(
    model,
    loader,
    optimizer,
    device,
    noise_std: float,
    grad_clip: float,
    loss_fn,
    channel_weight: torch.Tensor,
    use_last_step_residual: bool,
    mask_loss_on_masked_only: bool,
    mask_visible_loss_weight: float,
):
    model.train()
    losses = []
    cw = channel_weight.to(device).view(1, 1, -1)
    for xb, yb_aux, yb in loader:
        xb = xb.to(device)
        yb_aux = yb_aux.to(device)
        yb = yb.to(device)

        if noise_std > 0:
            xb = xb + noise_std * torch.randn_like(xb)

        optimizer.zero_grad()
        pred = decode_with_last_step_residual(model, xb, yb_aux, use_last_step_residual, training=True)
        loss_raw = loss_fn(pred, yb)
        time_mask = get_model_time_mask(model, pred)
        loss = reduce_weighted_loss(
            loss_raw=loss_raw,
            channel_weight=cw,
            time_mask=time_mask,
            mask_loss_on_masked_only=mask_loss_on_masked_only,
            mask_visible_loss_weight=mask_visible_loss_weight,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def eval_metrics(model, loader, device, y_scaler: SignalScaler, use_last_step_residual: bool):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb_aux, yb in loader:
            xb = xb.to(device)
            yb_aux = yb_aux.to(device)
            pred = decode_with_last_step_residual(model, xb, yb_aux, use_last_step_residual, training=False).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    y_pred_scaled = np.concatenate(preds, axis=0)
    y_true_scaled = np.concatenate(trues, axis=0)

    # Primary metrics should be computed in standardized space to avoid domination
    # from high-amplitude channels in raw scale.
    mse = float(mean_squared_error(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1)))
    mae = float(mean_absolute_error(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1)))
    r2 = float(r2_score(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1)))

    c = y_true_scaled.shape[-1]
    per_channel_mse = {}
    for i in range(c):
        per_channel_mse[f"ch_{i}"] = float(
            mean_squared_error(y_true_scaled[:, :, i].reshape(-1), y_pred_scaled[:, :, i].reshape(-1))
        )

    # Keep raw-scale metrics as secondary references for engineering inspection.
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)
    raw_mse = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    raw_mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
    raw_r2 = float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))

    return {
        "metric_space": "standardized",
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "per_channel_mse": per_channel_mse,
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "raw_r2": raw_r2,
    }


def eval_with_channel_mask(
    model,
    loader,
    device,
    y_scaler: SignalScaler,
    missing_channels: list[int],
    use_last_step_residual: bool,
):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb_aux, yb in loader:
            xb = xb.clone()
            if missing_channels:
                xb[:, :, missing_channels] = 0.0
            xb = xb.to(device)
            yb_aux = yb_aux.to(device)
            pred = decode_with_last_step_residual(model, xb, yb_aux, use_last_step_residual, training=False).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    y_pred_scaled = np.concatenate(preds, axis=0)
    y_true_scaled = np.concatenate(trues, axis=0)

    # Robustness comparison uses standardized-space errors for fairness across channels.
    mse = float(mean_squared_error(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1)))
    mae = float(mean_absolute_error(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1)))
    return mse, mae


def robustness_report(
    model,
    test_loader,
    device,
    y_scaler,
    n_channels,
    random_missing_list,
    n_missing_random,
    repeats,
    use_last_step_residual: bool,
):
    baseline = eval_metrics(model, test_loader, device, y_scaler, use_last_step_residual)
    base_mse = baseline["mse"]

    loo = []
    for c in range(n_channels):
        mse, mae = eval_with_channel_mask(model, test_loader, device, y_scaler, [c], use_last_step_residual)
        loo.append({
            "missing_channel": int(c),
            "mse": mse,
            "mae": mae,
            "delta_mse_vs_base": float(mse - base_mse),
        })

    random_missing = {}
    for k in random_missing_list:
        vals = []
        kk = min(max(1, int(k)), n_channels)
        for _ in range(repeats):
            missing = sorted(random.sample(range(n_channels), kk))
            mse, mae = eval_with_channel_mask(model, test_loader, device, y_scaler, missing, use_last_step_residual)
            vals.append({"missing_channels": missing, "mse": mse, "mae": mae})
        random_missing[f"k_{kk}"] = {
            "mean_mse": float(np.mean([v["mse"] for v in vals])),
            "std_mse": float(np.std([v["mse"] for v in vals])),
            "samples": vals,
        }

    return {
        "baseline": baseline,
        "leave_one_channel_out": loo,
        "random_missing": random_missing,
        "n_missing_random": int(n_missing_random),
    }


def main():
    parser = argparse.ArgumentParser(description="DEAP multimodal forecasting/mask modeling")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--prediction_mode", type=str, default="decoder", choices=["decoder", "encoder_mask", "encoder_mask_only"])
    parser.add_argument("--init_from_checkpoint", type=str, default="")
    parser.add_argument("--init_load_mode", type=str, default="backbone", choices=["backbone", "all"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data", type=str, default="data/s01.dat")
    parser.add_argument("--selected_channels_1based", type=str, default="8,10,15,21,22,23,24,26,27,28,31,35,36,37,38,39,40")
    parser.add_argument("--max_subjects", type=int, default=1)
    parser.add_argument("--group_split", type=str, default="false")
    parser.add_argument("--subject_norm", type=str, default="false")
    parser.add_argument("--enforce_single_subject", type=str, default="true")
    parser.add_argument("--input_ratio", type=float, default=0.8)
    parser.add_argument("--outdir", type=str, default="outputs/deap_multimodal_decoder")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--n_bi_layers", type=int, default=1)
    parser.add_argument("--use_mimo", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--mimo_rank", type=int, default=2)
    parser.add_argument("--min_vram_gb_for_mimo", type=float, default=10.0)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--preconv_kernel", type=int, default=7)
    parser.add_argument("--disable_preconv", type=str, default="false")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--kan_hidden", type=int, default=64)
    parser.add_argument("--kan_grid_size", type=int, default=8)
    parser.add_argument("--encoder_random_mask_ratio", type=float, default=0.0)
    parser.add_argument("--encoder_eval_mask_ratio", type=float, default=0.0)
    parser.add_argument("--mask_loss_on_masked_only", type=str, default="false")
    parser.add_argument("--mask_visible_loss_weight", type=float, default=0.05)
    parser.add_argument("--mask_observed_residual", type=str, default="true")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--noise_std", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--use_channel_weight", type=str, default="true")
    parser.add_argument("--channel_weight_min", type=float, default=0.2)
    parser.add_argument("--channel_weight_max", type=float, default=5.0)
    parser.add_argument("--use_last_step_residual", type=str, default="true")
    parser.add_argument("--selection_metric", type=str, default="val_r2", choices=["val_loss", "val_r2"])

    parser.add_argument("--n_missing_random", type=int, default=2)
    parser.add_argument("--random_missing_list", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--robustness_repeats", type=int, default=8)

    args = parser.parse_args()
    cli_keys = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            k = tok[2:].split("=", 1)[0].strip()
            if k:
                cli_keys.add(k)

    if args.config:
        cfg = coerce_config_types(load_yaml_config(args.config))
        for k, v in cfg.items():
            if hasattr(args, k) and k not in cli_keys:
                setattr(args, k, v)

    if isinstance(args.group_split, str):
        args.group_split = args.group_split.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.subject_norm, str):
        args.subject_norm = args.subject_norm.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.enforce_single_subject, str):
        args.enforce_single_subject = args.enforce_single_subject.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.use_channel_weight, str):
        args.use_channel_weight = args.use_channel_weight.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.use_last_step_residual, str):
        args.use_last_step_residual = args.use_last_step_residual.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.mask_loss_on_masked_only, str):
        args.mask_loss_on_masked_only = args.mask_loss_on_masked_only.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.mask_observed_residual, str):
        args.mask_observed_residual = args.mask_observed_residual.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(args.disable_preconv, str):
        args.disable_preconv = args.disable_preconv.strip().lower() in {"1", "true", "yes", "y", "on"}

    if (
        args.prediction_mode == "encoder_mask_only"
        and args.mask_observed_residual
        and float(args.encoder_eval_mask_ratio) <= 0.0
    ):
        fallback = max(0.05, float(args.encoder_random_mask_ratio))
        args.encoder_eval_mask_ratio = fallback
        print(
            "[warn] encoder_eval_mask_ratio<=0 with mask_observed_residual=true may cause trivial perfect scores; "
            f"auto-set encoder_eval_mask_ratio={fallback:.3f}"
        )

    set_seed(args.seed)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    channels = parse_channels(args.selected_channels_1based)
    args.headdim = resolve_headdim(args.d_model, args.headdim)
    use_mimo, gpu_total_gb = resolve_mimo_mode(device, args.use_mimo, args.min_vram_gb_for_mimo)
    print(
        f"MIMO setting: cfg={args.use_mimo}, enabled={use_mimo}, "
        f"mimo_rank={args.mimo_rank}, gpu_total_gb={gpu_total_gb}"
    )

    x_full, y_aux, subject_ids = load_deap_dataset(args.data, channels, max_subjects=args.max_subjects)
    if args.prediction_mode == "encoder_mask_only":
        x_in = x_full
        y_target = x_full
        split_idx = int(x_full.shape[1])
    else:
        x_in, y_target, split_idx = split_prefix_suffix(x_full, args.input_ratio)
    subject_ids = np.asarray(subject_ids)
    n_subjects = len(np.unique(subject_ids))
    if args.enforce_single_subject and n_subjects != 1:
        raise ValueError(
            f"enforce_single_subject=true but found {n_subjects} subjects. "
            f"Set max_subjects=1 or enforce_single_subject=false explicitly."
        )

    if args.group_split and len(np.unique(subject_ids)) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(gss.split(x_in, y_aux, groups=subject_ids))
        x_train, x_test = x_in[tr_idx], x_in[te_idx]
        y_aux_train, y_aux_test = y_aux[tr_idx], y_aux[te_idx]
        y_train, y_test = y_target[tr_idx], y_target[te_idx]
        sid_train = subject_ids[tr_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        tr2_idx, va_idx = next(gss2.split(x_train, y_aux_train, groups=sid_train))
        x_train_core, x_val = x_train[tr2_idx], x_train[va_idx]
        y_aux_train_core, y_aux_val = y_aux_train[tr2_idx], y_aux_train[va_idx]
        y_train_core, y_val = y_train[tr2_idx], y_train[va_idx]
        sid_train_core = sid_train[tr2_idx]
        sid_val = sid_train[va_idx]
    else:
        x_train, x_test, y_aux_train, y_aux_test, y_train, y_test, sid_train, sid_test = train_test_split(
            x_in,
            y_aux,
            y_target,
            subject_ids,
            test_size=args.test_size,
            random_state=args.seed,
        )
        x_train_core, x_val, y_aux_train_core, y_aux_val, y_train_core, y_val, sid_train_core, sid_val = train_test_split(
            x_train,
            y_aux_train,
            y_train,
            sid_train,
            test_size=args.val_size,
            random_state=args.seed,
        )

    if args.subject_norm:
        x_train_core = apply_subject_wise_norm(x_train_core, sid_train_core)
        x_val = apply_subject_wise_norm(x_val, sid_val)
        x_test_norm = apply_subject_wise_norm(x_test, sid_test if not args.group_split else subject_ids[te_idx])
    else:
        x_scaler = SignalScaler()
        x_scaler.fit(x_train_core)
        x_train_core = x_scaler.transform(x_train_core)
        x_val = x_scaler.transform(x_val)
        x_test_norm = x_scaler.transform(x_test)

    y_scaler = SignalScaler()
    y_scaler.fit(y_train_core)
    y_train_scaled = y_scaler.transform(y_train_core)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    aux_scaler = LabelScaler()
    aux_scaler.fit(y_aux_train_core)
    y_aux_train_scaled = aux_scaler.transform(y_aux_train_core)
    y_aux_val_scaled = aux_scaler.transform(y_aux_val)
    y_aux_test_scaled = aux_scaler.transform(y_aux_test)

    train_ds = ForecastDataset(x_train_core.astype(np.float32), y_aux_train_scaled.astype(np.float32), y_train_scaled.astype(np.float32))
    val_ds = ForecastDataset(x_val.astype(np.float32), y_aux_val_scaled.astype(np.float32), y_val_scaled.astype(np.float32))
    test_ds = ForecastDataset(x_test_norm.astype(np.float32), y_aux_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    aux_dim = int(y_aux_train_scaled.shape[1])

    if args.prediction_mode == "encoder_mask_only" and args.use_last_step_residual:
        print("[warn] encoder_mask_only mode ignores temporal residual; force use_last_step_residual=false")
        args.use_last_step_residual = False

    if args.prediction_mode == "encoder_mask":
        model = MultiModalMambaKANEncoderMask(
            in_channels=len(channels),
            aux_dim=aux_dim,
            out_channels=len(channels),
            out_len=y_train_scaled.shape[1],
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            use_mimo=use_mimo,
            mimo_rank=args.mimo_rank,
            n_bi_layers=args.n_bi_layers,
            chunk_size=args.chunk_size,
            patch_size=args.patch_size,
            dropout=args.dropout,
            preconv_kernel=args.preconv_kernel,
            disable_preconv=args.disable_preconv,
            encoder_random_mask_ratio=args.encoder_random_mask_ratio,
            device=device,
        ).to(device)
    elif args.prediction_mode == "encoder_mask_only":
        model = MultiModalMambaKANEncoderMaskOnly(
            in_channels=len(channels),
            aux_dim=aux_dim,
            out_channels=len(channels),
            seq_len=y_train_scaled.shape[1],
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            use_mimo=use_mimo,
            mimo_rank=args.mimo_rank,
            n_bi_layers=args.n_bi_layers,
            chunk_size=args.chunk_size,
            patch_size=args.patch_size,
            dropout=args.dropout,
            preconv_kernel=args.preconv_kernel,
            disable_preconv=args.disable_preconv,
            encoder_random_mask_ratio=args.encoder_random_mask_ratio,
            encoder_eval_mask_ratio=args.encoder_eval_mask_ratio,
            mask_observed_residual=args.mask_observed_residual,
            device=device,
        ).to(device)
    else:
        model = MultiModalMambaKANDecoder(
            in_channels=len(channels),
            aux_dim=aux_dim,
            out_channels=len(channels),
            out_len=y_train_scaled.shape[1],
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            use_mimo=use_mimo,
            mimo_rank=args.mimo_rank,
            n_bi_layers=args.n_bi_layers,
            chunk_size=args.chunk_size,
            patch_size=args.patch_size,
            dropout=args.dropout,
            kan_hidden=args.kan_hidden,
            kan_grid_size=args.kan_grid_size,
            preconv_kernel=args.preconv_kernel,
            disable_preconv=args.disable_preconv,
            device=device,
        ).to(device)

    warm_start_info = None
    if str(args.init_from_checkpoint).strip():
        warm_start_info = warm_start_from_checkpoint(
            model=model,
            ckpt_path=str(args.init_from_checkpoint).strip(),
            load_mode=str(args.init_load_mode).strip().lower(),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=args.min_lr,
    )

    if args.selection_metric == "val_r2":
        best_val = float("-inf")
    else:
        best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": [], "lr": []}

    loss_elem_fn = build_loss_function(args.loss_type, args.huber_delta)
    # Weight channels by inverse variance in train targets to reduce large-channel domination.
    y_train_std = y_train_core.std(axis=(0, 1)).astype(np.float32)
    y_train_std = np.where(y_train_std < 1e-8, 1.0, y_train_std)
    w = 1.0 / (y_train_std**2)
    w = w / np.mean(w)
    if args.channel_weight_min > args.channel_weight_max:
        raise ValueError("channel_weight_min must be <= channel_weight_max")
    w = np.clip(w, args.channel_weight_min, args.channel_weight_max)
    w = w / np.mean(w)
    channel_weight = torch.tensor(w, dtype=torch.float32)

    for ep in range(1, args.epochs + 1):
        if args.use_channel_weight:
            tr_loss = run_epoch_weighted(
                model,
                train_loader,
                optimizer,
                device,
                args.noise_std,
                args.grad_clip,
                loss_elem_fn,
                channel_weight,
                args.use_last_step_residual,
                args.mask_loss_on_masked_only,
                args.mask_visible_loss_weight,
            )
        else:
            tr_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.noise_std,
                args.grad_clip,
                args.use_last_step_residual,
            )

        model.eval()
        va_losses = []
        va_preds = []
        va_trues = []
        with torch.no_grad():
            for xb, yb_aux, yb in val_loader:
                xb = xb.to(device)
                yb_aux = yb_aux.to(device)
                yb = yb.to(device)
                pred = decode_with_last_step_residual(model, xb, yb_aux, args.use_last_step_residual, training=False)
                va_preds.append(pred.detach().cpu())
                va_trues.append(yb.detach().cpu())
                time_mask = get_model_time_mask(model, pred)
                if args.use_channel_weight:
                    cw = channel_weight.to(device).view(1, 1, -1)
                    va_losses.append(
                        reduce_weighted_loss(
                            loss_raw=loss_elem_fn(pred, yb),
                            channel_weight=cw,
                            time_mask=time_mask,
                            mask_loss_on_masked_only=args.mask_loss_on_masked_only,
                            mask_visible_loss_weight=args.mask_visible_loss_weight,
                        ).item()
                    )
                else:
                    va_losses.append(
                        reduce_weighted_loss(
                            loss_raw=loss_elem_fn(pred, yb),
                            channel_weight=None,
                            time_mask=time_mask,
                            mask_loss_on_masked_only=args.mask_loss_on_masked_only,
                            mask_visible_loss_weight=args.mask_visible_loss_weight,
                        ).item()
                    )

        va_loss = float(np.mean(va_losses))
        y_val_pred = torch.cat(va_preds, dim=0).numpy()
        y_val_true = torch.cat(va_trues, dim=0).numpy()
        va_r2 = float(r2_score(y_val_true.reshape(-1), y_val_pred.reshape(-1)))
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_r2"].append(va_r2)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} "
            f"| val_r2={va_r2:.5f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = (va_r2 > best_val) if args.selection_metric == "val_r2" else (va_loss < best_val)
        if improved:
            best_val = va_r2 if args.selection_metric == "val_r2" else va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep} (patience={args.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    base_metrics = eval_metrics(model, test_loader, device, y_scaler, args.use_last_step_residual)
    robust = robustness_report(
        model,
        test_loader,
        device,
        y_scaler,
        n_channels=len(channels),
        random_missing_list=args.random_missing_list,
        n_missing_random=args.n_missing_random,
        repeats=args.robustness_repeats,
        use_last_step_residual=args.use_last_step_residual,
    )

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "task": "multimodal_signal_modeling",
        "prediction_mode": args.prediction_mode,
        "data_path": str(resolve_data_path(args.data)),
        "n_samples": int(x_full.shape[0]),
        "n_subjects": int(len(np.unique(subject_ids))),
        "selected_channels_1based": channels,
        "signal_length": int(x_full.shape[1]),
        "input_ratio": float(args.input_ratio),
        "split_index": int(split_idx),
        "prefix_len": int(x_in.shape[1] if args.prediction_mode == "encoder_mask_only" else x_in.shape[1]),
        "future_len": int(0 if args.prediction_mode == "encoder_mask_only" else y_target.shape[1]),
        "model_hyperparams": {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "headdim": args.headdim,
            "n_bi_layers": args.n_bi_layers,
            "use_mimo": use_mimo,
            "mimo_rank": args.mimo_rank,
            "mimo_cfg": args.use_mimo,
            "gpu_total_gb": gpu_total_gb,
            "min_vram_gb_for_mimo": args.min_vram_gb_for_mimo,
            "chunk_size": args.chunk_size,
            "patch_size": args.patch_size,
            "preconv_kernel": args.preconv_kernel,
            "disable_preconv": bool(args.disable_preconv),
            "dropout": args.dropout,
            "kan_hidden": args.kan_hidden,
            "kan_grid_size": args.kan_grid_size,
            "encoder_random_mask_ratio": args.encoder_random_mask_ratio,
            "encoder_eval_mask_ratio": args.encoder_eval_mask_ratio,
            "mask_loss_on_masked_only": bool(args.mask_loss_on_masked_only),
            "mask_visible_loss_weight": args.mask_visible_loss_weight,
            "mask_observed_residual": bool(args.mask_observed_residual),
            "bidirectional_mamba": True,
            "output_head": (
                "multi_head_linear"
                if args.prediction_mode == "decoder"
                else "patch_reconstruction_linear"
            ),
            "aux_input": "all_subjective_ratings",
            "aux_dim": aux_dim,
            "in_channels": len(channels),
            "out_channels": len(channels),
            "init_from_checkpoint": str(args.init_from_checkpoint),
            "init_load_mode": str(args.init_load_mode),
            "warm_start": warm_start_info,
        },
        "optim_hyperparams": {
            "device": str(device),
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "noise_std": args.noise_std,
            "grad_clip": args.grad_clip,
            "loss_type": args.loss_type,
            "huber_delta": args.huber_delta,
            "use_channel_weight": bool(args.use_channel_weight),
            "use_last_step_residual": bool(args.use_last_step_residual),
            "channel_weight_min": args.channel_weight_min,
            "channel_weight_max": args.channel_weight_max,
            "channel_weight": [float(x) for x in w.tolist()],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "selection_metric": args.selection_metric,
            "mask_loss_on_masked_only": bool(args.mask_loss_on_masked_only),
            "mask_visible_loss_weight": args.mask_visible_loss_weight,
            "mask_observed_residual": bool(args.mask_observed_residual),
        },
        "split": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
            "group_split": bool(args.group_split),
            "subject_norm": bool(args.subject_norm),
            "enforce_single_subject": bool(args.enforce_single_subject),
        },
        "metrics": base_metrics,
        "robustness": robust,
        "history": history,
    }

    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    torch.save(best_state if best_state is not None else model.state_dict(), out_dir / "best_model.pt")

    print("\nFinal metrics:")
    print(f"  mse={base_metrics['mse']:.6f}")
    print(f"  mae={base_metrics['mae']:.6f}")
    print(f"  r2={base_metrics['r2']:.6f}")
    print("Saved:")
    print(f"  {out_dir / 'best_model.pt'}")
    print(f"  {out_dir / 'metrics_report.json'}")


if __name__ == "__main__":
    main()
