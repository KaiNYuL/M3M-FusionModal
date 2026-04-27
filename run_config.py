import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from mamba3 import Mamba3Config, Mamba3LMHeadModel, get_device


def _load_config(config_path: Path) -> dict:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config format: {suffix}. Use .json/.yaml/.yml")


def _init_model_parameters(model: Mamba3LMHeadModel, init_cfg: dict) -> None:
    a_log_min = float(init_cfg.get("a_log_min", -4.0))
    a_log_max = float(init_cfg.get("a_log_max", -1.0))
    dt_bias_min = float(init_cfg.get("dt_bias_min", 0.001))
    dt_bias_max = float(init_cfg.get("dt_bias_max", 0.1))
    weight_std = float(init_cfg.get("weight_std", 0.02))
    d_skip_init = float(init_cfg.get("d_skip_init", 1.0))

    for name, p in model.named_parameters():
        if "A_log" in name:
            nn.init.uniform_(p, a_log_min, a_log_max)
        elif "D" in name and p.dim() == 1:
            nn.init.constant_(p, d_skip_init)
        elif "dt_bias" in name:
            nn.init.uniform_(p, dt_bias_min, dt_bias_max)
        elif "B_bias" in name or "C_bias" in name or "mimo" in name:
            # Keep constructor defaults (ones or ones/R) for these terms.
            continue
        elif p.dim() >= 2:
            nn.init.normal_(p, std=weight_std)


def _resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "auto":
        return get_device()
    return torch.device(device_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mamba3-minimal from a JSON/YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tuning.template.json"),
        help="Path to config file (.json/.yaml/.yml)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    runtime_cfg = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})
    init_cfg = cfg.get("init", {})
    exp_cfg = cfg.get("experiment", {})

    seed = int(runtime_cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = _resolve_device(str(runtime_cfg.get("device", "auto")))

    mamba_cfg = Mamba3Config(
        d_model=int(model_cfg.get("d_model", 256)),
        n_layer=int(model_cfg.get("n_layer", 6)),
        d_state=int(model_cfg.get("d_state", 128)),
        expand=int(model_cfg.get("expand", 2)),
        headdim=int(model_cfg.get("headdim", 64)),
        chunk_size=int(model_cfg.get("chunk_size", 64)),
        vocab_size=int(model_cfg.get("vocab_size", 50277)),
        use_mimo=bool(model_cfg.get("use_mimo", False)),
        mimo_rank=int(model_cfg.get("mimo_rank", 4)),
    )

    model = Mamba3LMHeadModel(mamba_cfg, device=device)
    _init_model_parameters(model, init_cfg)
    model.eval()

    batch_size = int(exp_cfg.get("batch_size", 1))
    seq_len = int(exp_cfg.get("seq_len", 128))
    run_generation = bool(exp_cfg.get("run_generation", True))

    # Chunked SSD requires sequence length divisible by chunk_size on training path.
    aligned_seq_len = (seq_len // mamba_cfg.chunk_size) * mamba_cfg.chunk_size
    if aligned_seq_len == 0:
        aligned_seq_len = mamba_cfg.chunk_size

    input_ids = torch.randint(
        0,
        mamba_cfg.vocab_size,
        (batch_size, aligned_seq_len),
        device=device,
    )

    with torch.no_grad():
        logits, _ = model(input_ids)

    n_params = sum(p.numel() for p in model.parameters())
    print("=== mamba3-minimal config run ===")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Params: {n_params:,}")
    print(f"Forward input shape: {tuple(input_ids.shape)}")
    print(f"Forward logits shape: {tuple(logits.shape)}")

    if run_generation:
        prompt = exp_cfg.get("prompt_token_ids", [1, 2, 3, 4])
        prompt_tensor = torch.tensor(prompt, device=device, dtype=torch.long)

        max_new_length = int(exp_cfg.get("max_new_length", 20))
        temperature = float(exp_cfg.get("temperature", 0.8))
        top_k = int(exp_cfg.get("top_k", 50))
        top_p = float(exp_cfg.get("top_p", 1.0))
        eos_token_id = int(exp_cfg.get("eos_token_id", 0))

        generated = prompt_tensor.tolist()
        with torch.no_grad():
            for token, _ in model.generate(
                prompt_tensor,
                max_new_length=max_new_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
            ):
                generated.append(token)

        print(f"Generated token ids: {generated}")


if __name__ == "__main__":
    main()
