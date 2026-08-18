#!/usr/bin/env python3
"""Fail-fast checks for ControlAI v1 MLX training configurations and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "sft": PROJECT_ROOT / "configs" / "lora_controlai_sft_v1.yaml",
    "dapt": PROJECT_ROOT / "configs" / "lora_controlai_dapt_v1.yaml",
    "sft-after-dapt": PROJECT_ROOT / "configs" / "lora_controlai_sft_after_dapt_v1.yaml",
}


def qwen_lora_parameter_count(architecture, config: dict) -> int:
    """Return exact LoRA A/B parameter count for the configured Qwen projections."""
    hidden = architecture.hidden_size
    head_dim = architecture.head_dim
    attention = architecture.num_attention_heads * head_dim
    key_value = architecture.num_key_value_heads * head_dim
    intermediate = architecture.intermediate_size
    dimensions = {
        "self_attn.q_proj": (hidden, attention),
        "self_attn.k_proj": (hidden, key_value),
        "self_attn.v_proj": (hidden, key_value),
        "self_attn.o_proj": (attention, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }
    lora = config["lora_parameters"]
    keys = lora.get("keys")
    if not keys:
        raise ValueError("lora_parameters.keys must explicitly pin target projections")
    unknown = sorted(set(keys) - dimensions.keys())
    if unknown:
        raise ValueError(f"unsupported LoRA projection keys: {unknown}")
    per_block = int(lora["rank"]) * sum(
        dimensions[key][0] + dimensions[key][1] for key in keys
    )
    return int(config["num_layers"]) * per_block


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    checksum_manifest = PROJECT_ROOT / "configs" / "training_artifacts_v1.sha256"
    for line in checksum_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        artifact = PROJECT_ROOT / relative.strip()
        if not artifact.exists():
            errors.append(f"checksummed artifact is missing: {artifact}")
            continue
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            errors.append(f"artifact checksum changed: {relative.strip()}")

    config_path = STAGES[args.stage]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = PROJECT_ROOT / config["data"]
    train_path, valid_path = data_dir / "train.jsonl", data_dir / "valid.jsonl"
    for path in (train_path, valid_path):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty data split: {path}")
    train = rows(train_path) if train_path.exists() else []
    valid = rows(valid_path) if valid_path.exists() else []
    optimizer_updates = config["iters"] // config["grad_accumulation_steps"]
    schedule = config.get("lr_schedule")
    if schedule:
        warmup = int(schedule.get("warmup", 0))
        decay_steps = int(schedule["arguments"][1])
        if abs((warmup + decay_steps) - optimizer_updates) > 2:
            errors.append(
                "learning-rate schedule length is not expressed in optimizer updates: "
                f"warmup+decay={warmup + decay_steps}, updates={optimizer_updates}"
            )

    trainable_parameters = None
    try:
        architecture = AutoConfig.from_pretrained(config["model"], local_files_only=True)
        if config["num_layers"] > architecture.num_hidden_layers:
            errors.append(
                f"num_layers={config['num_layers']} exceeds model depth {architecture.num_hidden_layers}"
            )
        trainable_parameters = qwen_lora_parameter_count(architecture, config)
    except Exception as exc:
        errors.append(f"model/config is not available locally: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        sample = train[: min(128, len(train))]
        lengths = []
        for row in sample:
            if "messages" in row:
                lengths.append(len(tokenizer.apply_chat_template(row["messages"], return_dict=False)))
            else:
                lengths.append(len(tokenizer.encode(row["text"], add_special_tokens=False)))
        if lengths and max(lengths) > config["max_seq_length"]:
            errors.append(f"sample contains {max(lengths)} tokens, above max_seq_length")
    except Exception as exc:
        errors.append(f"tokenizer preflight failed: {exc}")

    adapter_path = PROJECT_ROOT / config["adapter_path"]
    if adapter_path.exists() and any(adapter_path.iterdir()) and not args.allow_existing_output:
        errors.append(
            f"output already contains files: {adapter_path}; choose a new path or pass --allow-existing-output"
        )
    resume = config.get("resume_adapter_file")
    if resume and not (PROJECT_ROOT / resume).exists():
        errors.append(f"required resume adapter is missing: {PROJECT_ROOT / resume}")

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    if free_gb < 10:
        warnings.append(f"only {free_gb:.1f} GB disk space is free")
    if sys.prefix == sys.base_prefix:
        errors.append("virtual environment is not active")
    try:
        mlx_lm_version = importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        errors.append("mlx-lm is not installed in this environment")
        mlx_lm_version = "missing"

    print(f"stage: {args.stage}")
    print(f"config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"mlx-lm: {mlx_lm_version}")
    print(f"model: {config['model']}")
    print(f"train/valid rows: {len(train):,}/{len(valid):,}")
    print(f"iterations: {config['iters']:,}; gradient accumulation: {config['grad_accumulation_steps']}")
    print(f"optimizer updates: {optimizer_updates:,}")
    print(f"adapter: rank {config['lora_parameters']['rank']}, layers {config['num_layers']}")
    if trainable_parameters is not None:
        print(
            f"trainable LoRA parameters: {trainable_parameters:,} "
            f"(~{trainable_parameters * 4 / 1_000_000:.1f} MB as float32 weights)"
        )
    print(f"free disk: {free_gb:.1f} GB")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("preflight passed")
    print(f"train command: python -m mlx_lm.lora --config {config_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
