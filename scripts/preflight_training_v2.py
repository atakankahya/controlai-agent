#!/usr/bin/env python3
"""Preflight checks for ControlAI SFT v2 MLX training configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def qwen_lora_parameter_count(architecture, config: dict) -> int:
    """Return exact LoRA parameter count for configured projections."""
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
    keys = lora.get("keys", [])
    if not keys:
        raise ValueError("lora_parameters.keys must explicitly pin target projections")
    unknown = sorted(set(keys) - dimensions.keys())
    if unknown:
        raise ValueError(f"unsupported LoRA projection keys: {unknown}")
    per_block = int(lora["rank"]) * sum(
        dimensions[key][0] + dimensions[key][1] for key in keys
    )
    return int(config["num_layers"]) * per_block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "lora_controlai_sft_v2.yaml"
    )
    args = parser.parse_args()

    errors: list[str] = []
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    data_dir = PROJECT_ROOT / config["data"]
    train_path, valid_path = data_dir / "train.jsonl", data_dir / "valid.jsonl"
    for path in (train_path, valid_path):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty data split: {path}")

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_rows = [json.loads(line) for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    optimizer_updates = config["iters"] // config["grad_accumulation_steps"]
    schedule = config.get("lr_schedule")
    if schedule:
        warmup = int(schedule.get("warmup", 0))
        decay_steps = int(schedule["arguments"][1])
        if abs((warmup + decay_steps) - optimizer_updates) > 2:
            errors.append(
                f"Schedule mismatch: warmup({warmup}) + decay({decay_steps}) != updates({optimizer_updates})"
            )

    architecture = AutoConfig.from_pretrained(config["model"])
    trainable_params = qwen_lora_parameter_count(architecture, config)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    sample = train_rows[:100]
    lengths = []
    for row in sample:
        tokens = len(tokenizer.apply_chat_template(row["messages"], return_dict=False))
        lengths.append(tokens)

    max_len = max(lengths)
    p95_len = sorted(lengths)[int(0.95 * len(lengths))]

    print(f"Preflight SFT v2 Config Check:")
    print(f"  - Model: {config['model']}")
    print(f"  - Train records: {len(train_rows):,}")
    print(f"  - Valid records: {len(valid_rows):,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Iterations: {config['iters']} (Optimizer updates: {optimizer_updates})")
    print(f"  - Effective batch size: {config['batch_size'] * config['grad_accumulation_steps']}")
    print(f"  - Max sequence length: {config['max_seq_length']} (Sample p95: {p95_len}, max: {max_len})")
    print(f"  - Output adapter path: {config['adapter_path']}")

    if errors:
        print(f"\nPreflight FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nPREFLIGHT PASSED: READY FOR MLX LORA TRAINING!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
