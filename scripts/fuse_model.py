#!/usr/bin/env python3
"""Merge/Fuse ControlAI LoRA fine-tuned adapter into base model to produce a standalone model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse ControlAI LoRA adapter into base model")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        help="Base model path or Hugging Face repo ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters/controlai_qwen3_4b_sft_v2",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/controlai_fused",
        help="Target folder for standalone merged model",
    )

    args = parser.parse_args()

    print(f"Fusing LoRA adapter from {args.adapter_path} into {args.model}...")
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        args.model,
        "--adapter-path",
        args.adapter_path,
        "--save-path",
        args.save_path,
    ]

    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode == 0:
        print(f"\nModel successfully fused into standalone checkpoint at: {args.save_path}")
    else:
        print(f"\nError during fusion (exit code {res.returncode})")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
