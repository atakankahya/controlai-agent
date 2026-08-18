#!/usr/bin/env python3
"""Create a benchmarkable adapter directory from one saved MLX checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("iteration", type=int)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    checkpoint = args.adapter_dir / f"{args.iteration:07d}_adapters.safetensors"
    config = args.adapter_dir / "adapter_config.json"
    for path in (checkpoint, config):
        if not path.exists():
            parser.error(f"missing {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, args.output_dir / "adapters.safetensors")
    shutil.copy2(config, args.output_dir / "adapter_config.json")
    print(f"Materialized iteration {args.iteration} at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

