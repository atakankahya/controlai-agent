#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate .venv first: source .venv/bin/activate" >&2
  exit 1
fi

python scripts/preflight_training_v1.py sft
exec python -m mlx_lm.lora --config configs/lora_controlai_sft_v1.yaml

