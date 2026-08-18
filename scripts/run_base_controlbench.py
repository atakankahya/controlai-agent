"""Run raw base Qwen3-4B-Instruct model (without tools) on ControlBench v1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import generate, load
from transformers import AutoTokenizer

BENCHMARK_PATH = Path("benchmarks/controlbench_v1.jsonl")
OUTPUT_PATH = Path("benchmarks/responses/qwen3_4b_base_controlbench_v1.jsonl")


def main() -> int:
    model_name = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    print(f"Loading Base Foundation Model: {model_name}...")
    model, tokenizer = load(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    items = [json.loads(line) for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    responses = []

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(items)} ControlBench items on Base Model...")

    start_time = time.time()
    for idx, item in enumerate(items, 1):
        item_id = item["id"]
        prompt = item["prompt"]
        print(f"[{idx}/{len(items)}] Evaluating {item_id}...", end=" ", flush=True)

        messages = [{"role": "user", "content": prompt}]
        rendered = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        resp_text = generate(model, tokenizer, prompt=rendered, max_tokens=768, verbose=False).strip()
        elapsed = time.time() - t0

        print(f"done in {elapsed:.2f}s")
        responses.append({
            "id": item_id,
            "pillar": item["pillar"],
            "prompt": prompt,
            "response": resp_text,
            "tool_calls": [],
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(responses)} base responses to {OUTPUT_PATH} (Took {time.time() - start_time:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
