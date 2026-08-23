#!/usr/bin/env python3
"""Run benchmark evaluation using the ControlAI Tool-Calling Agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.agent import ControlAgent
from controlai_agent.engine import LocalEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/smoke_30.jsonl"),
        help="Benchmark JSONL file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        help="Base model path or HuggingFace repo",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional adapter checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/responses/agent_smoke_30.jsonl"),
        help="Output response JSONL file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Maximum generation tokens per step",
    )
    args = parser.parse_args()

    print(f"Loading benchmark from {args.benchmark}...")
    items = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Total benchmark items: {len(items)}")

    print(f"Initializing ControlAI Agent ({args.model})...")
    agent = ControlAgent(engine=LocalEngine(model_id=args.model, adapter_path=args.adapter_path))
    print("Agent ready.")

    responses = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for idx, item in enumerate(items, 1):
        item_id = item.get("id", f"item_{idx:03d}")
        prompt = item.get("prompt", "")
        print(f"[{idx}/{len(items)}] Evaluating {item_id}...", end=" ", flush=True)

        item_start = time.time()
        result = agent.run(prompt, max_tokens=args.max_tokens)
        elapsed = time.time() - item_start

        tool_names = [t.name for t in result.traces]
        print(f"done in {elapsed:.2f}s | Steps: {len(result.traces)} | Tools: {tool_names}")

        record = {
            "benchmark_id": item_id,
            "id": item_id,
            "family": item.get("family", ""),
            "prompt": prompt,
            "response": result.answer,
            "tool_calls": [
                {
                    "name": t.name,
                    "arguments": t.arguments,
                    "result": t.result,
                }
                for t in result.traces
            ],
            "total_steps": len(result.traces),
            "finish_reason": "stop",
        }
        responses.append(record)

    with args.output.open("w", encoding="utf-8") as stream:
        for resp in responses:
            stream.write(json.dumps(resp, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time
    print(f"\nSaved {len(responses)} agent responses to {args.output} (Took {total_time:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
