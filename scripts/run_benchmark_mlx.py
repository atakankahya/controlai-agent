"""Run a held-out ControlAI benchmark with one MLX-LM model.

The benchmark rubric and reference answer are intentionally never placed in the
model prompt. Results are appended one at a time so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_SYSTEM_PROMPT = (
    "You are an offline control-systems engineering assistant. Lead with the "
    "result and do not restate the problem. Answer only from "
    "the information supplied. Never invent a plant, parameters, controller "
    "coefficients, software output, or verification results. State necessary "
    "assumptions. Provide executable Python or MATLAB when requested. Avoid "
    "tutorial filler, emojis, and repeated conclusions. Obey the requested word limit."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        record["benchmark_id"]
        for record in load_jsonl(path)
        if isinstance(record.get("benchmark_id"), str)
    }


def user_prompt(record: dict[str, Any]) -> str:
    """Expose benchmark constraints without exposing the rubric or reference."""
    constraints = record.get("constraints", {})
    language = constraints.get("language", "English")
    max_words = constraints.get("max_words")
    lines = [record["prompt"], "", "Response constraints:", f"- Language: {language}"]
    if isinstance(max_words, int):
        lines.append(f"- Maximum length: {max_words} words")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        help="Hugging Face model id or local MLX model directory",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/v0.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional trained MLX LoRA adapter directory",
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N unfinished items (useful for a smoke test)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to chat templates that support it",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(
            "benchmarks/responses/controlai_qwen3_4b_v0_lora.jsonl"
            if args.adapter_path
            else "benchmarks/responses/qwen3_4b_instruct_v0_1.jsonl"
        )

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    benchmark = load_jsonl(args.benchmark)
    done = completed_ids(args.output)
    pending = [record for record in benchmark if record["id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("No unfinished benchmark items.")
        return 0

    print(f"Loading {args.model}")
    if args.adapter_path:
        print(f"Applying adapter {args.adapter_path}")
    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    sampler = make_sampler(temp=args.temperature)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output_handle:
        for index, record in enumerate(pending, start=1):
            benchmark_id = record["id"]
            print(f"[{index}/{len(pending)}] {benchmark_id}")
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_prompt(record)},
            ]
            template_kwargs = {"enable_thinking": True} if args.enable_thinking else {}
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False)

            started = time.perf_counter()
            pieces: list[str] = []
            final = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=args.max_tokens,
                sampler=sampler,
            ):
                pieces.append(response.text)
                final = response
            elapsed = time.perf_counter() - started

            result = {
                "benchmark_id": benchmark_id,
                "family": record["family"],
                "domain": record["domain"],
                "model": args.model,
                "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "enable_thinking": args.enable_thinking,
                "response_constraints": record.get("constraints", {}),
                "response": "".join(pieces),
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": getattr(final, "prompt_tokens", None),
                "generation_tokens": getattr(final, "generation_tokens", None),
                "generation_tokens_per_second": round(
                    getattr(final, "generation_tps", 0.0), 3
                ),
                "peak_memory_gb": round(getattr(final, "peak_memory", 0.0), 3),
                "finish_reason": getattr(final, "finish_reason", None),
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(
                f"  {result['generation_tokens']} tokens, "
                f"{result['generation_tokens_per_second']} tok/s, "
                f"{result['peak_memory_gb']} GB peak, {result['finish_reason']}"
            )

    print(f"Saved responses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
