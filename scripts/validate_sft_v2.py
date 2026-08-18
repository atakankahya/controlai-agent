#!/usr/bin/env python3
"""Independently validate ControlAI SFT v2 records and split invariants."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import cvxpy as cp
from scipy import linalg, signal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).casefold().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def similarity_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", " <NUM> ", text)
    return re.sub(r"(?:\s*<num>\s*,?){3,}", " <NUMSEQ> ", text)


def near_duplicate_errors(
    left_name: str,
    left: list[tuple[str, str]],
    right_name: str,
    right: list[tuple[str, str]],
    threshold: float = 0.90,
) -> list[str]:
    if not left or not right:
        return []
    texts = [similarity_text(text) for _, text in left + right]
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=100_000
    ).fit_transform(texts)
    left_matrix = matrix[: len(left)]
    right_matrix = matrix[len(left) :]
    distances, indices = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
        left_matrix
    ).kneighbors(right_matrix)
    errors = []
    for right_index, (distance, nearest) in enumerate(zip(distances[:, 0], indices[:, 0])):
        similarity = 1.0 - float(distance)
        if similarity >= threshold:
            errors.append(
                f"near-duplicate prompt across {left_name}/{right_name} "
                f"({left[int(nearest)][0]} vs {right[right_index][0]}, cosine={similarity:.3f})"
            )
    return errors


def check_underspecified_answer(answer: str, gt: dict) -> None:
    for term in gt.get("required_answer_terms", []):
        if term.casefold() not in answer.casefold():
            raise AssertionError(f"missing required term {term!r}")


def execute_python_block(code: str) -> None:
    stdout_buf = io.StringIO()
    globals_dict: dict[str, Any] = {}
    with contextlib.redirect_stdout(stdout_buf):
        exec(code, globals_dict, globals_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/training/sft_v2")
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/v1_dev.jsonl")
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_path = args.dataset_dir / "train.jsonl"
    valid_path = args.dataset_dir / "valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        print(f"Error: {train_path} or {valid_path} does not exist", file=sys.stderr)
        return 1

    train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [json.loads(line) for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]

    all_rows = train + valid
    errors: list[str] = []

    seen_ids: set[str] = set()
    families: dict[str, set[str]] = {"train": set(), "valid": set()}
    task_counts: dict[str, Counter[str]] = {"train": Counter(), "valid": Counter()}
    kind_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()

    for split_name, split_rows in (("train", train), ("valid", valid)):
        for idx, row in enumerate(split_rows):
            location = f"{split_name}[{idx}]"
            metadata = row.get("metadata", {})
            row_id = metadata.get("id")
            if not row_id:
                errors.append(f"{location}: missing metadata.id")
            elif row_id in seen_ids:
                errors.append(f"{location}: duplicate id {row_id}")
            else:
                seen_ids.add(row_id)

            family = metadata.get("family")
            if family:
                families[split_name].add(family)
            task_type = metadata.get("task_type")
            if task_type:
                task_counts[split_name][task_type] += 1
            template_id = metadata.get("template_id")
            if template_id:
                template_counts[template_id] += 1

            gt = row.get("ground_truth", {})
            if isinstance(gt, dict) and "kind" in gt:
                kind_counts[gt["kind"]] += 1

            messages = row.get("messages", [])
            if len(messages) != 3:
                errors.append(f"{location}: expected 3 messages, got {len(messages)}")
                continue

            if metadata.get("task_type") == "code" and metadata.get("code_language") == "python":
                code_match = re.search(r"```python\s*(.*?)\s*```", messages[2]["content"], re.DOTALL)
                if code_match:
                    try:
                        execute_python_block(code_match.group(1))
                    except Exception as exc:
                        errors.append(f"{location}: Python execution failed: {exc}")

            if metadata.get("task_type") == "underspecified" and isinstance(gt, dict):
                try:
                    check_underspecified_answer(messages[2]["content"], gt)
                except Exception as exc:
                    errors.append(f"{location}: underspecification check failed: {exc}")

            token_count = len(tokenizer.apply_chat_template(messages, return_dict=False))
            if token_count > args.max_seq_length:
                errors.append(f"{location}: {token_count} tokens exceeds max {args.max_seq_length}")

    # Check family separation (no leakage)
    overlap = families["train"] & families["valid"]
    if overlap:
        errors.append(f"Family split leakage: {sorted(overlap)}")

    bench_families = {b.get("family") for b in benchmark if "family" in b}
    bench_leak = (families["train"] | families["valid"]) & bench_families
    if bench_leak:
        errors.append(f"Benchmark leakage into train/valid: {sorted(bench_leak)}")

    underspecified_fraction = task_counts["train"]["underspecified"] / len(train) if train else 0
    if underspecified_fraction < 0.04:
        errors.append(f"underspecified train fraction {underspecified_fraction:.2%} is below 4%")

    prompt_splits = {
        "train": [(row["metadata"]["id"], row["messages"][1]["content"]) for row in train],
        "valid": [(row["metadata"]["id"], row["messages"][1]["content"]) for row in valid],
        "benchmark": [(row["id"], row["prompt"]) for row in benchmark],
    }
    errors.extend(near_duplicate_errors("train", prompt_splits["train"], "valid", prompt_splits["valid"]))
    errors.extend(near_duplicate_errors("train", prompt_splits["train"], "benchmark", prompt_splits["benchmark"]))
    errors.extend(near_duplicate_errors("valid", prompt_splits["valid"], "benchmark", prompt_splits["benchmark"]))

    max_template = max(template_counts.values(), default=0)
    concentration = max_template / len(all_rows) if all_rows else 0

    print(f"Validated records: {len(all_rows):,}")
    print(f"Train / Valid: {len(train):,} / {len(valid):,}")
    print(f"Train families: {len(families['train'])}")
    print(f"Valid families: {len(families['valid'])}")
    print(f"Ground-truth kinds: {len(kind_counts)}")
    print(f"Train task types: {dict(sorted(task_counts['train'].items()))}")
    print(f"Max template concentration: {concentration:.2%}")

    if errors:
        print(f"\nValidation failed with {len(errors)} errors:", file=sys.stderr)
        for err in errors[:30]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nALL SFT V2 QUALITY GATES AND INVARIANTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
