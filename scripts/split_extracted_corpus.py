#!/usr/bin/env python3
"""Split extracted units into knowledge, quarantine, and review pools."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "extracted_corpus.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "pools"

CONTROL_SIGNALS = [
    "adaptive control",
    "bode",
    "closed loop",
    "controllability",
    "controller",
    "discrete time",
    "dynamic system",
    "estimator",
    "feedback",
    "frequency response",
    "gain margin",
    "h infinity",
    "kalman",
    "laplace",
    "linear quadratic",
    "loop shaping",
    "lyapunov",
    "model predictive",
    "nyquist",
    "observability",
    "observer",
    "phase margin",
    "pid",
    "pole placement",
    "riccati",
    "robust control",
    "root locus",
    "state feedback",
    "state space",
    "system identification",
    "transfer function",
]


def normalize_for_matching(text: str) -> str:
    text = text.lower().replace("-", " ")
    return re.sub(r"\s+", " ", text)


def document_relevance(rows: list[dict]) -> tuple[int, list[str]]:
    text = normalize_for_matching("\n".join(row["text"] for row in rows))
    matches = [signal for signal in CONTROL_SIGNALS if signal in text]
    return len(matches), matches


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    by_document: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)

    relevance = {
        document_id: document_relevance(document_rows)
        for document_id, document_rows in by_document.items()
    }

    seen_hashes: dict[str, str] = {}
    pools: dict[str, list[dict]] = {"knowledge": [], "quarantine": [], "review": []}
    reasons = Counter()

    for row in rows:
        score, matches = relevance[row["document_id"]]
        row["document_relevance_score"] = score
        row["document_relevance_signals"] = matches

        reason = None
        if not row["text"]:
            pool, reason = "review", "empty_text"
        elif row.get("needs_ocr_review"):
            pool, reason = "review", "possible_ocr_or_sparse_page"
        elif row["text_sha256"] in seen_hashes:
            pool, reason = "review", "exact_text_duplicate"
            row["duplicate_unit_id"] = seen_hashes[row["text_sha256"]]
        elif row["split_policy"] == "quarantine_problem_or_solution":
            pool, reason = "quarantine", "problem_solution_or_exam"
        elif row.get("corpus_tier") in {
            "metadata_only",
            "canonical_excerpt",
            "canonical_scan_pending_ocr",
        }:
            pool, reason = "review", "partial_or_metadata_only_source"
        elif row["content_role"] == "code":
            pool, reason = "knowledge", "code_candidate"
        elif row.get("corpus_tier") == "foundation":
            pool, reason = "knowledge", "foundation_knowledge_candidate"
        elif score == 0:
            pool, reason = "review", "no_strong_control_signal"
        else:
            pool, reason = "knowledge", "control_knowledge_candidate"

        if row["text_sha256"] and row["text_sha256"] not in seen_hashes:
            seen_hashes[row["text_sha256"]] = row["unit_id"]
        row["pool"] = pool
        row["pool_reason"] = reason
        pools[pool].append(row)
        reasons[reason] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, pool_rows in pools.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", pool_rows)

    summary = {
        "input_units": len(rows),
        "documents": len(by_document),
        "pool_counts": {name: len(pool_rows) for name, pool_rows in pools.items()},
        "reason_counts": dict(sorted(reasons.items())),
        "review_documents": len({row["document_id"] for row in pools["review"]}),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Pools: {args.output_dir}")


if __name__ == "__main__":
    main()
