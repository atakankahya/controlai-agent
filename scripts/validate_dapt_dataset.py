#!/usr/bin/env python3
"""Validate ControlAI DAPT splits, provenance, deduplication, and token counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer


def normalized_hash(text: str) -> str:
    text = re.sub(r"\s+", " ", text).casefold().strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    errors = []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
                continue
            row["_location"] = f"{path}:{line_number}"
            rows.append(row)
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/training/dapt_v1"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    train, errors = load(args.dataset / "train.jsonl")
    valid, valid_errors = load(args.dataset / "valid.jsonl")
    errors.extend(valid_errors)
    summary_path = args.dataset / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    seen_hashes = {}
    documents = {"train": set(), "valid": set()}
    source_counts = {"train": {}, "valid": {}}
    for split, rows in (("train", train), ("valid", valid)):
        for row in rows:
            location = row["_location"]
            for field in (
                "text",
                "token_count",
                "document_id",
                "source_id",
                "source_pool",
                "text_sha256",
                "simhash64",
            ):
                if row.get(field) in (None, ""):
                    errors.append(f"{location}: missing {field}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            digest = normalized_hash(text)
            if digest != row.get("text_sha256"):
                errors.append(f"{location}: text_sha256 mismatch")
            if digest in seen_hashes:
                errors.append(
                    f"{location}: exact duplicate also present at {seen_hashes[digest]}"
                )
            else:
                seen_hashes[digest] = location
            if row.get("source_pool") == "mathworks_r2026a":
                errors.append(f"{location}: proprietary MathWorks text is disallowed")
            if row.get("document_id"):
                documents[split].add(f"{row.get('source_pool')}::{row['document_id']}")
            pool = row.get("source_pool", "unknown")
            source_counts[split][pool] = source_counts[split].get(pool, 0) + 1

    leaked_documents = documents["train"] & documents["valid"]
    if leaked_documents:
        errors.append(
            f"document split leakage: {len(leaked_documents)} document ids occur in both splits"
        )

    expected = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_tokens": sum(row.get("token_count", 0) for row in train),
        "valid_tokens": sum(row.get("token_count", 0) for row in valid),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary {key}={summary.get(key)!r}, actual={value}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )
    # Full token recount is deliberate: a stale count changes training estimates.
    for row in train + valid:
        actual = len(tokenizer.encode(row["text"], add_special_tokens=False))
        if actual != row.get("token_count"):
            errors.append(
                f"{row['_location']}: token_count={row.get('token_count')}, actual={actual}"
            )

    print(f"train rows: {len(train):,}")
    print(f"validation rows: {len(valid):,}")
    print(f"unique normalized texts: {len(seen_hashes):,}")
    print(f"train documents: {len(documents['train']):,}")
    print(f"validation documents: {len(documents['valid']):,}")
    print(f"train tokens: {expected['train_tokens']:,}")
    print(f"validation tokens: {expected['valid_tokens']:,}")
    print("source rows:")
    for split in ("train", "valid"):
        print(f"  {split}: {dict(sorted(source_counts[split].items()))}")

    if errors:
        print("validation failed:", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} additional errors", file=sys.stderr)
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
