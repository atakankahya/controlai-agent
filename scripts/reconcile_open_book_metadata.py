#!/usr/bin/env python3
"""Backfill publisher metadata and demote bulk-discovered books to supplemental."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "sources" / "discovered_open_books.jsonl"
DEFAULT_LOCK = PROJECT_ROOT / "data" / "raw" / "open_books" / "source_lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    candidates = {
        row["pdf_url"]: row
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    records = json.loads(args.lock.read_text(encoding="utf-8"))
    publisher_counts: dict[str, int] = {}
    matched = 0
    for record in records:
        candidate = candidates.get(record["url"])
        publishers = (candidate or {}).get("publishers", [])
        record["publishers"] = publishers
        record["corpus_tier"] = "supplemental_open_book"
        if candidate:
            matched += 1
        for publisher in publishers or ["unknown"]:
            publisher_counts[publisher] = publisher_counts.get(publisher, 0) + 1

    args.lock.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "matched_to_candidates": matched,
                "corpus_tier": "supplemental_open_book",
                "publisher_counts": dict(sorted(publisher_counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
