#!/usr/bin/env python3
"""Reject front-matter-only canonical book PDF candidates and refresh summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "sources" / "resolved_canonical_books.jsonl"
SUMMARY = ROOT / "data" / "sources" / "resolved_canonical_books_summary.json"


def is_front_matter(candidate: dict) -> bool:
    url = (candidate.get("pdf_url") or "").lower()
    return "frontmatter" in url or "/bfm" in url or "bfm%3a" in url


def main() -> None:
    rows = [json.loads(line) for line in CATALOG.read_text(encoding="utf-8").splitlines()]
    rejected = []
    for row in rows:
        candidates = row.get("pdf_candidates") or []
        direct = [
            candidate
            for candidate in candidates
            if candidate.get("version") != "openalex_content" and not is_front_matter(candidate)
        ]
        if candidates and not direct:
            rejected.append({"rank": row["rank"], "title": row["title"], "candidates": candidates})
            row["pdf_candidates"] = []
            if row["status"] == "oa_candidate":
                row["status"] = "metadata_only"

    CATALOG.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    summary = {
        "targets": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "targets_with_pdf_candidates": sum(bool(row.get("pdf_candidates")) for row in rows),
        "total_pdf_candidates": sum(len(row.get("pdf_candidates") or []) for row in rows),
        "front_matter_only_targets_rejected": len(rejected),
        "rejected": rejected,
        "api": "https://api.openalex.org/works",
        "catalog_api": "https://openlibrary.org/search.json",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
