#!/usr/bin/env python3
"""List or download discovered arXiv PDFs with resume and provenance records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sources" / "discovered_arxiv.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "arxiv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id) + ".pdf"


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def select_records(
    records: list[dict],
    excluded_domains: set[str],
    max_papers: int | None,
    per_domain: int | None,
) -> list[dict]:
    selected = [
        record
        for record in records
        if set(record["matched_domains"]) - excluded_domains
    ]
    selected.sort(key=lambda row: (row["matched_domains"], row["title"]))
    if per_domain is not None:
        balanced = {}
        domains = sorted(
            {domain for record in selected for domain in record["matched_domains"]}
            - excluded_domains
        )
        for domain in domains:
            matches = [record for record in selected if domain in record["matched_domains"]]
            for record in matches[:per_domain]:
                balanced[record["arxiv_id"]] = record
        selected = sorted(balanced.values(), key=lambda row: (row["matched_domains"], row["title"]))
    return selected[:max_papers] if max_papers is not None else selected


def download_pdf(record: dict, output_dir: Path) -> dict:
    filename = safe_filename(record["arxiv_id"])
    destination = output_dir / filename
    partial = destination.with_suffix(".pdf.part")
    if not destination.exists():
        request = urllib.request.Request(
            record["pdf_url"], headers={"User-Agent": "controlai-literature-fetcher/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if partial.read_bytes()[:5] != b"%PDF-":
                raise ValueError("downloaded content is not a PDF")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    return {
        "source_id": f"arxiv_{record['arxiv_id']}",
        "arxiv_id": record["arxiv_id"],
        "title": record["title"],
        "authors": record.get("authors", []),
        "filename": filename,
        "url": record["pdf_url"],
        "pdf_url": record["pdf_url"],
        "corpus_tier": "research_paper",
        "coverage": record["matched_domains"],
        "matched_domains": record["matched_domains"],
        "license": record["license_url"],
        "license_url": record["license_url"],
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-domain", action="append", default=["foundations"])
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--per-domain", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=3.2)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    records = select_records(
        load_records(args.input), set(args.exclude_domain), args.max_papers, args.per_domain
    )
    domain_counts = Counter(domain for row in records for domain in row["matched_domains"])
    print(f"Selected papers: {len(records)}")
    print("Domains:", json.dumps(dict(sorted(domain_counts.items())), indent=2))
    print(f"Output: {args.output_dir}")
    if not args.download:
        print("Dry run only. Add --download to fetch PDFs.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "source_lock.json"
    existing = {}
    if lock_path.exists():
        existing = {row["arxiv_id"]: row for row in json.loads(lock_path.read_text(encoding="utf-8"))}

    failures = []
    for index, record in enumerate(records, start=1):
        try:
            existing[record["arxiv_id"]] = download_pdf(record, args.output_dir)
            status = "ok"
        except Exception as error:
            failures.append({"arxiv_id": record["arxiv_id"], "error": str(error)})
            status = f"error: {error}"
        lock_path.write_text(
            json.dumps(sorted(existing.values(), key=lambda row: row["arxiv_id"]), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{index}/{len(records)}] {record['arxiv_id']}: {status}")
        if index < len(records):
            time.sleep(args.delay_seconds)

    print(f"Downloaded/verified: {len(existing)}")
    print(f"Failures: {len(failures)}")
    if failures:
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
