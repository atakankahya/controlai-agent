#!/usr/bin/env python3
"""Download ranked open-book candidates with Scrapy and verify each PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import scrapy
from scrapy.crawler import CrawlerProcess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sources" / "discovered_open_books.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "open_books"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "sources" / "downloaded_open_books_summary.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slug(text: str, limit: int = 80) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (text[:limit].rstrip("_") or "untitled")


def source_id(candidate: dict) -> str:
    identity = candidate.get("oai_identifier") or candidate["pdf_url"]
    return f"{candidate['source_catalog']}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def filename(candidate: dict) -> str:
    url_hash = hashlib.sha256(candidate["pdf_url"].encode()).hexdigest()[:10]
    return f"{slug(candidate['title'])}_{url_hash}.pdf"


def load_candidates(path: Path, min_score: int, max_books: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [row for row in rows if row["relevance_score"] >= min_score]
    rows.sort(key=lambda row: (-row["relevance_score"], row["title"].lower(), row["pdf_url"]))
    return rows[:max_books] if max_books else rows


class OpenBookDownloadSpider(scrapy.Spider):
    name = "open_book_downloads"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.5,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "RETRY_TIMES": 4,
        "DOWNLOAD_TIMEOUT": 180,
        "USER_AGENT": "controlai-open-corpus-research/0.1",
        "LOG_LEVEL": "INFO",
    }

    def __init__(
        self,
        candidates: list[dict],
        output_dir: str,
        summary_path: str,
        max_mib: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.candidates = candidates
        self.output_dir = Path(output_dir)
        self.summary_path = Path(summary_path)
        self.max_bytes = int(max_mib) * 1024 * 1024
        self.lock_path = self.output_dir / "source_lock.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        if self.lock_path.exists():
            self.records = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.known_urls = {record["url"] for record in self.records}
        self.known_hashes = {record["sha256"] for record in self.records}
        self.downloaded = 0
        self.skipped = 0
        self.duplicates = 0
        self.failures = []

    async def start(self):
        for candidate in self.candidates:
            if candidate["pdf_url"] in self.known_urls:
                self.skipped += 1
                continue
            yield scrapy.Request(
                candidate["pdf_url"],
                callback=self.save_pdf,
                errback=self.download_failed,
                cb_kwargs={"candidate": candidate},
                meta={"download_maxsize": self.max_bytes, "download_warnsize": self.max_bytes},
                dont_filter=True,
            )

    def save_pdf(self, response: scrapy.http.Response, candidate: dict):
        data = response.body
        if not data.startswith(b"%PDF"):
            self.failures.append(
                {
                    "title": candidate["title"],
                    "url": candidate["pdf_url"],
                    "error": f"not a PDF (content-type={response.headers.get('Content-Type')!r})",
                }
            )
            return

        digest = sha256_bytes(data)
        if digest in self.known_hashes:
            self.duplicates += 1
            return

        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                page_count = document.page_count
                if page_count < 1:
                    raise ValueError("PDF has no pages")
        except Exception as error:
            self.failures.append(
                {
                    "title": candidate["title"],
                    "url": candidate["pdf_url"],
                    "error": f"invalid PDF: {type(error).__name__}: {error}",
                }
            )
            return

        name = filename(candidate)
        destination = self.output_dir / name
        partial = destination.with_suffix(".pdf.part")
        partial.write_bytes(data)
        partial.replace(destination)

        record = {
            "source_id": source_id(candidate),
            "title": candidate["title"],
            "authors": candidate.get("creators", []) or candidate.get("contributors", []),
            "filename": name,
            "url": candidate["pdf_url"],
            "catalog": candidate["source_catalog"],
            "oai_identifier": candidate.get("oai_identifier"),
            "publishers": candidate.get("publishers", []),
            "license": candidate.get("licenses", []),
            "corpus_tier": "supplemental_open_book",
            "coverage": candidate.get("matched_phrases", []),
            "relevance_score": candidate["relevance_score"],
            "bytes": len(data),
            "page_count": page_count,
            "sha256": digest,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.records.append(record)
        self.known_urls.add(record["url"])
        self.known_hashes.add(digest)
        self.downloaded += 1
        yield record

    def download_failed(self, failure):
        candidate = failure.request.cb_kwargs["candidate"]
        self.failures.append(
            {
                "title": candidate["title"],
                "url": candidate["pdf_url"],
                "error": failure.getErrorMessage(),
            }
        )

    def closed(self, reason: str) -> None:
        self.records.sort(key=lambda record: (record["title"].lower(), record["url"]))
        self.lock_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary = {
            "selected_candidates": len(self.candidates),
            "downloaded_this_run": self.downloaded,
            "already_downloaded": self.skipped,
            "exact_content_duplicates": self.duplicates,
            "failures": self.failures,
            "total_downloaded_books": len(self.records),
            "total_pages": sum(record["page_count"] for record in self.records),
            "total_bytes": sum(record["bytes"] for record in self.records),
            "close_reason": reason,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-score", type=int, default=20)
    parser.add_argument("--max-books", type=int, default=25, help="0 downloads every selected candidate")
    parser.add_argument("--max-mib", type=int, default=150, help="Reject a single PDF larger than this")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = load_candidates(args.input, args.min_score, args.max_books)
    print(f"Selected {len(candidates)} candidates with score >= {args.min_score}")
    for candidate in candidates[:20]:
        print(f"- {candidate['relevance_score']:3d} {candidate['title']}")
    if len(candidates) > 20:
        print(f"... and {len(candidates) - 20} more")
    if args.dry_run:
        print("Dry run only. Remove --dry-run to download and verify PDFs.")
        return

    process = CrawlerProcess()
    process.crawl(
        OpenBookDownloadSpider,
        candidates=candidates,
        output_dir=str(args.output),
        summary_path=str(args.summary),
        max_mib=args.max_mib,
    )
    process.start()
    if args.summary.exists():
        print(args.summary.read_text(encoding="utf-8"))
        print(f"PDFs: {args.output}")


if __name__ == "__main__":
    main()
