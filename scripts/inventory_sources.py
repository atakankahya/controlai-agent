#!/usr/bin/env python3
"""Inventory useful documents inside downloaded source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "sources"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "inventory.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "inventory_summary.json"

USEFUL_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".m",
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".srt",
    ".vtt",
}

ROLE_PATTERNS = [
    ("solution", re.compile(r"((^|[/_.-])(sol|solution|solutions|answer|answers)([/_.-]|$)|(?:ps|hw|exam|quiz)\d*[_-]?sol)", re.I)),
    ("exam", re.compile(r"(^|[/_.-])(exam|quiz|midterm|final|test)\d*([/_.-]|$)", re.I)),
    ("assignment", re.compile(r"((^|[/_.-])(assignment|problem|problems|prob|pset|homework)\d*([/_.-]|$)|(^|[/_.-])(hw|ps)\d+([/_.-]|$))", re.I)),
    ("lecture", re.compile(r"(^|[/_.-])(lecture|lectures|lec|chapter|chap|notes)([/_.-]|$)", re.I)),
    ("syllabus", re.compile(r"(^|[/_.-])(syllabus|calendar|readings)([/_.-]|$)", re.I)),
    ("transcript", re.compile(r"\.(srt|vtt)$", re.I)),
]


def stable_id(source_id: str, member_path: str) -> str:
    key = f"{source_id}\0{member_path}".encode()
    return hashlib.sha256(key).hexdigest()[:20]


def classify_role(path: str, extension: str) -> str:
    if extension in {".m", ".py", ".ipynb"}:
        return "code"
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(path):
            return role
    if extension in {".html", ".htm"}:
        return "course_page"
    if extension == ".pdf":
        return "document"
    return "text"


def load_source_lock(input_dir: Path) -> dict[str, dict]:
    lock_path = input_dir / "source_lock.json"
    records = json.loads(lock_path.read_text(encoding="utf-8"))
    return {record["filename"]: record for record in records}


def zip_records(archive: Path, source: dict) -> list[dict]:
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename)
            extension = member_path.suffix.lower()
            if extension not in USEFUL_EXTENSIONS:
                continue
            normalized_path = member_path.as_posix()
            records.append(
                {
                    "document_id": stable_id(source["source_id"], normalized_path),
                    "source_id": source["source_id"],
                    "source_title": source.get("title"),
                    "source_authors": source.get("authors", []),
                    "corpus_tier": source.get("corpus_tier"),
                    "source_coverage": source.get("coverage", []),
                    "container": archive.name,
                    "member_path": normalized_path,
                    "extension": extension,
                    "content_role": classify_role(normalized_path, extension),
                    "uncompressed_bytes": member.file_size,
                    "compressed_bytes": member.compress_size,
                    "archive_crc32": f"{member.CRC:08x}",
                }
            )
    return records


def standalone_record(path: Path, source: dict) -> dict:
    return {
        "document_id": stable_id(source["source_id"], path.name),
        "source_id": source["source_id"],
        "source_title": source.get("title"),
        "source_authors": source.get("authors", []),
        "corpus_tier": source.get("corpus_tier"),
        "source_coverage": source.get("coverage", []),
        "container": path.name,
        "member_path": path.name,
        "extension": path.suffix.lower(),
        "content_role": classify_role(path.name, path.suffix.lower()),
        "uncompressed_bytes": path.stat().st_size,
        "compressed_bytes": None,
        "archive_crc32": None,
    }


def mark_archive_duplicates(records: list[dict]) -> None:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for record in records:
        record["exact_duplicate_of"] = None
        crc = record.get("archive_crc32")
        if crc:
            key = (record["extension"], record["uncompressed_bytes"], crc)
            groups[key].append(record)
    for group in groups.values():
        canonical = sorted(group, key=lambda record: record["document_id"])[0]
        for record in group:
            if record is not canonical:
                record["exact_duplicate_of"] = canonical["document_id"]


def build_summary(records: list[dict]) -> dict:
    unique_records = [record for record in records if not record["exact_duplicate_of"]]
    return {
        "total_useful_files": len(records),
        "unique_by_archive_checksum": len(unique_records),
        "exact_duplicates": len(records) - len(unique_records),
        "by_extension": dict(sorted(Counter(r["extension"] for r in records).items())),
        "by_role": dict(sorted(Counter(r["content_role"] for r in records).items())),
        "by_source": dict(sorted(Counter(r["source_id"] for r in records).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    source_lock = load_source_lock(args.input)
    records = []
    for filename, source in sorted(source_lock.items()):
        artifact = args.input / filename
        if artifact.suffix.lower() == ".zip":
            records.extend(zip_records(artifact, source))
        elif artifact.suffix.lower() in USEFUL_EXTENSIONS:
            records.append(standalone_record(artifact, source))

    mark_archive_duplicates(records)
    records.sort(key=lambda record: (record["source_id"], record["member_path"]))
    summary = build_summary(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Inventory: {args.output}")
    print(f"Summary:   {args.summary}")


if __name__ == "__main__":
    main()
