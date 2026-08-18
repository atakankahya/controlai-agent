#!/usr/bin/env python3
"""List or download approved source artifacts from the source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sources" / "download_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "sources"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def human_size(byte_count: int | None) -> str:
    if byte_count is None:
        return "unknown"
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("sources"), list):
        raise ValueError(f"Unsupported manifest: {path}")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Manifest defaults must be an object: {path}")
    return [{**defaults, **source} for source in data["sources"]]


def select_sources(sources: list[dict], requested: list[str]) -> list[dict]:
    if not requested:
        return sources
    by_id = {source["source_id"]: source for source in sources}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown source_id: {', '.join(unknown)}")
    return [by_id[source_id] for source_id in requested]


def validate_artifact(path: Path, filename: str) -> None:
    """Reject an HTML redirect or error page saved under a PDF filename."""
    if Path(filename).suffix.casefold() == ".pdf":
        with path.open("rb") as stream:
            signature = stream.read(5)
        if signature != b"%PDF-":
            raise ValueError(
                f"Expected a PDF for {filename}, received signature {signature!r}"
            )
        try:
            with pymupdf.open(path) as document:
                if document.page_count < 1:
                    raise ValueError(f"PDF has no readable pages: {filename}")
        except (pymupdf.FileDataError, RuntimeError) as error:
            raise ValueError(f"Unreadable PDF {filename}: {error}") from error


def curl_download(url: str, destination: Path) -> None:
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--proto",
        "=https",
        "--retry",
        "3",
        "--retry-delay",
        "2",
    ]
    if destination.exists() and destination.stat().st_size > 0:
        try:
            subprocess.run(
                command
                + ["--continue-at", "-", "--output", str(destination), url],
                check=True,
            )
            return
        except subprocess.CalledProcessError:
            print(f"[restart] server cannot resume {destination.name}")
            destination.unlink()
    subprocess.run(command + ["--output", str(destination), url], check=True)


def download(source: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source["filename"]
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        print(f"[skip] {source['source_id']}: {destination.name} already exists")
    else:
        print(f"[download] {source['source_id']} -> {destination}")
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "controlai-source-fetcher/0.1"}
        )
        try:
            with urllib.request.urlopen(request, context=TLS_CONTEXT) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            validate_artifact(partial, source["filename"])
            partial.replace(destination)
        except urllib.error.HTTPError as error:
            if error.code not in {403, 429}:
                partial.unlink(missing_ok=True)
                raise
            curl_download(source["url"], partial)
            validate_artifact(partial, source["filename"])
            partial.replace(destination)
        except urllib.error.URLError as error:
            partial.unlink(missing_ok=True)
            if not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise
            curl_download(source["url"], partial)
            validate_artifact(partial, source["filename"])
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    validate_artifact(destination, source["filename"])
    actual_bytes = destination.stat().st_size
    expected_bytes = source.get("expected_bytes")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"Size mismatch for {source['source_id']}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )

    return {
        "source_id": source["source_id"],
        "title": source.get("title"),
        "authors": source.get("authors", []),
        "filename": source["filename"],
        "url": source["url"],
        "license": source["license"],
        "corpus_tier": source.get("corpus_tier"),
        "release": source.get("release"),
        "intended_use": source.get("intended_use"),
        "coverage": source.get("coverage", []),
        "bytes": actual_bytes,
        "sha256": sha256(destination),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", action="append", default=[], help="Download only this source_id; repeatable")
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Skip this source_id; repeatable",
    )
    parser.add_argument("--download", action="store_true", help="Actually download; otherwise only list")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record successful files and continue after a failed source",
    )
    args = parser.parse_args()

    all_sources = load_manifest(args.manifest)
    sources = select_sources(all_sources, args.source)
    known_ids = {source["source_id"] for source in all_sources}
    unknown_exclusions = sorted(set(args.exclude_source) - known_ids)
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded source_id: {', '.join(unknown_exclusions)}")
    excluded = set(args.exclude_source)
    sources = [source for source in sources if source["source_id"] not in excluded]
    known_total = sum(source.get("expected_bytes") or 0 for source in sources)

    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output}")
    for source in sources:
        print(
            f"- {source['source_id']}: {human_size(source.get('expected_bytes'))} "
            f"[{source['license']}]"
        )
    print(f"Known total: {human_size(known_total)} + unknown sizes")

    if not args.download:
        print("Dry run only. Add --download to fetch these files.")
        return 0

    lock_path = args.output / "source_lock.json"
    existing_records = []
    if lock_path.exists():
        existing_records = json.loads(lock_path.read_text(encoding="utf-8"))
    by_source_id = {record["source_id"]: record for record in existing_records}
    failures: list[tuple[str, str]] = []
    for source in sources:
        try:
            record = download(source, args.output)
        except Exception as error:
            if not args.continue_on_error:
                raise
            failures.append((source["source_id"], str(error)))
            print(f"[error] {source['source_id']}: {error}", file=sys.stderr)
            continue
        by_source_id[record["source_id"]] = record
        lock_records = [by_source_id[source_id] for source_id in sorted(by_source_id)]
        lock_path.write_text(
            json.dumps(lock_records, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Wrote checksums and provenance: {lock_path}")
    if failures:
        print(f"Completed with {len(failures)} failed source(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
