#!/usr/bin/env python3
"""Download bounded document/code collections from approved author/university pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import certifi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sources" / "web_collections_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "web_collections"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
USER_AGENT = "controlai-source-fetcher/0.1"
ALLOWED_SUFFIXES = {".pdf", ".m", ".py", ".ipynb", ".zip"}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, context=TLS_CONTEXT) as response:
            return response.read()
    except urllib.error.URLError:
        result = subprocess.run(
            [
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--user-agent", USER_AGENT, url,
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout


def discover_files(collection: dict) -> list[str]:
    if collection.get("files"):
        return collection["files"]
    html = fetch_bytes(collection["landing_url"]).decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    pattern = re.compile(collection["link_pattern"], re.IGNORECASE)
    return sorted(dict.fromkeys(link for link in parser.links if pattern.search(link)))


def safe_url(base_url: str, link: str) -> str:
    url = urllib.parse.urljoin(base_url, link)
    base = urllib.parse.urlparse(base_url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != base.hostname:
        raise ValueError(f"Link escaped approved HTTPS host: {url}")
    if Path(parsed.path).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported collection file type: {url}")
    return url


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value[:100]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    data = fetch_bytes(url)
    if destination.suffix.lower() == ".pdf" and not data.startswith(b"%PDF-"):
        raise ValueError(f"Expected PDF but received different content: {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--collection", action="append", default=[])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    collections = manifest["collections"]
    if args.collection:
        requested = set(args.collection)
        collections = [item for item in collections if item["collection_id"] in requested]
        missing = requested - {item["collection_id"] for item in collections}
        if missing:
            raise ValueError(f"Unknown collection: {', '.join(sorted(missing))}")
    else:
        collections = [item for item in collections if item.get("enabled", True)]

    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "source_lock.json"
    existing = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else []
    records = {item["source_id"]: item for item in existing}
    failures = []

    for collection in collections:
        files = discover_files(collection)
        expected = collection.get("expected_count")
        if expected is not None and len(files) != expected:
            raise ValueError(
                f"{collection['collection_id']}: expected {expected} files, discovered {len(files)}"
            )
        print(f"{collection['collection_id']}: {len(files)} approved files")
        for link in files:
            url = safe_url(collection["base_url"], link)
            original_name = Path(urllib.parse.urlparse(url).path).name
            filename = f"{collection['collection_id']}__{original_name}"
            source_id = f"{collection['collection_id']}__{slug(original_name)}"
            destination = args.output / filename
            print(f"- {url}")
            if not args.download:
                continue
            try:
                download_file(url, destination)
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                failures.append(
                    {
                        "collection_id": collection["collection_id"],
                        "url": url,
                        "error": str(error),
                    }
                )
                print(f"  [failed] {error}")
                continue
            records[source_id] = {
                "source_id": source_id,
                "collection_id": collection["collection_id"],
                "title": collection["title"],
                "authors": collection.get("authors", []),
                "filename": filename,
                "url": url,
                "landing_url": collection["landing_url"],
                "license": collection["license"],
                "corpus_tier": collection.get("corpus_tier"),
                "coverage": collection.get("coverage", []),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }

    if args.download:
        lock_path.write_text(
            json.dumps([records[key] for key in sorted(records)], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote checksums and provenance: {lock_path}")
        failure_path = args.output / "download_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        print(f"Failures: {len(failures)} ({failure_path})")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
