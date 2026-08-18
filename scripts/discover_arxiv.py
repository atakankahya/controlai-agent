#!/usr/bin/env python3
"""Discover control papers through the official arXiv Atom API."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = PROJECT_ROOT / "data" / "sources" / "arxiv_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sources" / "discovered_arxiv.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "sources" / "discovered_arxiv_summary.json"
API_URL = "https://export.arxiv.org/api/query"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def child_text(entry: ET.Element, tag: str) -> str:
    child = entry.find(f"{ATOM}{tag}")
    return clean_text(child.text if child is not None else None)


def parse_entry(entry: ET.Element, query: dict) -> dict:
    entry_url = child_text(entry, "id")
    arxiv_id = entry_url.rsplit("/", 1)[-1]
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    links = {
        link.attrib.get("title") or link.attrib.get("rel", "link"): link.attrib.get("href")
        for link in entry.findall(f"{ATOM}link")
    }
    categories = [category.attrib["term"] for category in entry.findall(f"{ATOM}category")]
    license_element = entry.find(f"{ARXIV}license")
    return {
        "arxiv_id": base_id,
        "versioned_arxiv_id": arxiv_id,
        "title": child_text(entry, "title"),
        "abstract": child_text(entry, "summary"),
        "authors": [
            child_text(author, "name") for author in entry.findall(f"{ATOM}author")
        ],
        "published": child_text(entry, "published"),
        "updated": child_text(entry, "updated"),
        "categories": categories,
        "primary_category": (
            entry.find(f"{ARXIV}primary_category").attrib.get("term")
            if entry.find(f"{ARXIV}primary_category") is not None
            else None
        ),
        "abstract_url": entry_url,
        "pdf_url": links.get("pdf") or f"https://arxiv.org/pdf/{base_id}",
        "license_url": license_element.attrib.get("href") if license_element is not None else None,
        "matched_query_ids": [query["id"]],
        "matched_domains": [query["domain"]],
    }


def fetch_query(query: dict, per_query: int, timeout: int) -> list[dict]:
    parameters = urllib.parse.urlencode(
        {
            "search_query": query["query"],
            "start": 0,
            "max_results": per_query,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{parameters}",
        headers={"User-Agent": "controlai-literature-discovery/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        root = ET.fromstring(response.read())
    return [parse_entry(entry, query) for entry in root.findall(f"{ATOM}entry")]


def merge_record(existing: dict, incoming: dict) -> None:
    existing["matched_query_ids"] = sorted(
        set(existing["matched_query_ids"] + incoming["matched_query_ids"])
    )
    existing["matched_domains"] = sorted(
        set(existing["matched_domains"] + incoming["matched_domains"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--delay-seconds", type=float, default=3.2)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--fetch", action="store_true", help="Call arXiv; otherwise list planned queries")
    args = parser.parse_args()

    config = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = config["queries"]
    print(f"Queries: {len(queries)}")
    print(f"Maximum raw results: {len(queries) * args.per_query}")
    for query in queries:
        print(f"- {query['id']} [{query['domain']}]: {query['query']}")
    if not args.fetch:
        print("Dry run only. Add --fetch to call the arXiv API.")
        return

    works: dict[str, dict] = {}
    query_counts = {}
    for index, query in enumerate(queries, start=1):
        results = fetch_query(query, args.per_query, args.timeout)
        query_counts[query["id"]] = len(results)
        for result in results:
            if result["arxiv_id"] in works:
                merge_record(works[result["arxiv_id"]], result)
            else:
                works[result["arxiv_id"]] = result
        print(f"Fetched {index}/{len(queries)}: {query['id']} -> {len(results)}")
        if index < len(queries):
            time.sleep(args.delay_seconds)

    records = sorted(works.values(), key=lambda row: (row["matched_domains"], row["title"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "queries": len(queries),
        "per_query": args.per_query,
        "raw_results": sum(query_counts.values()),
        "unique_works": len(records),
        "query_counts": query_counts,
        "domain_counts": dict(
            sorted(Counter(domain for row in records for domain in row["matched_domains"]).items())
        ),
        "with_explicit_license_url": sum(bool(row["license_url"]) for row in records),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
