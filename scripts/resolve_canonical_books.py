#!/usr/bin/env python3
"""Resolve a curated canonical control-book list against OpenAlex OA locations."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sources" / "canonical_control_books.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sources" / "resolved_canonical_books.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "sources" / "resolved_canonical_books_summary.json"
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENLIBRARY_SEARCH = "https://openlibrary.org/search.json"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def surnames(authors: list[str]) -> set[str]:
    result = set()
    for author in authors:
        parts = normalize(author).split()
        if parts:
            result.add(parts[-1])
    return result


def candidate_authors(work: dict) -> list[str]:
    return [
        authorship.get("author", {}).get("display_name", "")
        for authorship in work.get("authorships", [])
        if authorship.get("author", {}).get("display_name")
    ]


def match_score(target: dict, work: dict) -> tuple[float, float, int]:
    title_similarity = SequenceMatcher(
        None, normalize(target["title"]), normalize(work.get("display_name", ""))
    ).ratio()
    overlap = len(surnames(target["authors"]) & surnames(candidate_authors(work)))
    score = title_similarity + min(overlap, 2) * 0.12
    return score, title_similarity, overlap


def pdf_locations(work: dict) -> list[dict]:
    locations = list(work.get("locations") or [])
    best = work.get("best_oa_location")
    if best:
        locations.append(best)
    results = []
    seen = set()
    for location in locations:
        url = location.get("pdf_url")
        if not url or url in seen:
            continue
        normalized_url = url.lower()
        if "frontmatter" in normalized_url or "/bfm" in normalized_url or "bfm%3a" in normalized_url:
            continue
        seen.add(url)
        source = location.get("source") or {}
        results.append(
            {
                "pdf_url": url,
                "landing_page_url": location.get("landing_page_url"),
                "license": location.get("license"),
                "version": location.get("version"),
                "source_name": source.get("display_name"),
                "host_organization_name": source.get("host_organization_name"),
            }
        )
    content_pdf = (work.get("content_urls") or {}).get("pdf")
    if (
        results
        and content_pdf
        and content_pdf not in seen
        and (work.get("open_access") or {}).get("is_oa")
    ):
        results.append(
            {
                "pdf_url": content_pdf,
                "landing_page_url": work.get("id"),
                "license": None,
                "version": "openalex_content",
                "source_name": "OpenAlex",
                "host_organization_name": "OpenAlex",
            }
        )
    return results


def load_local_books() -> list[dict]:
    books = []
    for lock_path in (
        PROJECT_ROOT / "data" / "raw" / "core_books" / "source_lock.json",
        PROJECT_ROOT / "data" / "raw" / "open_books" / "source_lock.json",
    ):
        if lock_path.exists():
            books.extend(json.loads(lock_path.read_text(encoding="utf-8")))
    return books


def local_match(target: dict, local_books: list[dict]) -> dict | None:
    target_title = normalize(target["title"])
    target_surnames = surnames(target["authors"])
    for book in local_books:
        if normalize(book.get("title") or "") != target_title:
            continue
        local_surnames = surnames(book.get("authors") or [])
        if not target_surnames or not local_surnames or target_surnames & local_surnames:
            return {
                "source_id": book.get("source_id"),
                "filename": book.get("filename"),
                "url": book.get("url"),
                "sha256": book.get("sha256"),
            }
    return None


def resolve_book(session: requests.Session, target: dict, per_page: int) -> dict:
    query = f"{target['title']} {target['authors'][0]}"
    response = session.get(
        OPENALEX_WORKS,
        params={"search": query, "filter": "type:book", "per-page": per_page},
        timeout=60,
    )
    response.raise_for_status()
    works = response.json().get("results", [])
    ranked = []
    for work in works:
        score, title_similarity, author_overlap = match_score(target, work)
        ranked.append((score, title_similarity, author_overlap, work))
    ranked.sort(key=lambda row: (row[0], row[3].get("cited_by_count", 0)), reverse=True)

    accepted = None
    for score, title_similarity, author_overlap, work in ranked:
        if title_similarity >= 0.94 or (title_similarity >= 0.82 and author_overlap >= 1):
            accepted = (score, title_similarity, author_overlap, work)
            break

    result = dict(target)
    result["query"] = query
    if not accepted:
        result.update({"status": "unresolved", "openalex_match": None, "pdf_candidates": []})
        return result

    score, title_similarity, author_overlap, work = accepted
    result.update(
        {
            "status": "oa_candidate" if pdf_locations(work) else "metadata_only",
            "openalex_match": {
                "id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("display_name"),
                "authors": candidate_authors(work),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "is_oa": (work.get("open_access") or {}).get("is_oa"),
                "title_similarity": round(title_similarity, 4),
                "author_surname_overlap": author_overlap,
                "combined_match_score": round(score, 4),
            },
            "pdf_candidates": pdf_locations(work),
        }
    )
    return result


def resolve_openlibrary(session: requests.Session, target: dict, per_page: int) -> dict | None:
    response = session.get(
        OPENLIBRARY_SEARCH,
        params={
            "title": target["title"],
            "author": target["authors"][0],
            "limit": per_page,
            "fields": (
                "key,title,author_name,first_publish_year,isbn,ia,"
                "public_scan_b,ebook_access"
            ),
        },
        timeout=60,
    )
    response.raise_for_status()
    ranked = []
    for work in response.json().get("docs", []):
        title_similarity = SequenceMatcher(
            None, normalize(target["title"]), normalize(work.get("title", ""))
        ).ratio()
        overlap = len(surnames(target["authors"]) & surnames(work.get("author_name") or []))
        score = title_similarity + min(overlap, 2) * 0.12
        ranked.append((score, title_similarity, overlap, work))
    ranked.sort(key=lambda row: row[0], reverse=True)
    for score, title_similarity, author_overlap, work in ranked:
        if title_similarity >= 0.94 or (title_similarity >= 0.82 and author_overlap >= 1):
            ebook_access = work.get("ebook_access")
            public_scan = bool(work.get("public_scan_b"))
            return {
                "key": work.get("key"),
                "title": work.get("title"),
                "authors": work.get("author_name") or [],
                "first_publish_year": work.get("first_publish_year"),
                "isbn": (work.get("isbn") or [])[:12],
                "internet_archive_ids": work.get("ia") or [],
                "public_scan": public_scan,
                "ebook_access": ebook_access,
                "full_text_access": (
                    "public" if public_scan or ebook_access == "public" else "restricted_or_none"
                ),
                "title_similarity": round(title_similarity, 4),
                "author_surname_overlap": author_overlap,
                "combined_match_score": round(score, 4),
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-books", type=int, default=0, help="0 resolves all targets")
    parser.add_argument("--min-rank", type=int, default=1, help="First canonical rank to resolve")
    parser.add_argument("--per-page", type=int, default=10)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument(
        "--merge", action="store_true", help="Replace resolved ranks in an existing output"
    )
    args = parser.parse_args()

    targets = json.loads(args.input.read_text(encoding="utf-8"))["books"]
    targets = [target for target in targets if target["rank"] >= args.min_rank]
    if args.max_books:
        targets = targets[: args.max_books]
    local_books = load_local_books()
    session = requests.Session()
    session.headers.update({"User-Agent": "controlai-canonical-resolver/0.1"})
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    results = []
    for index, target in enumerate(targets, start=1):
        local = local_match(target, local_books)
        if local:
            resolved = dict(target)
            resolved.update(
                {
                    "status": "already_local",
                    "local_copy": local,
                    "openalex_match": None,
                    "openlibrary_match": None,
                    "pdf_candidates": [],
                }
            )
            results.append(resolved)
            print(
                f"[{index}/{len(targets)}] {target['title']}: already_local "
                "(0 PDF candidates)"
            )
            continue
        try:
            resolved = resolve_book(session, target, args.per_page)
            openlibrary = resolve_openlibrary(session, target, args.per_page)
            resolved["openlibrary_match"] = openlibrary
            if resolved["status"] not in {"oa_candidate"} and openlibrary:
                resolved["status"] = (
                    "openlibrary_public_candidate"
                    if openlibrary["full_text_access"] == "public"
                    else "cataloged_no_open_fulltext"
                )
        except Exception as error:
            resolved = dict(target)
            resolved.update(
                {
                    "status": "resolver_error",
                    "error": f"{type(error).__name__}: {error}",
                    "openalex_match": None,
                    "openlibrary_match": None,
                    "pdf_candidates": [],
                }
            )
            if local:
                resolved["local_copy"] = local
                resolved["status"] = "already_local"
        results.append(resolved)
        print(
            f"[{index}/{len(targets)}] {target['title']}: {resolved['status']} "
            f"({len(resolved.get('pdf_candidates', []))} PDF candidates)"
        )
        if index < len(targets):
            time.sleep(args.delay_seconds)

    if args.merge and args.output.exists():
        previous = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merged = {result["rank"]: result for result in previous}
        merged.update({result["rank"]: result for result in results})
        results = [merged[rank] for rank in sorted(merged)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    statuses = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    summary = {
        "targets": len(results),
        "status_counts": dict(sorted(statuses.items())),
        "targets_with_pdf_candidates": sum(bool(result.get("pdf_candidates")) for result in results),
        "total_pdf_candidates": sum(len(result.get("pdf_candidates", [])) for result in results),
        "api": OPENALEX_WORKS,
        "catalog_api": OPENLIBRARY_SEARCH,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Resolved catalog: {args.output}")


if __name__ == "__main__":
    main()
