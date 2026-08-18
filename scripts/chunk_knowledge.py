#!/usr/bin/env python3
"""Create tokenizer-aware, provenance-preserving chunks from the knowledge pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "pools" / "knowledge.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chunks" / "knowledge_chunks.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "chunks" / "summary.json"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def text_hash(text: str) -> str:
    canonical = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    paragraphs = [
        part
        for part in paragraphs
        if not re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,8})", part, flags=re.I)
    ]
    return paragraphs or ([text.strip()] if text.strip() else [])


def token_windows(text: str, tokenizer, max_tokens: int, overlap_tokens: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    stride = max_tokens - overlap_tokens
    windows = []
    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + max_tokens]
        if not window:
            break
        decoded = tokenizer.decode(window, skip_special_tokens=True).strip()
        while len(tokenizer.encode(decoded, add_special_tokens=False)) > max_tokens:
            window = window[:-1]
            decoded = tokenizer.decode(window, skip_special_tokens=True).strip()
        windows.append(decoded)
        if start + max_tokens >= len(token_ids):
            break
    return [window for window in windows if window]


def page_blocks(row: dict, tokenizer, max_tokens: int, overlap_tokens: int) -> list[dict]:
    blocks = []
    for paragraph in split_paragraphs(row["text"]):
        for piece in token_windows(paragraph, tokenizer, max_tokens, overlap_tokens):
            blocks.append(
                {
                    "text": piece,
                    "tokens": len(tokenizer.encode(piece, add_special_tokens=False)),
                    "page_number": row["page_number"],
                    "unit_id": row["unit_id"],
                }
            )
    return blocks


def emit_chunk(document_rows: list[dict], blocks: list[dict], ordinal: int, tokenizer) -> dict:
    first = document_rows[0]
    text = "\n\n".join(block["text"] for block in blocks).strip()
    pages = [block["page_number"] for block in blocks if block["page_number"] is not None]
    chunk_key = f"{first['document_id']}:{ordinal}:{text_hash(text)}"
    return {
        "chunk_id": hashlib.sha256(chunk_key.encode()).hexdigest()[:24],
        "document_id": first["document_id"],
        "source_id": first["source_id"],
        "source_title": first.get("source_title"),
        "source_authors": first.get("source_authors", []),
        "corpus_tier": first.get("corpus_tier"),
        "source_coverage": first.get("source_coverage", []),
        "container": first["container"],
        "member_path": first["member_path"],
        "content_role": first["content_role"],
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "source_unit_ids": list(dict.fromkeys(block["unit_id"] for block in blocks)),
        "text": text,
        "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        "text_sha256": text_hash(text),
    }


def block_token_count(blocks: list[dict], tokenizer) -> int:
    if not blocks:
        return 0
    text = "\n\n".join(block["text"] for block in blocks)
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_document(
    rows: list[dict], tokenizer, min_tokens: int, target_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[dict]:
    rows.sort(key=lambda row: (row["page_number"] is None, row["page_number"] or 0, row["unit_id"]))
    all_blocks = []
    for row in rows:
        all_blocks.extend(page_blocks(row, tokenizer, max_tokens, overlap_tokens))

    chunk_blocks = []
    current = []
    current_tokens = 0
    for block in all_blocks:
        candidate = current + [block]
        candidate_tokens = block_token_count(candidate, tokenizer)
        if current and candidate_tokens > max_tokens:
            chunk_blocks.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens = block_token_count(current, tokenizer)
        if current_tokens >= target_tokens:
            chunk_blocks.append(current)
            current = []
            current_tokens = 0
    if current:
        chunk_blocks.append(current)

    if len(chunk_blocks) > 1 and block_token_count(chunk_blocks[-1], tokenizer) < min_tokens:
        previous, tail = chunk_blocks[-2], chunk_blocks[-1]
        if block_token_count(previous + tail, tokenizer) <= max_tokens:
            chunk_blocks[-2:] = [previous + tail]
        else:
            while (
                block_token_count(tail, tokenizer) < min_tokens
                and len(previous) > 1
                and block_token_count(previous[:-1], tokenizer) >= min_tokens
            ):
                tail.insert(0, previous.pop())

    return [
        emit_chunk(rows, blocks, ordinal, tokenizer)
        for ordinal, blocks in enumerate(chunk_blocks, start=1)
    ]


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument(
        "--drop-below-tokens",
        type=int,
        default=40,
        help="Discard isolated fragments shorter than this after chunking",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.overlap_tokens < args.max_tokens:
        parser.error("--overlap-tokens must be non-negative and smaller than --max-tokens")
    if not 0 < args.min_tokens <= args.target_tokens <= args.max_tokens:
        parser.error("Require 0 < min tokens <= target tokens <= max tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )
    by_document: dict[str, list[dict]] = defaultdict(list)
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_document[row["document_id"]].append(row)

    document_ids = sorted(by_document)
    if args.max_documents is not None:
        document_ids = document_ids[: args.max_documents]

    chunks = []
    for index, document_id in enumerate(document_ids, start=1):
        chunks.extend(
            chunk_document(
                by_document[document_id],
                tokenizer,
                args.min_tokens,
                args.target_tokens,
                args.max_tokens,
                args.overlap_tokens,
            )
        )
        if index % 50 == 0 or index == len(document_ids):
            print(f"Chunked {index}/{len(document_ids)} documents")

    short_chunks_removed = sum(
        chunk["token_count"] < args.drop_below_tokens for chunk in chunks
    )
    chunks = [
        chunk for chunk in chunks if chunk["token_count"] >= args.drop_below_tokens
    ]
    raw_chunk_count = len(chunks)
    unique_chunks = []
    seen_hashes = set()
    for chunk in chunks:
        if chunk["text_sha256"] in seen_hashes:
            continue
        seen_hashes.add(chunk["text_sha256"])
        unique_chunks.append(chunk)
    chunks = unique_chunks
    duplicate_chunks_removed = raw_chunk_count - len(chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [chunk["token_count"] for chunk in chunks]
    duplicate_hashes = Counter(chunk["text_sha256"] for chunk in chunks)
    summary = {
        "tokenizer": args.tokenizer,
        "documents": len(document_ids),
        "chunks": len(chunks),
        "short_chunks_removed": short_chunks_removed,
        "duplicate_chunks_removed": duplicate_chunks_removed,
        "total_tokens": sum(token_counts),
        "min_tokens": min(token_counts, default=0),
        "median_tokens": percentile(token_counts, 0.5),
        "p90_tokens": percentile(token_counts, 0.9),
        "max_tokens": max(token_counts, default=0),
        "chunks_over_limit": sum(count > args.max_tokens for count in token_counts),
        "exact_duplicate_chunks_beyond_first": sum(count - 1 for count in duplicate_hashes.values()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Chunks:  {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
