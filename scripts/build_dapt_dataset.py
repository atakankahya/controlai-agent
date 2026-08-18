#!/usr/bin/env python3
"""Build a deterministic, provenance-preserving ControlAI DAPT dataset.

The default profile targets roughly 20 million high-quality tokens. Lower-value
supplemental books and research papers are capped so they cannot swamp canonical
books, courses, and mathematical foundations. Proprietary MathWorks text is not
included; it remains a local RAG/tool-reference layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "dapt_v1"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class Pool:
    name: str
    path: Path
    token_cap: int | None
    priority: int


DEFAULT_POOLS = (
    Pool(
        "canonical_books",
        PROJECT_ROOT / "data/processed/core_books_chunks/knowledge_chunks.jsonl",
        None,
        0,
    ),
    Pool(
        "control_courses_and_code",
        PROJECT_ROOT / "data/processed/chunks/knowledge_chunks.jsonl",
        None,
        1,
    ),
    Pool(
        "advanced_control_courses",
        PROJECT_ROOT / "data/processed/advanced_control_chunks/knowledge_chunks.jsonl",
        None,
        2,
    ),
    Pool(
        "university_web_collections",
        PROJECT_ROOT / "data/processed/web_collections_chunks/knowledge_chunks.jsonl",
        None,
        3,
    ),
    Pool(
        "math_signal_numerical_foundations",
        PROJECT_ROOT / "data/processed/general_foundations_chunks/knowledge_chunks.jsonl",
        6_000_000,
        4,
    ),
    Pool(
        "control_research",
        PROJECT_ROOT / "data/processed/arxiv_chunks/knowledge_chunks.jsonl",
        4_000_000,
        5,
    ),
    Pool(
        "supplemental_open_books",
        PROJECT_ROOT / "data/processed/open_books_chunks/knowledge_chunks.jsonl",
        5_000_000,
        6,
    ),
)


def canonical_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).casefold().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_fraction(value: str) -> float:
    raw = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def simhash64(text: str) -> int:
    """Conservative near-duplicate fingerprint over sampled word 4-grams."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    features = [" ".join(words[index : index + 4]) for index in range(0, len(words) - 3, 2)]
    if not features:
        features = words
    accumulator = [0] * 64
    for feature in features:
        value = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            accumulator[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, score in enumerate(accumulator):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def remove_near_duplicates(
    rows: list[dict], max_hamming: int
) -> tuple[list[dict], Counter[str]]:
    """Keep priority-first rows; use four 16-bit LSH bands for candidates."""
    kept: list[dict] = []
    fingerprints: list[int] = []
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    removed: Counter[str] = Counter()
    mask = (1 << 16) - 1
    for row in rows:
        fingerprint = simhash64(row["text"])
        candidate_indices: set[int] = set()
        for band in range(4):
            value = (fingerprint >> (band * 16)) & mask
            candidate_indices.update(bands[(band, value)])
        duplicate = any(
            (fingerprint ^ fingerprints[index]).bit_count() <= max_hamming
            for index in candidate_indices
        )
        if duplicate:
            removed[row["source_pool"]] += 1
            continue
        row["simhash64"] = f"{fingerprint:016x}"
        index = len(kept)
        kept.append(row)
        fingerprints.append(fingerprint)
        for band in range(4):
            value = (fingerprint >> (band * 16)) & mask
            bands[(band, value)].append(index)
    return kept, removed


def quality_reason(text: str, token_count: int) -> str | None:
    if token_count < 80:
        return "too_short"
    if not text:
        return "empty"
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    if printable / len(text) < 0.98:
        return "nonprintable_noise"
    alphabetic = sum(character.isalpha() for character in text)
    if alphabetic / len(text) < 0.35:
        return "low_alphabetic_ratio"
    if re.search(r"(.)\1{12,}", text):
        return "repeated_character_noise"
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 25:
        return "too_few_words"
    return None


def rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: invalid chunk record")
            yield row


def select_pool(pool: Pool, tokenizer) -> tuple[list[dict], Counter[str]]:
    candidates = []
    rejected: Counter[str] = Counter()
    for row in rows(pool.path):
        text = canonical_text(row["text"])
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        reason = quality_reason(text, token_count)
        if reason:
            rejected[reason] += 1
            continue
        candidates.append(
            {
                "text": text,
                "token_count": token_count,
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "source_id": row.get("source_id"),
                "source_title": row.get("source_title"),
                "source_pool": pool.name,
                "source_priority": pool.priority,
                "text_sha256": text_hash(text),
            }
        )

    # Stable hash sampling avoids a leading-document bias when a pool is capped.
    candidates.sort(key=lambda row: stable_fraction(row["text_sha256"]))
    if pool.token_cap is None:
        return candidates, rejected
    selected = []
    used = 0
    for row in candidates:
        if used + row["token_count"] > pool.token_cap and selected:
            continue
        selected.append(row)
        used += row["token_count"]
        if used >= pool.token_cap:
            break
    return selected, rejected


def write_jsonl(path: Path, dataset: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in dataset:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_key(row: dict) -> str:
    """Namespace document ids because independent source pools reuse filenames."""
    identity = row.get("document_id") or row["text_sha256"]
    return f"{row['source_pool']}::{identity}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--validation-percent",
        type=float,
        default=1.0,
        help="Document-level validation percentage",
    )
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=3,
        help="Drop SimHash candidates at or below this Hamming distance",
    )
    args = parser.parse_args()
    if not 0 < args.validation_percent < 20:
        parser.error("--validation-percent must be between 0 and 20")
    if not 0 <= args.near_duplicate_hamming <= 8:
        parser.error("--near-duplicate-hamming must be between 0 and 8")

    missing = [str(pool.path) for pool in DEFAULT_POOLS if not pool.path.exists()]
    if missing:
        raise FileNotFoundError("Missing chunk pools:\n" + "\n".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )
    selected = []
    pool_stats = {}
    for pool in DEFAULT_POOLS:
        pool_rows, rejected = select_pool(pool, tokenizer)
        selected.extend(pool_rows)
        pool_stats[pool.name] = {
            "rows_before_cross_pool_dedup": len(pool_rows),
            "tokens_before_cross_pool_dedup": sum(
                row["token_count"] for row in pool_rows
            ),
            "rejected": dict(sorted(rejected.items())),
            "token_cap": pool.token_cap,
        }
        print(
            f"{pool.name}: {len(pool_rows):,} rows, "
            f"{pool_stats[pool.name]['tokens_before_cross_pool_dedup']:,} tokens"
        )

    # Higher-priority canonical sources win cross-pool exact duplicates.
    selected.sort(key=lambda row: (row["source_priority"], row["text_sha256"]))
    unique = []
    seen_hashes: set[str] = set()
    duplicates_by_pool: Counter[str] = Counter()
    for row in selected:
        if row["text_sha256"] in seen_hashes:
            duplicates_by_pool[row["source_pool"]] += 1
            continue
        seen_hashes.add(row["text_sha256"])
        unique.append(row)

    unique, near_duplicates_by_pool = remove_near_duplicates(
        unique, args.near_duplicate_hamming
    )

    # Stratify by source pool and keep entire documents together. A pure hash
    # threshold can accidentally leave a small canonical pool unrepresented.
    documents_by_pool: dict[str, set[str]] = defaultdict(set)
    for row in unique:
        documents_by_pool[row["source_pool"]].add(document_key(row))
    valid_documents: set[str] = set()
    for pool_name, document_ids in documents_by_pool.items():
        ordered = sorted(
            document_ids,
            key=lambda value: stable_fraction(f"validation:{pool_name}:{value}"),
        )
        count = max(1, round(len(ordered) * args.validation_percent / 100.0))
        valid_documents.update(ordered[:count])

    train = []
    valid = []
    for row in unique:
        identity = document_key(row)
        destination = valid if identity in valid_documents else train
        destination.append(row)

    # Stable shuffle prevents source blocks while preserving reproducibility.
    train.sort(key=lambda row: stable_fraction(f"train:{row['text_sha256']}"))
    valid.sort(key=lambda row: stable_fraction(f"valid:{row['text_sha256']}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    summary = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "profile": "high_quality_approximately_20m",
        "proprietary_mathworks_included": False,
        "validation_split": "source_pool_stratified_document_hash",
        "validation_percent": args.validation_percent,
        "train_rows": len(train),
        "train_tokens": sum(row["token_count"] for row in train),
        "valid_rows": len(valid),
        "valid_tokens": sum(row["token_count"] for row in valid),
        "unique_documents": len({document_key(row) for row in unique}),
        "cross_pool_exact_duplicates_removed": sum(duplicates_by_pool.values()),
        "duplicates_removed_by_pool": dict(sorted(duplicates_by_pool.items())),
        "near_duplicate_hamming_threshold": args.near_duplicate_hamming,
        "near_duplicates_removed": sum(near_duplicates_by_pool.values()),
        "near_duplicates_removed_by_pool": dict(
            sorted(near_duplicates_by_pool.items())
        ),
        "pools": pool_stats,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
