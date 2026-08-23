#!/usr/bin/env python3
"""Bridge the processed corpus into the retrieval index.

`scripts/` builds a large pipeline -- raw downloads, extraction, chunking --
whose output lands in `data/processed/*_chunks/knowledge_chunks.jsonl` and
feeds *training dataset generation*. `ControlRAGIndex` was built separately and
only ever read `data/user_docs/`. Nothing connected the two, so retrieval saw
9,976 chunks of course notes plus two textbooks while 70,422 chunks of
canonical control literature -- Doyle/Francis/Tannenbaum, Astrom & Murray,
Rawlings/Mayne/Diehl, Sontag, Liberzon, Boyd, Soderstrom & Stoica -- sat on
disk unread.

This script merges them. The processed schema is richer than the index's: it
carries `source_title` and `source_authors`, which make far better citations
than the filename scrubbing `display_source_name` has to do for user uploads.

    python scripts/ingest_processed_corpus.py            # everything
    python scripts/ingest_processed_corpus.py --tiers core_books arxiv
    python scripts/ingest_processed_corpus.py --dry-run

Rebuilds BM25. Run `python -m controlai_rag.retriever --build` afterwards to
regenerate the dense index over the merged corpus.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_rag.index import INDEX_DIR, tokenize_corpus
from controlai_rag.textfix import repair

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MIN_CHARS = 120


def _tier_files(tiers: list[str] | None) -> list[tuple[str, Path]]:
    found = []
    for path in sorted(PROCESSED_DIR.glob("*_chunks/knowledge_chunks.jsonl")):
        tier = path.parent.name.removesuffix("_chunks")
        if tiers and tier not in tiers:
            continue
        found.append((tier, path))
    return found


def _to_index_chunk(tier: str, raw: dict) -> dict | None:
    text = repair(str(raw.get("text", "")).strip())
    if len(text) < MIN_CHARS:
        return None

    title = str(raw.get("source_title") or raw.get("source_id") or tier).strip()
    container = str(raw.get("container") or raw.get("member_path") or raw.get("document_id") or "")
    page = raw.get("page_start")
    try:
        page = int(page) if page not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    return {
        # Namespaced so a chunk id can never collide with one from another
        # tier or with the existing user_docs ids.
        "chunk_id": f"{tier}:{raw.get('chunk_id')}",
        "text": text,
        "source_path": f"data/processed/{tier}_chunks/{container}",
        "metadata": {
            "page": page,
            "page_end": raw.get("page_end"),
            # `filename` stays populated because the rest of the codebase reads
            # it; `source_title` is what citations should actually use.
            "filename": container or f"{title}.pdf",
            "source_title": title,
            "source_authors": raw.get("source_authors"),
            "corpus_tier": raw.get("corpus_tier") or tier,
            "source_coverage": raw.get("source_coverage"),
            "doc_type": "processed",
            "ingest_tier": tier,
        },
        "_sha": raw.get("text_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="*", help="only these tiers (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be added, write nothing")
    args = parser.parse_args()

    chunks_path = INDEX_DIR / "chunks.json"
    existing = json.loads(chunks_path.read_text(encoding="utf-8"))
    existing_ids = {c["chunk_id"] for c in existing}
    # Dedupe against what is already indexed, and across tiers, by content hash.
    import hashlib

    seen_hashes = {
        hashlib.sha256(c["text"].encode("utf-8")).hexdigest() for c in existing
    }
    print(f"existing index: {len(existing)} chunks")

    added: list[dict] = []
    for tier, path in _tier_files(args.tiers):
        kept = skipped_short = skipped_dupe = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunk = _to_index_chunk(tier, json.loads(line))
                if chunk is None:
                    skipped_short += 1
                    continue
                sha = chunk.pop("_sha", None) or hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
                if sha in seen_hashes or chunk["chunk_id"] in existing_ids:
                    skipped_dupe += 1
                    continue
                seen_hashes.add(sha)
                existing_ids.add(chunk["chunk_id"])
                added.append(chunk)
                kept += 1
        print(f"  {tier:22} +{kept:6d}   (short {skipped_short}, duplicate {skipped_dupe})")

    total = len(existing) + len(added)
    print(f"\nwould index {total} chunks ({len(existing)} existing + {len(added)} new)")
    if args.dry_run:
        return 0

    backup = chunks_path.with_suffix(".json.pre-corpus")
    if not backup.exists():
        shutil.copy2(chunks_path, backup)
        print(f"backed up existing index to {backup.name}")

    merged = existing + added
    print("writing chunks.json ...")
    chunks_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")

    from rank_bm25 import BM25Okapi

    print(f"building BM25 over {len(merged)} chunks (this takes a few minutes) ...")
    bm25 = BM25Okapi([tokenize_corpus(c["text"]) for c in merged])
    (INDEX_DIR / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    print("done. Now run: python -m controlai_rag.retriever --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
