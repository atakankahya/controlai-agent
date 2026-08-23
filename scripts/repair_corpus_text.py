#!/usr/bin/env python3
"""One-off migration: repair PDF font-map damage in the existing RAG index.

The chunks were extracted before `controlai_rag.textfix` existed, so the damage
is baked into `data/rag_index/chunks.json`. The corruption is deterministic, so
the chunks can be repaired in place -- no re-extraction of the source PDFs, and
chunk ids stay stable, which keeps the dense index aligned.

Rebuilds BM25 here; run `python -m controlai_rag.retriever --build` afterwards
to regenerate the embeddings from the repaired text.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_rag.index import INDEX_DIR, tokenize_corpus
from controlai_rag.textfix import looks_damaged, repair


def main() -> int:
    chunks_path = INDEX_DIR / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    backup = chunks_path.with_suffix(".json.pre-textfix")
    if not backup.exists():
        shutil.copy2(chunks_path, backup)
        print(f"Backed up original to {backup.name}")

    damaged = repaired = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if not looks_damaged(text):
            continue
        damaged += 1
        fixed = repair(text)
        if fixed != text:
            chunk["text"] = fixed
            repaired += 1

    print(f"{damaged} damaged chunks found, {repaired} repaired, {len(chunks)} total")
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    from rank_bm25 import BM25Okapi

    print("Rebuilding BM25 over the repaired text ...")
    bm25 = BM25Okapi([tokenize_corpus(c["text"]) for c in chunks])
    (INDEX_DIR / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    print("Done. Now run: python -m controlai_rag.retriever --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
