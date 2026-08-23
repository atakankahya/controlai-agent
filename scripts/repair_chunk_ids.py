#!/usr/bin/env python3
"""One-off migration: give every indexed chunk a unique id.

`chunk_document` used to restart its counter on every page, so the corpus held
154 distinct chunk ids across 9,976 chunks. Anything keyed on chunk_id -- the
dense retriever's vector lookup, in particular -- silently resolved to the
wrong row.

`controlai_rag.chunker` now generates page-qualified ids, but the existing
index was built before that. Rewriting the ids in place is enough: the text is
untouched, and `chunks.json` and `embeddings.npz` were written in the same
order, so the vectors stay valid and nothing needs re-embedding.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_rag.index import INDEX_DIR


def main() -> int:
    chunks_path = INDEX_DIR / "chunks.json"
    vectors_path = INDEX_DIR / "embeddings.npz"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    before = len({c["chunk_id"] for c in chunks})
    new_ids = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "doc")
        page = meta.get("page")
        index = meta.get("chunk_index", 0)
        page_part = f"_p{int(page):05d}" if page is not None else ""
        new_ids.append(f"{filename}{page_part}_c{int(index):04d}")

    if len(set(new_ids)) != len(new_ids):
        # Fall back to a positional suffix rather than trade one collision
        # for another.
        seen: dict[str, int] = {}
        for i, cid in enumerate(new_ids):
            if cid in seen:
                seen[cid] += 1
                new_ids[i] = f"{cid}_{seen[cid]:03d}"
            else:
                seen[cid] = 0

    for chunk, cid in zip(chunks, new_ids):
        chunk["chunk_id"] = cid
    print(f"chunk ids: {before} unique -> {len(set(new_ids))} unique across {len(chunks)} chunks")

    backup = chunks_path.with_suffix(".json.pre-idfix")
    if not backup.exists():
        shutil.copy2(chunks_path, backup)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    if vectors_path.exists():
        data = np.load(vectors_path, allow_pickle=False)
        vectors = data["vectors"]
        if len(vectors) != len(chunks):
            print(f"WARNING: {len(vectors)} vectors vs {len(chunks)} chunks -- rebuild the dense index")
        else:
            np.savez_compressed(vectors_path, vectors=vectors, chunk_ids=np.array(new_ids, dtype="U"))
            print(f"Rewrote {vectors_path.name} with matching ids (no re-embedding needed)")

    from rank_bm25 import BM25Okapi

    from controlai_rag.index import tokenize_corpus

    print("Rebuilding BM25 ...")
    (INDEX_DIR / "bm25.pkl").write_bytes(
        pickle.dumps(BM25Okapi([tokenize_corpus(c["text"]) for c in chunks]))
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
