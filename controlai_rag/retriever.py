"""Hybrid retrieval over the local control-engineering library.

The previous agent grounded its answers on BM25 alone, gated on a raw score
threshold of 2.5. BM25 scores are unbounded and corpus-dependent, so that gate
passed nearly everything: asking about the Bode sensitivity integral retrieved
Routh-Hurwitz tables at score 19.7 and injected them as authoritative context.

This layer fixes both halves of that. Lexical and dense rankings are fused with
reciprocal rank fusion, and the result is gated on cosine similarity -- a
bounded, comparable quantity -- so that when the library genuinely has nothing
relevant, nothing is injected and the model answers from its own knowledge
instead of from a mismatched passage.

Build the dense side with:  python -m controlai_rag.retriever --build
Without it, retrieval degrades to lexical-only rather than failing.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from controlai_rag.index import INDEX_DIR, display_source_name, get_shared_index

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"

# Cosine similarity below this means the corpus has nothing useful on the topic.
# Re-measure with `./run.sh --calibrate` whenever the corpus changes size
# materially -- this is a property of the model *and* the corpus, not a
# constant. Against the 80,370-chunk corpus:
#   in-domain worst best-match   0.678   (mean 0.748, 12 probes across
#                                         classical/modern/optimal/robust/
#                                         nonlinear/estimation/MPC)
#   off-domain best  best-match  0.571   (mean 0.432, 5 probes)
# 0.62 sits in that 0.108-wide gap. The off-domain outlier is a pollen-allergy
# query at 0.571, pulled up by biomedical material in the open_books tier --
# everything genuinely unrelated lands near 0.42.
# Overridable because the Space embeds queries with the bf16 transformers
# checkpoint rather than the MLX 4-bit one the index was built with; the two
# agree closely, but this is the knob if the gate turns out mis-set there.
MIN_COSINE = float(os.environ.get("CONTROLAI_MIN_COSINE", "0.62"))
# Standard RRF constant; damps the influence of any single ranker's tail.
RRF_K = 60

# A back-of-book index page matches almost any control query -- it contains
# every term in the field -- but carries no explanation whatsoever. Measured
# across the corpus, index-entry density ("Thermal systems, 100,136-39") is 0
# for the median chunk and 6.7 at the 99th percentile, while index pages run
# above 30, so this cleanly separates them without touching equation-dense
# prose.
_INDEX_ENTRY_RE = re.compile(r"[A-Za-z)],\s*\d")
_MAX_INDEX_DENSITY = 12.0  # entries per 1000 characters
_MIN_USEFUL_CHARS = 120


def _is_low_value(text: str) -> bool:
    """True for chunks that can match well but cannot inform an answer."""
    stripped = text.strip()
    if len(stripped) < _MIN_USEFUL_CHARS:
        return True
    density = 1000.0 * len(_INDEX_ENTRY_RE.findall(stripped)) / len(stripped)
    if density > _MAX_INDEX_DENSITY:
        return True
    # Mojibake from a failed PDF extraction: mostly characters outside the
    # Latin/Greek/mathematical ranges any real passage is written in.
    exotic = sum(1 for c in stripped if ord(c) > 0x2200)
    return exotic / len(stripped) > 0.25


class HybridRetriever:
    """Fuses BM25 and dense retrieval, and declines to return weak matches."""

    def __init__(self, index=None, embeddings_path: Path = EMBEDDINGS_PATH) -> None:
        self.index = index or get_shared_index()
        self.embeddings_path = embeddings_path
        self.vectors: np.ndarray | None = None
        self.vector_ids: list[str] = []
        self._row_of: dict[str, int] = {}
        self._embedder = None
        self._load_vectors()

    def _load_vectors(self) -> None:
        if not self.embeddings_path.exists():
            print(f"[retriever] no dense index at {self.embeddings_path}; lexical-only mode")
            return
        data = np.load(self.embeddings_path, allow_pickle=False)
        self.vectors = data["vectors"].astype(np.float32)
        self.vector_ids = [str(x) for x in data["chunk_ids"]]
        self._row_of = {cid: i for i, cid in enumerate(self.vector_ids)}
        if len(self.vector_ids) != len(self.index.chunks):
            print(
                f"[retriever] dense index covers {len(self.vector_ids)} chunks but the "
                f"corpus holds {len(self.index.chunks)}; rebuild to include the rest"
            )

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None and len(self.vector_ids) > 0

    def _dense_rank(self, query: str, depth: int) -> list[tuple[str, float]]:
        if not self.has_dense:
            return []
        from controlai_rag.embeddings import get_embedder

        if self._embedder is None:
            self._embedder = get_embedder()
        q = self._embedder.encode_query(query)
        sims = self.vectors @ q
        top = np.argpartition(-sims, min(depth, len(sims) - 1))[:depth]
        top = top[np.argsort(-sims[top])]
        return [(self.vector_ids[i], float(sims[i])) for i in top]

    def search(self, query: str, top_k: int = 4, depth: int = 40) -> list[dict[str, Any]]:
        """Return the passages worth showing the model, best first.

        An empty list is a valid and common answer: it means the library has
        nothing on this question.
        """
        by_id = {c["chunk_id"]: c for c in self.index.chunks}

        lexical = self.index.search(query, top_k=depth)
        dense = self._dense_rank(query, depth)
        if not lexical and not dense:
            return []

        cosine = {cid: score for cid, score in dense}
        # Score the lexical candidates densely as well. Without this, a chunk
        # that BM25 ranked first but that fell outside the dense top-`depth`
        # has no similarity, and the gate below drops it for lacking a score
        # rather than for being irrelevant -- which silently discarded the best
        # keyword matches in the corpus.
        if self.has_dense and lexical:
            missing = [h["chunk_id"] for h in lexical if h["chunk_id"] not in cosine]
            rows = [(cid, self._row_of[cid]) for cid in missing if cid in self._row_of]
            if rows:
                if self._embedder is None:
                    from controlai_rag.embeddings import get_embedder

                    self._embedder = get_embedder()
                q = self._embedder.encode_query(query)
                sub = self.vectors[[r for _, r in rows]] @ q
                cosine.update({cid: float(score) for (cid, _), score in zip(rows, sub)})

        fused: dict[str, float] = {}
        for rank, hit in enumerate(lexical):
            fused[hit["chunk_id"]] = fused.get(hit["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[dict[str, Any]] = []
        for chunk_id, fusion_score in ordered:
            chunk = by_id.get(chunk_id)
            if chunk is None or _is_low_value(str(chunk.get("text", ""))):
                continue
            similarity = cosine.get(chunk_id)
            # With a dense index available, similarity is the gate. Without one
            # there is no bounded relevance signal, so lexical order stands and
            # the caller gets fewer, higher-ranked passages instead.
            if self.has_dense:
                if similarity is None or similarity < MIN_COSINE:
                    continue
            elif len(results) >= 2:
                break
            meta = chunk.get("metadata", {})
            # Chunks bridged in from data/processed carry a real bibliographic
            # title, which beats anything that can be recovered from a
            # filename. display_source_name is the fallback for user uploads,
            # whose filenames carry owner initials and course codes.
            title = meta.get("source_title")
            if title:
                label, is_published = str(title), meta.get("corpus_tier") != "user_docs"
            else:
                label, is_published = display_source_name(meta.get("filename", "unknown"))
            results.append(
                {
                    "chunk_id": chunk_id,
                    "label": label,
                    "is_published_work": is_published,
                    "page": chunk["metadata"].get("page"),
                    "text": chunk["text"],
                    "similarity": round(similarity, 4) if similarity is not None else None,
                    "fusion_score": round(fusion_score, 5),
                }
            )
            if len(results) >= top_k:
                break
        return results


    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed newly ingested chunks so they are searchable immediately.

        Documents uploaded through the web UI are appended to the lexical index
        live. Without this they would have no vector, and since the relevance
        gate requires a cosine score, they would be silently unreachable until
        the whole dense index was rebuilt.
        """
        if not self.has_dense or not chunks:
            return 0
        from controlai_rag.embeddings import get_embedder

        if self._embedder is None:
            self._embedder = get_embedder()
        texts = [str(c.get("text", "")) for c in chunks]
        new_vectors = self._embedder.encode_documents(texts, progress_every=0)
        self.vectors = np.vstack([self.vectors, new_vectors.astype(np.float32)])
        self.vector_ids.extend(str(c["chunk_id"]) for c in chunks)
        return len(chunks)


_shared: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _shared
    if _shared is None:
        _shared = HybridRetriever()
    return _shared


def build(output: Path = EMBEDDINGS_PATH, slab: int = 4000) -> None:
    """Embed every chunk in the corpus and persist the vectors.

    Checkpointed. Embedding an 80k-chunk corpus takes over half an hour, and
    writing the result only at the end meant a single interruption threw all of
    it away -- observed directly at 72,008 of 80,370 chunks. Progress is now
    flushed to a sidecar file every `slab` chunks, and a re-run picks up from
    whatever is already there, so an interrupt costs a few minutes at most.
    """
    from controlai_rag.embeddings import get_embedder

    index = get_shared_index()
    chunks = index.chunks
    embedder = get_embedder()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(".partial.npz")

    done: dict[str, np.ndarray] = {}
    if partial_path.exists():
        cached = np.load(partial_path, allow_pickle=False)
        done = {str(cid): vec for cid, vec in zip(cached["chunk_ids"], cached["vectors"])}
        print(f"Resuming: {len(done)} chunks already embedded")

    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"Embedding {len(todo)} of {len(chunks)} chunks with {embedder.model_id} ...")

    def flush() -> None:
        ids = list(done.keys())
        np.savez_compressed(
            partial_path,
            vectors=np.stack([done[i] for i in ids]).astype(np.float16),
            chunk_ids=np.array(ids, dtype="U"),
        )

    for start in range(0, len(todo), slab):
        batch = todo[start : start + slab]
        vectors = embedder.encode_documents(
            [str(c.get("text", "")) for c in batch], progress_every=0
        )
        for chunk, vector in zip(batch, vectors):
            done[chunk["chunk_id"]] = vector.astype(np.float16)
        flush()
        print(f"  {len(done)}/{len(chunks)} embedded (checkpointed)", flush=True)

    # Emit in corpus order so the vector rows line up with chunks.json.
    ordered = [c["chunk_id"] for c in chunks if c["chunk_id"] in done]
    np.savez_compressed(
        output,
        vectors=np.stack([done[i] for i in ordered]).astype(np.float16),
        chunk_ids=np.array(ordered, dtype="U"),
    )
    partial_path.unlink(missing_ok=True)
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(ordered)} vectors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or probe the dense retrieval index")
    parser.add_argument("--build", action="store_true", help="embed the corpus and write the index")
    parser.add_argument("--query", type=str, help="run a test query against the current index")
    args = parser.parse_args()
    if args.build:
        build()
    if args.query:
        for hit in get_retriever().search(args.query):
            print(f"{hit['similarity']}  {hit['label']} p.{hit['page']}  {hit['text'][:110]!r}")
