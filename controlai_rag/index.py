"""Local Offline Hybrid BM25 Index with metadata preservation and citation support."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from controlai_rag.chunker import Chunk

INDEX_DIR = Path("data/rag_index")


def tokenize_corpus(text: str) -> list[str]:
    # Lowercase and extract alphanumeric tokens + mathematical symbols
    return re.findall(r"\b\w+\b|[+\-*/^_]", text.lower())


class ControlRAGIndex:
    """Fast, local, offline search index over control engineering documents."""

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index_dir = index_dir
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self._load_if_exists()

    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = [c.to_dict() for c in chunks]
        corpus = [tokenize_corpus(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.save()

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """Add new chunks to the existing index and rebuild BM25."""
        new_dict_chunks = [c.to_dict() for c in new_chunks]
        self.chunks.extend(new_dict_chunks)
        corpus = [tokenize_corpus(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        self.save()

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with (self.index_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with (self.index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f)

    def _load_if_exists(self) -> bool:
        chunks_file = self.index_dir / "chunks.json"
        bm25_file = self.index_dir / "bm25.pkl"
        if chunks_file.exists() and bm25_file.exists():
            try:
                with chunks_file.open("r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                with bm25_file.open("rb") as f:
                    self.bm25 = pickle.load(f)
                return True
            except Exception as exc:
                print(f"Warning: Failed to load existing index: {exc}")
        return False

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize_corpus(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
            chunk = self.chunks[idx]
            if source_filter and source_filter.lower() not in chunk["source_path"].lower():
                continue
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 3),
                "source": chunk["source_path"],
                "filename": chunk["metadata"].get("filename", "unknown"),
                "page": chunk["metadata"].get("page"),
                "text": chunk["text"],
            })
            if len(results) >= top_k:
                break

        return results
