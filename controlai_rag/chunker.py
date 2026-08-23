"""Structure-preserving semantic chunker for mathematical and engineering text."""

from __future__ import annotations

import re
from typing import Any

from controlai_rag.document_loader import Document


class Chunk:
    def __init__(self, text: str, chunk_id: str, source_path: str, metadata: dict[str, Any]) -> None:
        self.text = text
        self.chunk_id = chunk_id
        self.source_path = source_path
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }


def _chunk_id(doc: Document, chunk_index: int) -> str:
    """A chunk id that is unique across the whole corpus.

    `chunk_document` is called once per page, so a counter that restarts at
    zero for each call made every page's first chunk `<file>_c0000`. The corpus
    ended up with 154 distinct ids for 9,976 chunks, which silently broke
    anything keyed on chunk_id. Including the page makes the id unique, since
    (filename, page, index-within-page) is.
    """
    filename = doc.metadata.get("filename", "doc")
    page = doc.metadata.get("page")
    page_part = f"_p{int(page):05d}" if page is not None else ""
    return f"{filename}{page_part}_c{chunk_index:04d}"


def chunk_document(doc: Document, max_words: int = 350, overlap_words: int = 50) -> list[Chunk]:
    """Split document into coherent chunks with overlap, preserving paragraphs."""
    paragraphs = re.split(r"\n\s*\n", doc.content)
    chunks: list[Chunk] = []

    current_words: list[str] = []
    chunk_index = 0

    for para in paragraphs:
        para_words = para.strip().split()
        if not para_words:
            continue

        if len(current_words) + len(para_words) <= max_words:
            current_words.extend(para_words)
        else:
            if current_words:
                chunk_text = " ".join(current_words)
                chunk_id = _chunk_id(doc, chunk_index)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_path=doc.source_path,
                    metadata=dict(doc.metadata, chunk_index=chunk_index),
                ))
                chunk_index += 1
                # Overlap
                current_words = current_words[-overlap_words:] + para_words
            else:
                current_words = para_words

    if current_words:
        chunk_text = " ".join(current_words)
        chunk_id = _chunk_id(doc, chunk_index)
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_path=doc.source_path,
            metadata=dict(doc.metadata, chunk_index=chunk_index),
        ))

    return chunks
