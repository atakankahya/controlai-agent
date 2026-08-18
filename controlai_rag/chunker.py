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
                chunk_id = f"{doc.metadata.get('filename', 'doc')}_c{chunk_index:04d}"
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
        chunk_id = f"{doc.metadata.get('filename', 'doc')}_c{chunk_index:04d}"
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_path=doc.source_path,
            metadata=dict(doc.metadata, chunk_index=chunk_index),
        ))

    return chunks
