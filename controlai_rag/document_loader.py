"""Document loader supporting PDF, Markdown, Text, and JSONL corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class Document:
    def __init__(self, content: str, source_path: str, metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.source_path = source_path
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }


def load_pdf(path: Path) -> list[Document]:
    docs = []
    try:
        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    content=text.strip(),
                    source_path=str(path),
                    metadata={"page": idx, "filename": path.name, "doc_type": "pdf"},
                ))
    except Exception as exc:
        print(f"Warning: Failed to load PDF {path}: {exc}")
    return docs


def load_text(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return [Document(
                content=text.strip(),
                source_path=str(path),
                metadata={"filename": path.name, "doc_type": path.suffix.lstrip(".")},
            )]
    except Exception as exc:
        print(f"Warning: Failed to load text file {path}: {exc}")
    return []


def load_single_file(path: Path) -> list[Document]:
    """Load a single file based on its extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".md", ".txt"):
        return load_text(path)
    return []


def load_directory(dir_path: Path) -> list[Document]:
    docs = []
    if not dir_path.exists():
        return docs

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdf(file_path))
        elif suffix in (".md", ".txt", ".rst", ".py", ".m", ".json"):
            docs.extend(load_text(file_path))
    return docs
