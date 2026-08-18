"""CLI tool to ingest PDFs, Markdown, and Text files into local RAG index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from controlai_rag.chunker import chunk_document
from controlai_rag.document_loader import load_directory
from controlai_rag.index import ControlRAGIndex

USER_DOCS_DIR = Path("data/user_docs")
SOURCES_DIR = Path("data/sources")


def ingest_documents(directories: list[Path]) -> int:
    index = ControlRAGIndex()
    all_chunks = []

    print("ControlAI RAG Ingestion Pipeline")
    print("=" * 50)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Scanning directory: {directory} ...")
        docs = load_directory(directory)
        print(f"  Found {len(docs)} document pages/files.")
        for doc in docs:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No documents found to index. Drop files (.pdf, .md, .txt) into data/user_docs/.")
        return 0

    print(f"\nBuilding BM25 index across {len(all_chunks)} total chunks...")
    index.build_from_chunks(all_chunks)
    print(f"Index successfully saved to data/rag_index/ ({len(all_chunks)} chunks).")
    return len(all_chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ControlAI local offline RAG index.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        nargs="+",
        default=[USER_DOCS_DIR, SOURCES_DIR],
        help="Directories containing PDF/MD/TXT documents to index",
    )
    args = parser.parse_args()
    ingest_documents(args.docs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
