"""Local Offline Hybrid BM25 Index with metadata preservation and citation support."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from controlai_rag.chunker import Chunk

INDEX_DIR = Path("data/rag_index")


def tokenize_corpus(text: str) -> list[str]:
    # Lowercase and extract alphanumeric tokens + mathematical symbols
    return re.findall(r"\b\w+\b|[+\-*/^_]", text.lower())


# Only chunk body text is tokenized into BM25, so a query naming a source
# ("what does Nise say about stability") cannot match the book it refers to.
# These constants drive a post-retrieval boost for chunks whose *filename*
# matches a distinctive query term.
SOURCE_MATCH_BOOST = 1.6
# A filename token is only distinctive enough to boost on if it appears in at
# most this fraction of the indexed files (floored at 2 files, so the rule
# still works on a small corpus). An author surname like "nise" or "ogata"
# appears in exactly one file; generic domain words appear in many.
SOURCE_TOKEN_MAX_FILE_FRACTION = 0.02
# Words that identify the subject rather than a specific source. Even if the
# corpus grows large enough for these to slip under the fraction cutoff, they
# must never trigger a source boost.
GENERIC_SOURCE_TOKENS = {
    "control", "controls", "systems", "system", "engineering", "theory",
    "lecture", "lectures", "notes", "chapter", "solutions", "solution",
    "exercise", "exercises", "book", "textbook", "txtbk", "edition", "vol",
    "part", "final", "exam", "slides", "course", "intro", "introduction",
}


class ControlRAGIndex:
    """Fast, local, offline search index over control engineering documents."""

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index_dir = index_dir
        self.chunks: list[dict[str, Any]] = []
        self.bm25: Any | None = None
        self._distinctive_source_tokens: set[str] = set()
        self._load_if_exists()
        self._build_source_token_map()

    def _build_source_token_map(self) -> None:
        """Index which filename tokens are distinctive enough to boost on.

        Author surnames and title words unique to a few files ("nise", "ogata",
        "kharitonov") identify a source; words common across the corpus do not.
        """
        if not self.chunks:
            return
        files_per_token: dict[str, set[str]] = {}
        all_files: set[str] = set()
        for chunk in self.chunks:
            fname = str(chunk.get("metadata", {}).get("filename", ""))
            if not fname:
                continue
            all_files.add(fname)
            for token in set(tokenize_corpus(fname)):
                files_per_token.setdefault(token, set()).add(fname)

        if not all_files:
            return
        cutoff = max(2, int(len(all_files) * SOURCE_TOKEN_MAX_FILE_FRACTION))
        self._distinctive_source_tokens = {
            token
            for token, files in files_per_token.items()
            if len(files) <= cutoff
            and len(token) > 2
            and not token.isdigit()
            and token not in GENERIC_SOURCE_TOKENS
        }

    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        if BM25Okapi is None:
            return
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
        # A newly uploaded document may introduce a new author/title, so the
        # distinctive-source vocabulary has to be recomputed alongside BM25.
        self._build_source_token_map()
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
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Re-rank a widened candidate pool so that a chunk from an explicitly
        # named source can be promoted above a body-text-only BM25 match.
        query_source_tokens = set(tokens) & self._distinctive_source_tokens
        pool = ranked[: max(top_k * 10, 60)]
        rescored: list[tuple[float, int]] = []
        for idx in pool:
            base = float(scores[idx])
            if base <= 0.0:
                continue
            score = base
            if query_source_tokens:
                fname_tokens = set(tokenize_corpus(str(self.chunks[idx].get("metadata", {}).get("filename", ""))))
                if query_source_tokens & fname_tokens:
                    score *= SOURCE_MATCH_BOOST
            rescored.append((score, idx))
        rescored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for score, idx in rescored:
            chunk = self.chunks[idx]
            if source_filter and source_filter.lower() not in chunk["source_path"].lower():
                continue
            fname = chunk["metadata"].get("filename", "unknown")
            source_name, is_published = display_source_name(fname)
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 3),
                "source": chunk["source_path"],
                "filename": fname,
                # User-facing label -- never show the raw filename in an answer.
                "source_name": source_name,
                "is_published_work": is_published,
                "page": chunk["metadata"].get("page"),
                "text": chunk["text"],
            })
            if len(results) >= top_k:
                break

        return results


# A single process-wide index instance. The agent, the retrieval tool, and the
# document-upload endpoint must all read and mutate the SAME in-memory object:
# with separate instances, a document uploaded through the web UI is written to
# disk but stays invisible to the running agent until the server restarts.
_shared_index: ControlRAGIndex | None = None


def get_shared_index(index_dir: Path = INDEX_DIR) -> ControlRAGIndex:
    """Return the process-wide shared RAG index, loading it on first use."""
    global _shared_index
    if _shared_index is None:
        _shared_index = ControlRAGIndex(index_dir)
    return _shared_index


# --- Human-readable source names -------------------------------------------
# Indexed filenames carry private organisational cruft -- owner initials
# ("JD_", "B&B_"), course codes ("AMC", "PLMMR"), and scan artefacts
# ("txtbk", "DEFINITIVO", a doubled ".pdf.pdf"). Those must never reach a user
# as a citation, so every hit also carries a cleaned display name plus whether
# it is a published work (citable by author/title) or personal course notes
# (referred to generically).

_PUBLISHED_SOURCES: dict[str, str] = {
    "norman s. nise - control systems engineering": "Nise, *Control Systems Engineering*",
    "ogata modern control engineering 5th txtbk": "Ogata, *Modern Control Engineering* (5th ed.)",
}

# Course code -> the subject it stands for, so a generic remainder such as
# "CAM Course Notes Part 2" still resolves to something meaningful.
_COURSE_CODES: dict[str, str] = {
    "SAS": "Safety in Automation Systems",
    "AMC": "Advanced and Multivariable Control",
    "CIR": "Control of Industrial Robots",
    "PLMMR": "Perception, Localization and Mapping for Mobile Robots",
    "NC": "Networked Control",
    "CAM": "Computer-Aided Manufacturing",
    "PSC": "Production Systems Control",
    "MIDA": "Model Identification and Data Analysis",
    "MIDA1": "Model Identification and Data Analysis",
    "ACEHV": "Autonomous and Connected Electric/Hybrid Vehicles",
    "ACHEV": "Autonomous and Connected Electric/Hybrid Vehicles",
    "ACAV": "Autonomous and Connected Vehicles",
    "DDCSD": "Data-Driven Control System Design",
    "SACI": "Industrial Automation and Communication Systems",
    "ICT": "Information and Communication Technology",
}

_OWNER_PREFIX_RE = re.compile(r"^(?:B&B|BBB|JD|LP|EC|AC|FG|RB|XX)[_\-\s]+", re.IGNORECASE)
_NOISE_RE = re.compile(
    r"\b(?:txtbk|definitivo|margini\s+larghi|theory\s+notes|practice\s+notes|final)\b|\(.*?\)",
    re.IGNORECASE,
)


def display_source_name(filename: str) -> tuple[str, bool]:
    """Map an indexed filename to (display name, is_published_work)."""
    if not filename:
        return ("local reference", False)

    stem = str(filename)
    while True:
        lowered = stem.lower()
        for ext in (".pdf", ".md", ".txt", ".json", ".jsonl"):
            if lowered.endswith(ext):
                stem = stem[: -len(ext)]
                break
        else:
            break

    key = " ".join(stem.split()).lower()
    if key in _PUBLISHED_SOURCES:
        return (_PUBLISHED_SOURCES[key], True)

    name = _OWNER_PREFIX_RE.sub("", stem)

    # A leading all-caps token is a course code; swap it for its subject.
    subject = ""
    parts = name.replace("_", " ").split()
    if parts:
        head = parts[0].strip(":-").upper()
        if head in _COURSE_CODES:
            subject = _COURSE_CODES[head]
            parts = parts[1:]
        elif len(parts) > 1 and 2 <= len(head) <= 6 and head.isalpha() and parts[0].isupper():
            parts = parts[1:]

    remainder = _NOISE_RE.sub("", " ".join(parts))
    remainder = " ".join(remainder.replace("_", " ").split()).strip(" -–—")

    generic = remainder.lower() in {
        "", "course notes", "lecture notes", "notes", "exercise sessions",
        "summary", "course notes part 1", "course notes part 2", "lectures",
    }
    if subject and generic:
        label = subject
    elif subject and remainder:
        label = subject if remainder.lower() in subject.lower() else f"{subject} - {remainder}"
    else:
        label = remainder or subject or "local reference"

    return (f"{label} (course notes)", False)
