#!/usr/bin/env python3
"""Build and inspect a small, deterministic MLX semantic-retrieval pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index" / "pilot"
MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
MODEL_REVISION = "6c3ae70858513f1a78e9cdca3cae330d9075cd2a"
TASK_INSTRUCTION = (
    "Given a control-systems engineering query, retrieve technically relevant "
    "passages, equations, and executable MATLAB or Python examples."
)

CHUNK_FILES = [
    PROJECT_ROOT / "data" / "processed" / "chunks" / "knowledge_chunks.jsonl",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "web_collections_chunks"
    / "knowledge_chunks.jsonl",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "core_books_chunks"
    / "knowledge_chunks.jsonl",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arxiv_chunks"
    / "knowledge_chunks.jsonl",
]

TOPICS = {
    "controllability": [
        "controllability",
        "controllability matrix",
        "kalman rank",
        "reachable subspace",
        "ctrb",
    ],
    "h_infinity": [
        "h infinity",
        "hinfinity",
        "mixed sensitivity",
        "weighting function",
        "small gain",
        "hinfsyn",
    ],
    "mpc": [
        "model predictive control",
        "receding horizon",
        "finite horizon",
        "input constraints",
        "state constraints",
        "terminal cost",
    ],
}

DEMO_QUERIES = [
    "How do I test controllability of a continuous-time LTI system and compute the controllability matrix?",
    "Explain mixed-sensitivity H-infinity synthesis using weighting functions on S, KS, and T.",
    "How does model predictive control enforce input and state constraints over a finite horizon?",
]


def load_chunks() -> list[dict]:
    chunks = []
    for path in CHUNK_FILES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("text"):
                    row["chunk_file"] = str(path.relative_to(PROJECT_ROOT))
                    chunks.append(row)
    return chunks


def keyword_score(text: str, terms: list[str]) -> int:
    lowered = text.lower().replace("-", " ")
    return sum(lowered.count(term.replace("-", " ")) for term in terms)


def select_pilot_chunks(chunks: list[dict], per_topic: int, distractors: int) -> list[dict]:
    selected: dict[str, dict] = {}
    for topic, terms in TOPICS.items():
        ranked = sorted(
            chunks,
            key=lambda row: (
                keyword_score(row["text"], terms),
                row["token_count"],
                row["chunk_id"],
            ),
            reverse=True,
        )
        for row in (candidate for candidate in ranked if keyword_score(candidate["text"], terms) > 0):
            copy = dict(row)
            copy["pilot_topic"] = topic
            selected.setdefault(copy["chunk_id"], copy)
            if sum(item.get("pilot_topic") == topic for item in selected.values()) >= per_topic:
                break

    for row in sorted(chunks, key=lambda item: item["chunk_id"]):
        if row["chunk_id"] in selected:
            continue
        copy = dict(row)
        copy["pilot_topic"] = "distractor"
        selected[copy["chunk_id"]] = copy
        distractors -= 1
        if distractors == 0:
            break
    return list(selected.values())


def query_text(query: str) -> str:
    return f"Instruct: {TASK_INSTRUCTION}\nQuery:{query}"


def embed_text(model, tokenizer, text: str, max_tokens: int) -> np.ndarray:
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    token_ids = token_ids[-max_tokens:]
    hidden = model.model(mx.array([token_ids]))
    vector = hidden[0, -1].astype(mx.float32)
    vector = vector / mx.sqrt(mx.sum(vector * vector))
    mx.eval(vector)
    return np.asarray(vector, dtype=np.float32)


def build_index(index_dir: Path, per_topic: int, distractors: int, max_tokens: int) -> None:
    chunks = select_pilot_chunks(load_chunks(), per_topic, distractors)
    if not chunks:
        raise RuntimeError("No chunks were found. Build the processed corpus first.")

    print(f"Loading {MODEL_ID} at {MODEL_REVISION[:8]}...")
    model, tokenizer = load(MODEL_ID, revision=MODEL_REVISION)
    mx.reset_peak_memory()
    started = time.perf_counter()
    vectors = []
    for index, chunk in enumerate(chunks, start=1):
        vectors.append(embed_text(model, tokenizer, chunk["text"], max_tokens))
        if index % 10 == 0 or index == len(chunks):
            print(f"Embedded {index}/{len(chunks)} chunks")
    elapsed = time.perf_counter() - started

    matrix = np.stack(vectors).astype(np.float16)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", matrix)
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task_instruction": TASK_INSTRUCTION,
        "chunks": len(chunks),
        "dimensions": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "chunks_per_second": len(chunks) / elapsed,
        "peak_mlx_memory_bytes": mx.get_peak_memory(),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def search(index_dir: Path, queries: list[str], top_k: int, max_tokens: int) -> None:
    vectors = np.load(index_dir / "embeddings.npy").astype(np.float32)
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream]

    print(f"Loading {MODEL_ID} for query embedding...")
    model, tokenizer = load(MODEL_ID, revision=MODEL_REVISION)
    for query in queries:
        query_vector = embed_text(model, tokenizer, query_text(query), max_tokens)
        scores = vectors @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        print(f"\nQUERY: {query}")
        for rank, index in enumerate(indices, start=1):
            chunk = chunks[int(index)]
            excerpt = " ".join(chunk["text"].split())[:260]
            print(
                f"{rank}. score={scores[index]:.4f} topic={chunk['pilot_topic']} "
                f"source={chunk['source_id']}\n   {excerpt}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-topic", type=int, default=15)
    parser.add_argument("--distractors", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.build:
        build_index(args.index_dir, args.per_topic, args.distractors, args.max_tokens)
    queries = args.query or DEMO_QUERIES
    search(args.index_dir, queries, args.top_k, args.max_tokens)


if __name__ == "__main__":
    main()
