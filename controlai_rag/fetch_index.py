"""Download the prebuilt retrieval index from the Hugging Face Hub.

The three index artefacts total ~455 MB -- far past GitHub's 100 MB per-file
limit -- so they live in a Hub dataset repo instead of in git. The repo is
private: the index carries the full extracted text of commercial textbooks,
which is fine to hold locally and wrong to redistribute. Set HF_TOKEN (or run
`huggingface-cli login`) with an account that can read it.

Without the index the retriever has nothing to search. Building one from your
own documents instead is `./run.sh --build-index`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from controlai_rag.index import INDEX_DIR

REPO_ID = os.environ.get("CONTROLAI_INDEX_REPO", "atakankahya/controlai-rag-index")

# chunks.json and bm25.pkl are the lexical side; embeddings.npz is the dense
# side. Missing the last one degrades retrieval to BM25 silently, so it counts
# as required here rather than optional.
FILES = ("chunks.json", "bm25.pkl", "embeddings.npz")


def fetch(index_dir: Path = INDEX_DIR, force: bool = False) -> Path:
    """Place the index files in `index_dir`, downloading what is missing."""
    from huggingface_hub import hf_hub_download

    index_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = index_dir / name
        if target.exists() and not force:
            print(f"  {name}: already present, skipping")
            continue
        print(f"  {name}: downloading...", flush=True)
        cached = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        # Copy rather than symlink into the cache: the index is mutated in place
        # by uploads (HybridRetriever.add_chunks) and by scripts/repair_*.py.
        target.write_bytes(Path(cached).read_bytes())
        print(f"  {name}: {target.stat().st_size / 1e6:.0f} MB")
    return index_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = parser.parse_args()

    print(f"Fetching retrieval index from {REPO_ID}")
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        print(f"\nCould not fetch the index: {exc}", file=sys.stderr)
        print(
            "\nThe dataset repo is private. Authenticate with an account that can\n"
            "read it (`huggingface-cli login`, or set HF_TOKEN), or build your own\n"
            "index from data/user_docs/ with `./run.sh --build-index`.",
            file=sys.stderr,
        )
        return 1
    print("Index ready in", INDEX_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
