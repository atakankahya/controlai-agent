#!/usr/bin/env python3
"""Measure the relevance gap and suggest a MIN_COSINE for the current corpus.

`MIN_COSINE` in `controlai_rag/retriever.py` is the gate that decides whether a
retrieved passage is worth putting in front of the model. It is not a universal
constant: it depends on the embedding model *and* on the corpus. A larger corpus
raises every query's best match, off-domain ones included, so the threshold has
to be re-measured whenever the index changes size materially.

The method is to score two sets of probes -- questions the corpus should be able
to answer, and questions it definitely cannot -- and put the threshold in the gap
between them. If there is no gap, the report says so rather than inventing one.

    python scripts/calibrate_retrieval.py
    python scripts/calibrate_retrieval.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Topics a control-engineering library is expected to cover. Deliberately spans
# classical, modern, optimal, robust, nonlinear, estimation and MPC so the
# threshold is not tuned to one corner of the field.
IN_DOMAIN = [
    "Routh-Hurwitz stability criterion table construction",
    "root locus asymptotes and breakaway points",
    "Nyquist stability criterion encirclements and the Z = N + P rule",
    "Kalman filter measurement update equations",
    "why does LQG have no guaranteed stability margins",
    "Bode sensitivity integral waterbed effect right half plane zero",
    "terminal cost and terminal region for model predictive control stability",
    "small gain theorem and when it is conservative",
    "sliding mode control chattering and the boundary layer",
    "PBH rank test for controllability and stabilizability",
    "describing function analysis of a limit cycle",
    "persistent excitation in system identification",
]

# Questions with no plausible answer in a control library. If any of these
# clears the threshold, the gate is too loose and the model will be handed a
# confident irrelevance.
OUT_OF_DOMAIN = [
    "how do I bake sourdough bread with a levain starter",
    "best hiking trails in Patagonia in November",
    "React useState hook rerender behaviour in strict mode",
    "who won the 1998 FIFA World Cup final",
    "symptoms and treatment of seasonal pollen allergy",
]


def best_similarities(retriever, embedder, query: str, depth: int = 60, top: int = 5) -> list[float]:
    """Top similarities for `query`, after the low-value filter is applied.

    Filtering first matters: index pages match nearly any control query and
    would otherwise set the threshold from chunks that can never be returned.
    """
    from controlai_rag.retriever import _is_low_value

    by_id = {c["chunk_id"]: c for c in retriever.index.chunks}
    sims = retriever.vectors @ embedder.encode_query(query)
    order = np.argsort(-sims)[:depth]
    kept = [
        float(sims[i])
        for i in order
        if not _is_low_value(str(by_id.get(retriever.vector_ids[i], {}).get("text", "")))
    ]
    return kept[:top]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="also write the raw measurements here")
    args = parser.parse_args()

    from controlai_rag.embeddings import get_embedder
    from controlai_rag.retriever import MIN_COSINE, get_retriever

    retriever = get_retriever()
    if not retriever.has_dense:
        print("No dense index. Run: python -m controlai_rag.retriever --build")
        return 1
    embedder = get_embedder()
    print(f"corpus: {len(retriever.index.chunks)} chunks, {len(retriever.vector_ids)} vectors\n")

    report: dict[str, dict[str, list[float]]] = {"in_domain": {}, "out_of_domain": {}}
    for label, queries, key in (
        ("IN ", IN_DOMAIN, "in_domain"),
        ("OUT", OUT_OF_DOMAIN, "out_of_domain"),
    ):
        for query in queries:
            tops = best_similarities(retriever, embedder, query)
            report[key][query] = tops
            shown = np.round(tops, 3) if tops else "none"
            print(f"{label} {str(shown):32} {query[:58]}")
        print()

    # A passage is only useful if it clears the gate, so what matters for the
    # in-domain set is its *weakest* best-match, and for the out-of-domain set
    # its strongest.
    in_best = [max(v) for v in report["in_domain"].values() if v]
    out_best = [max(v) for v in report["out_of_domain"].values() if v]
    floor, ceiling = min(in_best), max(out_best)

    print(f"in-domain  worst best-match : {floor:.3f}  (mean {np.mean(in_best):.3f})")
    print(f"off-domain best  best-match : {ceiling:.3f}  (mean {np.mean(out_best):.3f})")
    print(f"current MIN_COSINE          : {MIN_COSINE}")

    if floor > ceiling:
        suggestion = round((floor + ceiling) / 2, 2)
        print(f"\ngap of {floor - ceiling:.3f} -> suggested MIN_COSINE = {suggestion}")
        if not (ceiling < MIN_COSINE < floor):
            print(f"the current value sits outside that gap; update it in controlai_rag/retriever.py")
    else:
        print(
            f"\nNO GAP: an off-domain query scores {ceiling:.3f} while an in-domain one "
            f"scores only {floor:.3f}. No single threshold separates them -- tighten "
            f"_is_low_value, or accept losing the weakest in-domain topics by setting the "
            f"threshold above {ceiling:.2f}."
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
