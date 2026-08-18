#!/usr/bin/env python3
"""Generate the upgraded ControlAI SFT v2 dataset with Chain of Thought (CoT), textbook cases, and STEM replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_data.generators import (
    generate_advanced_v1,
    generate_classical_optimal_v1,
    generate_code_v1,
    generate_extended_v1,
    generate_linear_v1,
    generate_safety_behavior_v1,
)
from scripts.build_stem_replay_dataset import build_all_stem_records
from scripts.build_textbook_sft_dataset import build_all_textbook_records

BENCHMARK_FAMILIES = {
    "iid_scalar_jump_mean_square_test",
    "kharitonov_cubic_interval_stability",
    "matlab_exact_zoh_state_space",
    "minimum_norm_redundant_control_allocation",
    "scalar_affine_nonlinear_dynamic_inversion",
    "scalar_control_barrier_safety_filter",
}

VALID_FAMILIES = {
    "box_constrained_scalar_mpc_horizon2",
    "discrete_cycle_consensus_step_size",
    "finite_horizon_scalar_lqr_recursion",
    "noise_free_arx_least_squares",
    "relative_degree_and_zero_dynamics",
    "scalar_kalman_time_update",
    "textbook_inverted_pendulum_design",
    "stem_rlc_circuit_transient",
}


def stable_fraction(value: str) -> float:
    raw = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/training/sft_v2")
    )
    args = parser.parse_args()

    records: list[dict] = []

    # 1. Synthetic control generators with rich CoT
    records.extend(generate_linear_v1(args.count_per_family, args.seed))
    records.extend(generate_classical_optimal_v1(args.count_per_family, args.seed + 10_000))
    records.extend(generate_advanced_v1(args.count_per_family, args.seed + 20_000))
    records.extend(generate_code_v1(args.count_per_family, args.seed + 30_000))
    records.extend(generate_extended_v1(args.count_per_family, args.seed + 40_000))
    records.extend(generate_safety_behavior_v1(args.count_per_family, args.seed + 50_000))

    # 2. Textbook case studies with CoT and verified code
    records.extend(build_all_textbook_records(args.count_per_family, args.seed + 60_000))

    # 3. Base STEM replay derivations
    records.extend(build_all_stem_records(args.count_per_family, args.seed + 70_000))

    families = sorted({row["metadata"]["family"] for row in records})
    missing_split_families = (BENCHMARK_FAMILIES | VALID_FAMILIES) - set(families)
    if missing_split_families:
        raise ValueError(f"Configured split families are missing: {sorted(missing_split_families)}")

    train = [
        row for row in records
        if row["metadata"]["family"] not in (BENCHMARK_FAMILIES | VALID_FAMILIES)
    ]
    valid = [
        row for row in records if row["metadata"]["family"] in VALID_FAMILIES
    ]
    benchmark_candidates = [
        row for row in records if row["metadata"]["family"] in BENCHMARK_FAMILIES
    ]

    train.sort(key=lambda row: stable_fraction(f"train:{row['metadata']['id']}"))
    valid.sort(key=lambda row: stable_fraction(f"valid:{row['metadata']['id']}"))
    benchmark_candidates.sort(key=lambda row: stable_fraction(f"bench:{row['metadata']['id']}"))

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "benchmark_candidates.jsonl", benchmark_candidates)

    domains = Counter(row["metadata"]["domain"] for row in records)
    summary = {
        "schema_version": 2,
        "status": "ready_for_training",
        "total_records": len(records),
        "total_families": len(families),
        "train_records": len(train),
        "valid_records": len(valid),
        "train_families": sorted(set(families) - BENCHMARK_FAMILIES - VALID_FAMILIES),
        "valid_families": sorted(VALID_FAMILIES),
        "benchmark_candidate_records": len(benchmark_candidates),
        "benchmark_families": sorted(BENCHMARK_FAMILIES),
        "domains": dict(sorted(domains.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
