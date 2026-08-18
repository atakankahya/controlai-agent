"""Validate the held-out ControlAI benchmark and selected numeric references."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DOMAINS = {
    "classical_control",
    "state_space",
    "estimation_filtering",
    "optimal_control",
    "robust_control",
    "mpc",
    "nonlinear_control",
    "adaptive_control",
    "system_identification",
    "sampled_data",
}
TASK_TYPES = {"concept", "derivation", "numerical", "code", "critique", "design", "underspecified"}
DIFFICULTIES = {"foundation", "intermediate", "advanced"}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"missing file: {path}"]

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{path}:{line_number}"
            if not line.strip():
                errors.append(f"{location}: blank lines are not allowed")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{location}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict):
                errors.append(f"{location}: record must be an object")
                continue
            record["_location"] = location
            records.append(record)
    return records, errors


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def validate_record(record: dict[str, Any]) -> list[str]:
    location = record["_location"]
    errors: list[str] = []

    for field in ("id", "domain", "family", "task_type", "difficulty", "prompt"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{location}: {field} must be non-empty text")

    if record.get("domain") not in DOMAINS:
        errors.append(f"{location}: unsupported domain {record.get('domain')!r}")
    if record.get("task_type") not in TASK_TYPES:
        errors.append(f"{location}: unsupported task_type {record.get('task_type')!r}")
    if record.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{location}: unsupported difficulty {record.get('difficulty')!r}")

    constraints = record.get("constraints")
    if not isinstance(constraints, dict):
        errors.append(f"{location}: constraints must be an object")
    else:
        if constraints.get("language") != "English":
            errors.append(f"{location}: benchmark language must be English")
        max_words = constraints.get("max_words")
        if not isinstance(max_words, int) or max_words < 50:
            errors.append(f"{location}: max_words must be an integer of at least 50")

    rubric = record.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        errors.append(f"{location}: rubric must be a non-empty list")
    else:
        total = 0
        for index, item in enumerate(rubric):
            if not isinstance(item, dict):
                errors.append(f"{location}: rubric[{index}] must be an object")
                continue
            if not isinstance(item.get("criterion"), str) or not item["criterion"].strip():
                errors.append(f"{location}: rubric[{index}].criterion must be text")
            points = item.get("points")
            if not isinstance(points, int) or points <= 0:
                errors.append(f"{location}: rubric[{index}].points must be positive")
            else:
                total += points
        if total != 10:
            errors.append(f"{location}: rubric must total 10 points, received {total}")

    if not isinstance(record.get("reference"), dict) or not record["reference"]:
        errors.append(f"{location}: reference must be a non-empty object")
    return errors


def validate_numeric_references(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_id = {record.get("id"): record for record in records}

    controllability = by_id.get("state_space_controllability_001")
    if controllability:
        ref = controllability["reference"]
        a = np.asarray(ref["example_A"], dtype=float)
        b = np.asarray(ref["example_B"], dtype=float)
        matrix = np.hstack([b, a @ b])
        rank = int(np.linalg.matrix_rank(matrix))
        if rank != ref["expected_rank"] or not np.allclose(matrix, ref["expected_matrix"]):
            errors.append(f"{controllability['_location']}: incorrect controllability reference")

    routh = by_id.get("classical_routh_001")
    if routh:
        ref = routh["reference"]
        threshold = 6.0 * 8.0
        if not close(ref["upper_bound"], threshold) or ref["lower_bound"] != 0.0:
            errors.append(f"{routh['_location']}: incorrect Routh interval reference")

    kalman = by_id.get("estimation_scalar_kf_001")
    if kalman:
        ref = kalman["reference"]
        innovation = 5.0 - 2.0
        innovation_covariance = 4.0 + 1.0
        gain = 4.0 / innovation_covariance
        posterior_state = 2.0 + gain * innovation
        posterior_covariance = (1.0 - gain) * 4.0
        expected = [innovation, innovation_covariance, gain, posterior_state, posterior_covariance]
        stored = [ref["innovation"], ref["innovation_covariance"], ref["kalman_gain"], ref["posterior_state"], ref["posterior_covariance"]]
        if not np.allclose(expected, stored):
            errors.append(f"{kalman['_location']}: incorrect Kalman reference")

    lqr = by_id.get("optimal_scalar_lqr_001")
    if lqr:
        ref = lqr["reference"]
        stabilizing_p = 1.0 + math.sqrt(2.0)
        closed_loop_pole = 1.0 - stabilizing_p
        if not close(ref["stabilizing_P"], stabilizing_p) or not close(ref["closed_loop_pole"], closed_loop_pole):
            errors.append(f"{lqr['_location']}: incorrect scalar LQR reference")

    mpc = by_id.get("mpc_scalar_constrained_001")
    if mpc:
        ref = mpc["reference"]
        unconstrained = -2.0 / 1.1
        constrained = float(np.clip(unconstrained, -1.0, 1.0))
        next_state = 2.0 + constrained
        cost = next_state**2 + 0.1 * constrained**2
        stored = [ref["unconstrained_u"], ref["optimal_u"], ref["next_state"], ref["optimal_cost"]]
        if not np.allclose([unconstrained, constrained, next_state, cost], stored):
            errors.append(f"{mpc['_location']}: incorrect MPC reference")

    sampled = by_id.get("sampled_zoh_integrator_001")
    if sampled:
        ref = sampled["reference"]
        pole = ref["A_d"] - ref["B_d"] * ref["feedback_gain"]
        if not close(pole, ref["closed_loop_pole"]) or (abs(pole) < 1.0) != ref["stable"]:
            errors.append(f"{sampled['_location']}: incorrect sampled-data reference")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("benchmarks/v0.jsonl"))
    args = parser.parse_args()

    records, errors = load_jsonl(args.path)
    seen_ids: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    domain_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for record in records:
        errors.extend(validate_record(record))
        record_id = record.get("id")
        family = record.get("family")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(f"{record['_location']}: duplicate id also at {seen_ids[record_id]}")
            seen_ids[record_id] = record["_location"]
        if isinstance(family, str):
            if family in seen_families:
                errors.append(f"{record['_location']}: duplicate family also at {seen_families[family]}")
            seen_families[family] = record["_location"]
        if record.get("domain") in DOMAINS:
            domain_counts[record["domain"]] += 1
        if record.get("task_type") in TASK_TYPES:
            type_counts[record["task_type"]] += 1

    missing_domains = DOMAINS - set(domain_counts)
    if missing_domains:
        errors.append(f"benchmark is missing domains: {', '.join(sorted(missing_domains))}")
    errors.extend(validate_numeric_references(records))

    print(f"records checked: {len(records)}")
    print(f"unique families held out: {len(seen_families)}")
    print("domain counts:")
    for domain, count in sorted(domain_counts.items()):
        print(f"- {domain}: {count}")
    print("task-type counts:")
    for task_type, count in sorted(type_counts.items()):
        print(f"- {task_type}: {count}")

    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
