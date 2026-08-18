"""Validate ControlAI SFT drafts and detect benchmark-family leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROLES = ["system", "user", "assistant"]


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
                errors.append(f"{location}: record must be a JSON object")
                continue
            record["_location"] = location
            records.append(record)

    return records, errors


def benchmark_families(path: Path) -> tuple[set[str], list[str]]:
    records, errors = load_jsonl(path)
    families = {
        record.get("family")
        for record in records
        if isinstance(record.get("family"), str)
    }
    return families, errors


def validate_messages(record: dict[str, Any]) -> list[str]:
    location = record["_location"]
    messages = record.get("messages")
    if not isinstance(messages, list):
        return [f"{location}: messages must be a list"]

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    errors: list[str] = []
    if roles != EXPECTED_ROLES or len(messages) != len(EXPECTED_ROLES):
        errors.append(
            f"{location}: expected roles {EXPECTED_ROLES}, received {roles}"
        )

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{location}: messages[{index}] must be an object")
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{location}: messages[{index}].content must be non-empty")
    return errors


def validate_metadata(
    record: dict[str, Any], held_out_families: set[str]
) -> list[str]:
    location = record["_location"]
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return [f"{location}: metadata must be an object"]

    errors: list[str] = []
    for field in ("id", "domain", "family", "source_type", "status"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            errors.append(f"{location}: metadata.{field} must be non-empty text")

    family = metadata.get("family")
    if family in held_out_families:
        errors.append(
            f"{location}: benchmark-family leakage detected for {family!r}"
        )
    if metadata.get("status") not in {"draft", "approved", "rejected"}:
        errors.append(f"{location}: unsupported metadata.status")
    return errors


def validate_stability_record(record: dict[str, Any]) -> list[str]:
    """Independently verify a real 2x2 Hurwitz decision using trace and determinant."""
    metadata = record["metadata"]
    if metadata.get("family") != "continuous_lti_eigenvalue_stability":
        return []

    location = record["_location"]
    ground_truth = metadata.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return [f"{location}: stability record is missing ground_truth"]

    try:
        matrix = np.asarray(ground_truth["A"], dtype=float)
        claimed = ground_truth["asymptotically_stable"]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{location}: invalid stability ground_truth ({exc})"]

    if matrix.shape != (2, 2):
        return [f"{location}: expected a 2x2 A matrix, received {matrix.shape}"]
    if not isinstance(claimed, bool):
        return [f"{location}: asymptotically_stable must be boolean"]

    trace = float(np.trace(matrix))
    determinant = float(np.linalg.det(matrix))
    independently_stable = trace < 0.0 and determinant > 0.0
    if claimed != independently_stable:
        return [
            f"{location}: claimed stability={claimed}, but the independent 2x2 "
            f"Hurwitz test gives {independently_stable} "
            f"(trace={trace:.6g}, determinant={determinant:.6g})"
        ]
    return []


def complex_values(pairs: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(real, imag) for real, imag in pairs])


def same_roots(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare small root sets without depending on eigensolver ordering."""
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.shape != right.shape:
        return False
    left = left[np.lexsort((np.imag(left), np.real(left)))]
    right = right[np.lexsort((np.imag(right), np.real(right)))]
    return bool(np.allclose(left, right, rtol=1e-7, atol=1e-8))


def validate_numeric_ground_truth(record: dict[str, Any]) -> list[str]:
    """Recompute every numeric v0 ground truth independently of answer text."""
    location = record["_location"]
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return []
    gt = metadata.get("ground_truth")
    if not isinstance(gt, dict):
        return []
    kind = gt.get("kind")
    try:
        if kind == "second_order":
            a1, a0 = float(gt["a1"]), float(gt["a0"])
            wn = np.sqrt(a0)
            zeta = a1 / (2 * wn)
            poles = np.roots([1.0, a1, a0])
            valid = (
                np.isclose(wn, gt["omega_n"])
                and np.isclose(zeta, gt["zeta"])
                and same_roots(poles, complex_values(gt["poles"]))
            )
        elif kind == "observability":
            A = np.asarray(gt["A"], dtype=float)
            C = np.asarray(gt["C"], dtype=float)
            O = np.vstack([C, C @ A])
            rank = int(np.linalg.matrix_rank(O))
            valid = (
                np.allclose(O, gt["O"])
                and rank == gt["rank"]
                and (rank == A.shape[0]) == gt["observable"]
            )
        elif kind == "state_feedback":
            A = np.asarray(gt["A"], dtype=float)
            B = np.asarray(gt["B"], dtype=float)
            K = np.asarray(gt["K"], dtype=float)
            Acl = A - B @ K
            poles = np.linalg.eigvals(Acl)
            valid = (
                np.allclose(Acl, gt["Acl"])
                and same_roots(poles, complex_values(gt["eigenvalues"]))
                and bool(np.all(np.real(poles) < 0)) == gt["stable"]
            )
        elif kind == "discrete_poles":
            radius = max(abs(float(pole)) for pole in gt["poles"])
            valid = np.isclose(radius, gt["spectral_radius"]) and (
                radius < 1
            ) == gt["stable"]
        elif kind == "first_order_frequency":
            gain = float(gt["gain"])
            x = float(gt["tau"]) * float(gt["omega"])
            magnitude = gain / np.sqrt(1 + x * x)
            phase = -np.degrees(np.arctan(x))
            valid = np.isclose(magnitude, gt["magnitude"]) and np.isclose(
                phase, gt["phase_deg"]
            )
        elif kind == "transfer_properties":
            num = np.asarray(gt["numerator"], dtype=float)
            den = np.asarray(gt["denominator"], dtype=float)
            poles = np.roots(den)
            zeros = np.roots(num)
            valid = (
                same_roots(poles, np.asarray(gt["poles"], dtype=complex))
                and same_roots(zeros, np.asarray([gt["zero"]], dtype=complex))
                and np.isclose(num[-1] / den[-1], gt["dc_gain"])
            )
        else:
            return []
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return [f"{location}: invalid {kind!r} ground truth ({exc})"]
    if not valid:
        return [f"{location}: independent verification failed for {kind!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/v0.jsonl"),
    )
    args = parser.parse_args()

    held_out_families, errors = benchmark_families(args.benchmark)
    records: list[dict[str, Any]] = []
    for path in args.paths:
        loaded, load_errors = load_jsonl(path)
        records.extend(loaded)
        errors.extend(load_errors)

    seen_ids: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}
    family_counts: Counter[str] = Counter()
    stability_checks = 0
    numeric_checks = 0

    for record in records:
        errors.extend(validate_messages(record))
        errors.extend(validate_metadata(record, held_out_families))

        metadata = record.get("metadata")
        messages = record.get("messages")
        if not isinstance(metadata, dict) or not isinstance(messages, list):
            continue

        record_id = metadata.get("id")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(
                    f"{record['_location']}: duplicate id also found at "
                    f"{seen_ids[record_id]}"
                )
            else:
                seen_ids[record_id] = record["_location"]

        family = metadata.get("family")
        if isinstance(family, str):
            family_counts[family] += 1
            if family == "continuous_lti_eigenvalue_stability":
                stability_checks += 1
                errors.extend(validate_stability_record(record))

        ground_truth = metadata.get("ground_truth")
        if isinstance(ground_truth, dict) and ground_truth.get("kind") in {
            "second_order",
            "observability",
            "state_feedback",
            "discrete_poles",
            "first_order_frequency",
            "transfer_properties",
        }:
            numeric_checks += 1
            errors.extend(validate_numeric_ground_truth(record))

        user_messages = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if user_messages and isinstance(user_messages[0], str):
            prompt_hash = normalized_hash(user_messages[0])
            if prompt_hash in seen_prompts:
                errors.append(
                    f"{record['_location']}: duplicate prompt also found at "
                    f"{seen_prompts[prompt_hash]}"
                )
            else:
                seen_prompts[prompt_hash] = record["_location"]

    print(f"records checked: {len(records)}")
    print(f"unique ids: {len(seen_ids)}")
    print(f"benchmark families held out: {len(held_out_families)}")
    print(f"independent stability checks: {stability_checks}")
    print(f"other independent numeric checks: {numeric_checks}")
    print("family counts:")
    for family, count in sorted(family_counts.items()):
        print(f"- {family}: {count}")

    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
