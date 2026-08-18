"""Generate verified SFT examples for continuous-time LTI stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SYSTEM_PROMPT = (
    "You are a concise control-systems engineering assistant. "
    "State assumptions, show the decisive calculation, and provide executable code "
    "when requested."
)


def format_number(value: complex) -> str:
    """Format a real or complex eigenvalue without numerical noise."""
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(imag) < 1e-10:
        return f"{real:.6g}"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.6g} {sign} {abs(imag):.6g}j"


def make_record(index: int, matrix: np.ndarray, seed: int) -> dict:
    eigenvalues = np.linalg.eigvals(matrix)
    stable = bool(np.all(np.real(eigenvalues) < 0.0))

    matrix_text = repr(matrix.tolist())
    eigenvalue_text = ", ".join(format_number(value) for value in eigenvalues)
    conclusion = (
        "All eigenvalues have strictly negative real parts; therefore, the origin "
        "is asymptotically stable"
        if stable
        else "At least one eigenvalue has a positive real part; therefore, the origin "
        "is unstable"
    )

    prompt = (
        "For the continuous-time autonomous LTI system x_dot = A x with "
        f"A = {matrix_text}, determine asymptotic stability from the eigenvalues. "
        "Provide minimal NumPy code that verifies the conclusion."
    )
    answer = (
        f"The eigenvalues are {eigenvalue_text}. {conclusion}.\n\n"
        "```python\n"
        "import numpy as np\n\n"
        f"A = np.array({matrix_text}, dtype=float)\n"
        "eigenvalues = np.linalg.eigvals(A)\n"
        "stable = np.all(np.real(eigenvalues) < 0.0)\n"
        "print(eigenvalues)\n"
        "print(\"asymptotically stable:\", stable)\n"
        "```"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "id": f"lti_eigenvalue_stability_{index:04d}",
            "domain": "state_space",
            "family": "continuous_lti_eigenvalue_stability",
            "tool": "python_numpy",
            "source_type": "programmatically_generated",
            "generator": "scripts/generate_stability_sft.py",
            "seed": seed,
            "verification": ["numpy.linalg.eigvals"],
            "ground_truth": {
                "A": matrix.tolist(),
                "eigenvalues": [
                    [float(np.real(value)), float(np.imag(value))]
                    for value in eigenvalues
                ],
                "asymptotically_stable": stable,
            },
            "status": "draft",
        },
    }


def generate(count: int, seed: int) -> list[dict]:
    """Generate a roughly balanced set of stable and unstable examples."""
    rng = np.random.default_rng(seed)
    stable_target = count // 2
    unstable_target = count - stable_target
    stable_count = 0
    unstable_count = 0
    seen: set[tuple[int, ...]] = set()
    records: list[dict] = []

    while len(records) < count:
        integer_matrix = rng.integers(-4, 5, size=(2, 2))
        key = tuple(int(value) for value in integer_matrix.flat)
        if key in seen:
            continue
        seen.add(key)

        matrix = integer_matrix.astype(float)
        eigenvalues = np.linalg.eigvals(matrix)
        real_parts = np.real(eigenvalues)

        # Avoid marginal or numerically ambiguous examples in this first dataset.
        if np.any(np.abs(real_parts) < 0.25):
            continue

        stable = bool(np.all(real_parts < 0.0))
        if stable and stable_count >= stable_target:
            continue
        if not stable and unstable_count >= unstable_target:
            continue

        records.append(make_record(len(records) + 1, matrix, seed))
        stable_count += int(stable)
        unstable_count += int(not stable)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft_draft/generated_stability.jsonl"),
    )
    args = parser.parse_args()

    if args.count < 2:
        parser.error("--count must be at least 2")

    records = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stable_count = sum(
        record["metadata"]["ground_truth"]["asymptotically_stable"]
        for record in records
    )
    print(f"wrote: {args.output}")
    print(f"records: {len(records)}")
    print(f"stable: {stable_count}")
    print(f"unstable: {len(records) - stable_count}")


if __name__ == "__main__":
    main()
