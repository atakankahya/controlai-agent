"""Shared schema helpers for original, solver-verified ControlAI SFT data."""

from __future__ import annotations

import hashlib
import json
from typing import Any


GENERATOR_VERSION = "1.0.0"
SYSTEM_PROMPT = (
    "You are an offline control-systems engineering assistant. Lead with the "
    "result, state assumptions and conventions, show the decisive calculation, "
    "and never invent a plant, parameter, software output, or verification result. "
    "Use executable Python or MATLAB only when requested."
)


def solution_digest(ground_truth: dict[str, Any]) -> str:
    payload = json.dumps(ground_truth, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_record(
    *,
    record_id: str,
    domain: str,
    family: str,
    task_type: str,
    difficulty: str,
    template_id: str,
    prompt: str,
    answer: str,
    ground_truth: dict[str, Any],
    source_refs: list[str],
    verifier: str,
    tool: str = "analytic",
    code_language: str | None = None,
    code_execution: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "id": record_id,
            "domain": domain,
            "family": family,
            "split_group": family,
            "task_type": task_type,
            "difficulty": difficulty,
            "template_id": template_id,
            "language": "English",
            "tool": tool,
            "code_language": code_language,
            "code_execution": code_execution,
            "source_type": "original_solver_generated",
            "source_refs": source_refs,
            "generator": "controlai_data.generators.v1",
            "generator_version": GENERATOR_VERSION,
            "verification": {
                "verifier": verifier,
                "status": "passed_at_generation",
            },
            "ground_truth": ground_truth,
            "solution_sha256": solution_digest(ground_truth),
            "status": "approved",
        },
    }
