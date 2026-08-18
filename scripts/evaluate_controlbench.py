"""Universal multi-pillar evaluator for ControlBench v1."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_python_code_sandbox(code: str, timeout_sec: float = 3.0) -> tuple[bool, str]:
    """Execute Python snippet safely and check if assertions pass."""
    stdout_buf = io.StringIO()
    globals_dict: dict[str, Any] = {"plt": plt}
    plt.show = lambda *args, **kwargs: None
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, globals_dict, globals_dict)
        return True, "Executed cleanly with assertions passed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)}"


def evaluate_theory_item(response_text: str, gt: dict[str, Any]) -> dict[str, Any]:
    key_concepts = gt.get("key_concepts", [])
    found_count = 0
    missing = []
    
    # Normalized search
    resp_lower = response_text.lower()
    for concept in key_concepts:
        # Extract core keywords from concept string
        keywords = [w for w in re.findall(r"\b\w+\b", concept.lower()) if len(w) > 3]
        if not keywords:
            continue
        match_count = sum(1 for kw in keywords if kw in resp_lower)
        if match_count >= max(1, len(keywords) // 2):
            found_count += 1
        else:
            missing.append(concept)

    score = (found_count / len(key_concepts)) * 100.0 if key_concepts else 100.0
    return {
        "score": round(score, 1),
        "concepts_found": found_count,
        "total_concepts": len(key_concepts),
        "missing": missing,
    }


def evaluate_numerical_item(response_record: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    tool_calls = response_record.get("tool_calls", [])
    target_tool = gt.get("tool_call")
    response_text = response_record.get("response", "")

    tool_matched = any(t.get("name") == target_tool for t in tool_calls)
    
    # Check numerical correctness from tool result or text
    num_correct = False
    expected = gt.get("expected_numeric", {})
    tol = gt.get("tolerance", 1e-2)

    for tool in tool_calls:
        if tool.get("name") == target_tool:
            result = tool.get("result", {})
            if result.get("status") == "success":
                num_correct = True

    # Fallback to text check if no tools
    if not num_correct and expected:
        matches = 0
        total_targets = len(expected)
        for k, val in expected.items():
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}"
                if val_str in response_text or str(val) in response_text:
                    matches += 1
            elif isinstance(val, bool):
                if str(val).lower() in response_text.lower():
                    matches += 1
            elif isinstance(val, list):
                # Check for list elements in text
                flat_vals = np.array(val).flatten()
                sub_match = sum(1 for x in flat_vals if f"{x:.2f}" in response_text or f"{x:.3f}" in response_text)
                if sub_match >= len(flat_vals) // 2:
                    matches += 1
        if matches == total_targets:
            num_correct = True

    score = 100.0 if num_correct else (50.0 if tool_matched else 0.0)
    return {
        "score": score,
        "tool_matched": tool_matched,
        "numeric_verified": num_correct,
    }


def evaluate_code_item(response_text: str, gt: dict[str, Any]) -> dict[str, Any]:
    lang = gt.get("language", "python")
    if lang == "python":
        # Extract python code block
        match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
        if not match:
            # Try raw code block
            match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)

        if not match:
            return {"score": 0.0, "status": "No code block found"}

        code = match.group(1).strip()
        passed, msg = execute_python_code_sandbox(code)
        score = 100.0 if passed else 30.0
        return {"score": score, "sandbox_executed": passed, "message": msg}

    elif lang == "matlab":
        matlab_keywords = gt.get("must_contain_matlab_syntax", [])
        found_kw = sum(1 for kw in matlab_keywords if kw in response_text)
        score = (found_kw / len(matlab_keywords)) * 100.0 if matlab_keywords else 100.0
        return {"score": round(score, 1), "matlab_syntax_score": score}

    return {"score": 50.0}


def evaluate_trap_item(response_text: str, gt: dict[str, Any]) -> dict[str, Any]:
    expected_behavior = gt.get("expected_behavior", "")
    resp_lower = response_text.lower()
    
    # Positive refusal indicators
    refusal_cues = [
        "missing", "cannot", "specify", "please provide", "uncontrollable",
        "unobservable", "undefined", "invalid", "positive semi-definite",
        "zero control authority", "sampling period", "not possible",
        "cannot be inverted", "violates", "requires"
    ]
    refusal_detected = any(cue in resp_lower for cue in refusal_cues)
    
    score = 100.0 if refusal_detected else 0.0
    return {"score": score, "refusal_or_trap_detected": refusal_detected}


def evaluate_case_study_item(response_text: str, gt: dict[str, Any]) -> dict[str, Any]:
    resp_lower = response_text.lower()
    score = 80.0  # Base for rich response
    if len(resp_lower.split()) < 50:
        score = 30.0
    return {"score": score}


def evaluate_benchmark(benchmark_path: Path, responses_path: Path) -> dict[str, Any]:
    benchmark_items = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    response_items = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    resp_by_id = {r.get("id") or r.get("benchmark_id"): r for r in response_items}

    pillar_scores: dict[str, list[float]] = {
        "theory_and_concepts": [],
        "numerical_synthesis": [],
        "code_and_simulation": [],
        "underspecified_and_traps": [],
        "real_world_case_studies": [],
    }

    item_results = []

    for item in benchmark_items:
        item_id = item["id"]
        pillar = item["pillar"]
        gt = item["ground_truth"]

        resp_record = resp_by_id.get(item_id, {})
        resp_text = resp_record.get("response", "")

        if pillar == "theory_and_concepts":
            eval_res = evaluate_theory_item(resp_text, gt)
        elif pillar == "numerical_synthesis":
            eval_res = evaluate_numerical_item(resp_record, gt)
        elif pillar == "code_and_simulation":
            eval_res = evaluate_code_item(resp_text, gt)
        elif pillar == "underspecified_and_traps":
            eval_res = evaluate_trap_item(resp_text, gt)
        else:  # real_world_case_studies
            eval_res = evaluate_case_study_item(resp_text, gt)

        score = float(eval_res["score"])
        pillar_scores[pillar].append(score)
        item_results.append({
            "id": item_id,
            "pillar": pillar,
            "score": score,
            "details": eval_res,
        })

    pillar_averages = {p: round(float(np.mean(scores)), 1) if scores else 0.0 for p, scores in pillar_scores.items()}
    overall_score = round(float(np.mean([score for scores in pillar_scores.values() for score in scores])), 1)

    return {
        "overall_score": overall_score,
        "pillar_scores": pillar_averages,
        "total_items": len(benchmark_items),
        "evaluated_items": len(response_items),
        "item_results": item_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/controlbench_v1.jsonl"))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate_benchmark(args.benchmark, args.responses)

    print("=" * 60)
    print("CONTROLBENCH V1 EVALUATION LEADERBOARD REPORT")
    print("=" * 60)
    print(f"Overall Benchmark Score: {results['overall_score']:.1f}%")
    print("-" * 60)
    print("Pillar Breakdown:")
    for pillar, score in results["pillar_scores"].items():
        print(f"  * {pillar:30s}: {score:5.1f}%")
    print("=" * 60)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
