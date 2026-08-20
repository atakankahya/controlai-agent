#!/usr/bin/env python3
"""Broad answer-quality smoke test across every control-engineering domain.

Retrieval tests (scripts/test_rag_knowledge.py) only prove the right passage was
found. This runs the FULL agent and grades the answer that actually reaches the
user, checking the failure modes seen in practice:

  * rendering    -- doubled LaTeX backslashes, orphaned <tool_call> tags,
                    raw JSON leaking into prose
  * citations    -- raw indexed filenames / extensions / course codes shown to
                    the user instead of clean source names
  * substance    -- the canned "analysis complete" placeholder, empty answers,
                    answers too short to be useful
  * tool hygiene -- the same tool re-run over and over, failed tool calls

Usage:
    python3 scripts/eval_answer_quality.py                 # all cases
    python3 scripts/eval_answer_quality.py --filter robot  # subset
    python3 scripts/eval_answer_quality.py --limit 5
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.orchestrator import ControlAIAgent

PLACEHOLDER = "computational analysis has been completed"

# (domain, question). Deliberately spans applied domains and answer *shapes*:
# numeric-with-tool, conceptual-from-RAG, design/applied prose, and definitional.
CASES: list[tuple[str, str]] = [
    ("classical", "Compute the gain margin, phase margin and crossover frequencies for G(s) = 10/(s*(s+1)*(s+5))."),
    ("classical", "Explain what the phase margin tells you about a closed-loop system's damping."),
    ("modern", "Design an LQR controller for A=[[0,1],[-2,-3]], B=[[0],[1]], Q=diag([10,1]), R=1 and simulate the step response."),
    ("modern", "What is the difference between controllability and stabilizability?"),
    ("mpc", "What is quasi-infinite horizon MPC and why is the terminal region needed?"),
    ("mpc", "How does MPC handle actuator saturation compared to anti-windup PID?"),
    ("estimation", "Explain the difference between the Kalman filter time update and measurement update."),
    ("nonlinear", "What is a control barrier function and how does it enforce safety?"),
    ("robust", "State the small gain theorem and when it is conservative."),
    ("aerospace", "How is gain scheduling used in an aircraft flight control system across the flight envelope?"),
    ("aerospace", "Explain the short-period and phugoid modes of aircraft longitudinal dynamics."),
    ("automotive", "How does an electronic stability program use yaw rate feedback to correct oversteer?"),
    ("automotive", "Design considerations for an adaptive cruise control spacing policy."),
    ("robotics", "Explain impedance control for a robot manipulator in contact with the environment."),
    ("robotics", "How does a mobile robot fuse odometry and lidar for localization?"),
    ("automation", "What is cascade control in process automation and when does it help?"),
    ("automation", "Explain integral windup in an industrial PID loop and how to prevent it."),
    ("power", "How is field-oriented control used for induction motor drives?"),
]

RAW_FILENAME_PAT = re.compile(
    r"\.pdf\b|\.md\b|\.jsonl?\b|\bJD_|\bB&B_|\bBBB[_\s]|\bLP_|\bEC_|\bAC_|\bFG_|\bRB_|\bXX_|txtbk|DEFINITIVO",
    re.IGNORECASE,
)
DOUBLE_BS_PAT = re.compile(r"\\\\(?=[a-zA-Z|{}()])")


def grade(question: str, result) -> tuple[list[str], dict]:
    """Return (list of problems, metrics) for one answer."""
    text = result.final_response or ""
    problems: list[str] = []

    if not text.strip():
        problems.append("EMPTY answer")
    elif PLACEHOLDER in text.lower():
        problems.append("PLACEHOLDER non-answer")
    elif len(text.split()) < 40:
        problems.append(f"THIN answer ({len(text.split())} words)")

    if DOUBLE_BS_PAT.search(text):
        problems.append("DOUBLED LaTeX backslash")
    if "<tool_call>" in text or "</tool_call>" in text:
        problems.append("LEAKED tool_call tag")
    if re.search(r'\{\s*"name"\s*:', text):
        problems.append("LEAKED raw JSON")
    if RAW_FILENAME_PAT.search(text):
        problems.append("RAW filename in citation")

    names = [t.tool_name for t in result.tool_traces]
    failed = [t.tool_name for t in result.tool_traces if t.result.get("status") == "error"]
    if failed:
        problems.append(f"TOOL ERROR: {', '.join(sorted(set(failed)))}")
    for n in set(names):
        if names.count(n) > 2:
            problems.append(f"REPEATED tool x{names.count(n)}: {n}")

    return problems, {"tools": names, "words": len(text.split()), "plots": len(result.plots)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filter", default="", help="only run cases whose domain or text matches")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = [c for c in CASES if args.filter.lower() in (c[0] + " " + c[1]).lower()]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Loading agent...\n")
    agent = ControlAIAgent()

    failures: list[tuple[str, str, list[str]]] = []
    for i, (domain, q) in enumerate(cases, 1):
        t0 = time.time()
        try:
            res = agent.run(q)
        except Exception as exc:  # a crash is itself a finding
            failures.append((domain, q, [f"EXCEPTION: {exc}"]))
            print(f"[{i}/{len(cases)}] {domain:<11} EXCEPTION  {exc}")
            continue
        problems, meta = grade(q, res)
        status = "ok  " if not problems else "FAIL"
        print(
            f"[{i}/{len(cases)}] {domain:<11} {status} "
            f"{time.time()-t0:5.1f}s  words={meta['words']:<4} "
            f"plots={meta['plots']} tools={','.join(meta['tools']) or '-'}"
        )
        for p in problems:
            print(f"              - {p}")
        if problems:
            failures.append((domain, q, problems))

    print("\n" + "=" * 74)
    print(f"{len(cases) - len(failures)}/{len(cases)} answers clean")
    if failures:
        print("\nIssues by type:")
        counts: dict[str, int] = {}
        for _, _, ps in failures:
            for p in ps:
                key = p.split(":")[0].split("(")[0].strip()
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>3}  {k}")
    print("=" * 74)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
