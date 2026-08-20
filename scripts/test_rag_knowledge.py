#!/usr/bin/env python3
"""Knowledge-retrieval smoke test for the local RAG index.

Runs a fixed set of canonical control-engineering queries against the live
BM25 index used by the agent (controlai_rag.index.ControlRAGIndex) and checks
that each query's top hits actually contain at least one expected keyword.
Also reports whether each query's best score clears the 2.5 relevance
threshold that ControlAIAgent._get_grounded_instruction uses to decide
whether to inject retrieved text into the system prompt -- a query can
retrieve "correct" chunks yet still never get grounded into an answer if its
score sits below that bar.

Usage:
    python3 scripts/test_rag_knowledge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_rag.index import ControlRAGIndex

GROUNDING_SCORE_THRESHOLD = 2.5

# (query, keywords where at least one must appear in a top-k hit's text)
# ControlAI is a general control-engineering agent, so this suite deliberately
# spans every application domain -- aerospace, automotive, robotics, industrial
# automation, power -- not just classical/modern theory.
TEST_CASES: list[tuple[str, list[str]]] = [
    # --- Core theory ---
    ("controllability matrix rank test", ["controllab", "rank"]),
    ("observability of linear time invariant systems", ["observ"]),
    ("continuous algebraic Riccati equation LQR", ["riccati", "lqr", "quadratic"]),
    ("discrete algebraic Riccati equation DARE", ["riccati", "discrete"]),
    ("zero order hold ZOH discretization", ["zero-order", "zero order", "hold", "discret"]),
    ("Lyapunov stability of nonlinear systems", ["lyapunov", "stab"]),
    ("gain margin phase margin frequency response", ["gain margin", "phase margin"]),
    ("Kalman filter state estimation", ["kalman", "estimat"]),
    ("PID controller tuning", ["pid", "proportional"]),
    ("root locus method", ["root locus"]),
    ("PBH test for controllability", ["pbh", "popov"]),
    ("model predictive control constrained optimization", ["model predictive", "mpc", "horizon"]),
    ("H-infinity robust control small gain theorem", ["h-infinity", "h infinity", "small gain", "hinf"]),
    ("control barrier function safety filter", ["barrier", "safety"]),
    ("state feedback pole placement", ["pole placement", "state feedback"]),
    # --- Application domains ---
    ("aircraft flight control longitudinal dynamics", ["aircraft", "flight", "longitudinal", "pitch"]),
    ("quadrotor UAV attitude control", ["quadrotor", "uav", "attitude", "drone"]),
    ("vehicle dynamics yaw rate stability control", ["vehicle", "yaw", "tire", "steering"]),
    ("automotive cruise control design", ["cruise", "vehicle", "throttle", "speed"]),
    ("robot manipulator kinematics and Jacobian", ["manipulator", "jacobian", "kinematic", "robot"]),
    ("mobile robot localization and odometry", ["odometry", "localiz", "mobile robot", "slam"]),
    ("industrial process control valve saturation", ["valve", "process", "saturat", "actuator"]),
    ("cascade control loop in process automation", ["cascade", "process", "inner loop", "secondary"]),
    ("electric motor drive speed control", ["motor", "drive", "torque", "induction"]),
    ("system identification from input output data", ["identification", "arx", "least squares", "estimat"]),
]


def run() -> int:
    index = ControlRAGIndex()
    if not index.chunks or not index.bm25:
        print("FAIL: RAG index did not load (no chunks / no BM25 model). Is data/rag_index/ populated?")
        return 1

    print(f"RAG index loaded: {len(index.chunks)} chunks\n")

    passed = 0
    grounded = 0
    for query, keywords in TEST_CASES:
        hits = index.search(query, top_k=5)
        best_score = hits[0]["score"] if hits else 0.0
        matched = any(
            kw.lower() in hit["text"].lower()
            for hit in hits
            for kw in keywords
        )
        would_ground = best_score > GROUNDING_SCORE_THRESHOLD
        grounded += int(would_ground)
        passed += int(matched)

        status = "PASS" if matched else "FAIL"
        ground_tag = "grounds" if would_ground else "below threshold"
        print(f"[{status}] '{query}'  best_score={best_score:.2f} ({ground_tag})")
        if hits:
            top = hits[0]
            excerpt = " ".join(top["text"].split())[:160]
            print(f"        top hit: [{top['filename']} p.{top['page']}] {excerpt}...")
        else:
            print("        no hits returned")
        print()

    total = len(TEST_CASES)
    print("=" * 70)
    print(f"Keyword relevance: {passed}/{total} queries retrieved an on-topic chunk")
    print(f"Grounding trigger:  {grounded}/{total} queries would clear the score>{GROUNDING_SCORE_THRESHOLD} auto-grounding bar")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run())
