#!/usr/bin/env python3
"""Generate the behavioral SFT dataset the deployed adapter is missing.

The current adapter (sft_v2 + agent traces) was trained exclusively on
fully-specified control problems mapped to the matching design tool. As a
result the model:
  * routes ANY matrix-shaped question to a control-design tool (asked to
    multiply two matrices, it ran an LQR synthesis),
  * fabricates missing parameters (invented B/Q/R three separate times)
    because it never saw a single example of asking for a missing value,
  * carries numbers over from a previous, different problem in multi-turn
    conversations.

This script generates verified trajectories for exactly those behaviors:
  1. plain linear-algebra routing   -> matrix_arithmetic tool traces
  2. missing-parameter refusal      -> no tool call; state what's missing, ask
  3. poisoned-history faithfulness  -> multi-turn: full problem in turn 1,
                                       different request in turn 2 answered
                                       WITHOUT reusing turn-1 numbers
  4. conceptual questions           -> prose answer, no unnecessary code tool

Every numeric answer is produced by executing the real registered tool, so
the dataset is verified by construction.

Usage:
    python3 scripts/generate_behavior_sft_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from controlai_agent.registry import registry
import controlai_agent.tools  # noqa: F401  (register all tools)

OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "behavior_sft_v1"
RNG_SEED = 20260819

SYSTEM_PROMPT = (
    "You are an offline control-systems engineering assistant. Lead with the result, "
    "state assumptions and conventions, show the decisive calculation, and never invent "
    "a plant, parameter, software output, or verification result. Use executable Python "
    "or MATLAB only when requested."
)


def fmt_num(x: float) -> str:
    return f"{x:.6g}"


def fmt_matrix(mat: list[list[float]]) -> str:
    return "[" + ", ".join("[" + ", ".join(fmt_num(v) for v in row) + "]" for row in mat) + "]"


def fmt_matrix_latex(mat) -> str:
    arr = np.atleast_2d(np.array(mat, dtype=float))
    rows = [" & ".join(fmt_num(v) for v in row) for row in arr]
    return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"


def rand_int_matrix(rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    return [[float(rng.randint(-9, 9)) for _ in range(cols)] for _ in range(rows)]


def rand_nonsingular(rng: random.Random, n: int) -> list[list[float]]:
    while True:
        m = rand_int_matrix(rng, n, n)
        if abs(np.linalg.det(np.array(m))) > 0.5:
            return m


def matrix_to_text(mat: list[list[float]], rng: random.Random) -> str:
    """Render a matrix as the user might actually type it, garbling included."""
    style = rng.random()
    body = "[" + ", ".join("[" + ", ".join(fmt_num(v) for v in row) + "]" for row in mat) + "]"
    if style < 0.7:
        return body
    # space-separated rows, occasionally with a dropped comma (real user input)
    rows = []
    for row in mat:
        if rng.random() < 0.5:
            rows.append("[" + " ".join(fmt_num(v) for v in row) + "]")
        else:
            rows.append("[" + ", ".join(fmt_num(v) for v in row) + "]")
    return "[" + ", ".join(rows) + "]"


def tool_call_msg(name: str, args: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>",
    }


def tool_result_msg(name: str, result: dict[str, Any]) -> dict[str, str]:
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


def make_example(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}


# ---------------------------------------------------------------------------
# 1. Plain linear-algebra routing
# ---------------------------------------------------------------------------

MULT_PHRASES = [
    "{a} times {b}",
    "Multiply {a} by {b}",
    "Compute the matrix product of A = {a} and B = {b}",
    "A = {a} times A = {b}",
    "What is {a} * {b}?",
    "{a} multiplied by {b}",
]

UNARY_TEMPLATES = {
    "inverse": ["Invert the matrix {a}", "What is the inverse of {a}?", "Compute {a}^-1"],
    "determinant": ["Determinant of {a}", "Compute det({a})", "What is the determinant of the matrix {a}?"],
    "rank": ["What is the rank of {a}?", "Compute the rank of the matrix {a}"],
    "transpose": ["Transpose {a}", "What is {a} transposed?"],
    "eigenvalues": ["Eigenvalues of {a}", "Find the eigenvalues of the matrix {a}", "Compute the spectrum of {a}"],
    "trace": ["What is the trace of {a}?", "Compute the trace of {a}"],
}


def gen_multiply(rng: random.Random) -> dict[str, Any] | None:
    n = rng.choice([2, 2, 3, 3, 4])
    k = rng.choice([n, n, rng.choice([1, 2, 3])])
    A = rand_int_matrix(rng, n, k)
    B = rand_int_matrix(rng, k, rng.choice([1, 2, 3, n]))
    args = {"operation": "multiply", "matrix_a": A, "matrix_b": B}
    result = registry.execute("matrix_arithmetic", args)
    if result.get("status") != "success":
        return None
    prompt = rng.choice(MULT_PHRASES).format(a=matrix_to_text(A, rng), b=matrix_to_text(B, rng))
    product = result["result"]
    final = (
        f"The matrix product is:\n$$AB = {fmt_matrix_latex(product)}$$\n"
        f"({len(A)}x{len(A[0])} times {len(B)}x{len(B[0])} gives a {len(product)}x{len(product[0])} result; "
        f"entry $(i,j)$ is the dot product of row $i$ of the first matrix with column $j$ of the second.)"
    )
    return make_example([
        {"role": "user", "content": prompt},
        tool_call_msg("matrix_arithmetic", args),
        tool_result_msg("matrix_arithmetic", result),
        {"role": "assistant", "content": final},
    ])


def gen_unary(rng: random.Random) -> dict[str, Any] | None:
    op = rng.choice(list(UNARY_TEMPLATES.keys()))
    n = rng.choice([2, 2, 3, 3])
    A = rand_nonsingular(rng, n) if op == "inverse" else (
        rand_int_matrix(rng, n, n) if op in ("determinant", "eigenvalues", "trace") else rand_int_matrix(rng, n, rng.choice([n, n + 1]))
    )
    args = {"operation": op, "matrix_a": A}
    result = registry.execute("matrix_arithmetic", args)
    if result.get("status") != "success":
        return None
    prompt = rng.choice(UNARY_TEMPLATES[op]).format(a=matrix_to_text(A, rng))
    res = result["result"]
    if op == "inverse":
        final = f"The inverse is:\n$$A^{{-1}} = {fmt_matrix_latex(res)}$$\n(det $= {fmt_num(result['determinant'])} \\neq 0$, so the matrix is invertible.)"
    elif op == "transpose":
        final = f"The transpose is:\n$$A^T = {fmt_matrix_latex(res)}$$"
    elif op == "eigenvalues":
        eig_strs = [f"{fmt_num(re_)}" + (f" {'+' if im >= 0 else '-'} {fmt_num(abs(im))}j" if abs(im) > 1e-12 else "") for re_, im in res]
        final = f"The eigenvalues are:\n$$\\lambda = \\{{{', '.join(eig_strs)}\\}}$$\nSpectral radius: $\\rho(A) = {fmt_num(result['spectral_radius'])}$."
    elif op == "rank":
        full = "full rank" if res == min(len(A), len(A[0])) else f"rank-deficient (max possible {min(len(A), len(A[0]))})"
        final = f"$\\text{{rank}}(A) = {res}$ -- the matrix is {full}."
    else:
        final = f"$\\text{{{op}}}(A) = {fmt_num(res)}$."
    return make_example([
        {"role": "user", "content": prompt},
        tool_call_msg("matrix_arithmetic", args),
        tool_result_msg("matrix_arithmetic", result),
        {"role": "assistant", "content": final},
    ])


# ---------------------------------------------------------------------------
# 2. Missing-parameter refusal (single-turn)
# ---------------------------------------------------------------------------

def gen_refusal_single(rng: random.Random) -> dict[str, Any]:
    n = rng.choice([2, 3])
    A = rand_int_matrix(rng, n, n)
    kind = rng.choice(["lqr_no_b", "lqr_no_qr", "place_no_poles", "zoh_no_ts", "margins_no_tf", "kalman_no_cov"])

    if kind == "lqr_no_b":
        prompt = rng.choice([
            f"Design an LQR controller for A={fmt_matrix(A)}.",
            f"Solve the LQR problem for the system with state matrix A={fmt_matrix(A)} and simulate the closed loop.",
            f"Find the optimal LQR gain for A={fmt_matrix(A)}, Q=diag({', '.join(str(rng.randint(1, 20)) for _ in range(n))}), R=1.",
        ])
        answer = (
            f"I can't run the LQR synthesis yet: the input matrix $B$ ({n}x m) has not been provided, and I "
            f"won't assume one -- the gain $K$ depends entirely on how the input enters the dynamics.\n\n"
            f"What I can tell you from $A$ alone: the open-loop eigenvalues determine whether the plant is "
            f"stable before feedback, and controllability of $(A, B)$ must be checked once $B$ is known.\n\n"
            f"Please provide $B$ (and $Q$, $R$ if you want other than default weights), and I will compute "
            f"$P$, $K$, and the closed-loop poles."
        )
    elif kind == "lqr_no_qr":
        B = [[0.0] for _ in range(n - 1)] + [[1.0]]
        prompt = f"Compute the LQR gain for A={fmt_matrix(A)}, B={fmt_matrix(B)}."
        answer = (
            "The LQR gain is not defined until the cost is: I need the state penalty $Q$ (positive "
            "semidefinite, {n}x{n}) and the control penalty $R$ (positive definite). Different $Q, R$ give "
            "completely different gains for the same plant.\n\nTell me $Q$ and $R$ -- or tell me what "
            "behavior you want (e.g. fast state 1, cheap control) and I will propose weights and say so "
            "explicitly as a design choice."
        ).replace("{n}", str(n))
    elif kind == "place_no_poles":
        B = [[0.0] for _ in range(n - 1)] + [[1.0]]
        prompt = f"Place the closed-loop poles for A={fmt_matrix(A)}, B={fmt_matrix(B)} using state feedback."
        answer = (
            "Pole placement needs target pole locations, and none were given. Where do you want the "
            f"closed-loop poles? Provide {n} locations (complex ones in conjugate pairs, e.g. $-2 \\pm 3j$), "
            "or give a specification (settling time, damping ratio) and I will derive pole locations from it "
            "and state that derivation explicitly."
        )
    elif kind == "zoh_no_ts":
        B = [[0.0] for _ in range(n - 1)] + [[1.0]]
        prompt = f"Discretize A={fmt_matrix(A)}, B={fmt_matrix(B)} using zero-order hold."
        answer = (
            "Exact ZOH discretization requires the sampling period $T_s$, which was not given -- $A_d = "
            "e^{A T_s}$ changes completely with it. What is $T_s$ (in seconds)? As a rule of thumb it should "
            "give 10-30 samples per dominant closed-loop time constant."
        )
    elif kind == "margins_no_tf":
        prompt = rng.choice([
            "Compute the gain and phase margins of my loop.",
            "What are the stability margins of the system?",
        ])
        answer = (
            "I need the open-loop transfer function to compute margins -- numerator and denominator "
            "coefficients of $L(s)$, or its poles/zeros/gain, or the state-space matrices. Please provide "
            "one of those and I will compute GM, PM, and both crossover frequencies exactly."
        )
    else:  # kalman_no_cov
        prompt = f"Run a Kalman filter time update for A={fmt_matrix(A)}."
        answer = (
            "A Kalman time update needs more than $A$: the current state estimate $\\hat{x}$, its covariance "
            "$P$, and the process noise covariance $Q_w$ (plus $B$ and $u$ if there is a known input). "
            "Which values should I use? I will not assume noise statistics -- they determine the filter "
            "entirely."
        )
    return make_example([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])


# ---------------------------------------------------------------------------
# 3. Poisoned-history faithfulness (multi-turn)
# ---------------------------------------------------------------------------

def _full_lqr_turn(rng: random.Random) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """A complete verified LQR exchange to serve as turn 1 (the 'poison')."""
    A = [[0.0, 1.0], [float(-rng.randint(1, 5)), float(-rng.randint(1, 5))]]
    B = [[0.0], [1.0]]
    q1, q2 = rng.choice([(10, 1), (20, 1), (5, 2), (100, 1)])
    Q = [[float(q1), 0.0], [0.0, float(q2)]]
    R = [[1.0]]
    args = {"A": A, "B": B, "Q": Q, "R": R}
    result = registry.execute("continuous_lqr", args)
    prompt = f"Design an LQR controller for A={fmt_matrix(A)}, B={fmt_matrix(B)}, Q=diag([{q1}, {q2}]), R=1."
    K = result["K"]
    poles = result["closed_loop_poles"]
    pole_strs = [f"{fmt_num(p[0])}" + (f" {'+' if p[1] >= 0 else '-'} {fmt_num(abs(p[1]))}j" if abs(p[1]) > 1e-9 else "") for p in poles]
    final = (
        f"Solving the CARE gives the optimal gain $K = {fmt_matrix_latex(K)}$ with closed-loop poles "
        f"$\\{{{', '.join(pole_strs)}\\}}$ -- the closed loop is asymptotically stable."
    )
    turn = [
        {"role": "user", "content": prompt},
        tool_call_msg("continuous_lqr", args),
        tool_result_msg("continuous_lqr", result),
        {"role": "assistant", "content": final},
    ]
    return turn, args


def gen_poisoned_matrix_question(rng: random.Random) -> dict[str, Any] | None:
    """Turn 1: full LQR. Turn 2: plain matrix question -> matrix tool only."""
    turn1, _ = _full_lqr_turn(rng)
    n = rng.choice([2, 3])
    X = rand_int_matrix(rng, n, n)
    Y = rand_int_matrix(rng, n, n)
    args = {"operation": "multiply", "matrix_a": X, "matrix_b": Y}
    result = registry.execute("matrix_arithmetic", args)
    if result.get("status") != "success":
        return None
    prompt = rng.choice(MULT_PHRASES).format(a=matrix_to_text(X, rng), b=matrix_to_text(Y, rng))
    final = f"The matrix product is:\n$$XY = {fmt_matrix_latex(result['result'])}$$"
    return make_example(turn1 + [
        {"role": "user", "content": prompt},
        tool_call_msg("matrix_arithmetic", args),
        tool_result_msg("matrix_arithmetic", result),
        {"role": "assistant", "content": final},
    ])


def gen_poisoned_refusal(rng: random.Random) -> dict[str, Any]:
    """Turn 1: full 2-state LQR. Turn 2: new 3-state A with no B -> refuse,
    explicitly declining to reuse the old problem's matrices."""
    turn1, old_args = _full_lqr_turn(rng)
    A3 = rand_int_matrix(rng, 3, 3)
    prompt = rng.choice([
        f"Now design an LQR controller for A={fmt_matrix(A3)}.",
        f"Do the same for A={fmt_matrix(A3)}.",
        f"LQR for A={fmt_matrix(A3)} please.",
    ])
    answer = (
        "This is a new 3-state system, and I only have its $A$ matrix. The $B$, $Q$, and $R$ from the "
        "previous problem belong to a different 2-state plant -- they do not carry over, and resizing them "
        "to fit would be inventing a model.\n\nTo run this synthesis I need: $B$ (3 x m), and the weights "
        "$Q$ (3x3 PSD) and $R$ (m x m PD), or a description of the actuation and objectives so I can "
        "propose weights as an explicit design choice. What is $B$ for this system?"
    )
    return make_example(turn1 + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])


def gen_poisoned_concept_question(rng: random.Random) -> dict[str, Any]:
    """Turn 1: full LQR. Turn 2: conceptual question -> prose, no tools, no old numbers."""
    turn1, _ = _full_lqr_turn(rng)
    concept, answer = rng.choice(CONCEPT_QA)
    return make_example(turn1 + [
        {"role": "user", "content": concept},
        {"role": "assistant", "content": answer},
    ])


# ---------------------------------------------------------------------------
# 4. Conceptual questions -> prose, no unnecessary tools
# ---------------------------------------------------------------------------

CONCEPT_QA: list[tuple[str, str]] = [
    (
        "What is the difference between controllability and stabilizability?",
        "**Controllability** means every state can be driven anywhere in finite time: $\\text{rank}[B\\ AB\\ \\cdots\\ A^{n-1}B] = n$, or equivalently the PBH test $\\text{rank}[\\lambda I - A\\ \\ B] = n$ for every eigenvalue $\\lambda$.\n\n**Stabilizability** is weaker: only the *unstable* modes must be controllable -- PBH needs to hold only for $\\text{Re}(\\lambda) \\geq 0$. Stable uncontrollable modes are acceptable because they decay on their own.\n\nEvery controllable system is stabilizable; the converse fails. Stabilizability (with detectability) is exactly what LQR/LQG existence requires -- full controllability is sufficient but not necessary.",
    ),
    (
        "Why does a phase margin around 45-60 degrees correspond to good damping?",
        "For a second-order-dominant loop, phase margin and closed-loop damping are linked approximately by $\\zeta \\approx PM/100$ (PM in degrees), so $PM = 45^\\circ$--$60^\\circ$ maps to $\\zeta \\approx 0.45$--$0.6$: fast response with modest overshoot (roughly 10-25%).\n\nBelow ~30° the closed loop rings badly and is fragile to delay ($\\Delta\\phi = \\omega_{gc} T_d$ eats margin directly); much above 70° the response is sluggish. The linear rule degrades when higher-order dynamics or RHP zeros distort the phase near crossover -- then the exact relation, not the approximation, must be used.",
    ),
    (
        "What is integral windup in a PID loop and how is it prevented?",
        "When the actuator saturates, the plant stops responding to further increases in the control signal, but the integrator keeps accumulating error. The integral term 'winds up' far beyond what the actuator can deliver; after the error changes sign it takes a long time to unwind, causing large overshoot and slow recovery.\n\nStandard remedies: **conditional integration** (freeze the integrator while saturated), **back-calculation** (feed the difference between commanded and saturated actuator output back to discharge the integrator through gain $1/T_t$), or designing with actuator limits explicitly (MPC). Back-calculation is the common industrial choice because it degrades gracefully.",
    ),
    (
        "When would you choose MPC over a well-tuned PID controller?",
        "Choose **MPC** when the problem has structure PID cannot represent: hard constraints on inputs/states that the controller should respect *by design* rather than by saturation, strong multivariable interaction (coupled MIMO loops), a good process model with significant dead time, or an economic objective over a horizon.\n\nStay with **PID** when the loop is essentially SISO, fast, and model-poor: PID needs almost no model, runs at microsecond rates on a PLC, and its failure modes are well understood on the plant floor. A practical rule: constraints and coupling pay for MPC's modeling and maintenance cost; without them, they don't.",
    ),
    (
        "What does the separation principle say for observer-based control?",
        "For a linear system with state feedback $u = -K\\hat{x}$ driven by an observer with gain $L$, the closed-loop eigenvalues are exactly the union of the state-feedback poles $\\lambda(A - BK)$ and the observer poles $\\lambda(A - LC)$: the estimation error dynamics decouple from the regulation dynamics.\n\nSo $K$ and $L$ may be designed independently -- pole placement or LQR for $K$, pole placement or a Kalman filter for $L$ (the LQG combination). The caveat: separation guarantees *nominal* stability only. The combined loop can have poor robustness margins (Doyle's 1978 counterexample: LQG has no guaranteed margins), so robustness must be checked on the assembled loop.",
    ),
    (
        "Explain the waterbed effect in feedback design.",
        "Bode's sensitivity integral: for an open-loop stable, minimum-phase loop with relative degree ≥ 2, $\\int_0^\\infty \\ln|S(j\\omega)|\\, d\\omega = 0$ (and $= \\pi \\sum \\text{Re}(p_i)$ with unstable poles $p_i$). Pushing $|S|$ below 1 over one band necessarily pushes it above 1 somewhere else -- like pressing on a waterbed.\n\nConsequences: disturbance rejection improved in one band is paid for with amplification elsewhere; RHP zeros make it worse by confining where the 'bulge' can go. It is a conservation law, not a design flaw -- good loop shaping *places* the sensitivity bulge where disturbances and model error are least harmful.",
    ),
]


def gen_concept(rng: random.Random) -> dict[str, Any]:
    q, a = rng.choice(CONCEPT_QA)
    return make_example([
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ])


# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(RNG_SEED)
    examples: list[dict[str, Any]] = []

    generators = [
        (gen_multiply, 220),
        (gen_unary, 200),
        (gen_refusal_single, 260),
        (gen_poisoned_matrix_question, 130),
        (gen_poisoned_refusal, 130),
        (gen_poisoned_concept_question, 60),
        (gen_concept, 60),
    ]
    for fn, count in generators:
        made = 0
        attempts = 0
        while made < count and attempts < count * 4:
            attempts += 1
            ex = fn(rng)
            if ex is not None:
                examples.append(ex)
                made += 1
        print(f"{fn.__name__}: {made}")

    rng.shuffle(examples)
    n_valid = max(1, len(examples) // 10)
    valid, train = examples[:n_valid], examples[n_valid:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (OUTPUT_DIR / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(train)} train / {len(valid)} valid to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
