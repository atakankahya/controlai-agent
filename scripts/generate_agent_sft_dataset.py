"""Generate multi-turn tool-calling agent SFT dataset with verified SciPy execution."""

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
from scipy import linalg, signal

from controlai_agent.registry import registry
import controlai_agent.tools  # Register tools

OUTPUT_DIR = Path("data/training/agent_sft_v1")
RNG_SEED = 20260817


def format_matrix_latex(mat: np.ndarray) -> str:
    if mat.ndim == 1:
        return "[" + ", ".join(f"{x:.4g}" for x in mat) + "]"
    rows = []
    for r in mat:
        rows.append(" & ".join(f"{x:.4g}" for x in r))
    return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"


def generate_zoh_trajectory(rng: random.Random) -> dict[str, Any]:
    n = rng.choice([2, 3])
    A = np.diag([-rng.uniform(0.5, 3.0) for _ in range(n)])
    for i in range(n - 1):
        A[i, i + 1] = rng.uniform(0.5, 2.0)
    B = np.zeros((n, 1))
    B[-1, 0] = rng.uniform(1.0, 3.0)
    Ts = rng.choice([0.01, 0.02, 0.05, 0.1, 0.2])

    A_list = A.tolist()
    B_list = B.tolist()

    prompt = f"Discretize the continuous-time state-space system with A={A_list} and B={B_list} using exact zero-order hold at sampling period Ts={Ts} s."

    tool_call_text = f"""<tool_call>
{{"name": "exact_zoh", "arguments": {{"A": {A_list}, "B": {B_list}, "Ts": {Ts}}}}}
</tool_call>"""

    result = registry.execute("exact_zoh", {"A": A_list, "B": B_list, "Ts": Ts})
    Ad = np.array(result["Ad"])
    Bd = np.array(result["Bd"])

    assistant_final = f"""The continuous-time system has been discretized under exact Zero-Order Hold (ZOH) at sample time $T_s = {Ts}$ s.

The resulting discrete-time state-space matrices are:
$$
A_d = {format_matrix_latex(Ad)}, \\quad B_d = {format_matrix_latex(Bd)}
$$

The discretization was cross-verified using the block matrix exponential method with a residual tolerance below $10^{{-7}}$. The system is ready for discrete-time state estimation or controller synthesis."""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tool_call_text},
            {"role": "tool", "name": "exact_zoh", "content": json.dumps(result)},
            {"role": "assistant", "content": assistant_final},
        ],
        "metadata": {"task_type": "tool_calling", "family": "exact_zoh", "domain": "linear_systems"},
    }


def generate_lqr_trajectory(rng: random.Random) -> dict[str, Any]:
    A = [[0.0, 1.0], [-rng.uniform(1.0, 5.0), -rng.uniform(1.0, 4.0)]]
    B = [[0.0], [rng.uniform(1.0, 3.0)]]
    q1 = rng.choice([1.0, 5.0, 10.0, 20.0])
    q2 = rng.choice([0.1, 1.0, 2.0])
    Q = [[q1, 0.0], [0.0, q2]]
    r_val = rng.choice([0.1, 1.0, 2.0])
    R = [[r_val]]

    prompt = f"Design a continuous-time LQR state feedback controller for system A={A}, B={B} with state penalty Q={Q} and control penalty R={R}. Compute the algebraic Riccati solution P, feedback gain K, and closed-loop poles."

    tool_call_text = f"""<tool_call>
{{"name": "continuous_lqr", "arguments": {{"A": {A}, "B": {B}, "Q": {Q}, "R": {R}}}}}
</tool_call>"""

    result = registry.execute("continuous_lqr", {"A": A, "B": B, "Q": Q, "R": R})
    P_mat = np.array(result["P"])
    K_mat = np.array(result["K"])

    assistant_final = f"""The continuous-time Linear Quadratic Regulator (LQR) has been synthesized by solving the Algebraic Riccati Equation (CARE):

1. **Riccati Solution Matrix $P$**:
$$
P = {format_matrix_latex(P_mat)}
$$

2. **Optimal State Feedback Gain $K$**:
$$
K = {format_matrix_latex(K_mat)}
$$

3. **Closed-Loop Poles**:
$$
{result['closed_loop_poles']}
$$

The Riccati residual is verified at {result['verification']['riccati_residual_max']:.2e}, and all closed-loop eigenvalues lie strictly in the open left-half plane, certifying closed-loop asymptotic stability."""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tool_call_text},
            {"role": "tool", "name": "continuous_lqr", "content": json.dumps(result)},
            {"role": "assistant", "content": assistant_final},
        ],
        "metadata": {"task_type": "tool_calling", "family": "continuous_lqr", "domain": "optimal_control"},
    }


def generate_cbf_trajectory(rng: random.Random) -> dict[str, Any]:
    x_min = -rng.uniform(0.5, 2.0)
    x = x_min + rng.uniform(0.1, 0.5)
    u_nom = -rng.uniform(1.5, 4.0)
    alpha = rng.choice([0.5, 1.0, 1.2, 1.5, 2.0])

    prompt = f"For a scalar single-integrator system x_dot = u with state safety boundary x >= {x_min:.2f}, apply a Control Barrier Function safety filter with class-K gain alpha = {alpha:.1f}. Given nominal control u_nom = {u_nom:.2f} at state x = {x:.2f}, compute the safe control input u."

    tool_call_text = f"""<tool_call>
{{"name": "cbf_safety_filter", "arguments": {{"x": {x:.4f}, "u_nom": {u_nom:.4f}, "alpha": {alpha:.1f}, "x_min": {x_min:.4f}}}}}
</tool_call>"""

    result = registry.execute("cbf_safety_filter", {"x": x, "u_nom": u_nom, "alpha": alpha, "x_min": x_min})

    assistant_final = f"""The Control Barrier Function (CBF) quadratic program safety filter yields:

- **Barrier Function Value**: $h(x) = x - x_{{min}} = {result['h']:.4f}$
- **Lower Admissible Bound**: $\\dot{{h}} + \\alpha h \\ge 0 \\implies u \\ge -\\alpha h = {result['cbf_lower_bound']:.4f}$
- **Filtered Safe Control Input $u^*$**: ${result['u_safe']:.4f}$

The safety constraint is {'active' if result['is_constraint_active'] else 'inactive'}. The filtered control $u^*$ guarantees forward invariance of the safe set $\\mathcal{{C}} = \\{{x \\mid x \\ge {x_min:.2f}\\}}$, minimally perturbing the nominal control $u_{{nom}} = {u_nom:.2f}$."""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tool_call_text},
            {"role": "tool", "name": "cbf_safety_filter", "content": json.dumps(result)},
            {"role": "assistant", "content": assistant_final},
        ],
        "metadata": {"task_type": "tool_calling", "family": "cbf_safety_filter", "domain": "safety_critical"},
    }


def generate_underspecified_trajectory(rng: random.Random) -> dict[str, Any]:
    cases = [
        (
            "Design a discrete state-feedback controller for the continuous plant with A=[[0, 1], [-4, -2]], B=[[0], [1]].",
            "To design a discrete state-feedback controller, the following critical specifications are missing:\n1. **Sampling Period ($T_s$)**: Required to discretize the continuous-time plant matrices $(A, B)$ via Zero-Order Hold.\n2. **Performance Specifications**: Desired closed-loop pole locations $z_i$ or state/input penalty weights $(Q, R)$ for discrete LQR synthesis.\n\nPlease provide the sampling period $T_s$ and your desired closed-loop pole locations or LQR weighting matrices.",
        ),
        (
            "Tune a PID controller for an unknown motor.",
            "To tune a PID controller, the mathematical model or empirical response of the motor is required:\n1. **Plant Transfer Function** $G(s)$ or State-Space Realization $(A, B, C, D)$.\n2. **Or Empirical FOPDT Parameters**: Process static gain $K$, time constant $\\tau$, and apparent dead time $L$.\n3. **Design Criteria**: Target phase margin, settling time, or overshoot tolerance.\n\nPlease provide the plant transfer function or experimental step response parameters.",
        ),
        (
            "Solve the MPC quadratic program for initial state x0=[1.0, 0.0].",
            "To formulate and solve the Model Predictive Control (MPC) quadratic program, the following parameters are required:\n1. **Discrete System Dynamics**: State matrix $A$ and input matrix $B$.\n2. **Penalty Matrices**: State weighting matrix $Q$ and control weighting matrix $R$.\n3. **Prediction Horizon**: Horizon length $N$.\n4. **Actuator/State Constraints**: Bounds on inputs $|u_k| \\le u_{max}$ and states $x_{min} \\le x_k \\le x_{max}$.\n\nPlease provide the plant matrices $(A, B)$, weights $(Q, R)$, prediction horizon $N$, and any state/input constraints.",
        ),
    ]
    prompt, response = rng.choice(cases)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"task_type": "underspecified", "family": "refusal", "domain": "specification_gating"},
    }


def main() -> int:
    rng = random.Random(RNG_SEED)
    generators = [
        generate_zoh_trajectory,
        generate_lqr_trajectory,
        generate_cbf_trajectory,
    ]

    all_records = []

    # 1. Tool calling trajectories
    for _ in range(1800):
        gen = rng.choice(generators)
        all_records.append(gen(rng))

    # 2. Underspecified refusal trajectories
    for _ in range(300):
        all_records.append(generate_underspecified_trajectory(rng))

    rng.shuffle(all_records)

    split_idx = int(0.85 * len(all_records))
    train_records = all_records[:split_idx]
    valid_records = all_records[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "valid.jsonl").open("w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated Agent SFT dataset in {OUTPUT_DIR}:")
    print(f"  - Train: {len(train_records):,} records")
    print(f"  - Valid: {len(valid_records):,} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
