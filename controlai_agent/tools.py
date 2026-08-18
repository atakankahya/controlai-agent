"""Deterministic control engineering tools backed by NumPy, SciPy, and CVXPY."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import linalg, signal

from controlai_agent.registry import registry


@registry.register(
    name="exact_zoh",
    description="Discretize continuous-time state-space matrices (A, B) under exact Zero-Order Hold (ZOH) at sample time Ts.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Continuous system matrix A (n x n)",
            },
            "B": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Continuous input matrix B (n x m)",
            },
            "Ts": {
                "type": "number",
                "description": "Sample time in seconds (Ts > 0)",
            },
        },
        "required": ["A", "B", "Ts"],
    },
)
def exact_zoh(A: list[list[float]], B: list[list[float]], Ts: float) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    n = A_mat.shape[0]
    m = B_mat.shape[1] if B_mat.ndim > 1 else 1
    C_dummy = np.eye(n)
    D_dummy = np.zeros((n, m))
    Ad, Bd, _, _, _ = signal.cont2discrete((A_mat, B_mat, C_dummy, D_dummy), Ts, method="zoh")
    return {
        "Ad": Ad.tolist(),
        "Bd": Bd.tolist(),
        "sample_time": Ts,
        "method": "exact_zoh",
    }


@registry.register(
    name="eigen_analysis",
    description="Compute eigenvalues, eigenvectors, and asymptotic stability of a continuous or discrete system matrix A.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "System matrix A (n x n)",
            },
            "discrete": {
                "type": "boolean",
                "description": "True if discrete-time (criterion |lambda| < 1), False if continuous (Re(lambda) < 0)",
            },
        },
        "required": ["A"],
    },
)
def eigen_analysis(A: list[list[float]], discrete: bool = False) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    eigenvalues = np.linalg.eigvals(A_mat)
    if discrete:
        stable = bool(np.all(np.abs(eigenvalues) < 1.0))
        criterion = "|lambda_i| < 1"
    else:
        stable = bool(np.all(np.real(eigenvalues) < 0.0))
        criterion = "Re(lambda_i) < 0"

    formatted_eigs = []
    for eig in eigenvalues:
        if abs(eig.imag) < 1e-9:
            formatted_eigs.append(f"{eig.real:.6g}")
        else:
            sign = "+" if eig.imag >= 0 else "-"
            formatted_eigs.append(f"{eig.real:.6g} {sign} {abs(eig.imag):.6g}j")

    return {
        "eigenvalues": [[float(e.real), float(e.imag)] for e in eigenvalues],
        "eigenvalues_formatted": formatted_eigs,
        "is_stable": stable,
        "criterion": criterion,
        "time_domain": "discrete" if discrete else "continuous",
    }


@registry.register(
    name="controllability_analysis",
    description="Compute controllability matrix, rank, and PBH modal controllability test for pair (A, B).",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
        "required": ["A", "B"],
    },
)
def controllability_analysis(A: list[list[float]], B: list[list[float]]) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    n = A_mat.shape[0]
    blocks = [B_mat]
    current = B_mat
    for _ in range(1, n):
        current = A_mat @ current
        blocks.append(current)
    C_mat = np.hstack(blocks)
    rank = int(np.linalg.matrix_rank(C_mat))
    is_controllable = rank == n
    return {
        "controllability_matrix": C_mat.tolist(),
        "rank": rank,
        "state_dimension": n,
        "is_controllable": is_controllable,
    }


@registry.register(
    name="observability_analysis",
    description="Compute observability matrix, rank, and PBH modal observability test for pair (A, C).",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "C": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
        "required": ["A", "C"],
    },
)
def observability_analysis(A: list[list[float]], C: list[list[float]]) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    C_mat = np.array(C, dtype=float)
    n = A_mat.shape[0]
    blocks = [C_mat]
    current = C_mat
    for _ in range(1, n):
        current = current @ A_mat
        blocks.append(current)
    O_mat = np.vstack(blocks)
    rank = int(np.linalg.matrix_rank(O_mat))
    is_observable = rank == n
    return {
        "observability_matrix": O_mat.tolist(),
        "rank": rank,
        "state_dimension": n,
        "is_observable": is_observable,
    }


@registry.register(
    name="continuous_lqr",
    description="Solve Continuous-time Linear Quadratic Regulator (CARE) problem: min integral (x^T Q x + u^T R u) dt.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "R": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
        "required": ["A", "B", "Q", "R"],
    },
)
def continuous_lqr(
    A: list[list[float]], B: list[list[float]], Q: list[list[float]], R: list[list[float]]
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    Q_mat = np.array(Q, dtype=float)
    R_mat = np.array(R, dtype=float)
    P = linalg.solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)
    K = np.linalg.solve(R_mat, B_mat.T @ P)
    A_cl = A_mat - B_mat @ K
    poles = np.linalg.eigvals(A_cl)
    residual = float(np.max(np.abs(A_mat.T @ P + P @ A_mat - P @ B_mat @ np.linalg.inv(R_mat) @ B_mat.T @ P + Q_mat)))
    return {
        "P": P.tolist(),
        "K": K.tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in poles],
        "is_stable": bool(np.all(np.real(poles) < 0)),
        "riccati_residual": residual,
    }


@registry.register(
    name="discrete_lqr",
    description="Solve Discrete-time Linear Quadratic Regulator (DARE) problem: min sum (x_k^T Q x_k + u_k^T R u_k).",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "R": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
        "required": ["A", "B", "Q", "R"],
    },
)
def discrete_lqr(
    A: list[list[float]], B: list[list[float]], Q: list[list[float]], R: list[list[float]]
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    Q_mat = np.array(Q, dtype=float)
    R_mat = np.array(R, dtype=float)
    P = linalg.solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)
    K = np.linalg.solve(R_mat + B_mat.T @ P @ B_mat, B_mat.T @ P @ A_mat)
    A_cl = A_mat - B_mat @ K
    poles = np.linalg.eigvals(A_cl)
    return {
        "P": P.tolist(),
        "K": K.tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in poles],
        "is_stable": bool(np.all(np.abs(poles) < 1.0)),
    }


@registry.register(
    name="cbf_safety_filter",
    description="Scalar Control Barrier Function (CBF) quadratic program safety filter for single integrator x_dot=u with constraint x >= x_min.",
    parameters_schema={
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "Current state"},
            "u_nom": {"type": "number", "description": "Nominal desired control input"},
            "alpha": {"type": "number", "description": "CBF class-K gain parameter"},
            "x_min": {"type": "number", "description": "Safety lower bound x >= x_min"},
        },
        "required": ["x", "u_nom", "alpha", "x_min"],
    },
)
def cbf_safety_filter(x: float, u_nom: float, alpha: float, x_min: float) -> dict[str, Any]:
    h = x - x_min
    lower_bound = -alpha * h
    u_safe = max(u_nom, lower_bound)
    active = bool(u_nom < lower_bound)
    return {
        "h": h,
        "cbf_lower_bound": lower_bound,
        "u_safe": u_safe,
        "is_constraint_active": active,
        "h_dot_plus_alpha_h": u_safe + alpha * h,
    }


@registry.register(
    name="dynamic_inversion",
    description="Compute exact Nonlinear Dynamic Inversion (NDI) input u for scalar affine system x_dot = a*x + b*u to track target error dynamics x_dot = -gain*(x - r).",
    parameters_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Open-loop plant coefficient a"},
            "b": {"type": "number", "description": "Control effectiveness coefficient b (b != 0)"},
            "x": {"type": "number", "description": "Current state x"},
            "reference": {"type": "number", "description": "Setpoint reference r"},
            "gain": {"type": "number", "description": "Tracking convergence gain k > 0"},
        },
        "required": ["a", "b", "x", "reference", "gain"],
    },
)
def dynamic_inversion(a: float, b: float, x: float, reference: float, gain: float) -> dict[str, Any]:
    virtual_control = -gain * (x - reference)
    u = (virtual_control - a * x) / b
    x_dot = a * x + b * u
    return {
        "virtual_control_v": virtual_control,
        "control_input_u": u,
        "achieved_x_dot": x_dot,
        "residual": abs(x_dot - virtual_control),
    }


@registry.register(
    name="kharitonov_stability_test",
    description="Evaluate robust Hurwitz stability of a real interval polynomial using Kharitonov's Theorem (tests the 4 extreme polynomials).",
    parameters_schema={
        "type": "object",
        "properties": {
            "lower_bounds": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Lower coefficient bounds [a0^-, a1^-, a2^-, a3^-] in ascending powers",
            },
            "upper_bounds": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Upper coefficient bounds [a0^+, a1^+, a2^+, a3^+] in ascending powers",
            },
        },
        "required": ["lower_bounds", "upper_bounds"],
    },
)
def kharitonov_stability_test(lower_bounds: list[float], upper_bounds: list[float]) -> dict[str, Any]:
    lows = np.array(lower_bounds, dtype=float)
    highs = np.array(upper_bounds, dtype=float)
    polys = [
        [lows[0], lows[1], highs[2], highs[3]],
        [highs[0], highs[1], lows[2], lows[3]],
        [highs[0], lows[1], lows[2], highs[3]],
        [lows[0], highs[1], highs[2], lows[3]],
    ]
    pole_sets = []
    stable_flags = []
    for poly in polys:
        desc_poly = poly[::-1]  # descending for np.roots
        roots = np.roots(desc_poly)
        pole_sets.append([[float(r.real), float(r.imag)] for r in roots])
        stable_flags.append(bool(np.all(np.real(roots) < 0)))

    robustly_stable = all(stable_flags)
    return {
        "kharitonov_polynomials_ascending": polys,
        "stable_flags": stable_flags,
        "is_robustly_hurwitz": robustly_stable,
        "poles": pole_sets,
    }


@registry.register(
    name="minimum_norm_control_allocation",
    description="Compute minimum 2-norm control allocation for redundant actuators: min ||u||_2 subject to B*u = tau.",
    parameters_schema={
        "type": "object",
        "properties": {
            "B": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Actuator effectiveness row vector B (1 x m)",
            },
            "desired_tau": {
                "type": "number",
                "description": "Desired virtual control torque/force tau",
            },
        },
        "required": ["B", "desired_tau"],
    },
)
def minimum_norm_control_allocation(B: list[float], desired_tau: float) -> dict[str, Any]:
    B_vec = np.array(B, dtype=float)
    b_norm_sq = float(np.dot(B_vec, B_vec))
    u = (desired_tau / b_norm_sq) * B_vec
    achieved = float(np.dot(B_vec, u))
    return {
        "u": u.tolist(),
        "achieved_tau": achieved,
        "residual": abs(achieved - desired_tau),
        "norm_u": float(np.linalg.norm(u)),
    }


@registry.register(
    name="jump_system_stability",
    description="Compute second-moment contraction factor E[a^2] = sum(p_i * a_i^2) for i.i.d. scalar jump linear systems.",
    parameters_schema={
        "type": "object",
        "properties": {
            "probabilities": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Probabilities p_i summing to 1",
            },
            "multipliers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Multipliers a_i for each jump mode",
            },
        },
        "required": ["probabilities", "multipliers"],
    },
)
def jump_system_stability(probabilities: list[float], multipliers: list[float]) -> dict[str, Any]:
    p = np.array(probabilities, dtype=float)
    a = np.array(multipliers, dtype=float)
    expected_square = float(np.sum(p * (a**2)))
    is_stable = expected_square < 1.0
    return {
        "expected_squared_multiplier": expected_square,
        "is_mean_square_stable": is_stable,
        "contraction_margin": 1.0 - expected_square,
    }


@registry.register(
    name="place_state_feedback",
    description="Compute state feedback gain matrix K such that eig(A - B*K) matches target desired poles.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "desired_poles": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Target closed-loop pole locations",
            },
        },
        "required": ["A", "B", "desired_poles"],
    },
)
def place_state_feedback(
    A: list[list[float]], B: list[list[float]], desired_poles: list[float]
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    des = np.array(desired_poles, dtype=float)
    placed = signal.place_poles(A_mat, B_mat, des)
    K = placed.gain_matrix
    closed_poles = np.linalg.eigvals(A_mat - B_mat @ K)
    return {
        "K": K.tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in closed_poles],
        "target_poles": desired_poles,
    }
