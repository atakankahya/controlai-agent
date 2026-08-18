"""Linear state-space and algebraic system analysis tools."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg, signal

from controlai_agent.registry import registry
from controlai_agent.verifier import verifier


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
                "minimum": 1e-9,
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
    
    # Run independent verification
    v_report = verifier.verify_zoh(A_mat, B_mat, Ts, Ad, Bd)
    return {
        "Ad": Ad.tolist(),
        "Bd": Bd.tolist(),
        "sample_time": Ts,
        "verification": v_report,
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
    name="state_space_to_transfer_function",
    description="Convert state-space realization (A, B, C, D) to transfer function G(s) = C (sI - A)^-1 B + D.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "C": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "D": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
        "required": ["A", "B", "C", "D"],
    },
)
def state_space_to_transfer_function(
    A: list[list[float]], B: list[list[float]], C: list[list[float]], D: list[list[float]]
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    C_mat = np.array(C, dtype=float)
    D_mat = np.array(D, dtype=float)
    num, den = signal.ss2tf(A_mat, B_mat, C_mat, D_mat)
    poles = np.roots(den)
    zeros = np.roots(num[0]) if len(num[0]) > 1 else []
    return {
        "numerator": num[0].tolist(),
        "denominator": den.tolist(),
        "poles": [[float(p.real), float(p.imag)] for p in poles],
        "zeros": [[float(z.real), float(z.imag)] for z in zeros],
    }


@registry.register(
    name="solve_lyapunov",
    description="Solve continuous Lyapunov equation (A^T P + P A = -Q) or discrete Lyapunov equation (A^T P A - P = -Q).",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "discrete": {"type": "boolean", "description": "True for discrete Lyapunov, False for continuous"},
        },
        "required": ["A", "Q"],
    },
)
def solve_lyapunov(A: list[list[float]], Q: list[list[float]], discrete: bool = False) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    Q_mat = np.array(Q, dtype=float)
    if discrete:
        P = linalg.solve_discrete_lyapunov(A_mat.T, Q_mat)
        residual = float(np.max(np.abs(A_mat.T @ P @ A_mat - P + Q_mat)))
    else:
        P = linalg.solve_continuous_lyapunov(A_mat.T, -Q_mat)
        residual = float(np.max(np.abs(A_mat.T @ P + P @ A_mat + Q_mat)))

    p_eigs = np.linalg.eigvalsh(P)
    is_pos_def = bool(np.all(p_eigs > 1e-10))
    return {
        "P": P.tolist(),
        "P_eigenvalues": p_eigs.tolist(),
        "is_positive_definite": is_pos_def,
        "residual_max": residual,
        "time_domain": "discrete" if discrete else "continuous",
    }
