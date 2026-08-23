"""Controller synthesis tools: LQR, Pole Placement, PID Tuning, and MPC QP."""

from __future__ import annotations

import math
from typing import Any

import cvxpy as cp
import numpy as np
from scipy import linalg, signal

from controlai_agent.registry import registry
from controlai_agent.verifier import verifier


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

    v_report = verifier.verify_care(A_mat, B_mat, Q_mat, R_mat, P, K)
    return {
        "P": P.tolist(),
        "K": K.tolist(),
        # Returned so the answer never has to derive it. The model was
        # observed writing A - BK as [[-6, 1], [-5, -6]] for a double
        # integrator whose true closed loop is [[0, 1], [-6, -5]] -- correct
        # gain, wrong write-up. Handing it the computed matrix removes the
        # arithmetic from the answer entirely.
        "closed_loop_A": (A - B @ K).tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in poles],
        "is_stable": bool(np.all(np.real(poles) < 0)),
        "verification": v_report,
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

    v_report = verifier.verify_dare(A_mat, B_mat, Q_mat, R_mat, P, K)
    return {
        "P": P.tolist(),
        "K": K.tolist(),
        "closed_loop_A": (A - B @ K).tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in poles],
        "is_stable": bool(np.all(np.abs(poles) < 1.0)),
        "verification": v_report,
    }


@registry.register(
    name="place_state_feedback",
    description=(
        "Compute state feedback gain matrix K such that eig(A - B*K) matches target desired poles. "
        "Each entry of desired_poles is either a real number (a real pole) or a [real, imag] pair "
        "(one half of a complex-conjugate pair -- e.g. targeting a damping ratio/natural frequency "
        "needs poles like [-2, 3] and [-2, -3], both halves listed explicitly)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "desired_poles": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    ]
                },
                "description": "Target closed-loop pole locations: a number for a real pole, or [real, imag] for a complex one.",
            },
        },
        "required": ["A", "B", "desired_poles"],
    },
)
def place_state_feedback(
    A: list[list[float]], B: list[list[float]], desired_poles: list[float | list[float]]
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    # Each pole is a plain number (real pole) or a [real, imag] pair.
    desired_poles = [complex(p[0], p[1]) if isinstance(p, (list, tuple)) else complex(p) for p in desired_poles]
    # dtype=complex, not float: targeting a specific damping ratio / natural
    # frequency means passing a complex-conjugate pole pair, which is the
    # normal case for a 2nd-order-or-higher design, not an edge case. Forcing
    # float here silently discarded the imaginary part, turning a legitimate
    # conjugate pair into the SAME real pole listed twice -- which then fails
    # outright for a SISO system since place_poles cannot repeat a pole more
    # than rank(B) times.
    des = np.array(desired_poles, dtype=complex)
    placed = signal.place_poles(A_mat, B_mat, des)
    # K is mathematically real for real A, B with a properly conjugate-paired
    # desired_poles; drop the negligible numerical imaginary residue so the
    # result is JSON-serializable (a genuinely unpaired complex pole is
    # rejected by place_poles itself before this line is reached).
    K = placed.gain_matrix.real
    closed_poles = np.linalg.eigvals(A_mat - B_mat @ K)

    v_report = verifier.verify_pole_placement(A_mat, B_mat, K, list(des))
    return {
        "K": K.tolist(),
        "closed_loop_A": (A - B @ K).tolist(),
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in closed_poles],
        "target_poles": [[float(p.real), float(p.imag)] for p in des],
        "verification": v_report,
    }


@registry.register(
    name="pid_tune_fopdt",
    description="Tune standard PID gains (Kp, Ki, Kd) for First-Order Plus Dead-Time (FOPDT) plant G(s) = K*e^(-L s)/(T s + 1) using Chien-Hrones-Reswick (CHR) method.",
    parameters_schema={
        "type": "object",
        "properties": {
            "K_plant": {"type": "number", "description": "Static process gain K"},
            "T_tau": {"type": "number", "minimum": 1e-6, "description": "Time constant T in seconds"},
            "L_delay": {"type": "number", "minimum": 1e-6, "description": "Apparent dead time / delay L in seconds"},
            "tuning_objective": {
                "type": "string",
                "enum": ["setpoint_tracking_0_overshoot", "setpoint_tracking_20_overshoot", "disturbance_rejection"],
                "default": "setpoint_tracking_0_overshoot",
            },
        },
        "required": ["K_plant", "T_tau", "L_delay"],
    },
)
def pid_tune_fopdt(
    K_plant: float,
    T_tau: float,
    L_delay: float,
    tuning_objective: str = "setpoint_tracking_0_overshoot",
) -> dict[str, Any]:
    a = (K_plant * L_delay) / T_tau
    if tuning_objective == "setpoint_tracking_0_overshoot":
        kp = 0.6 / a
        ti = T_tau
        td = 0.5 * L_delay
    elif tuning_objective == "setpoint_tracking_20_overshoot":
        kp = 0.95 / a
        ti = 1.35 * T_tau
        td = 0.47 * L_delay
    else:  # disturbance rejection
        kp = 1.2 / a
        ti = 2.0 * L_delay
        td = 0.42 * L_delay

    ki = kp / ti
    kd = kp * td

    return {
        "Kp": kp,
        "Ki": ki,
        "Kd": kd,
        "Ti_integral_time": ti,
        "Td_derivative_time": td,
        "tuning_rule": "Chien-Hrones-Reswick (CHR)",
        "objective": tuning_objective,
    }


@registry.register(
    name="mpc_solve_qp",
    description="Solve finite-horizon Model Predictive Control (MPC) quadratic program with box state/input constraints via CVXPY and Clarabel.",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Discrete state matrix A"},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Discrete input matrix B"},
            "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "State penalty matrix Q"},
            "R": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Input penalty matrix R"},
            "x0": {"type": "array", "items": {"type": "number"}, "description": "Initial state vector"},
            "horizon": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Prediction horizon N"},
            "u_limit": {"type": "number", "description": "Input constraint magnitude |u_k| <= u_limit"},
        },
        "required": ["A", "B", "Q", "R", "x0", "horizon"],
    },
)
def mpc_solve_qp(
    A: list[list[float]],
    B: list[list[float]],
    Q: list[list[float]],
    R: list[list[float]],
    x0: list[float],
    horizon: int,
    u_limit: float | None = None,
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    B_mat = np.array(B, dtype=float)
    Q_mat = np.array(Q, dtype=float)
    R_mat = np.array(R, dtype=float)
    x0_vec = np.array(x0, dtype=float)

    nx = A_mat.shape[0]
    nu = B_mat.shape[1] if B_mat.ndim > 1 else 1

    x = cp.Variable((nx, horizon + 1))
    u = cp.Variable((nu, horizon))

    cost = 0
    constraints = [x[:, 0] == x0_vec]

    for k in range(horizon):
        cost += cp.quad_form(x[:, k], Q_mat) + cp.quad_form(u[:, k], R_mat)
        constraints.append(x[:, k + 1] == A_mat @ x[:, k] + B_mat @ u[:, k])
        if u_limit is not None:
            constraints.append(cp.norm_inf(u[:, k]) <= u_limit)

    cost += cp.quad_form(x[:, horizon], Q_mat)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    opt_cost = prob.solve(solver="CLARABEL")

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": "error", "error": f"MPC solver failed: {prob.status}"}

    u_opt = u.value.tolist()
    x_opt = x.value.tolist()

    return {
        "status": "success",
        "first_control_move": u.value[:, 0].tolist(),
        "optimal_cost": float(opt_cost),
        "predicted_state_trajectory": x_opt,
        "optimal_input_trajectory": u_opt,
        "solver": "CLARABEL",
    }
