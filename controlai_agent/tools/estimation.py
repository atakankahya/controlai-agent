"""Estimation, Kalman filtering, and system identification tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from controlai_agent.registry import registry


@registry.register(
    name="kalman_time_update",
    description="Perform Kalman filter time update (state and covariance prediction) for x[k+1] = A x[k] + B u[k] + w[k].",
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "x_plus": {"type": "array", "items": {"type": "number"}, "description": "Prior state estimate x_hat+[k-1]"},
            "u": {"type": "array", "items": {"type": "number"}, "description": "Control input u[k-1]"},
            "P_plus": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Prior error covariance P+[k-1]"},
            "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Process noise covariance Q"},
        },
        "required": ["A", "x_plus", "P_plus", "Q"],
    },
)
def kalman_time_update(
    A: list[list[float]],
    P_plus: list[list[float]],
    Q: list[list[float]],
    x_plus: list[float],
    B: list[list[float]] | None = None,
    u: list[float] | None = None,
) -> dict[str, Any]:
    A_mat = np.array(A, dtype=float)
    P_mat = np.array(P_plus, dtype=float)
    Q_mat = np.array(Q, dtype=float)
    x_vec = np.array(x_plus, dtype=float)

    x_minus = A_mat @ x_vec
    if B is not None and u is not None:
        B_mat = np.array(B, dtype=float)
        u_vec = np.array(u, dtype=float)
        x_minus += B_mat @ u_vec

    P_minus = A_mat @ P_mat @ A_mat.T + Q_mat
    return {
        "x_minus_predicted": x_minus.tolist(),
        "P_minus_predicted": P_minus.tolist(),
        "is_symmetric": bool(np.max(np.abs(P_minus - P_minus.T)) < 1e-10),
    }


@registry.register(
    name="kalman_measurement_update",
    description="Perform Kalman filter measurement update using Joseph-form covariance update to ensure numerical stability and symmetry.",
    parameters_schema={
        "type": "object",
        "properties": {
            "x_minus": {"type": "array", "items": {"type": "number"}, "description": "Prior state estimate x_hat-[k]"},
            "P_minus": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Prior error covariance P-[k]"},
            "H": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Measurement matrix H"},
            "R": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Measurement noise covariance R"},
            "z": {"type": "array", "items": {"type": "number"}, "description": "Sensor measurement vector z[k]"},
        },
        "required": ["x_minus", "P_minus", "H", "R", "z"],
    },
)
def kalman_measurement_update(
    x_minus: list[float],
    P_minus: list[list[float]],
    H: list[list[float]],
    R: list[list[float]],
    z: list[float],
) -> dict[str, Any]:
    x_m = np.array(x_minus, dtype=float)
    P_m = np.array(P_minus, dtype=float)
    H_mat = np.array(H, dtype=float)
    R_mat = np.array(R, dtype=float)
    z_vec = np.array(z, dtype=float)

    innovation = z_vec - H_mat @ x_m
    S = H_mat @ P_m @ H_mat.T + R_mat
    K = P_m @ H_mat.T @ np.linalg.inv(S)
    x_plus = x_m + K @ innovation

    # Joseph stabilized form
    I = np.eye(len(x_m))
    P_plus = (I - K @ H_mat) @ P_m @ (I - K @ H_mat).T + K @ R_mat @ K.T

    return {
        "innovation": innovation.tolist(),
        "innovation_covariance_S": S.tolist(),
        "kalman_gain_K": K.tolist(),
        "x_plus_updated": x_plus.tolist(),
        "P_plus_updated": P_plus.tolist(),
    }


@registry.register(
    name="least_squares_arx",
    description="Batch ordinary least squares parameter estimation for ARX / linear regression model Y = Phi * theta + e.",
    parameters_schema={
        "type": "object",
        "properties": {
            "Phi": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Regressor matrix (N x p)"},
            "Y": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Measurement output vector (N x 1)"},
        },
        "required": ["Phi", "Y"],
    },
)
def least_squares_arx(Phi: list[list[float]], Y: list[list[float]]) -> dict[str, Any]:
    Phi_mat = np.array(Phi, dtype=float)
    Y_mat = np.array(Y, dtype=float)

    Gram = Phi_mat.T @ Phi_mat
    theta = np.linalg.solve(Gram, Phi_mat.T @ Y_mat)
    residual = Y_mat - Phi_mat @ theta
    res_norm = float(np.linalg.norm(residual))
    cond = float(np.linalg.cond(Gram))

    return {
        "theta_estimated": theta.tolist(),
        "residual_2norm": res_norm,
        "gram_matrix_condition_number": cond,
        "is_identifiable": cond < 1e12,
    }
