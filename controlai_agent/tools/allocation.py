"""Actuator allocation and fault detection tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from controlai_agent.registry import registry
from controlai_agent.verifier import verifier


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
    
    v_report = verifier.verify_allocation(B_vec, u, desired_tau)
    return {
        "u": u.tolist(),
        "achieved_tau": float(np.dot(B_vec, u)),
        "norm_u": float(np.linalg.norm(u)),
        "verification": v_report,
    }


@registry.register(
    name="actuator_fault_isolation",
    description="Isolate single actuator effectiveness loss from torque error residual: r = tau_measured - B * u_cmd.",
    parameters_schema={
        "type": "object",
        "properties": {
            "B": {"type": "array", "items": {"type": "number"}, "description": "Nominal actuator effectiveness vector"},
            "command": {"type": "array", "items": {"type": "number"}, "description": "Commanded actuator vector u_cmd"},
            "measured_tau": {"type": "number", "description": "Actual achieved torque tau_meas"},
        },
        "required": ["B", "command", "measured_tau"],
    },
)
def actuator_fault_isolation(B: list[float], command: list[float], measured_tau: float) -> dict[str, Any]:
    B_vec = np.array(B, dtype=float)
    u_vec = np.array(command, dtype=float)
    expected_tau = float(np.dot(B_vec, u_vec))
    residual = float(measured_tau - expected_tau)

    # Candidate loss fractions assuming actuator i failed
    candidate_losses = []
    for i in range(len(B_vec)):
        denom = B_vec[i] * u_vec[i]
        loss_fraction = float(-residual / denom) if abs(denom) > 1e-9 else None
        candidate_losses.append(loss_fraction)

    return {
        "expected_tau": expected_tau,
        "measured_tau": measured_tau,
        "torque_residual": residual,
        "is_fault_detected": abs(residual) > 1e-4,
        "candidate_actuator_loss_fractions": candidate_losses,
    }
