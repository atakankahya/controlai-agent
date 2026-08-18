"""Nonlinear and safety-critical control tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from controlai_agent.registry import registry
from controlai_agent.verifier import verifier


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
    
    v_report = verifier.verify_cbf(x, u_safe, alpha, x_min, u_nom)
    return {
        "h": h,
        "cbf_lower_bound": lower_bound,
        "u_safe": u_safe,
        "is_constraint_active": active,
        "verification": v_report,
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
    name="feedback_linearization_second_order",
    description="Compute feedback-linearizing control input for x1_dot = x2, x2_dot = f(x1, x2) + b*u to achieve linear dynamics y_ddot + k2*y_dot + k1*y = 0.",
    parameters_schema={
        "type": "object",
        "properties": {
            "f_drift": {"type": "number", "description": "Nonlinear drift term value f(x1, x2) at current state"},
            "b": {"type": "number", "description": "Control input gain b != 0"},
            "k1": {"type": "number", "description": "Position gain k1 > 0"},
            "k2": {"type": "number", "description": "Velocity damping gain k2 > 0"},
            "x1": {"type": "number", "description": "Current state x1 (y)"},
            "x2": {"type": "number", "description": "Current state x2 (y_dot)"},
        },
        "required": ["f_drift", "b", "k1", "k2", "x1", "x2"],
    },
)
def feedback_linearization_second_order(
    f_drift: float, b: float, k1: float, k2: float, x1: float, x2: float
) -> dict[str, Any]:
    v_des = -k1 * x1 - k2 * x2
    u = (v_des - f_drift) / b
    closed_poles = np.roots([1.0, k2, k1])
    return {
        "virtual_acceleration_v": v_des,
        "control_input_u": u,
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in closed_poles],
        "is_stable": bool(np.all(np.real(closed_poles) < 0)),
    }
