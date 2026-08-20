"""Time-domain simulation and plotting artifact generation tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend for artifact generation
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from controlai_agent.registry import registry

ARTIFACT_DIR = Path("outputs/plots")


def _step_metrics(t_out: np.ndarray, y_out: np.ndarray) -> dict[str, Any]:
    """Standard transient response metrics shared by the simulation tools."""
    y_final = float(y_out[-1])
    y_peak = float(np.max(y_out))
    t_peak = float(t_out[int(np.argmax(y_out))])
    overshoot_pct = (
        float(max(0.0, (y_peak - y_final) / abs(y_final) * 100.0)) if abs(y_final) > 1e-6 else 0.0
    )

    idx_10 = np.where(y_out >= 0.1 * y_final)[0]
    idx_90 = np.where(y_out >= 0.9 * y_final)[0]
    rise_time = float(t_out[idx_90[0]] - t_out[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else None

    settled = np.where(np.abs(y_out - y_final) > 0.02 * abs(y_final))[0]
    settling_time = float(t_out[settled[-1]]) if len(settled) > 0 else 0.0

    return {
        "final_value": y_final,
        "peak_value": y_peak,
        "peak_time_seconds": t_peak,
        "overshoot_percentage": overshoot_pct,
        "rise_time_seconds": rise_time,
        "settling_time_2pct_seconds": settling_time,
    }


@registry.register(
    name="simulate_state_feedback_response",
    description=(
        "Simulate the step response of a state-space system directly from matrices, optionally under "
        "state feedback u = -Kx. USE THIS (never a hand-derived transfer function) whenever you have "
        "A, B and a gain K from continuous_lqr, discrete_lqr, or place_state_feedback: it builds the "
        "closed-loop system A - B*K internally, so you never have to expand closed-loop polynomial "
        "coefficients by hand. Returns the closed-loop matrix, poles, damping, the exact closed-loop "
        "transfer function coefficients, transient metrics, and a PNG plot."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Open-loop state matrix A"},
            "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Input matrix B"},
            "K": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Optional state feedback gain K (from LQR/pole placement). If given, simulates closed loop A - B*K. Omit for open-loop.",
            },
            "C": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Optional output matrix C. Defaults to [[1, 0, ..., 0]] (measures the first state).",
            },
            "D": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Optional feedthrough matrix D (defaults to zero)."},
            "normalize_dc_gain": {
                "type": "boolean",
                "default": False,
                "description": "If true, scale the input by a precompensator so the step response settles at 1.0 (removes steady-state offset inherent to pure state feedback).",
            },
            "sim_time": {"type": "number", "default": 10.0, "description": "Total simulation time in seconds"},
            "plot_title": {"type": "string", "default": "Closed-Loop Step Response", "description": "Title for the saved plot"},
        },
        "required": ["A", "B"],
    },
)
def simulate_state_feedback_response(
    A: list[list[float]],
    B: list[list[float]],
    K: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
    D: list[list[float]] | None = None,
    normalize_dc_gain: bool = False,
    sim_time: float = 10.0,
    plot_title: str = "Closed-Loop Step Response",
) -> dict[str, Any]:
    A_mat = np.atleast_2d(np.array(A, dtype=float))
    B_mat = np.array(B, dtype=float)
    if B_mat.ndim == 1:
        B_mat = B_mat.reshape(-1, 1)

    n = A_mat.shape[0]
    if A_mat.shape[0] != A_mat.shape[1]:
        return {"status": "error", "error": f"A must be square, got shape {A_mat.shape}."}
    if B_mat.shape[0] != n:
        return {"status": "error", "error": f"B row count {B_mat.shape[0]} does not match A dimension {n}."}

    # Closed loop under u = -Kx (the step is then applied as the reference input)
    K_mat = None
    if K is not None:
        K_mat = np.atleast_2d(np.array(K, dtype=float))
        if K_mat.shape[1] != n:
            return {"status": "error", "error": f"K must have {n} columns to match the state dimension, got {K_mat.shape}."}
        A_eff = A_mat - B_mat @ K_mat
    else:
        A_eff = A_mat

    C_mat = np.atleast_2d(np.array(C, dtype=float)) if C is not None else np.eye(1, n)
    D_mat = np.atleast_2d(np.array(D, dtype=float)) if D is not None else np.zeros((C_mat.shape[0], B_mat.shape[1]))

    poles = np.linalg.eigvals(A_eff)

    # Optional precompensator so the closed loop actually tracks a unit step
    dc_scale = 1.0
    if normalize_dc_gain:
        try:
            dc = float(-(C_mat @ np.linalg.solve(A_eff, B_mat) - D_mat).ravel()[0])
            if abs(dc) > 1e-12:
                dc_scale = 1.0 / dc
        except np.linalg.LinAlgError:
            dc_scale = 1.0

    sys = signal.StateSpace(A_eff, B_mat * dc_scale, C_mat, D_mat)
    t = np.linspace(0, sim_time, 1000)
    t_out, y_out = signal.step(sys, T=t)
    y_out = np.asarray(y_out, dtype=float).ravel()

    num, den = signal.ss2tf(A_eff, B_mat, C_mat, D_mat)

    metrics = _step_metrics(t_out, y_out)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / f"state_feedback_step_{abs(hash((str(A), str(B), str(K), sim_time))) % 10**8:08d}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(t_out, y_out, color="#58a6ff", linewidth=2.0, label="Response $y(t)$")
    y_final = metrics["final_value"]
    ax.axhline(y_final, color="#f85149", linestyle="--", alpha=0.8, label=f"Final Value ({y_final:.4f})")
    ax.axhline(y_final * 1.02, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y_final * 0.98, color="gray", linestyle=":", alpha=0.5, label="2% Settling Band")
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time [seconds]", fontsize=10)
    ax.set_ylabel("Output Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    wn = [float(abs(p)) for p in poles]
    zeta = [float(-np.real(p) / abs(p)) if abs(p) > 1e-12 else 0.0 for p in poles]

    return {
        "status": "success",
        "mode": "closed_loop_state_feedback" if K_mat is not None else "open_loop",
        "closed_loop_A": A_eff.tolist(),
        "poles": [[float(p.real), float(p.imag)] for p in poles],
        "natural_frequencies_rad_s": wn,
        "damping_ratios": zeta,
        "is_stable": bool(np.all(np.real(poles) < 0)),
        "closed_loop_tf_numerator": np.asarray(num).ravel().tolist(),
        "closed_loop_tf_denominator": np.asarray(den).ravel().tolist(),
        "dc_precompensator_applied": dc_scale if normalize_dc_gain else None,
        **metrics,
        "plot_path": str(plot_path),
        "plot_artifact_path": str(plot_path),
    }


@registry.register(
    name="simulate_step_response",
    description=(
        "Simulate the unit step response of a continuous transfer function G(s) = num(s)/den(s), compute "
        "rise time, overshoot, settling time, and save a high-resolution PNG plot artifact. Only use this "
        "when the system is genuinely given to you as a transfer function -- if you have state-space "
        "matrices A, B and/or a feedback gain K, call simulate_state_feedback_response instead rather "
        "than deriving closed-loop coefficients yourself."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Numerator coefficients in descending powers",
            },
            "denominator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Denominator coefficients in descending powers",
            },
            "sim_time": {"type": "number", "default": 10.0, "description": "Total simulation time in seconds"},
            "plot_title": {"type": "string", "default": "Closed-Loop Step Response", "description": "Title for the saved plot"},
            "plot_filename": {"type": "string", "default": "step_response.png", "description": "Filename for the PNG plot artifact"},
        },
        "required": ["numerator", "denominator"],
    },
)
def simulate_step_response(
    numerator: list[float],
    denominator: list[float],
    sim_time: float = 10.0,
    plot_title: str = "Closed-Loop Step Response",
    plot_filename: str = "step_response.png",
) -> dict[str, Any]:
    sys = signal.TransferFunction(numerator, denominator)
    t = np.linspace(0, sim_time, 1000)
    t_out, y_out = signal.step(sys, T=t)
    y_out = np.asarray(y_out, dtype=float).ravel()

    metrics = _step_metrics(t_out, y_out)
    y_final = metrics["final_value"]

    # Generate Matplotlib PNG Plot
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / plot_filename

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(t_out, y_out, "b-", linewidth=2.0, label="Response $y(t)$")
    ax.axhline(y_final, color="r", linestyle="--", alpha=0.7, label=f"Final Value ({y_final:.3f})")
    ax.axhline(y_final * 1.02, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y_final * 0.98, color="gray", linestyle=":", alpha=0.5, label="2% Settling Band")
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time [seconds]", fontsize=10)
    ax.set_ylabel("Output Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "status": "success",
        **metrics,
        # "plot_path" is the key the orchestrator looks for when surfacing
        # generated figures to the UI; "plot_artifact_path" is kept for
        # backward compatibility with existing benchmark/eval artifacts.
        "plot_path": str(plot_path),
        "plot_artifact_path": str(plot_path),
    }
