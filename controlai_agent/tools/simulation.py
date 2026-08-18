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


@registry.register(
    name="simulate_step_response",
    description="Simulate the unit step response of a continuous transfer function G(s) = num(s)/den(s), compute rise time, overshoot, settling time, and save a high-resolution PNG plot artifact.",
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

    # Transient metrics
    y_final = float(y_out[-1])
    y_peak = float(np.max(y_out))
    t_peak = float(t_out[int(np.argmax(y_out))])
    overshoot_pct = float(max(0.0, (y_peak - y_final) / abs(y_final) * 100.0)) if abs(y_final) > 1e-6 else 0.0

    # 10% to 90% Rise Time
    idx_10 = np.where(y_out >= 0.1 * y_final)[0]
    idx_90 = np.where(y_out >= 0.9 * y_final)[0]
    rise_time = float(t_out[idx_90[0]] - t_out[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else None

    # 2% Settling Time
    settled_indices = np.where(np.abs(y_out - y_final) > 0.02 * abs(y_final))[0]
    settling_time = float(t_out[settled_indices[-1]]) if len(settled_indices) > 0 else 0.0

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
        "final_value": y_final,
        "peak_value": y_peak,
        "peak_time_seconds": t_peak,
        "overshoot_percentage": overshoot_pct,
        "rise_time_seconds": rise_time,
        "settling_time_2pct_seconds": settling_time,
        "plot_artifact_path": str(plot_path),
    }
