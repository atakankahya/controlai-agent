"""Classical frequency-domain, root-locus, and stability margin tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from controlai_agent.registry import registry


@registry.register(
    name="bode_analysis",
    description="Compute frequency response magnitude (linear & dB), phase (degrees), and bandwidth for transfer function G(s) = num(s) / den(s).",
    parameters_schema={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Numerator polynomial coefficients in descending powers",
            },
            "denominator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Denominator polynomial coefficients in descending powers",
            },
            "omega_min": {"type": "number", "default": 0.01, "description": "Minimum frequency in rad/s"},
            "omega_max": {"type": "number", "default": 100.0, "description": "Maximum frequency in rad/s"},
            "num_points": {"type": "integer", "default": 50, "description": "Number of log-spaced frequency points"},
        },
        "required": ["numerator", "denominator"],
    },
)
def bode_analysis(
    numerator: list[float],
    denominator: list[float],
    omega_min: float = 0.01,
    omega_max: float = 100.0,
    num_points: int = 50,
) -> dict[str, Any]:
    w = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
    sys = signal.TransferFunction(numerator, denominator)
    w_out, mag_db, phase_deg = signal.bode(sys, w)
    
    # DC gain and peak resonance
    dc_gain = float(numerator[-1] / denominator[-1]) if abs(denominator[-1]) > 1e-12 else None
    peak_mag_db = float(np.max(mag_db))
    peak_omega = float(w_out[int(np.argmax(mag_db))])

    return {
        "dc_gain": dc_gain,
        "peak_magnitude_db": peak_mag_db,
        "peak_frequency_rad_s": peak_omega,
        "frequencies_sample": w_out[::10].tolist(),
        "magnitudes_db_sample": mag_db[::10].tolist(),
        "phases_deg_sample": phase_deg[::10].tolist(),
    }


@registry.register(
    name="stability_margins",
    description="Compute classical SISO Gain Margin (ratio & dB), Phase Margin (degrees), Gain Crossover frequency, and Phase Crossover frequency.",
    parameters_schema={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop transfer function numerator coefficients",
            },
            "denominator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop transfer function denominator coefficients",
            },
        },
        "required": ["numerator", "denominator"],
    },
)
def stability_margins(numerator: list[float], denominator: list[float]) -> dict[str, Any]:
    w = np.logspace(-3, 4, 2000)
    sys = signal.TransferFunction(numerator, denominator)
    _, mag, phase_deg = signal.bode(sys, w)
    mag_linear = 10.0 ** (mag / 20.0)
    
    # Gain crossover (where |L| = 1 or mag_db = 0)
    idx_gc = int(np.argmin(np.abs(mag_linear - 1.0)))
    omega_gc = float(w[idx_gc])
    pm_deg = float(180.0 + phase_deg[idx_gc])
    
    # Phase crossover (where phase = -180 deg)
    idx_pc = int(np.argmin(np.abs(phase_deg - (-180.0))))
    omega_pc = float(w[idx_pc])
    gm_ratio = float(1.0 / mag_linear[idx_pc]) if mag_linear[idx_pc] > 1e-12 else float("inf")
    gm_db = float(20.0 * math.log10(gm_ratio)) if gm_ratio > 0 and not math.isinf(gm_ratio) else None

    is_stable = bool(pm_deg > 0 and (gm_db is None or gm_db > 0))

    return {
        "gain_margin_ratio": gm_ratio,
        "gain_margin_db": gm_db,
        "phase_margin_deg": pm_deg,
        "gain_crossover_freq_rad_s": omega_gc,
        "phase_crossover_freq_rad_s": omega_pc,
        "is_closed_loop_stable": is_stable,
    }


@registry.register(
    name="routh_hurwitz_analysis",
    description="Construct Routh-Hurwitz array for polynomial p(s) = a_n s^n + ... + a_0 and determine Hurwitz stability and right-half-plane pole count.",
    parameters_schema={
        "type": "object",
        "properties": {
            "coefficients": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Polynomial coefficients in descending order [a_n, a_n-1, ..., a_0]",
            },
        },
        "required": ["coefficients"],
    },
)
def routh_hurwitz_analysis(coefficients: list[float]) -> dict[str, Any]:
    coeffs = np.array(coefficients, dtype=float)
    n = len(coeffs) - 1
    m = (n + 2) // 2
    routh = np.zeros((n + 1, m))
    
    routh[0, : len(coeffs[0::2])] = coeffs[0::2]
    routh[1, : len(coeffs[1::2])] = coeffs[1::2]
    
    for i in range(2, n + 1):
        for j in range(m - 1):
            if abs(routh[i - 1, 0]) < 1e-12:
                routh[i - 1, 0] = 1e-6  # small epsilon perturbation
            routh[i, j] = (routh[i - 1, 0] * routh[i - 2, j + 1] - routh[i - 2, 0] * routh[i - 1, j + 1]) / routh[i - 1, 0]

    first_col = routh[:, 0].tolist()
    sign_changes = 0
    for i in range(len(first_col) - 1):
        if first_col[i] * first_col[i + 1] < 0:
            sign_changes += 1

    roots = np.roots(coeffs)
    is_hurwitz = bool(sign_changes == 0 and np.all(coeffs > 0))

    return {
        "routh_first_column": first_col,
        "sign_changes_rhp_poles": sign_changes,
        "is_hurwitz_stable": is_hurwitz,
        "roots": [[float(r.real), float(r.imag)] for r in roots],
    }
