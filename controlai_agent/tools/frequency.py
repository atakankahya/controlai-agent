"""Classical frequency-domain, root-locus, and stability margin tools."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from controlai_agent.registry import registry

ARTIFACT_DIR = Path("outputs/plots")


@registry.register(
    name="expand_polynomial_from_roots",
    description=(
        "Expand a transfer function or characteristic polynomial given in factored/root form -- "
        "e.g. G(s) = K / (s*(s+1)*(s+5)), roots at s = 0, -1, -5 -- into exact polynomial "
        "coefficients in descending powers. ALWAYS call this instead of multiplying the factors "
        "out by hand before calling bode_analysis, stability_margins, simulate_step_response, "
        "routh_hurwitz_analysis, root_locus, or any other tool that takes numerator/denominator "
        "coefficients: hand-expanding factors is the single most common source of a silently "
        "wrong tool call, since nothing downstream can verify an argument that was already wrong "
        "going in. A factor '(s + a)' contributes the root -a; '(s - a)' contributes root a."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "roots": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The roots of the polynomial, one per linear factor -- e.g. [0, -1, -5] for s*(s+1)*(s+5)",
            },
            "gain": {
                "type": "number",
                "default": 1.0,
                "description": (
                    "Multiplies every returned coefficient. Leave this at 1 when expanding a "
                    "DENOMINATOR -- the overall constant K belongs in the numerator, not scaled "
                    "into the denominator. For G(s) = 10/(s(s+1)(s+5)), expand roots [0,-1,-5] "
                    "with gain 1 to get the denominator [1,6,5,0] and pass numerator [10] "
                    "separately. Passing gain=10 here instead yields [10,60,50,0], which is the "
                    "same transfer function scaled by 1/10 and silently wrong."
                ),
            },
        },
        "required": ["roots"],
    },
)
def expand_polynomial_from_roots(roots: list[float], gain: float = 1.0) -> dict[str, Any]:
    coefficients = (gain * np.poly(roots)).tolist()
    return {
        "status": "success",
        "coefficients_descending": [float(c) for c in coefficients],
        "degree": len(roots),
    }


@registry.register(
    name="bode_analysis",
    description=(
        "Compute the frequency response of G(s) = num(s)/den(s): magnitude in dB, phase in "
        "degrees, resonant peak, DC gain, and the gain/phase margins with both crossover "
        "frequencies."
    ),
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
        # Margins are included even though `stability_margins` is the tool that
        # advertises them. Asked for a phase margin, the model was observed
        # reaching for bode_analysis, getting back only sampled curves, and
        # concluding the margin "cannot be determined" -- a correct reading of
        # a sampled plot, and a useless answer. Computing them here makes that
        # routing choice harmless rather than fatal.
        "margins": stability_margins(numerator, denominator),
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


@registry.register(
    name="nyquist_analysis",
    description=(
        "Compute the Nyquist plot of an open-loop transfer function L(s) = num(s)/den(s), count "
        "encirclements of the critical point -1+0j, and apply the Nyquist stability criterion "
        "Z = N + P to determine closed-loop stability. Saves a PNG Nyquist diagram."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop numerator coefficients in descending powers",
            },
            "denominator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop denominator coefficients in descending powers",
            },
            "omega_max": {"type": "number", "default": 100.0, "description": "Maximum frequency in rad/s"},
        },
        "required": ["numerator", "denominator"],
    },
)
def nyquist_analysis(
    numerator: list[float],
    denominator: list[float],
    omega_max: float = 100.0,
) -> dict[str, Any]:
    num = np.array(numerator, dtype=float)
    den = np.array(denominator, dtype=float)

    # Open-loop poles: P is the count in the open right-half plane. Poles on
    # the imaginary axis (e.g. an integrator at the origin) are excluded --
    # the standard Nyquist contour indents around them.
    ol_poles = np.roots(den) if len(den) > 1 else np.array([])
    P = int(np.sum(np.real(ol_poles) > 1e-9))
    n_origin = int(np.sum(np.abs(ol_poles) < 1e-9))

    w = np.logspace(-3, np.log10(max(omega_max, 1e-2)), 4000)
    _, H = signal.freqresp(signal.TransferFunction(num, den), w=w)

    # Z (closed-loop RHP poles) and P (open-loop RHP poles) are both exact
    # root counts, so the encirclement count follows exactly as N = Z - P.
    # Numerically integrating the winding of L(jw)+1 instead is unreliable for
    # systems with poles on the imaginary axis (a type-1 integrator here),
    # where the Nyquist contour must indent around the origin -- that shortcut
    # yields impossible results such as N = -1 with Z = -1.
    closed_loop_poles = (
        np.roots(np.polyadd(den, np.pad(num, (len(den) - len(num), 0))))
        if len(den) >= len(num)
        else np.array([])
    )
    Z = int(np.sum(np.real(closed_loop_poles) > 1e-9)) if closed_loop_poles.size else 0
    N = Z - P

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / f"nyquist_{abs(hash((str(numerator), str(denominator)))) % 10**8:08d}.png"

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=140)
    ax.plot(H.real, H.imag, color="#58a6ff", linewidth=1.8, label="$L(j\\omega)$, $\\omega > 0$")
    ax.plot(H.real, -H.imag, color="#58a6ff", linewidth=1.0, linestyle="--", alpha=0.6, label="$\\omega < 0$ (mirror)")
    ax.plot(-1.0, 0.0, "x", color="#f85149", markersize=11, markeredgewidth=2.5, label="Critical point $-1+0j$")
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.set_title("Nyquist Diagram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.grid(True, linestyle=":", alpha=0.45)
    lim = float(min(max(3.0, np.percentile(np.abs(H), 92)), 25.0))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "status": "success",
        "encirclements_N": N,
        "open_loop_rhp_poles_P": P,
        "open_loop_poles_at_origin": n_origin,
        "closed_loop_rhp_poles_Z": Z,
        "closed_loop_poles": [[float(p.real), float(p.imag)] for p in closed_loop_poles],
        "is_closed_loop_stable": bool(Z == 0),
        "criterion": "Z = N + P (Z = closed-loop RHP poles, N = clockwise encirclements of -1, P = open-loop RHP poles)",
        "plot_path": str(plot_path),
    }


@registry.register(
    name="root_locus_analysis",
    description=(
        "Compute the root locus of the closed-loop characteristic equation 1 + k*L(s) = 0 as the gain "
        "k sweeps from 0 to infinity, for open-loop L(s) = num(s)/den(s). Returns open-loop poles and "
        "zeros, asymptote angles and centroid, real-axis breakaway points, the imaginary-axis crossing "
        "gain (critical gain for stability), and a PNG root locus plot."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop numerator coefficients in descending powers",
            },
            "denominator": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Open-loop denominator coefficients in descending powers",
            },
            "k_max": {"type": "number", "default": 100.0, "description": "Maximum gain k to sweep"},
        },
        "required": ["numerator", "denominator"],
    },
)
def root_locus_analysis(
    numerator: list[float],
    denominator: list[float],
    k_max: float = 100.0,
) -> dict[str, Any]:
    num = np.array(numerator, dtype=float)
    den = np.array(denominator, dtype=float)

    ol_zeros = np.roots(num) if len(num) > 1 else np.array([])
    ol_poles = np.roots(den) if len(den) > 1 else np.array([])
    n_p, n_z = len(ol_poles), len(ol_zeros)

    # Asymptotes for the n_p - n_z branches heading to infinity
    excess = n_p - n_z
    asymptote_angles, centroid = [], None
    if excess > 0:
        centroid = float((np.sum(ol_poles).real - np.sum(ol_zeros).real) / excess)
        asymptote_angles = [float((180.0 * (2 * i + 1)) / excess) for i in range(excess)]

    # Sweep gain and collect closed-loop roots of den + k*num
    gains = np.concatenate([[0.0], np.logspace(-3, np.log10(max(k_max, 1e-2)), 600)])
    locus: list[np.ndarray] = []
    for k in gains:
        poly = np.polyadd(den, k * np.pad(num, (max(0, len(den) - len(num)), 0)))
        locus.append(np.roots(poly))

    # Imaginary-axis crossing: first gain where any root's real part turns >= 0
    k_critical, w_crossing = None, None
    for k, roots in zip(gains, locus):
        if roots.size and np.any(np.real(roots) > 1e-9):
            k_critical = float(k)
            crossing = roots[np.argmax(np.real(roots))]
            w_crossing = float(abs(crossing.imag))
            break

    # Breakaway/break-in points: real roots of d/ds[-den/num] = 0. Only those
    # lying ON the locus count -- a real point belongs to the locus iff an odd
    # number of real poles and zeros lie strictly to its right, so the
    # remaining stationary points must be discarded.
    real_singularities = [float(p.real) for p in ol_poles if abs(p.imag) < 1e-8]
    real_singularities += [float(z.real) for z in ol_zeros if abs(z.imag) < 1e-8]

    def _on_real_axis_locus(sigma: float) -> bool:
        to_right = sum(1 for v in real_singularities if v > sigma + 1e-9)
        return to_right % 2 == 1

    breakaway: list[float] = []
    try:
        dnum, dden = np.polyder(num), np.polyder(den)
        crit = np.polysub(np.polymul(dden, num), np.polymul(den, dnum))
        for r in np.roots(crit):
            if abs(r.imag) < 1e-8 and _on_real_axis_locus(float(r.real)):
                breakaway.append(round(float(r.real), 6))
    except Exception:
        pass

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / f"root_locus_{abs(hash((str(numerator), str(denominator)))) % 10**8:08d}.png"

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=140)
    max_branches = max((r.size for r in locus), default=0)
    for b in range(max_branches):
        pts = np.array([r[b] for r in locus if r.size > b])
        ax.plot(pts.real, pts.imag, color="#58a6ff", linewidth=1.0, alpha=0.85)
    if n_p:
        ax.plot(ol_poles.real, ol_poles.imag, "x", color="#f85149", markersize=10, markeredgewidth=2.2, label="Open-loop poles")
    if n_z:
        ax.plot(ol_zeros.real, ol_zeros.imag, "o", mfc="none", color="#3fb950", markersize=9, markeredgewidth=2.0, label="Open-loop zeros")
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.set_title("Root Locus", fontsize=12, fontweight="bold")
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.grid(True, linestyle=":", alpha=0.45)
    if n_p or n_z:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "status": "success",
        "open_loop_poles": [[float(p.real), float(p.imag)] for p in ol_poles],
        "open_loop_zeros": [[float(z.real), float(z.imag)] for z in ol_zeros],
        "num_asymptotes": excess,
        "asymptote_centroid": centroid,
        "asymptote_angles_deg": asymptote_angles,
        "breakaway_points_real_axis": sorted(set(breakaway)),
        "critical_gain_k_at_instability": k_critical,
        "imaginary_axis_crossing_freq_rad_s": w_crossing,
        "is_stable_for_all_swept_gains": bool(k_critical is None),
        "plot_path": str(plot_path),
    }
