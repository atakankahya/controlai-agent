"""Robust control and uncertain system tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from controlai_agent.registry import registry


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
    name="small_gain_test",
    description="Apply the Small Gain Theorem: certifies closed-loop input-output stability if induced norm product ||G1|| * ||G2|| < 1.",
    parameters_schema={
        "type": "object",
        "properties": {
            "gamma1": {"type": "number", "minimum": 0.0, "description": "H-infinity or induced norm bound of operator G1"},
            "gamma2": {"type": "number", "minimum": 0.0, "description": "H-infinity or induced norm bound of operator G2"},
        },
        "required": ["gamma1", "gamma2"],
    },
)
def small_gain_test(gamma1: float, gamma2: float) -> dict[str, Any]:
    product = gamma1 * gamma2
    certified = bool(product < 1.0)
    return {
        "norm_product": product,
        "is_stability_certified": certified,
        "stability_margin": 1.0 - product if certified else 0.0,
    }


@registry.register(
    name="multiplicative_uncertainty_test",
    description="Check robust stability condition max |W(jw) * T(jw)| < 1 for output multiplicative plant uncertainty over frequency samples.",
    parameters_schema={
        "type": "object",
        "properties": {
            "weights": {"type": "array", "items": {"type": "number"}, "description": "Multiplicative weight |W(jw)| samples"},
            "complementary_sensitivity": {"type": "array", "items": {"type": "number"}, "description": "Nominal |T(jw)| samples"},
        },
        "required": ["weights", "complementary_sensitivity"],
    },
)
def multiplicative_uncertainty_test(
    weights: list[float], complementary_sensitivity: list[float]
) -> dict[str, Any]:
    w_arr = np.array(weights, dtype=float)
    t_arr = np.array(complementary_sensitivity, dtype=float)
    products = w_arr * t_arr
    peak = float(np.max(products))
    is_satisfied = bool(peak < 1.0)
    return {
        "pointwise_products": products.tolist(),
        "peak_weighted_product": peak,
        "is_robustly_stable_on_grid": is_satisfied,
    }
