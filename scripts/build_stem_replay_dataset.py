#!/usr/bin/env python3
"""Base STEM and Mathematical Reasoning replay dataset generator.

Prevents catastrophic forgetting by mixing in step-by-step Chain of Thought (CoT)
linear algebra, calculus, ODE, Laplace transforms, and physical systems reasoning.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_data.schema import make_record


def n(val: float) -> str:
    if abs(val) < 1e-12:
        return "0.0"
    return f"{val:.6g}"


def mat_str(arr: np.ndarray) -> str:
    return repr(np.asarray(arr, dtype=float).tolist())


def generate_laplace_partial_fractions(count: int, seed: int) -> list[dict]:
    """Generate partial fraction expansion of rational transfer functions with step-by-step math."""
    records = []
    for i in range(count):
        p1 = -(1.0 + 0.5 * (i % 4))
        p2 = -(3.0 + 0.5 * (i % 5))
        z = 2.0 + 0.25 * (i % 3)
        # F(s) = (s + z) / ((s - p1)(s - p2)) = A / (s - p1) + B / (s - p2)
        # A = (p1 + z) / (p1 - p2)
        # B = (p2 + z) / (p2 - p1)
        A_res = (p1 + z) / (p1 - p2)
        B_res = (p2 + z) / (p2 - p1)

        prompt = (
            f"Compute the inverse Laplace transform $f(t) = \\mathcal{{L}}^{{-1}}\\{{F(s)\\}}$ for $t \\ge 0$, where "
            f"$$F(s) = \\frac{{s + {n(z)}}}{{(s + {n(-p1)})(s + {n(-p2)})}}$$. "
            f"Show the step-by-step partial fraction expansion and state the time-domain signal."
        )

        answer = (
            f"### 1. Partial Fraction Decomposition\n"
            f"Write $F(s)$ in terms of first-order partial fractions with distinct poles $s_1 = {n(p1)}$ and $s_2 = {n(p2)}$:\n"
            f"$$F(s) = \\frac{{s + {n(z)}}}{{(s - ({n(p1)}))(s - ({n(p2)}))}} = \\frac{{C_1}}{{s - ({n(p1)})}} + \\frac{{C_2}}{{s - ({n(p2)})}}$$\n\n"
            f"### 2. Residue Calculation\n"
            f"Applying the residue formula:\n"
            f"$$C_1 = \\lim_{{s \\to {n(p1)}}} (s - ({n(p1)})) F(s) = \\frac{{{n(p1)} + {n(z)}}}{{{n(p1)} - ({n(p2)})}} = \\frac{{{n(p1 + z)}}}{{{n(p1 - p2)}}} = {n(A_res)}$$\n"
            f"$$C_2 = \\lim_{{s \\to {n(p2)}}} (s - ({n(p2)})) F(s) = \\frac{{{n(p2)} + {n(z)}}}{{{n(p2)} - ({n(p1)})}} = \\frac{{{n(p2 + z)}}}{{{n(p2 - p1)}}} = {n(B_res)}$$\n\n"
            f"Thus, the expanded expression is:\n"
            f"$$F(s) = \\frac{{{n(A_res)}}}{{s + {n(-p1)}}} + \\frac{{{n(B_res)}}}{{s + {n(-p2)}}}$$\n\n"
            f"### 3. Inverse Laplace Transform\n"
            f"Using the standard pair $\\mathcal{{L}}^{{-1}}\\{{\\frac{{1}}{{s + a}}\\}} = e^{{-at}} u(t)$:\n"
            f"$$f(t) = ({n(A_res)} e^{{{n(p1)} t}} + {n(B_res)} e^{{{n(p2)} t}}) u(t)$$\n"
            f"**Conclusion:** The causal time-domain signal is $f(t) = {n(A_res)} e^{{{n(p1)} t}} + {n(B_res)} e^{{{n(p2)} t}}$ for $t \\ge 0$."
        )

        record = make_record(
            record_id=f"stem_laplace_partial_fraction_{i:04d}",
            domain="classical_optimal_control",
            family="stem_laplace_partial_fraction",
            task_type="derivation",
            difficulty="foundation",
            template_id=f"laplace_partial_fraction_prompt_{i % 4}",
            prompt=prompt,
            answer=answer,
            ground_truth={
                "kind": "laplace_inversion",
                "p1": p1,
                "p2": p2,
                "z": z,
                "C1": A_res,
                "C2": B_res,
            },
            source_refs=["stanford_ee263_course_reader", "astrom_murray_feedback_systems_1e"],
            verifier="verify_laplace_inversion",
            tool="requirements_analysis",
        )
        records.append(record)
    return records


def generate_rlc_circuit_cases(count: int, seed: int) -> list[dict]:
    """Generate series RLC circuit frequency response & damping ratio analysis."""
    records = []
    for i in range(count):
        R = 10.0 + 2.0 * (i % 5)  # Ohms
        L = 0.05 + 0.01 * (i % 4)  # Henry
        C = 0.001 + 0.0002 * (i % 3)  # Farad

        # Series RLC: L * ddot(q) + R * dot(q) + (1/C)*q = v(t)
        # Characteristic eq: s^2 + (R/L)s + 1/(LC) = s^2 + 2*zeta*wn*s + wn^2
        wn = 1.0 / math.sqrt(L * C)
        zeta = R / (2.0 * math.sqrt(L / C))
        wd = wn * math.sqrt(abs(1.0 - zeta**2)) if zeta < 1.0 else 0.0

        if zeta < 1.0:
            regime = "underdamped"
        elif abs(zeta - 1.0) < 1e-4:
            regime = "critically damped"
        else:
            regime = "overdamped"

        prompt = (
            f"Analyze a series RLC circuit with resistor $R = {n(R)}$ $\\Omega$, inductor $L = {n(L)}$ H, "
            f"and capacitor $C = {n(C)}$ F. "
            f"Find the natural frequency $\\omega_n$, damping ratio $\\zeta$, damped natural frequency $\\omega_d$, "
            f"and characterize the transient response regime."
        )

        answer = (
            f"### 1. Differential Equation and Standard Form\n"
            f"Applying Kirchhoff's Voltage Law (KVL) around the series loop with charge $q(t)$:\n"
            f"$$L \\frac{{d^2q}}{{dt^2}} + R \\frac{{dq}}{{dt}} + \\frac{{1}}{{C}} q = v(t)$$\n\n"
            f"Dividing by $L$ puts the equation into canonical second-order form $\\ddot{{q}} + 2\\zeta \\omega_n \\dot{{q}} + \\omega_n^2 q = \\frac{{1}}{{L}} v(t)$:\n"
            f"$$\\ddot{{q}} + \\frac{{{n(R)}}}{{{n(L)}}} \\dot{{q}} + \\frac{{1}}{{{n(L)} \\times {n(C)}}} q = \\frac{{1}}{{{n(L)}}} v(t)$$\n\n"
            f"### 2. Parameter Extraction\n"
            f"1. **Natural frequency $\\omega_n$**:\n"
            f"$$\\omega_n = \\frac{{1}}{{\\sqrt{{LC}}}} = \\frac{{1}}{{\\sqrt{{{n(L)} \\times {n(C)}}}}} = {n(wn)}\\text{{ rad/s}}$$\n\n"
            f"2. **Damping ratio $\\zeta$**:\n"
            f"$$2\\zeta \\omega_n = \\frac{{R}}{{L}} \\implies \\zeta = \\frac{{R}}{{2 \\sqrt{{L/C}}}} = \\frac{{{n(R)}}}{{2 \\sqrt{{{n(L)} / {n(C)}}}}} = {n(zeta)}$$\n\n"
            f"3. **Damped frequency $\\omega_d$**:\n"
            f"$$\\omega_d = \\omega_n \\sqrt{{|1 - \\zeta^2|}} = {n(wd)}\\text{{ rad/s}}$$\n\n"
            f"### 3. Conclusion\n"
            f"Since $\\zeta = {n(zeta)}$ ({'< 1' if zeta < 1.0 else '> 1'}), the system is **{regime}**."
        )

        record = make_record(
            record_id=f"stem_rlc_circuit_{i:04d}",
            domain="classical_optimal_control",
            family="stem_rlc_circuit_transient",
            task_type="numerical",
            difficulty="foundation",
            template_id=f"rlc_circuit_prompt_{i % 4}",
            prompt=prompt,
            answer=answer,
            ground_truth={
                "kind": "second_order_transient",
                "R": R,
                "L": L,
                "C": C,
                "wn": wn,
                "zeta": zeta,
                "wd": wd,
                "regime": regime,
            },
            source_refs=["stanford_ee263_course_reader", "astrom_murray_feedback_systems_1e"],
            verifier="verify_second_order_transient",
            tool="requirements_analysis",
        )
        records.append(record)
    return records


def build_all_stem_records(count_per_case: int = 50, seed: int = 20260902) -> list[dict]:
    """Compile all STEM replay records."""
    records = []
    records.extend(generate_laplace_partial_fractions(count_per_case, seed))
    records.extend(generate_rlc_circuit_cases(count_per_case, seed + 1000))
    return records


if __name__ == "__main__":
    cases = build_all_stem_records(40)
    print(f"Generated {len(cases)} STEM replay SFT records.")
