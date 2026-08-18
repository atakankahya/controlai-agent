"""Family-diverse, solver-backed classical/optimal SFT generators with Chain of Thought (CoT)."""

from __future__ import annotations

import math
from collections.abc import Callable

import cvxpy as cp
import numpy as np
from scipy import linalg

from controlai_data.schema import make_record

OPENERS = (
    "Provide a compact engineering calculation for the following case.",
    "Compute the requested quantities and state the decisive criterion.",
    "Work this classical/optimal control problem step-by-step with verified values.",
    "Derive the requested objects from the stated model; show intermediate equations.",
)
FBS = ["astrom_murray_feedback_systems_1e"]
OPTIMAL = ["boyd_lmi_system_control", "mit_ocw_16_323_optimal_control"]
ESTIMATION = ["stanford_ee263_course_reader"]
MPC = ["cvxpy_docs", "rawlings_mayne_diehl_mpc_2e"]


def n(value: float) -> str:
    if abs(value) < 5e-11:
        value = 0.0
    return f"{value:.6g}"


def mat(value: np.ndarray) -> str:
    return repr(np.asarray(value, dtype=float).tolist())


def roots(values: np.ndarray) -> list[list[float]]:
    values = np.asarray(values, dtype=complex)
    values = values[np.lexsort((values.imag, values.real))]
    return [[float(x.real), float(x.imag)] for x in values]


def first_order_step(rng: np.random.Generator, index: int) -> dict:
    del rng
    gain = 0.5 + 0.075 * index
    tau = 0.2 + 0.04 * index
    step = 0.5 + 0.25 * (index % 5)
    final = gain * step
    at_tau = final * (1 - math.exp(-1))
    settling_2 = -tau * math.log(0.02)
    prompt = (
        f"{OPENERS[index % 4]} A stable first-order plant is G(s) = {n(gain)}/"
        f"({n(tau)}s+1). For a step of amplitude {n(step)}, find the final value, "
        "the output at one time constant, and the exact 2% settling time."
    )
    answer = (
        f"### 1. Step Response Formulation\n"
        f"For plant $G(s) = \\frac{{{n(gain)}}}{{{n(tau)}s + 1}}$ with step input $u(t) = {n(step)}$, the output response is:\n"
        f"$$y(t) = K \\cdot U_0 (1 - e^{{-t/\\tau}}) = {n(gain)} \\times {n(step)} (1 - e^{{-t/{n(tau)}}}) = {n(final)} (1 - e^{{-t/{n(tau)}}})$$\n\n"
        f"### 2. Output Calculations\n"
        f"1. **Final steady-state value** ($t \\to \\infty$):\n"
        f"   $$y_{{ss}} = K U_0 = {n(final)}$$\n"
        f"2. **Output at one time constant** ($t = \\tau = {n(tau)}$ s):\n"
        f"   $$y(\\tau) = y_{{ss}} (1 - e^{{-1}}) = {n(final)} \\times {n(1 - math.exp(-1))} = {n(at_tau)}$$\n"
        f"3. **Exact 2% settling time** ($y(t_s) = 0.98 y_{{ss}} \\implies e^{{-t_s/\\tau}} = 0.02$):\n"
        f"   $$t_s = -\\tau \\ln(0.02) = -{n(tau)} \\times ({n(math.log(0.02))}) = {n(settling_2)}\\text{{ s}}$$\n\n"
        f"**Summary:** Final value = {n(final)}, y(tau) = {n(at_tau)}, t_s(2%) = {n(settling_2)} s."
    )
    return make_record(
        record_id=f"first_order_step_metrics_{index:05d}",
        domain="classical_control",
        family="first_order_exact_step_metrics",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"first_order_step_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "first_order_step",
            "gain": gain,
            "tau": tau,
            "step": step,
            "final": final,
            "at_tau": at_tau,
            "settling_2": settling_2,
        },
        source_refs=FBS,
        verifier="verify_first_order_step",
    )


def second_order_transient(rng: np.random.Generator, index: int) -> dict:
    del rng
    zeta = 0.12 + 0.012 * index
    wn = 0.8 + 0.09 * index
    wd = wn * math.sqrt(1 - zeta * zeta)
    overshoot = math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta)) * 100
    peak_time = math.pi / wd
    settling = -math.log(0.02) / (zeta * wn)
    poles = np.array([-zeta * wn + 1j * wd, -zeta * wn - 1j * wd])
    prompt = (
        f"{OPENERS[index % 4]} A standard second-order closed loop has zeta={n(zeta)} "
        f"and omega_n={n(wn)} rad/s. Compute its poles, damped frequency, percent "
        "overshoot, peak time, and exact 2% exponential-envelope time."
    )
    answer = (
        f"### 1. Second-Order System Parameters\n"
        f"Given $\\zeta = {n(zeta)}$ and $\\omega_n = {n(wn)}$ rad/s:\n"
        f"1. **Damped natural frequency**:\n"
        f"   $$\\omega_d = \\omega_n \\sqrt{{1 - \\zeta^2}} = {n(wn)} \\times \\sqrt{{1 - {n(zeta)}^2}} = {n(wd)}\\text{{ rad/s}}$$\n"
        f"2. **Closed-loop complex conjugate poles**:\n"
        f"   $$s_{{1,2}} = -\\zeta \\omega_n \\pm j \\omega_d = -{n(zeta * wn)} \\pm {n(wd)}j = {n(poles[0].real)} \\pm {n(abs(poles[0].imag))}j$$\n\n"
        f"### 2. Transient Step Specifications\n"
        f"1. **Percent overshoot ($M_p$)**:\n"
        f"   $$M_p = \\exp\\left(-\\frac{{\\pi \\zeta}}{{\\sqrt{{1 - \\zeta^2}}}}\\right) \\times 100\\% = {n(overshoot)}\\%$$\n"
        f"2. **Peak time ($t_p$)**:\n"
        f"   $$t_p = \\frac{{\\pi}}{{\\omega_d}} = \\frac{{\\pi}}{{{n(wd)}}} = {n(peak_time)}\\text{{ s}}$$\n"
        f"3. **Exact 2% settling time ($t_s$)**:\n"
        f"   $$t_s = -\\frac{{\\ln(0.02)}}{{\\zeta \\omega_n}} = \\frac{{{n(-math.log(0.02))}}}{{{n(zeta * wn)}}} = {n(settling)}\\text{{ s}}$$\n\n"
        f"**Summary:** Poles = {n(poles[0].real)} ± {n(abs(poles[0].imag))}j, omega_d = {n(wd)} rad/s, M_p = {n(overshoot)}%, t_p = {n(peak_time)} s, t_s(2%) = {n(settling)} s."
    )
    return make_record(
        record_id=f"second_order_transient_{index:05d}",
        domain="classical_control",
        family="underdamped_second_order_exact_metrics",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"second_order_transient_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "second_order_transient",
            "zeta": zeta,
            "omega_n": wn,
            "omega_d": wd,
            "overshoot_percent": overshoot,
            "peak_time": peak_time,
            "settling_2": settling,
            "poles": roots(poles),
        },
        source_refs=FBS,
        verifier="verify_second_order_transient",
    )


def closed_loop_tf(rng: np.random.Generator, index: int) -> dict:
    del rng
    a1 = 1.0 + 0.1 * index
    a0 = 2.0 + 0.15 * index
    b0 = 0.5 + 0.05 * index
    k = 0.4 + 0.08 * index
    den = [1.0, a1, a0 + k * b0]
    num = [k * b0]
    poles = np.roots(den)
    stable = bool(np.all(np.real(poles) < 0))
    prompt = (
        f"{OPENERS[index%4]} With unity negative feedback, P(s)={n(b0)}/(s^2+{n(a1)}s+{n(a0)}) and C(s)={n(k)}. "
        "Derive T(s), list its poles, and classify stability."
    )
    answer = (
        f"### 1. Closed-Loop Transfer Function Derivation\n"
        f"Under unity feedback, the complementary sensitivity / closed-loop transfer function is:\n"
        f"$$T(s) = \\frac{{C(s)P(s)}}{{1 + C(s)P(s)}} = \\frac{{{n(k)} \\times \\frac{{{n(b0)}}}{{s^2 + {n(a1)}s + {n(a0)}}}}}{{1 + {n(k)} \\times \\frac{{{n(b0)}}}{{s^2 + {n(a1)}s + {n(a0)}}}}}$$\n"
        f"$$T(s) = \\frac{{{n(num[0])}}}{{s^2 + {n(a1)}s + ({n(a0)} + {n(k * b0)})}} = \\frac{{{n(num[0])}}}{{s^2 + {n(a1)}s + {n(den[2])}}}$$\n\n"
        f"### 2. Pole Locations and Stability\n"
        f"Solving the characteristic equation $s^2 + {n(a1)}s + {n(den[2])} = 0$:\n"
        f"$$\\text{{Poles}} = \\{{{', '.join(f'{x.real:.6g} + {x.imag:.6g}j' if abs(x.imag) > 1e-9 else f'{x.real:.6g}' for x in poles)}\\}}$$\n\n"
        f"### 3. Conclusion\n"
        f"Since all poles have strictly negative real parts, the closed-loop system is **{'asymptotically stable' if stable else 'not asymptotically stable'}**."
    )
    return make_record(
        record_id=f"closed_loop_tf_{index:05d}",
        domain="classical_control",
        family="unity_feedback_closed_loop_polynomial",
        task_type="derivation",
        difficulty="foundation",
        template_id=f"closed_loop_tf_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "closed_loop_tf",
            "plant_num": [b0],
            "plant_den": [1.0, a1, a0],
            "gain": k,
            "closed_num": num,
            "closed_den": den,
            "poles": roots(poles),
            "stable": stable,
        },
        source_refs=FBS,
        verifier="verify_closed_loop_tf",
    )


def sensitivity_complex(rng: np.random.Generator, index: int) -> dict:
    del rng
    real = -0.7 + 0.06 * index
    imag = -1.2 + 0.05 * index
    L = complex(real, imag)
    S = 1 / (1 + L)
    T = L / (1 + L)
    prompt = (
        f"{OPENERS[index%4]} At one frequency the loop value is L(jw)={n(real)} {'+' if imag>=0 else '-'} {n(abs(imag))}j. "
        "Compute S=1/(1+L), T=L/(1+L), their magnitudes, and verify S+T=1."
    )
    answer = (
        f"### 1. Sensitivity Function $S(j\\omega) = \\frac{{1}}{{1 + L(j\\omega)}}$\n"
        f"$$1 + L = 1 + ({n(real)} + {n(imag)}j) = {n(1 + real)} + {n(imag)}j$$\n"
        f"$$S = \\frac{{1}}{{{n(1 + real)} + {n(imag)}j}} = {n(S.real)} {'+' if S.imag>=0 else '-'} {n(abs(S.imag))}j, \\quad |S| = {n(abs(S))}$$\n\n"
        f"### 2. Complementary Sensitivity $T(j\\omega) = \\frac{{L(j\\omega)}}{{1 + L(j\\omega)}}$\n"
        f"$$T = \\frac{{{n(real)} + {n(imag)}j}}{{{n(1 + real)} + {n(imag)}j}} = {n(T.real)} {'+' if T.imag>=0 else '-'} {n(abs(T.imag))}j, \\quad |T| = {n(abs(T))}$$\n\n"
        f"### 3. Identity Verification\n"
        f"$$S + T = ({n(S.real)} + {n(T.real)}) + ({n(S.imag)} + {n(T.imag)})j = 1.0 + 0.0j$$\n"
        f"**Summary:** S = {n(S.real)} {'+' if S.imag>=0 else '-'} {n(abs(S.imag))}j (|S|={n(abs(S))}), T = {n(T.real)} {'+' if T.imag>=0 else '-'} {n(abs(T.imag))}j (|T|={n(abs(T))}), S + T = 1."
    )
    return make_record(
        record_id=f"sensitivity_complex_{index:05d}",
        domain="robust_control",
        family="complex_sensitivity_complementary_sensitivity",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"sensitivity_complex_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "sensitivity_complex",
            "L": [real, imag],
            "S": [S.real, S.imag],
            "T": [T.real, T.imag],
            "S_mag": abs(S),
            "T_mag": abs(T),
        },
        source_refs=FBS,
        verifier="verify_sensitivity_complex",
    )


def lead_parameters(rng: np.random.Generator, index: int) -> dict:
    del rng
    phi_deg = 12.0 + 1.2 * index
    wm = 0.6 + 0.08 * index
    phi = math.radians(phi_deg)
    alpha = (1 - math.sin(phi)) / (1 + math.sin(phi))
    tau = 1 / (wm * math.sqrt(alpha))
    zero = -1 / tau
    pole = -1 / (alpha * tau)
    prompt = (
        f"{OPENERS[index%4]} Design the pole/zero locations of C(s)=(tau s+1)/(alpha tau s+1) for maximum phase lead "
        f"{n(phi_deg)} deg at omega_m={n(wm)} rad/s. Report alpha, tau, zero, and pole."
    )
    answer = (
        f"### 1. Lead Compensator Design Equations\n"
        f"For target maximum phase lead $\\phi_m = {n(phi_deg)}^\\circ$ at geometric mean frequency $\\omega_m = {n(wm)}$ rad/s:\n"
        f"1. **Alpha parameter**:\n"
        f"   $$\\alpha = \\frac{{1 - \\sin(\\phi_m)}}{{1 + \\sin(\\phi_m)}} = \\frac{{1 - \\sin({n(phi_deg)}^\\circ)}}{{1 + \\sin({n(phi_deg)}^\\circ)}} = {n(alpha)}$$\n"
        f"2. **Time constant $\\tau$**:\n"
        f"   $$\\tau = \\frac{{1}}{{\\omega_m \\sqrt{{\\alpha}}}} = \\frac{{1}}{{{n(wm)} \\times \\sqrt{{{n(alpha)}}}}} = {n(tau)}\\text{{ s}}$$\n\n"
        f"### 2. Pole and Zero Placement\n"
        f"- Compensator zero: $z = -1/\\tau = {n(zero)}$ rad/s\n"
        f"- Compensator pole: $p = -1/(\\alpha \\tau) = {n(pole)}$ rad/s\n\n"
        f"**Summary:** alpha = {n(alpha)}, tau = {n(tau)} s, zero = {n(zero)} rad/s, pole = {n(pole)} rad/s."
    )
    return make_record(
        record_id=f"lead_parameters_{index:05d}",
        domain="classical_control",
        family="lead_compensator_maximum_phase_parameters",
        task_type="design",
        difficulty="intermediate",
        template_id=f"lead_parameters_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "lead_parameters",
            "phi_deg": phi_deg,
            "omega_m": wm,
            "alpha": alpha,
            "tau": tau,
            "zero": zero,
            "pole": pole,
        },
        source_refs=FBS,
        verifier="verify_lead_parameters",
    )


def pid_characteristic(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.8 + 0.06 * index
    kp = 0.5 + 0.07 * index
    ki = 0.2 + 0.025 * index
    kd = 0.1 + 0.02 * index
    coeff = [1.0, a + kd, kp, ki]
    poles = np.roots(coeff)
    stable = bool(np.all(np.real(poles) < 0))
    prompt = (
        f"{OPENERS[index%4]} For P(s)=1/[s(s+{n(a)})] with unity negative feedback and C(s)=Kp+Ki/s+Kd s, "
        f"use Kp={n(kp)}, Ki={n(ki)}, Kd={n(kd)}. Derive the characteristic polynomial, compute poles, and assess stability."
    )
    answer = (
        f"### 1. Closed-Loop Characteristic Equation\n"
        f"The characteristic equation is $1 + C(s)P(s) = 0$:\n"
        f"$$1 + \\left(K_p + \\frac{{K_i}}{{s}} + K_d s\\right) \\frac{{1}}{{s(s + {n(a)})}} = 0$$\n"
        f"$$s^2(s + {n(a)}) + (K_d s^2 + K_p s + K_i) = 0$$\n"
        f"$$s^3 + ({n(a)} + {n(kd)}) s^2 + {n(kp)} s + {n(ki)} = s^3 + {n(coeff[1])} s^2 + {n(coeff[2])} s + {n(coeff[3])} = 0$$\n\n"
        f"### 2. Roots (Closed-Loop Poles)\n"
        f"Solving the cubic polynomial:\n"
        f"$$\\text{{Poles}} = \\{{{', '.join(f'{x.real:.6g}{x.imag:+.6g}j' for x in poles)}\\}}$$\n\n"
        f"### 3. Stability Assessment\n"
        f"Since all closed-loop poles have strictly negative real parts, the closed loop is **{'asymptotically stable' if stable else 'not asymptotically stable'}**."
    )
    return make_record(
        record_id=f"pid_characteristic_{index:05d}",
        domain="classical_control",
        family="pid_closed_loop_characteristic_polynomial",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"pid_characteristic_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "polynomial_stability",
            "coefficients": coeff,
            "poles": roots(poles),
            "stable": stable,
        },
        source_refs=FBS,
        verifier="verify_polynomial_stability",
    )


def routh_quartic(rng: np.random.Generator, index: int) -> dict:
    del rng
    a3 = 1.0 + 0.09 * index
    a2 = 1.5 + 0.07 * (index % 9)
    a1 = 0.3 + 0.08 * (index % 11)
    a0 = 0.2 + 0.03 * index
    b1 = (a3 * a2 - a1) / a3
    c1 = (b1 * a1 - a3 * a0) / b1
    first = [1.0, a3, b1, c1, a0]
    poles = np.roots([1, a3, a2, a1, a0])
    stable = bool(np.all(np.real(poles) < 0))
    routh_stable = all(x > 0 for x in first)
    assert stable == routh_stable
    prompt = (
        f"{OPENERS[index%4]} Apply Routh-Hurwitz to p(s)=s^4+{n(a3)}s^3+{n(a2)}s^2+{n(a1)}s+{n(a0)}. "
        "Report the full first column and the stability conclusion."
    )
    answer = (
        f"### 1. Routh Array Construction\n"
        f"For polynomial $p(s) = s^4 + {n(a3)} s^3 + {n(a2)} s^2 + {n(a1)} s + {n(a0)}$:\n"
        f"- Row $s^4$: $[1, {n(a2)}, {n(a0)}]$\n"
        f"- Row $s^3$: $[{n(a3)}, {n(a1)}, 0]$\n"
        f"- Row $s^2$: $b_1 = \\frac{{{n(a3)} \\times {n(a2)} - 1 \\times {n(a1)}}}{{{n(a3)}}} = {n(b1)}$, $b_2 = {n(a0)}$\n"
        f"- Row $s^1$: $c_1 = \\frac{{{n(b1)} \\times {n(a1)} - {n(a3)} \\times {n(a0)}}}{{{n(b1)}}} = {n(c1)}$\n"
        f"- Row $s^0$: $d_1 = {n(a0)}$\n\n"
        f"### 2. First Column Analysis\n"
        f"The Routh first column is **$[1, {n(a3)}, {n(b1)}, {n(c1)}, {n(a0)}]$**.\n\n"
        f"### 3. Conclusion\n"
        f"The first column entries are **{'strictly positive, so the polynomial is Hurwitz' if stable else 'not strictly positive, so the polynomial is not Hurwitz'}**."
    )
    return make_record(
        record_id=f"routh_quartic_{index:05d}",
        domain="classical_control",
        family="routh_quartic_numeric_classification",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"routh_quartic_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "routh_quartic",
            "coefficients": [1, a3, a2, a1, a0],
            "first_column": first,
            "poles": roots(poles),
            "stable": stable,
        },
        source_refs=FBS,
        verifier="verify_routh_quartic",
    )


def root_locus_point(rng: np.random.Generator, index: int) -> dict:
    del rng
    poles = sorted([-1.0 - 0.05 * index, -3.0 - 0.08 * index, -6.0 - 0.03 * index])
    zeros = [-2.0 - 0.04 * index]
    point = -0.4 - 0.11 * index
    count = sum(x > point for x in poles + zeros)
    on_locus = count % 2 == 1
    prompt = (
        f"{OPENERS[index%4]} A positive-gain root locus has real open-loop poles {poles} and real zeros {zeros}. "
        f"Is the real-axis point s={n(point)} on the locus? Apply the odd-count rule."
    )
    answer = (
        f"### 1. Real-Axis Root Locus Criterion (Angle Condition)\n"
        f"A point $s_0$ on the real axis belongs to the $180^\\circ$ (positive gain) root locus if and only if the total number of real open-loop poles and zeros to its right is **odd**.\n\n"
        f"### 2. Point Evaluation at $s = {n(point)}$\n"
        f"- Open-loop poles: {poles}\n"
        f"- Open-loop zeros: {zeros}\n"
        f"- Poles and zeros to the right of $s = {n(point)}$: {count}\n\n"
        f"### 3. Conclusion\n"
        f"There are {count} real open-loop poles and zeros to the right of {n(point)}. Because this count is **{'odd' if on_locus else 'even'}**, the point **{'is' if on_locus else 'is not'}** on the positive-gain root locus."
    )
    return make_record(
        record_id=f"root_locus_point_{index:05d}",
        domain="classical_control",
        family="root_locus_real_axis_membership",
        task_type="concept",
        difficulty="foundation",
        template_id=f"root_locus_point_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "root_locus_real_axis",
            "poles": poles,
            "zeros": zeros,
            "point": point,
            "right_count": count,
            "on_locus": on_locus,
        },
        source_refs=FBS,
        verifier="verify_root_locus_real_axis",
    )


def margins_from_crossovers(rng: np.random.Generator, index: int) -> dict:
    del rng
    mag_pc = 0.15 + 0.012 * index
    phase_gc = -110.0 - 1.1 * index
    gm = 1 / mag_pc
    gm_db = 20 * math.log10(gm)
    pm = 180 + phase_gc
    prompt = (
        f"{OPENERS[index%4]} At the phase-crossover frequency, |L|={n(mag_pc)}. "
        f"At the gain-crossover frequency, phase(L)={n(phase_gc)} deg. "
        "Compute gain margin as a ratio and dB, plus phase margin."
    )
    answer = (
        f"### 1. Stability Margin Formulas\n"
        f"1. **Gain Margin (GM)**: Defined at phase-crossover frequency $\\omega_{{pc}}$ where $\\angle L(j\\omega_{{pc}}) = -180^\\circ$:\n"
        f"   $$GM = \\frac{{1}}{{|L(j\\omega_{{pc}})|}} = \\frac{{1}}{{{n(mag_pc)}}} = {n(gm)}$$\n"
        f"   $$GM_{{dB}} = 20 \\log_{{10}}(GM) = 20 \\log_{{10}}({n(gm)}) = {n(gm_db)}\\text{{ dB}}$$\n\n"
        f"2. **Phase Margin (PM)**: Defined at gain-crossover frequency $\\omega_{{gc}}$ where $|L(j\\omega_{{gc}})| = 1$:\n"
        f"   $$PM = 180^\\circ + \\angle L(j\\omega_{{gc}}) = 180^\\circ + ({n(phase_gc)}^\\circ) = {n(pm)}^\\circ$$\n\n"
        f"**Summary:** GM = {n(gm)} ({n(gm_db)} dB), PM = {n(pm)} deg."
    )
    return make_record(
        record_id=f"margins_crossovers_{index:05d}",
        domain="classical_control",
        family="classical_margins_from_crossover_values",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"margins_crossovers_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "crossover_margins",
            "magnitude_at_phase_crossover": mag_pc,
            "phase_at_gain_crossover_deg": phase_gc,
            "gain_margin": gm,
            "gain_margin_db": gm_db,
            "phase_margin_deg": pm,
        },
        source_refs=FBS,
        verifier="verify_crossover_margins",
    )


def scalar_kf_predict(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.7 + 0.01 * index
    b = 0.4 + 0.015 * index
    x = 0.5 + 0.08 * index
    u = -0.3 + 0.025 * index
    P = 0.6 + 0.04 * index
    Q = 0.05 + 0.005 * index
    xm = a * x + b * u
    Pm = a * a * P + Q
    prompt = (
        f"{OPENERS[index%4]} Perform only the scalar Kalman time update for x[k+1]={n(a)}x[k]+{n(b)}u[k]+w[k]. "
        f"Given x_hat+={n(x)}, u={n(u)}, P+={n(P)}, Q={n(Q)}, compute x_hat- and P-."
    )
    answer = (
        f"### 1. Kalman State Prediction (Time Update)\n"
        f"$$\\hat{{x}}^-_k = a \\hat{{x}}^+_{{k-1}} + b u_{{k-1}} = ({n(a)})({n(x)}) + ({n(b)})({n(u)}) = {n(a * x)} + ({n(b * u)}) = {n(xm)}$$\n\n"
        f"### 2. Error Covariance Prediction\n"
        f"$$P^-_k = a^2 P^+_{{k-1}} + Q = ({n(a)})^2 ({n(P)}) + {n(Q)} = ({n(a*a)})({n(P)}) + {n(Q)} = {n(Pm)}$$\n\n"
        f"**Summary:** Predicted state x_hat- = {n(xm)}, predicted covariance P- = {n(Pm)}."
    )
    return make_record(
        record_id=f"scalar_kf_predict_{index:05d}",
        domain="estimation_filtering",
        family="scalar_kalman_time_update",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"scalar_kf_predict_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "scalar_kf_predict",
            "a": a,
            "b": b,
            "x": x,
            "u": u,
            "P": P,
            "Q": Q,
            "x_minus": xm,
            "P_minus": Pm,
        },
        source_refs=ESTIMATION,
        verifier="verify_scalar_kf_predict",
    )


def matrix_kf_update(rng: np.random.Generator, index: int) -> dict:
    del rng
    x = np.array([0.2 + 0.06 * index, -0.4 + 0.03 * index])
    base = np.array([[1.0 + 0.03 * index, 0.15], [0.15, 0.7 + 0.02 * index]])
    P = base @ base.T
    H = np.array([[1.0, 0.0 if index % 3 == 0 else 0.5]])
    R = 0.2 + 0.01 * index
    z = 0.5 + 0.05 * index
    innovation = float(z - (H @ x)[0])
    S = float((H @ P @ H.T)[0, 0] + R)
    K = P @ H.T / S
    xp = x + K[:, 0] * innovation
    I = np.eye(2)
    Pp = (I - K @ H) @ P @ (I - K @ H).T + K * R @ K.T
    prompt = (
        f"{OPENERS[index%4]} Perform a 2-state Kalman measurement update with x-={mat(x)}, P-={mat(P)}, "
        f"H={mat(H)}, R={n(R)}, z={n(z)}. Report innovation, S, K, x+, and Joseph-form P+."
    )
    answer = (
        f"### 1. Innovation and Innovation Covariance\n"
        f"- Innovation: $\\tilde{{y}} = z - H \\hat{{x}}^- = {n(z)} - {n(float((H @ x)[0]))} = {n(innovation)}$\n"
        f"- Innovation covariance: $S = H P^- H^T + R = {n(float((H @ P @ H.T)[0, 0]))} + {n(R)} = {n(S)}$\n\n"
        f"### 2. Kalman Gain and State Update\n"
        f"- Kalman Gain: $K = P^- H^T S^{{-1}} = {mat(K)}$\n"
        f"- Updated State: $\\hat{{x}}^+ = \\hat{{x}}^- + K \\tilde{{y}} = {mat(xp)}$\n\n"
        f"### 3. Joseph-Form Covariance Update\n"
        f"$$P^+ = (I - KH) P^- (I - KH)^T + K R K^T = {mat(Pp)}$$\n\n"
        f"**Summary:** Innovation = {n(innovation)}, S = {n(S)}, K = {mat(K)}, x+ = {mat(xp)}, P+ = {mat(Pp)}."
    )
    return make_record(
        record_id=f"matrix_kf_update_{index:05d}",
        domain="estimation_filtering",
        family="matrix_kalman_measurement_update_2state",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"matrix_kf_update_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "matrix_kf_update",
            "x_minus": x.tolist(),
            "P_minus": P.tolist(),
            "H": H.tolist(),
            "R": [[R]],
            "z": [z],
            "innovation": [innovation],
            "S": [[S]],
            "K": K.tolist(),
            "x_plus": xp.tolist(),
            "P_plus": Pp.tolist(),
        },
        source_refs=ESTIMATION,
        verifier="verify_matrix_kf_update",
    )


def covariance_propagation(rng: np.random.Generator, index: int) -> dict:
    del rng
    A = np.array([[1.0, 0.05 + 0.002 * index], [0.0, 0.8 + 0.003 * index]])
    P = np.array([[1.0 + 0.02 * index, 0.1], [0.1, 0.5 + 0.01 * index]])
    Q = np.diag([0.02 + 0.001 * index, 0.04 + 0.0015 * index])
    Pn = A @ P @ A.T + Q
    prompt = (
        f"{OPENERS[index%4]} Propagate covariance one step using P_next=A P A^T+Q for A={mat(A)}, P={mat(P)}, "
        f"Q={mat(Q)}. Report P_next and check symmetry."
    )
    answer = (
        f"### 1. Covariance Time Propagation Formula\n"
        f"$$P_{{k+1}} = A P_k A^T + Q$$\n\n"
        f"### 2. Matrix Multiplication and Addition\n"
        f"- $A P A^T = {mat(A @ P @ A.T)}$\n"
        f"- $P_{{next}} = A P A^T + Q = {mat(Pn)}$\n\n"
        f"### 3. Symmetry Verification\n"
        f"$$\\|P_{{next}} - P_{{next}}^T\\|_{{\\max}} = {n(float(np.max(np.abs(Pn - Pn.T))))} \\approx 0$$\n"
        f"**Summary:** P_next = {mat(Pn)}."
    )
    return make_record(
        record_id=f"covariance_propagation_{index:05d}",
        domain="estimation_filtering",
        family="linear_covariance_time_propagation",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"covariance_propagation_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "covariance_propagation",
            "A": A.tolist(),
            "P": P.tolist(),
            "Q": Q.tolist(),
            "P_next": Pn.tolist(),
        },
        source_refs=ESTIMATION,
        verifier="verify_covariance_propagation",
    )


def continuous_lqr_matrix(rng: np.random.Generator, index: int) -> dict:
    del rng
    a0 = 0.8 + 0.04 * index
    a1 = 0.3 + 0.025 * index
    A = np.array([[0, 1], [-a0, -a1]], float)
    B = np.array([[0], [1]], float)
    Q = np.diag([1 + 0.03 * index, 0.5 + 0.02 * index])
    R = np.array([[0.5 + 0.015 * index]])
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    poles = np.linalg.eigvals(A - B @ K)
    prompt = (
        f"{OPENERS[index%4]} Solve the continuous-time LQR for A={mat(A)}, B={mat(B)}, Q={mat(Q)}, R={mat(R)}. "
        "Report the stabilizing P, K=R^-1 B^T P, and closed-loop poles."
    )
    answer = (
        f"### 1. Continuous Algebraic Riccati Equation (CARE)\n"
        f"$$A^T P + P A - P B R^{{-1}} B^T P + Q = 0$$\n"
        f"Solving CARE yields the unique positive definite stabilizing matrix:\n"
        f"$$P = {mat(P)}$$\n\n"
        f"### 2. State-Feedback Gain Synthesis\n"
        f"$$K = R^{{-1}} B^T P = {mat(K)}$$\n\n"
        f"### 3. Closed-Loop Poles\n"
        f"Eigenvalues of $A - BK$:\n"
        f"$$\\lambda(A - BK) = \\{{{', '.join(f'{x.real:.6g}{x.imag:+.6g}j' for x in poles)}\\}}$$\n"
        f"All closed-loop poles have strictly negative real parts, guaranteeing asymptotic stability."
    )
    return make_record(
        record_id=f"continuous_lqr_matrix_{index:05d}",
        domain="optimal_control",
        family="continuous_matrix_lqr_care",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"continuous_lqr_matrix_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "continuous_lqr",
            "A": A.tolist(),
            "B": B.tolist(),
            "Q": Q.tolist(),
            "R": R.tolist(),
            "P": P.tolist(),
            "K": K.tolist(),
            "closed_poles": roots(poles),
        },
        source_refs=OPTIMAL,
        verifier="verify_continuous_lqr",
    )


def discrete_lqr_matrix(rng: np.random.Generator, index: int) -> dict:
    del rng
    T = 0.04 + 0.002 * index
    A = np.array([[1, T], [0, 1]], float)
    B = np.array([[0.5 * T * T], [T]], float)
    Q = np.diag([1 + 0.02 * index, 0.2 + 0.01 * index])
    R = np.array([[0.1 + 0.006 * index]])
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    poles = np.linalg.eigvals(A - B @ K)
    prompt = (
        f"{OPENERS[index%4]} Solve the infinite-horizon discrete LQR for A={mat(A)}, B={mat(B)}, Q={mat(Q)}, R={mat(R)}. "
        "Give P, K, and closed-loop poles under u=-Kx."
    )
    answer = (
        f"### 1. Discrete Algebraic Riccati Equation (DARE)\n"
        f"$$P = A^T P A - (A^T P B)(R + B^T P B)^{{-1}}(B^T P A) + Q$$\n"
        f"The stabilizing positive definite solution is:\n"
        f"$$P = {mat(P)}$$\n\n"
        f"### 2. Discrete LQR Gain\n"
        f"$$K = (R + B^T P B)^{{-1}} B^T P A = {mat(K)}$$\n\n"
        f"### 3. Closed-Loop Discrete Poles\n"
        f"$$\\lambda(A - BK) = \\{{{', '.join(f'{x.real:.6g}{x.imag:+.6g}j' for x in poles)}\\}}$$\n"
        f"All poles satisfy $|\\lambda_i| < 1$, strictly inside the unit circle."
    )
    return make_record(
        record_id=f"discrete_lqr_matrix_{index:05d}",
        domain="optimal_control",
        family="discrete_matrix_lqr_dare",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"discrete_lqr_matrix_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "discrete_lqr",
            "A": A.tolist(),
            "B": B.tolist(),
            "Q": Q.tolist(),
            "R": R.tolist(),
            "P": P.tolist(),
            "K": K.tolist(),
            "closed_poles": roots(poles),
        },
        source_refs=OPTIMAL,
        verifier="verify_discrete_lqr",
    )


def finite_horizon_lqr(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.8 + 0.005 * index
    b = 0.6 + 0.01 * (index % 7)
    q = 1.0 + 0.03 * index
    r = 0.4 + 0.02 * (index % 5)
    qf = 2.0 + 0.04 * index
    horizon = 3
    P = [0.0] * (horizon + 1)
    K = [0.0] * horizon
    P[horizon] = qf
    for k in range(horizon - 1, -1, -1):
        den = r + b * b * P[k + 1]
        K[k] = b * P[k + 1] * a / den
        P[k] = q + a * a * P[k + 1] - (a * b * P[k + 1]) ** 2 / den
    prompt = (
        f"{OPENERS[index%4]} For x[k+1]={n(a)}x[k]+{n(b)}u[k], minimize sum_0^2(q x_k^2+r u_k^2)+q_f x_3^2 with "
        f"q={n(q)}, r={n(r)}, q_f={n(qf)}. Run the scalar Riccati recursion and report K0..K2 and P0..P3."
    )
    answer = (
        f"### 1. Dynamic Programming Backward Riccati Recursion\n"
        f"Terminal condition: $P_3 = q_f = {n(qf)}$.\n"
        f"Backward recursion: $K_k = \\frac{{b P_{{k+1}} a}}{{r + b^2 P_{{k+1}}}}$, $P_k = q + a^2 P_{{k+1}} - K_k (r + b^2 P_{{k+1}}) K_k$.\n\n"
        f"### 2. Step-by-Step Values\n"
        f"- At $k=2$: $K_2 = {n(K[2])}$, $P_2 = {n(P[2])}$\n"
        f"- At $k=1$: $K_1 = {n(K[1])}$, $P_1 = {n(P[1])}$\n"
        f"- At $k=0$: $K_0 = {n(K[0])}$, $P_0 = {n(P[0])}$\n\n"
        f"**Summary:** Gains K = {list(map(float, K))}, Riccati sequence P = {list(map(float, P))}."
    )
    return make_record(
        record_id=f"finite_horizon_lqr_{index:05d}",
        domain="optimal_control",
        family="finite_horizon_scalar_lqr_recursion",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"finite_horizon_lqr_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "finite_horizon_lqr",
            "a": a,
            "b": b,
            "q": q,
            "r": r,
            "qf": qf,
            "horizon": horizon,
            "K": K,
            "P": P,
        },
        source_refs=OPTIMAL,
        verifier="verify_finite_horizon_lqr",
    )


def mpc_prediction_matrices(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.7 + 0.006 * index
    b = 0.4 + 0.009 * index
    horizon = 3
    F = np.array([[a**i] for i in range(1, horizon + 1)])
    G = np.zeros((horizon, horizon))
    for row in range(horizon):
        for col in range(row + 1):
            G[row, col] = a ** (row - col) * b
    prompt = (
        f"{OPENERS[index%4]} For x[k+1]={n(a)}x[k]+{n(b)}u[k] and horizon N=3, form X=F x0+G U where "
        "X=[x1,x2,x3]^T and U=[u0,u1,u2]^T."
    )
    answer = (
        f"### 1. State Prediction Recursion over Horizon N=3\n"
        f"- $x_1 = a x_0 + b u_0$\n"
        f"- $x_2 = a x_1 + b u_1 = a^2 x_0 + a b u_0 + b u_1$\n"
        f"- $x_3 = a x_2 + b u_2 = a^3 x_0 + a^2 b u_0 + a b u_1 + b u_2$\n\n"
        f"### 2. Matrix Stacking $X = F x_0 + G U$\n"
        f"$$F = \\begin{{bmatrix}} a \\\\ a^2 \\\\ a^3 \\end{{bmatrix}} = {mat(F)}$$\n"
        f"$$G = \\begin{{bmatrix}} b & 0 & 0 \\\\ ab & b & 0 \\\\ a^2b & ab & b \\end{{bmatrix}} = {mat(G)}$$\n\n"
        f"**Summary:** F = {mat(F)}, G = {mat(G)}."
    )
    return make_record(
        record_id=f"mpc_prediction_matrices_{index:05d}",
        domain="mpc",
        family="scalar_mpc_prediction_matrices_horizon3",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"mpc_prediction_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "mpc_prediction",
            "a": a,
            "b": b,
            "horizon": horizon,
            "F": F.tolist(),
            "G": G.tolist(),
        },
        source_refs=MPC,
        verifier="verify_mpc_prediction",
    )


def mpc_unconstrained(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.8 + 0.004 * index
    b = 0.5 + 0.008 * index
    x0 = -2.0 + 0.11 * index
    r = 0.15 + 0.01 * (index % 9)
    F = np.array([[a], [a * a]])
    G = np.array([[b, 0], [a * b, b]])
    H = G.T @ G + r * np.eye(2)
    g = G.T @ (F[:, 0] * x0)
    U = -np.linalg.solve(H, g)
    X = F[:, 0] * x0 + G @ U
    cost = float(X @ X + r * (U @ U))
    prompt = (
        f"{OPENERS[index%4]} Solve the unconstrained horizon-2 MPC problem for x[k+1]={n(a)}x[k]+{n(b)}u[k], "
        f"x0={n(x0)}, J=x1^2+x2^2+{n(r)}(u0^2+u1^2). Report U*, X*, and J*."
    )
    answer = (
        f"### 1. Quadratic Objective in Vector Form\n"
        f"With $X = F x_0 + G U$, the cost is $J = \\|X\\|^2 + r \\|U\\|^2 = U^T (G^T G + r I) U + 2 (G^T F x_0)^T U + x_0^T F^T F x_0$.\n\n"
        f"### 2. Unconstrained Analytic Minimization\n"
        f"Setting gradient to zero: $(G^T G + r I) U^* = -G^T F x_0$\n"
        f"- Hessian $H = G^T G + r I = {mat(H)}$\n"
        f"- Gradient vector $g = G^T F x_0 = {mat(g)}$\n"
        f"- Optimal input vector: $U^* = -H^{{-1}} g = {mat(U)}$\n"
        f"- Optimal predicted trajectory: $X^* = F x_0 + G U^* = {mat(X)}$\n"
        f"- Optimal cost: $J^* = {n(cost)}$\n\n"
        f"**Summary:** U* = {mat(U)}, X* = {mat(X)}, J* = {n(cost)}."
    )
    return make_record(
        record_id=f"mpc_unconstrained_{index:05d}",
        domain="mpc",
        family="unconstrained_scalar_mpc_horizon2",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"mpc_unconstrained_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "mpc_unconstrained",
            "a": a,
            "b": b,
            "x0": x0,
            "r": r,
            "F": F.tolist(),
            "G": G.tolist(),
            "U": U.tolist(),
            "X": X.tolist(),
            "cost": cost,
        },
        source_refs=MPC,
        verifier="verify_mpc_unconstrained",
    )


def mpc_box_qp(rng: np.random.Generator, index: int) -> dict:
    del rng
    x0 = -2.5 + 0.13 * index
    r = 0.08 + 0.004 * index
    limit = 0.35 + 0.015 * (index % 10)
    u = cp.Variable(2)
    x1 = x0 + u[0]
    x2 = x1 + u[1]
    objective = cp.Minimize(cp.square(x1) + cp.square(x2) + r * cp.sum_squares(u))
    constraints = [u >= -limit, u <= limit]
    problem = cp.Problem(objective, constraints)
    value = problem.solve(solver="CLARABEL")
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(problem.status)
    U = np.asarray(u.value)
    X = np.array([x0 + U[0], x0 + U[0] + U[1]])
    active = [abs(abs(float(v)) - limit) < 1e-6 for v in U]
    prompt = (
        f"{OPENERS[index%4]} Solve the horizon-2 MPC QP x1=x0+u0, x2=x1+u1, x0={n(x0)}, minimize x1^2+x2^2+{n(r)}(u0^2+u1^2), "
        f"subject to |u0|,|u1| <= {n(limit)}. Report U*, X*, active bounds, and cost."
    )
    answer = (
        f"### 1. Constrained Quadratic Program (QP) Formulation\n"
        f"$$\\min_{{u_0, u_1}} (x_0 + u_0)^2 + (x_0 + u_0 + u_1)^2 + {n(r)}(u_0^2 + u_1^2) \\quad \\text{{s.t.}} \\quad -{n(limit)} \\le u_0, u_1 \\le {n(limit)}$$\n\n"
        f"### 2. Numerical QP Solution\n"
        f"- Optimal control trajectory: $U^* = {mat(U)}$\n"
        f"- Optimal state trajectory: $X^* = {mat(X)}$\n"
        f"- Active constraints flag ($|u_k| = {n(limit)}$): {active}\n"
        f"- Optimal cost: $J^* = {n(float(value))}$\n\n"
        f"**Summary:** The convex QP gives U* = {mat(U)}, X* = {mat(X)}, active-input flags = {active}, and J* = {n(float(value))}."
    )
    return make_record(
        record_id=f"mpc_box_qp_{index:05d}",
        domain="mpc",
        family="box_constrained_scalar_mpc_horizon2",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"mpc_box_qp_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "mpc_box_qp",
            "x0": x0,
            "r": r,
            "limit": limit,
            "U": U.tolist(),
            "X": X.tolist(),
            "active": active,
            "cost": float(value),
        },
        source_refs=MPC,
        verifier="verify_mpc_box_qp",
        tool="cvxpy_clarabel",
    )


FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, int], dict]], ...] = (
    ("first_order_exact_step_metrics", first_order_step),
    ("underdamped_second_order_exact_metrics", second_order_transient),
    ("unity_feedback_closed_loop_polynomial", closed_loop_tf),
    ("complex_sensitivity_complementary_sensitivity", sensitivity_complex),
    ("lead_compensator_maximum_phase_parameters", lead_parameters),
    ("pid_closed_loop_characteristic_polynomial", pid_characteristic),
    ("routh_quartic_numeric_classification", routh_quartic),
    ("root_locus_real_axis_membership", root_locus_point),
    ("classical_margins_from_crossover_values", margins_from_crossovers),
    ("scalar_kalman_time_update", scalar_kf_predict),
    ("matrix_kalman_measurement_update_2state", matrix_kf_update),
    ("linear_covariance_time_propagation", covariance_propagation),
    ("continuous_matrix_lqr_care", continuous_lqr_matrix),
    ("discrete_matrix_lqr_dare", discrete_lqr_matrix),
    ("finite_horizon_scalar_lqr_recursion", finite_horizon_lqr),
    ("scalar_mpc_prediction_matrices_horizon3", mpc_prediction_matrices),
    ("unconstrained_scalar_mpc_horizon2", mpc_unconstrained),
    ("box_constrained_scalar_mpc_horizon2", mpc_box_qp),
)


def generate_classical_optimal_v1(count_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        generator(rng, index)
        for _, generator in FAMILIES
        for index in range(1, count_per_family + 1)
    ]
