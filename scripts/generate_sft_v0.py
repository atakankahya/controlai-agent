"""Generate ControlAI's first verified, balanced SFT pilot dataset.

The held-out benchmark families in benchmarks/v0.jsonl are intentionally not
reproduced here.  This pilot teaches response style and several neighboring
control-engineering skills while keeping the benchmark useful.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SYSTEM_PROMPT = (
    "You are a control-systems engineering assistant. Lead with the result, "
    "state assumptions, show the decisive calculation, never invent missing "
    "plant data or numerical results, and provide executable code when requested."
)


def number(value: float) -> str:
    if abs(value) < 5e-11:
        value = 0.0
    return f"{value:.6g}"


def complex_number(value: complex) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(imag) < 1e-9:
        return number(real)
    return f"{number(real)} {'+' if imag >= 0 else '-'} {number(abs(imag))}j"


def matrix_text(matrix: np.ndarray) -> str:
    return repr(np.asarray(matrix, dtype=float).tolist())


def matlab_matrix(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=float):
        rows.append(" ".join(number(float(value)) for value in np.atleast_1d(row)))
    return "[" + "; ".join(rows) + "]"


def record(
    record_id: str,
    domain: str,
    family: str,
    prompt: str,
    answer: str,
    ground_truth: dict,
    tool: str = "analytic",
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "id": record_id,
            "domain": domain,
            "family": family,
            "tool": tool,
            "source_type": "programmatically_generated_verified",
            "generator": "scripts/generate_sft_v0.py",
            "verification": ["independent_numeric_recomputation"],
            "ground_truth": ground_truth,
            "status": "approved",
        },
    }


def stability_records(rng: np.random.Generator, count: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < count:
        A = rng.integers(-5, 6, size=(2, 2)).astype(float)
        key = tuple(int(x) for x in A.flat)
        if key in seen:
            continue
        seen.add(key)
        poles = np.linalg.eigvals(A)
        if np.any(np.abs(np.real(poles)) < 0.15):
            continue
        stable = bool(np.all(np.real(poles) < 0))
        pole_text = ", ".join(complex_number(x) for x in poles)
        conclusion = "asymptotically stable" if stable else "unstable"
        prompt = (
            "For the continuous-time autonomous system x_dot = A x with "
            f"A = {matrix_text(A)}, determine stability from its eigenvalues. "
            "Show the decisive values and give minimal NumPy verification."
        )
        answer = (
            f"The eigenvalues are {pole_text}; therefore the origin is {conclusion}. "
            "Continuous-time asymptotic stability requires every eigenvalue to have "
            "strictly negative real part.\n\n"
            "```python\nimport numpy as np\n"
            f"A = np.array({matrix_text(A)}, dtype=float)\n"
            "p = np.linalg.eigvals(A)\n"
            "print(p)\nprint(np.all(np.real(p) < 0))\n```"
        )
        rows.append(
            record(
                f"v0_ct_stability_{len(rows)+1:04d}",
                "state_space",
                "continuous_lti_eigenvalue_stability",
                prompt,
                answer,
                {
                    "kind": "continuous_stability",
                    "A": A.tolist(),
                    "eigenvalues": [[float(x.real), float(x.imag)] for x in poles],
                    "asymptotically_stable": stable,
                },
                "python_numpy",
            )
        )
    return rows


def second_order_records(count: int) -> list[dict]:
    rows: list[dict] = []
    zetas = [0.2, 0.3, 0.35, 0.5, 0.7, 0.85, 1.0, 1.25]
    frequencies = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    combinations = [(zeta, wn) for zeta in zetas for wn in frequencies]
    for i, (zeta, wn) in enumerate(combinations[:count]):
        a1 = 2 * zeta * wn
        a0 = wn**2
        poles = np.roots([1.0, a1, a0])
        regime = (
            "underdamped" if zeta < 1 else "critically damped" if zeta == 1 else "overdamped"
        )
        prompt = (
            f"A second-order denominator is s^2 + {number(a1)} s + {number(a0)}. "
            "Find the natural frequency, damping ratio, poles, and damping regime."
        )
        answer = (
            f"Result: omega_n = {number(wn)} rad/s, zeta = {number(zeta)}, and the "
            f"system is {regime}. Matching s^2 + 2 zeta omega_n s + omega_n^2 gives "
            f"omega_n = sqrt({number(a0)}) and zeta = {number(a1)}/(2 omega_n). "
            f"The poles are {', '.join(complex_number(x) for x in poles)}."
        )
        rows.append(
            record(
                f"v0_second_order_{i+1:04d}",
                "classical_control",
                "second_order_parameter_interpretation",
                prompt,
                answer,
                {
                    "kind": "second_order",
                    "a1": a1,
                    "a0": a0,
                    "omega_n": wn,
                    "zeta": zeta,
                    "poles": [[float(x.real), float(x.imag)] for x in poles],
                },
            )
        )
    return rows


def observability_records(rng: np.random.Generator, count: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < count:
        A = rng.integers(-3, 4, size=(2, 2)).astype(float)
        C = rng.integers(-2, 3, size=(1, 2)).astype(float)
        key = tuple(int(x) for x in np.concatenate([A.flat, C.flat]))
        if key in seen or not np.any(C):
            continue
        seen.add(key)
        O = np.vstack([C, C @ A])
        rank = int(np.linalg.matrix_rank(O))
        observable = rank == 2
        use_matlab = len(rows) % 2 == 1
        language = "MATLAB" if use_matlab else "Python"
        prompt = (
            f"For x_dot = A x, y = C x with A = {matrix_text(A)} and "
            f"C = {matrix_text(C)}, form the observability matrix, report its rank, "
            f"and conclude observability. Include minimal {language} code."
        )
        if use_matlab:
            code = (
                "```matlab\n"
                f"A = {matlab_matrix(A)};\n"
                f"C = {matlab_matrix(C)};\n"
                "O = obsv(A,C);\nr = rank(O);\ndisp(O); disp(r);\n```"
            )
            tool = "matlab_control_system_toolbox"
        else:
            code = (
                "```python\nimport numpy as np\n"
                f"A = np.array({matrix_text(A)})\nC = np.array({matrix_text(C)})\n"
                "O = np.vstack((C, C @ A))\n"
                "print(O)\nprint(np.linalg.matrix_rank(O))\n```"
            )
            tool = "python_numpy"
        answer = (
            f"The observability matrix is O = [C; CA] = {matrix_text(O)}. Its rank "
            f"is {rank}; therefore the two-state realization is "
            f"{'observable' if observable else 'not observable'}.\n\n{code}"
        )
        rows.append(
            record(
                f"v0_observability_{len(rows)+1:04d}",
                "state_space",
                "observability_rank_numeric_2x2",
                prompt,
                answer,
                {
                    "kind": "observability",
                    "A": A.tolist(),
                    "C": C.tolist(),
                    "O": O.tolist(),
                    "rank": rank,
                    "observable": observable,
                },
                tool,
            )
        )
    return rows


def feedback_records(rng: np.random.Generator, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(count):
        A = rng.integers(-3, 4, size=(2, 2)).astype(float)
        B = rng.integers(-2, 3, size=(2, 1)).astype(float)
        K = rng.integers(-2, 4, size=(1, 2)).astype(float)
        if not np.any(B):
            B[1, 0] = 1
        Acl = A - B @ K
        poles = np.linalg.eigvals(Acl)
        stable = bool(np.all(np.real(poles) < 0))
        prompt = (
            "For x_dot = A x + B u and u = -Kx, use "
            f"A = {matrix_text(A)}, B = {matrix_text(B)}, K = {matrix_text(K)}. "
            "Compute the closed-loop matrix and poles, then classify stability."
        )
        answer = (
            f"A_cl = A - BK = {matrix_text(Acl)}. Its poles are "
            f"{', '.join(complex_number(x) for x in poles)}. Hence the closed loop is "
            f"{'asymptotically stable' if stable else 'not asymptotically stable'}, "
            "because all continuous-time poles must have strictly negative real parts."
        )
        rows.append(
            record(
                f"v0_state_feedback_{i+1:04d}",
                "state_space",
                "given_state_feedback_pole_check",
                prompt,
                answer,
                {
                    "kind": "state_feedback",
                    "A": A.tolist(),
                    "B": B.tolist(),
                    "K": K.tolist(),
                    "Acl": Acl.tolist(),
                    "eigenvalues": [[float(x.real), float(x.imag)] for x in poles],
                    "stable": stable,
                },
            )
        )
    return rows


def discrete_pole_records(count: int) -> list[dict]:
    candidates = [-1.4, -1.0, -0.8, -0.35, 0.0, 0.4, 0.75, 0.98, 1.0, 1.2]
    rows: list[dict] = []
    combinations = [
        (p1, p2)
        for i, p1 in enumerate(candidates)
        for p2 in candidates[i:]
    ]
    for i, (p1, p2) in enumerate(combinations[:count]):
        radius = max(abs(p1), abs(p2))
        stable = radius < 1
        boundary = math.isclose(radius, 1.0)
        if stable:
            conclusion = "asymptotically stable"
        elif boundary:
            conclusion = "not asymptotically stable (a pole lies on the unit circle)"
        else:
            conclusion = "unstable (a pole lies outside the unit circle)"
        prompt = (
            f"A discrete-time LTI system has poles z = {number(p1)} and z = "
            f"{number(p2)}. Determine asymptotic stability and state the criterion."
        )
        answer = (
            f"The largest pole magnitude is {number(radius)}. The system is {conclusion}. "
            "A discrete-time LTI system is asymptotically stable exactly when every pole "
            "satisfies |z_i| < 1; equality is not asymptotic stability."
        )
        rows.append(
            record(
                f"v0_discrete_poles_{i+1:04d}",
                "sampled_data",
                "discrete_pole_unit_circle_classification",
                prompt,
                answer,
                {
                    "kind": "discrete_poles",
                    "poles": [p1, p2],
                    "spectral_radius": radius,
                    "stable": stable,
                },
            )
        )
    return rows


def first_order_frequency_records(count: int) -> list[dict]:
    gains = [0.5, 1.0, 2.0, 4.0]
    taus = [0.1, 0.25, 0.5, 1.0, 2.0]
    omegas = [0.5, 1.0, 2.0, 4.0]
    rows: list[dict] = []
    combinations = [
        (gain, tau, omega)
        for gain in gains
        for tau in taus
        for omega in omegas
    ]
    for i, (gain, tau, omega) in enumerate(combinations[:count]):
        x = tau * omega
        magnitude = gain / math.sqrt(1 + x * x)
        phase_deg = -math.degrees(math.atan(x))
        prompt = (
            f"For G(s) = {number(gain)}/({number(tau)} s + 1), compute the "
            f"magnitude and phase of G(j omega) at omega = {number(omega)} rad/s."
        )
        answer = (
            "Using |G(jw)| = K/sqrt(1+(tau w)^2) and angle G(jw) = "
            f"-atan(tau w), the result is |G| = {number(magnitude)} and phase = "
            f"{number(phase_deg)} degrees. Here tau w = {number(x)}."
        )
        rows.append(
            record(
                f"v0_first_order_frequency_{i+1:04d}",
                "frequency_domain",
                "first_order_frequency_response_numeric",
                prompt,
                answer,
                {
                    "kind": "first_order_frequency",
                    "gain": gain,
                    "tau": tau,
                    "omega": omega,
                    "magnitude": magnitude,
                    "phase_deg": phase_deg,
                },
            )
        )
    return rows


def transfer_records(count: int) -> list[dict]:
    rows: list[dict] = []
    combinations = [
        (-float(p1), -float(p2), -float(zero), float(gain))
        for p1 in range(1, 6)
        for p2 in range(p1, 8)
        for zero in range(1, 7)
        for gain in range(1, 5)
    ]
    for i, (p1, p2, zero, gain) in enumerate(combinations[:count]):
        den = [1.0, -(p1 + p2), p1 * p2]
        num = [gain, -gain * zero]
        dc = num[1] / den[2]
        use_matlab = i % 2 == 0
        prompt = (
            f"For G(s) = ({number(num[0])} s + {number(num[1])}) / "
            f"(s^2 + {number(den[1])} s + {number(den[2])}), report poles, "
            f"zero, DC gain, and give minimal {'MATLAB' if use_matlab else 'Python-control'} code."
        )
        if use_matlab:
            code = (
                "```matlab\n"
                f"G = tf([{number(num[0])} {number(num[1])}], "
                f"[1 {number(den[1])} {number(den[2])}]);\n"
                "pole(G)\nzero(G)\ndcgain(G)\n```"
            )
            tool = "matlab_control_system_toolbox"
        else:
            code = (
                "```python\nimport control as ct\n"
                f"G = ct.tf([{number(num[0])}, {number(num[1])}], "
                f"[1, {number(den[1])}, {number(den[2])}])\n"
                "print(ct.poles(G))\nprint(ct.zeros(G))\nprint(ct.dcgain(G))\n```"
            )
            tool = "python_control"
        answer = (
            f"Poles: {number(p1)}, {number(p2)}. Zero: {number(zero)}. "
            f"DC gain G(0) = {number(num[1])}/{number(den[2])} = {number(dc)}.\n\n{code}"
        )
        rows.append(
            record(
                f"v0_transfer_properties_{i+1:04d}",
                "classical_control",
                "transfer_function_basic_properties",
                prompt,
                answer,
                {
                    "kind": "transfer_properties",
                    "numerator": num,
                    "denominator": den,
                    "poles": [p1, p2],
                    "zero": zero,
                    "dc_gain": dc,
                },
                tool,
            )
        )
    return rows


def underspecified_records() -> list[dict]:
    cases = [
        (
            "pid",
            "Tune a PID controller for my process and give final gains.",
            "Final PID gains cannot be computed without a plant model or experiment data. Provide the input-output definition, transfer function or step-response data, operating point, sample time, delay, actuator limits, noise level, and rise-time/overshoot/disturbance-rejection priorities. I would first validate the model, select a PID structure and discretization, tune against those targets, then verify margins, saturation, anti-windup, noise sensitivity, and nonlinear simulation before reporting implementable gains.",
        ),
        (
            "mpc",
            "Build an MPC controller and send the final matrices.",
            "The controller matrices are plant- and objective-specific, so there is no defensible numerical answer yet. Provide the discrete prediction model, sample time, measured and controlled outputs, manipulated variables, prediction/control horizons, state estimator, constraints, scaling, and cost weights. The next step is to check model dimensions and stabilizability, formulate the constrained optimization, simulate nominal and disturbed cases, and only then export the verified matrices.",
        ),
        (
            "observer",
            "Choose observer poles for my system.",
            "Observer poles cannot be selected without A, C, the sample/continuous-time convention, measurement-noise level, model uncertainty, and required estimation bandwidth. First verify observability, compare estimator bandwidth with plant and sensor bandwidth, then choose poles or design a Kalman filter from justified Q and R. Numerical poles should be reported only after noise amplification and discretization are checked.",
        ),
        (
            "system_identification",
            "Identify my plant from this description and give a transfer function.",
            "A transfer function cannot be identified from a verbal description alone. Supply time-aligned input/output records, units, sample time, operating regime, excitation design, sensor preprocessing, and separate validation experiments. I would inspect delays and nonlinearities, choose candidate orders, estimate on contiguous training runs, and compare free-run prediction plus residual whiteness and residual-input correlation before accepting a model.",
        ),
        (
            "reinforcement_learning_control",
            "Train an RL controller for the machine and deploy it.",
            "Deployment is not justified without a simulator or safe data source, observation/action definitions, timing, constraints, reward, reset conditions, uncertainty ranges, and a certified fallback controller. Start with an offline simulator and explicit safety envelope, evaluate across randomized models and faults, compare against a conventional baseline, then perform bounded hardware tests with supervision. I cannot claim a safe policy or deployment settings from the information given.",
        ),
        (
            "trajectory_tracking",
            "Give me a tracking controller for my robot.",
            "A final tracking law requires the robot dynamics or kinematics, coordinates, actuators, available measurements, trajectory regularity, constraints, sample time, and tracking/error specifications. The next step is to define the error system and operating domain, establish controllability or relative degree, select a method appropriate to uncertainty, and verify stability, saturation, and collision/safety constraints in simulation.",
        ),
        (
            "fault_tolerant_control",
            "Design a fault-tolerant controller with guaranteed performance.",
            "A guarantee cannot be made without a nominal model, enumerated fault set and bounds, fault-detection assumptions, redundancy, measured signals, performance metric, constraints, delays, and certification domain. First define the fault model and detectability, then synthesize the nominal/reconfigured laws and verify worst-case stability and performance for every stated fault. Controller coefficients would otherwise be invented.",
        ),
        (
            "nonlinear_dynamic_inversion",
            "Compute the nonlinear dynamic inversion law for my aircraft.",
            "The inversion law is undefined without the nonlinear equations, chosen outputs, control-effectiveness map, state estimates, flight envelope, actuator dynamics/limits, and reference-command model. Provide those items first. I would determine relative degree, check invertibility and singular regions, derive the nominal inverse, add tracking-error dynamics, and analyze internal dynamics and robustness before reporting a law.",
        ),
    ]
    rows: list[dict] = []
    for repeat in range(2):
        for name, prompt, answer in cases:
            if repeat:
                prompt = "Engineering review request: " + prompt
            rows.append(
                record(
                    f"v0_missing_requirements_{len(rows)+1:04d}",
                    name,
                    "requirements_before_unmodeled_control_design",
                    prompt,
                    answer,
                    {"kind": "requirements", "scenario": name},
                )
            )
    return rows


def build(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    rows.extend(stability_records(rng, 48))
    rows.extend(second_order_records(48))
    rows.extend(observability_records(rng, 48))
    rows.extend(feedback_records(rng, 48))
    rows.extend(discrete_pole_records(40))
    rows.extend(first_order_frequency_records(40))
    rows.extend(transfer_records(32))
    rows.extend(underspecified_records())
    assert len(rows) == 320
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v0"))
    args = parser.parse_args()

    rows = build(args.seed)
    # Parameter holdout for training-loss validation. The real benchmark remains
    # family-separated and is never read by this generator.
    valid = [row for i, row in enumerate(rows) if i % 10 == 0]
    train = [row for i, row in enumerate(rows) if i % 10 != 0]
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    print(f"train: {len(train)} -> {args.output_dir / 'train.jsonl'}")
    print(f"valid: {len(valid)} -> {args.output_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
