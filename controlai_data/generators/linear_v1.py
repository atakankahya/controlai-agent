"""Family-diverse, solver-backed linear/control SFT generators with Chain of Thought (CoT)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import linalg

from controlai_data.schema import make_record


SOURCE_LINEAR = ["dahleh_dahleh_verghese_dynamic_systems_control"]
SOURCE_FOUNDATIONS = ["stanford_ee263_course_reader"]
SOURCE_FEEDBACK = ["astrom_murray_feedback_systems_1e"]
SOURCE_OPT = ["boyd_lmi_system_control"]


def n(value: float) -> str:
    if abs(value) < 5e-11:
        value = 0.0
    return f"{value:.6g}"


def c(value: complex) -> str:
    if abs(value.imag) < 1e-9:
        return n(float(value.real))
    return f"{n(float(value.real))} {'+' if value.imag >= 0 else '-'} {n(abs(float(value.imag)))}j"


def mat(value: np.ndarray) -> str:
    return repr(np.asarray(value, dtype=float).tolist())


def ordered_roots(values: np.ndarray) -> list[list[float]]:
    values = np.asarray(values, dtype=complex)
    values = values[np.lexsort((values.imag, values.real))]
    return [[float(x.real), float(x.imag)] for x in values]


PROMPT_OPENERS = (
    "Compute the requested quantities and justify the conclusion.",
    "Work this control problem from the stated data; show the decisive calculation.",
    "Give a compact engineering analysis with the numerical result first.",
    "Check the following system without introducing unstated parameters.",
)


def triangular_stability(
    rng: np.random.Generator, index: int, discrete: bool
) -> dict:
    if discrete:
        diagonal_values = [-1.25, -0.8, -0.3, 0.2, 0.65, 0.92, 1.0, 1.15]
    else:
        diagonal_values = [-4.0, -2.0, -0.75, -0.1, 0.4, 1.5]
    diagonal = np.array(
        [diagonal_values[(index * step + step) % len(diagonal_values)] for step in (1, 3, 5)],
        dtype=float,
    )
    A = np.diag(diagonal)
    A[0, 1] = int(rng.integers(-3, 4))
    A[0, 2] = int(rng.integers(-3, 4))
    A[1, 2] = int(rng.integers(-3, 4))
    poles = np.linalg.eigvals(A)
    stable = bool(
        np.all(np.abs(poles) < 1) if discrete else np.all(np.real(poles) < 0)
    )
    criterion = "|lambda_i| < 1" if discrete else "Re(lambda_i) < 0"
    family = "discrete_triangular_eigenvalue_stability" if discrete else "continuous_triangular_eigenvalue_stability"
    conclusion = "asymptotically stable" if stable else "not asymptotically stable"
    time_domain_str = "discrete-time x[k+1] = A x" if discrete else "continuous-time x_dot = A x"
    
    prompt = (
        f"{PROMPT_OPENERS[index % len(PROMPT_OPENERS)]} For the "
        f"{time_domain_str} system, "
        f"A = {mat(A)}. Determine its poles and asymptotic stability."
    )
    
    pole_details = "\n".join(
        f"- lambda_{idx+1} = {c(p)}: {'|lambda| = ' + n(abs(p)) if discrete else 'Re(lambda) = ' + n(p.real)} "
        f"({'satisfies' if (abs(p) < 1 if discrete else p.real < 0) else 'violates'} {criterion})"
        for idx, p in enumerate(poles)
    )
    
    answer = (
        f"### 1. Eigenvalue Calculation\n"
        f"Because matrix $A$ is upper triangular, its eigenvalues (poles) are identically its diagonal entries:\n"
        f"$$\\lambda(A) = \\text{{diag}}(A) = \\{{{', '.join(c(x) for x in poles)}\\}}$$\n\n"
        f"### 2. Stability Criterion Evaluation ({'Discrete-Time' if discrete else 'Continuous-Time'})\n"
        f"For a {'discrete-time' if discrete else 'continuous-time'} linear system, asymptotic stability requires **{criterion}** for every pole:\n"
        f"{pole_details}\n\n"
        f"### 3. Conclusion\n"
        f"Result: the poles are {', '.join(c(x) for x in poles)}, so the system is **{conclusion}**."
    )
    
    gt = {
        "kind": "eigenvalue_stability",
        "time_domain": "discrete" if discrete else "continuous",
        "A": A.tolist(),
        "eigenvalues": ordered_roots(poles),
        "stable": stable,
    }
    return make_record(
        record_id=f"{family}_{index:05d}",
        domain="sampled_data" if discrete else "linear_systems",
        family=family,
        task_type="numerical",
        difficulty="foundation",
        template_id=f"{family}_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth=gt,
        source_refs=SOURCE_LINEAR,
        verifier="verify_eigenvalue_stability",
    )


def diagonal_transition(rng: np.random.Generator, index: int) -> dict:
    del rng
    combinations = [
        (-0.25 * first, -0.3 * second, time)
        for first in range(1, 6)
        for second in range(1, 6)
        for time in (0.2, 0.5, 1.0, 1.5)
    ]
    first, second, t = combinations[index - 1]
    values = np.array([first, second], dtype=float)
    A = np.diag(values)
    Phi = np.diag(np.exp(values * t))
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} For x_dot = A x with A = {mat(A)}, calculate the state-transition "
        f"matrix Phi(t) = exp(A t) at t = {n(t)} s."
    )
    answer = (
        f"### 1. Matrix Exponential for Diagonal Matrix\n"
        f"For a diagonal system matrix $A = \\text{{diag}}({n(first)}, {n(second)})$, the matrix exponential is obtained by exponentiating each diagonal entry independently:\n"
        f"$$\\Phi(t) = \\exp(A t) = \\begin{{bmatrix}} \\exp({n(first)} t) & 0 \\\\ 0 & \\exp({n(second)} t) \\end{{bmatrix}}$$\n\n"
        f"### 2. Numerical Evaluation at $t = {n(t)}$ s\n"
        f"- First diagonal entry: $\\exp({n(first)} \\times {n(t)}) = \\exp({n(first * t)}) = {n(math.exp(first * t))}$\n"
        f"- Second diagonal entry: $\\exp({n(second)} \\times {n(t)}) = \\exp({n(second * t)}) = {n(math.exp(second * t))}$\n\n"
        f"### 3. Result\n"
        f"$$\\Phi({n(t)}) = {mat(Phi)}$$\n"
        f"Thus the state trajectory evolves as $x({n(t)}) = \\Phi({n(t)}) x(0)$."
    )
    return make_record(
        record_id=f"diagonal_state_transition_{index:05d}",
        domain="linear_systems",
        family="diagonal_state_transition_exponential",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"diagonal_state_transition_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "matrix_exponential",
            "A": A.tolist(),
            "time": t,
            "Phi": Phi.tolist(),
        },
        source_refs=SOURCE_FOUNDATIONS,
        verifier="verify_matrix_exponential",
    )


def jordan_transition(rng: np.random.Generator, index: int) -> dict:
    del rng
    combinations = [
        (a, t)
        for a in (
            -3.0, -2.0, -1.5, -1.0, -0.5, 0.25, 0.5, 1.0,
            -4.0, -0.25, 0.75, 1.5, 2.0,
        )
        for t in (0.1, 0.25, 0.5, 1.0, 2.0)
    ]
    a, t = combinations[index - 1]
    A = np.array([[a, 1.0], [0.0, a]])
    exp_at = math.exp(a * t)
    Phi = exp_at * np.array([[1.0, t], [0.0, 1.0]])
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Find exp(A t) at t = {n(t)} for the Jordan block A = {mat(A)}. "
        "Do not diagonalize a defective matrix."
    )
    answer = (
        f"### 1. Matrix Decomposition (Jordan Block)\n"
        f"The $2 \\times 2$ defective Jordan block matrix $A$ can be decomposed as $A = {n(a)} I + N$, where:\n"
        f"$$N = \\begin{{bmatrix}} 0 & 1 \\\\ 0 & 0 \\end{{bmatrix}}, \\quad N^2 = \\begin{{bmatrix}} 0 & 0 \\\\ 0 & 0 \\end{{bmatrix}}$$\n\n"
        f"### 2. Series Expansion\n"
        f"Because ${n(a)} I$ and $N$ commute, $\\exp(A t) = \\exp({n(a)} t I) \\exp(N t)$:\n"
        f"$$\\exp(N t) = I + N t + \\frac{{1}}{{2!}} N^2 t^2 + \\cdots = \\begin{{bmatrix}} 1 & t \\\\ 0 & 1 \\end{{bmatrix}}$$\n"
        f"$$\\exp(A t) = \\exp({n(a)} t) \\begin{{bmatrix}} 1 & t \\\\ 0 & 1 \\end{{bmatrix}} = \\begin{{bmatrix}} e^{{{n(a)} t}} & t e^{{{n(a)} t}} \\\\ 0 & e^{{{n(a)} t}} \\end{{bmatrix}}$$\n\n"
        f"### 3. Numerical Evaluation at $t = {n(t)}$\n"
        f"With $\\exp({n(a)} \\times {n(t)}) = \\exp({n(a * t)}) = {n(exp_at)}$:\n"
        f"$$\\Phi({n(t)}) = {mat(Phi)}$$"
    )
    return make_record(
        record_id=f"jordan_state_transition_{index:05d}",
        domain="linear_systems",
        family="jordan_block_state_transition",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"jordan_state_transition_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "matrix_exponential",
            "A": A.tolist(),
            "time": t,
            "Phi": Phi.tolist(),
        },
        source_refs=SOURCE_FOUNDATIONS,
        verifier="verify_matrix_exponential",
    )


def diagonal_zoh(rng: np.random.Generator, index: int) -> dict:
    del rng
    p1 = -0.5 * (1 + index % 4)
    p2 = -1.0 - 0.2 * (index % 3)
    T = 0.05 + 0.01 * (index % 4)
    A = np.diag([p1, p2])
    B = np.array([[1.0], [2.0]])
    Ad = np.diag([math.exp(p1 * T), math.exp(p2 * T)])
    Bd = np.array([[(math.exp(p1 * T) - 1.0) / p1], [2.0 * (math.exp(p2 * T) - 1.0) / p2]])
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Discretize the diagonal plant with A = {mat(A)}, B = {mat(B)} under "
        f"exact zero-order hold at Ts = {n(T)} s."
    )
    answer = (
        f"### 1. Zero-Order Hold (ZOH) Formulation\n"
        f"For sampling period $T_s = {n(T)}$ s, the exact discrete-time state-space matrices are:\n"
        f"1. $A_d = \\exp(A T_s)$\n"
        f"2. $B_d = \\int_0^{{T_s}} \\exp(A \\tau) B d\\tau$\n\n"
        f"### 2. Analytical Integration for Diagonal Matrix\n"
        f"- For $A = \\text{{diag}}({n(p1)}, {n(p2)})$, $A_d = \\text{{diag}}(\\exp({n(p1)} T_s), \\exp({n(p2)} T_s)) = {mat(Ad)}$\n"
        f"- For input matrix $B = [{n(float(B[0,0]))}, {n(float(B[1,0]))}]^T$:\n"
        f"  $$B_{{d,1}} = \\int_0^{{{n(T)}}} e^{{{n(p1)} \\tau}} ({n(float(B[0,0]))}) d\\tau = \\frac{{{n(float(B[0,0]))}}}{{{n(p1)}}} (e^{{{n(p1 * T)}}} - 1) = {n(float(Bd[0,0]))}$$\n"
        f"  $$B_{{d,2}} = \\int_0^{{{n(T)}}} e^{{{n(p2)} \\tau}} ({n(float(B[1,0]))}) d\\tau = \\frac{{{n(float(B[1,0]))}}}{{{n(p2)}}} (e^{{{n(p2 * T)}}} - 1) = {n(float(Bd[1,0]))}$$\n\n"
        f"### 3. Result\n"
        f"$$A_d = {mat(Ad)}$$\n"
        f"$$B_d = {mat(Bd)}$$"
    )
    return make_record(
        record_id=f"diagonal_exact_zoh_{index:05d}",
        domain="sampled_data",
        family="diagonal_exact_zoh_discretization",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"diagonal_exact_zoh_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "zoh_discretization",
            "A": A.tolist(),
            "B": B.tolist(),
            "sample_time": T,
            "Ad": Ad.tolist(),
            "Bd": Bd.tolist(),
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_zoh_discretization",
    )


def double_integrator_zoh(rng: np.random.Generator, index: int) -> dict:
    del rng
    T = 0.015 + 0.005 * index
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Ad = np.array([[1.0, T], [0.0, 1.0]])
    Bd = np.array([[0.5 * T * T], [T]])
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Derive the exact ZOH model of the double integrator with A = {mat(A)}, "
        f"B = {mat(B)}, and sample time {n(T)} s."
    )
    answer = (
        f"### 1. Matrix Exponential of Nilpotent System\n"
        f"For the double integrator $A = \\begin{{bmatrix}} 0 & 1 \\\\ 0 & 0 \\end{{bmatrix}}$, we have $A^2 = 0$. The Taylor series terminates:\n"
        f"$$\\exp(A \\tau) = I + A \\tau = \\begin{{bmatrix}} 1 & \\tau \\\\ 0 & 1 \\end{{bmatrix}}$$\n\n"
        f"### 2. State-Space ZOH Matrices\n"
        f"1. Discrete state matrix at $T_s = {n(T)}$ s:\n"
        f"$$A_d = \\exp(A T_s) = \\begin{{bmatrix}} 1 & {n(T)} \\\\ 0 & 1 \\end{{bmatrix}} = {mat(Ad)}$$\n\n"
        f"2. Discrete input matrix:\n"
        f"$$B_d = \\int_0^{{T_s}} \\exp(A \\tau) B d\\tau = \\int_0^{{{n(T)}}} \\begin{{bmatrix}} 1 & \\tau \\\\ 0 & 1 \\end{{bmatrix}} \\begin{{bmatrix}} 0 \\\\ 1 \\end{{bmatrix}} d\\tau = \\int_0^{{{n(T)}}} \\begin{{bmatrix}} \\tau \\\\ 1 \\end{{bmatrix}} d\\tau = \\begin{{bmatrix}} \\frac{{1}}{{2}} T_s^2 \\\\ T_s \\end{{bmatrix}} = {mat(Bd)}$$\n\n"
        f"### 3. Result\n"
        f"$$A_d = {mat(Ad)}, \\quad B_d = {mat(Bd)}$$"
    )
    return make_record(
        record_id=f"double_integrator_zoh_{index:05d}",
        domain="sampled_data",
        family="double_integrator_exact_zoh",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"double_integrator_zoh_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "zoh_discretization",
            "A": A.tolist(),
            "B": B.tolist(),
            "sample_time": T,
            "Ad": Ad.tolist(),
            "Bd": Bd.tolist(),
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_zoh_discretization",
    )


def observability_rank(rng: np.random.Generator, index: int) -> dict:
    while True:
        A = rng.integers(-3, 4, size=(3, 3)).astype(float)
        C = rng.integers(-2, 3, size=(1, 3)).astype(float)
        if np.any(C):
            break
    O = np.vstack([C, C @ A, C @ A @ A])
    rank = int(np.linalg.matrix_rank(O))
    observable = rank == 3
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} For x_dot = A x, y = Cx with A = {mat(A)} and C = {mat(C)}, form "
        "the three-state observability matrix, compute its rank, and conclude."
    )
    answer = (
        f"### 1. Observability Matrix Construction\n"
        f"For a 3rd-order linear system ($n=3$), the observability matrix is $\\mathcal{{O}} = \\begin{{bmatrix}} C \\\\ CA \\\\ CA^2 \\end{{bmatrix}}$:\n"
        f"- $C = {mat(C)}$\n"
        f"- $CA = {mat(C @ A)}$\n"
        f"- $CA^2 = {mat(C @ A @ A)}$\n\n"
        f"Stacking the rows yields:\n"
        f"$$\\mathcal{{O}} = {mat(O)}$$\n\n"
        f"### 2. Rank Determination\n"
        f"Evaluating the matrix rank: $\\text{{rank}}(\\mathcal{{O}}) = {rank}$.\n\n"
        f"### 3. Conclusion\n"
        f"Since $\\text{{rank}}(\\mathcal{{O}}) = {rank}$ {'= 3 (full state rank)' if observable else '< 3 (rank deficient)'}, "
        f"the realization is **{'observable' if observable else 'not observable'}**."
    )
    return make_record(
        record_id=f"observability_rank_3x3_{index:05d}",
        domain="linear_systems",
        family="observability_rank_numeric_3x3",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"observability_rank_3x3_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "observability_rank",
            "A": A.tolist(),
            "C": C.tolist(),
            "O": O.tolist(),
            "rank": rank,
            "observable": observable,
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_observability_rank",
    )


def pbh_observability(rng: np.random.Generator, index: int) -> dict:
    del rng
    poles = np.array(
        [-1.0 - index % 3, -3.0 - (index * 2) % 4, -7.0 - 0.05 * index]
    )
    C = np.array([[1.0, 0.0 if index % 3 == 0 else 1.0, 0.0 if index % 4 == 0 else 2.0]])
    A = np.diag(poles)
    pbh_ranks = []
    for pole in poles:
        pbh_ranks.append(int(np.linalg.matrix_rank(np.vstack([pole * np.eye(3) - A, C]))))
    observable = all(rank == 3 for rank in pbh_ranks)
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Apply the PBH observability test to A = {mat(A)}, C = {mat(C)}. "
        "Report the rank at each distinct eigenvalue and the conclusion."
    )
    eval_steps = "\n".join(
        f"- At $\\lambda = {n(p)}$: $\\text{{rank}}\\begin{{bmatrix}} {n(p)}I - A \\\\ C \\end{{bmatrix}} = {r}$ out of 3"
        for p, r in zip(poles, pbh_ranks)
    )
    answer = (
        f"### 1. PBH Observability Test Criterion\n"
        f"A realization $(A, C)$ is observable iff $\\text{{rank}}\\begin{{bmatrix}} \\lambda I - A \\\\ C \\end{{bmatrix}} = n = 3$ for every eigenvalue $\\lambda \\in \\sigma(A)$.\n\n"
        f"### 2. Modal Rank Checks\n"
        f"{eval_steps}\n\n"
        f"### 3. Conclusion\n"
        f"The realization is **{'observable' if observable else 'not observable'}** because "
        f"{'every eigenvalue yields full column rank 3' if observable else 'at least one eigenvalue produces rank deficiency'}."
    )
    return make_record(
        record_id=f"pbh_observability_{index:05d}",
        domain="linear_systems",
        family="pbh_observability_diagonal_modes",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"pbh_observability_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "pbh_observability",
            "A": A.tolist(),
            "C": C.tolist(),
            "eigenvalues": poles.tolist(),
            "pbh_ranks": pbh_ranks,
            "observable": observable,
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_pbh_observability",
    )


def state_feedback_check(rng: np.random.Generator, index: int) -> dict:
    A = rng.integers(-3, 4, size=(2, 2)).astype(float)
    B = rng.integers(-2, 3, size=(2, 1)).astype(float)
    K = rng.integers(-2, 4, size=(1, 2)).astype(float)
    if not np.any(B):
        B[1, 0] = 1.0
    BK = B @ K
    Acl = A - BK
    poles = np.linalg.eigvals(Acl)
    stable = bool(np.all(np.real(poles) < 0))
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Check the proposed state feedback u = -Kx for A = {mat(A)}, B = "
        f"{mat(B)}, K = {mat(K)}. Compute A-BK and its poles; do not assume the "
        "proposal stabilizes the plant."
    )
    answer = (
        f"### 1. Closed-Loop System Matrix\n"
        f"With state feedback $u = -K x$, the closed-loop dynamics are $\\dot{{x}} = (A - BK)x$:\n"
        f"- $BK = {mat(B)} \\times {mat(K)} = {mat(BK)}$\n"
        f"- $A_{{cl}} = A - BK = {mat(Acl)}$\n\n"
        f"### 2. Eigenvalues / Closed-Loop Poles\n"
        f"Solving $\\det(\\lambda I - A_{{cl}}) = 0$:\n"
        f"$$\\lambda(A - BK) = \\{{{', '.join(c(x) for x in poles)}\\}}$$\n\n"
        f"### 3. Stability Assessment\n"
        f"Since all closed-loop poles have {'strictly negative real parts (Re(lambda) < 0)' if stable else 'non-negative real parts (Re(lambda) >= 0)'}, "
        f"the proposed state feedback is **{'stabilizing' if stable else 'not stabilizing'}**."
    )
    return make_record(
        record_id=f"state_feedback_check_{index:05d}",
        domain="linear_systems",
        family="proposed_state_feedback_stability_check",
        task_type="critique",
        difficulty="intermediate",
        template_id=f"state_feedback_check_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "state_feedback",
            "A": A.tolist(),
            "B": B.tolist(),
            "K": K.tolist(),
            "Acl": Acl.tolist(),
            "eigenvalues": ordered_roots(poles),
            "stable": stable,
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_state_feedback",
    )


def observer_error_check(rng: np.random.Generator, index: int) -> dict:
    A = rng.integers(-3, 4, size=(2, 2)).astype(float)
    C = rng.integers(-2, 3, size=(1, 2)).astype(float)
    L = rng.integers(-2, 4, size=(2, 1)).astype(float)
    if not np.any(C):
        C[0, 0] = 1.0
    LC = L @ C
    Ae = A - LC
    poles = np.linalg.eigvals(Ae)
    stable = bool(np.all(np.real(poles) < 0))
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} An observer uses gain L = {mat(L)} for A = {mat(A)}, C = {mat(C)}. "
        "Derive the estimation-error matrix, list its poles, and decide whether "
        "the error converges exponentially."
    )
    answer = (
        f"### 1. Observer Error Dynamics\n"
        f"For estimation error $e(t) = x(t) - \\hat{{x}}(t)$, the error dynamics follow $\\dot{{e}} = (A - LC)e$:\n"
        f"- $LC = {mat(L)} \\times {mat(C)} = {mat(LC)}$\n"
        f"- $A_e = A - LC = {mat(Ae)}$\n\n"
        f"### 2. Error Poles (Eigenvalues of A-LC)\n"
        f"Solving $\\det(\\lambda I - (A - LC)) = 0$:\n"
        f"$$\\lambda(A - LC) = \\{{{', '.join(c(x) for x in poles)}\\}}$$\n\n"
        f"### 3. Exponential Convergence Conclusion\n"
        f"Because all poles satisfy {'Re(lambda) < 0' if stable else 'Re(lambda) >= 0 for at least one mode'}, "
        f"the estimation error **{'converges exponentially' if stable else 'does not converge exponentially for every initial error'}**."
    )
    return make_record(
        record_id=f"observer_error_check_{index:05d}",
        domain="estimation_filtering",
        family="observer_error_dynamics_stability_check",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"observer_error_check_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "observer_error",
            "A": A.tolist(),
            "C": C.tolist(),
            "L": L.tolist(),
            "Ae": Ae.tolist(),
            "eigenvalues": ordered_roots(poles),
            "stable": stable,
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_observer_error",
    )


def continuous_lyapunov(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = np.array([-0.5 * (1 + index % 5), -1.0 * (1 + (index * 2) % 4)])
    q = np.array([1.0 + 0.05 * index, 0.5 + (index * 3) % 5])
    A = np.diag(a)
    Q = np.diag(q)
    P = linalg.solve_continuous_lyapunov(A.T, -Q)
    residual = A.T @ P + P @ A + Q
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Solve A^T P + PA = -Q for A = {mat(A)} and Q = {mat(Q)}. "
        "Verify positive definiteness and the residual."
    )
    answer = (
        f"### 1. Continuous Lyapunov Equation\n"
        f"The equation $A^T P + P A = -Q$ with diagonal matrices $A = \\text{{diag}}({n(a[0])}, {n(a[1])})$ and $Q = \\text{{diag}}({n(q[0])}, {n(q[1])})$ has diagonal solution $P = \\text{{diag}}(p_1, p_2)$:\n"
        f"- $2 a_1 p_1 = -q_1 \\implies p_1 = -\\frac{{{n(q[0])}}}{{2({n(a[0])})}} = {n(float(P[0,0]))}$\n"
        f"- $2 a_2 p_2 = -q_2 \\implies p_2 = -\\frac{{{n(q[1])}}}{{2({n(a[1])})}} = {n(float(P[1,1]))}$\n\n"
        f"### 2. Solution Matrix and Definiteness\n"
        f"$$P = {mat(P)}$$\n"
        f"The eigenvalues of $P$ are {', '.join(n(x) for x in np.linalg.eigvalsh(P))}. Since all eigenvalues are strictly positive, $P$ is positive definite ($P > 0$).\n\n"
        f"### 3. Residual Verification\n"
        f"$$A^T P + PA + Q = {mat(residual)}$$ (zero within floating-point tolerance)."
    )
    return make_record(
        record_id=f"continuous_lyapunov_equation_{index:05d}",
        domain="linear_systems",
        family="continuous_diagonal_lyapunov_equation",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"continuous_lyapunov_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "continuous_lyapunov",
            "A": A.tolist(),
            "Q": Q.tolist(),
            "P": P.tolist(),
            "residual": residual.tolist(),
        },
        source_refs=SOURCE_OPT,
        verifier="verify_continuous_lyapunov",
    )


def discrete_lyapunov(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = np.array([0.1 + 0.1 * (index % 5), -0.2 - 0.1 * ((index * 2) % 5)])
    q = np.array([1.0 + 0.05 * index, 0.5 + (index * 2) % 4])
    A = np.diag(a)
    Q = np.diag(q)
    P = linalg.solve_discrete_lyapunov(A.T, Q)
    residual = A.T @ P @ A - P + Q
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Solve the discrete Lyapunov equation A^T P A - P = -Q for A = "
        f"{mat(A)}, Q = {mat(Q)}. Check the residual and definiteness."
    )
    answer = (
        f"### 1. Discrete Lyapunov Equation\n"
        f"The equation $A^T P A - P = -Q$ with diagonal $A = \\text{{diag}}({n(a[0])}, {n(a[1])})$ and $Q = \\text{{diag}}({n(q[0])}, {n(q[1])})$ yields $P = \\text{{diag}}(p_1, p_2)$:\n"
        f"- $(a_1^2 - 1) p_1 = -q_1 \\implies p_1 = \\frac{{{n(q[0])}}}{{1 - ({n(a[0])})^2}} = {n(float(P[0,0]))}$\n"
        f"- $(a_2^2 - 1) p_2 = -q_2 \\implies p_2 = \\frac{{{n(q[1])}}}{{1 - ({n(a[1])})^2}} = {n(float(P[1,1]))}$\n\n"
        f"### 2. Solution Matrix and Definiteness\n"
        f"$$P = {mat(P)}$$\n"
        f"Eigenvalues of $P$ are {', '.join(n(x) for x in np.linalg.eigvalsh(P))}. Since all eigenvalues are strictly positive, $P$ is positive definite ($P > 0$).\n\n"
        f"### 3. Residual Verification\n"
        f"$$A^T P A - P + Q = {mat(residual)}$$ (zero to numerical rounding)."
    )
    return make_record(
        record_id=f"discrete_lyapunov_equation_{index:05d}",
        domain="sampled_data",
        family="discrete_diagonal_lyapunov_equation",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"discrete_lyapunov_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "discrete_lyapunov",
            "A": A.tolist(),
            "Q": Q.tolist(),
            "P": P.tolist(),
            "residual": residual.tolist(),
        },
        source_refs=SOURCE_OPT,
        verifier="verify_discrete_lyapunov",
    )


def controllability_gramian(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = np.array([-1.0 - index % 3, -2.0 - (index * 2) % 4])
    B = np.array([[1.0 + 0.05 * index], [1.0 + (index * 3) % 3]])
    A = np.diag(a)
    W = linalg.solve_continuous_lyapunov(A, -(B @ B.T))
    eig = np.linalg.eigvalsh(W)
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} For the stable continuous-time pair A = {mat(A)}, B = {mat(B)}, "
        "compute the infinite-horizon controllability Gramian and interpret its eigenvalues."
    )
    answer = (
        f"### 1. Controllability Gramian Equation\n"
        f"The infinite-horizon controllability Gramian $W_c$ satisfies the Lyapunov equation $A W_c + W_c A^T + B B^T = 0$.\n\n"
        f"### 2. Solution Matrix\n"
        f"$$W_c = {mat(W)}$$\n\n"
        f"### 3. Eigenvalue Analysis\n"
        f"Eigenvalues of $W_c$ are $\\lambda(W_c) = \\{{{', '.join(n(x) for x in eig)}\\}}.\n"
        f"Since {'all eigenvalues are strictly positive' if np.all(eig > 0) else 'W_c is singular'}, "
        f"the Gramian is **{'positive definite and both states are reachable' if np.all(eig > 0) else 'singular, revealing an unreachable direction'}**."
    )
    return make_record(
        record_id=f"controllability_gramian_{index:05d}",
        domain="linear_systems",
        family="continuous_controllability_gramian",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"controllability_gramian_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "controllability_gramian",
            "A": A.tolist(),
            "B": B.tolist(),
            "W": W.tolist(),
            "eigenvalues": eig.tolist(),
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_controllability_gramian",
    )


def observability_gramian(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = np.array([-0.5 - index % 4, -2.0 - (index * 3) % 3])
    C = np.array([[1.0 + 0.05 * index, 0.0 if index % 5 == 0 else 1.0]])
    A = np.diag(a)
    W = linalg.solve_continuous_lyapunov(A.T, -(C.T @ C))
    eig = np.linalg.eigvalsh(W)
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} For A = {mat(A)}, C = {mat(C)}, compute the infinite-horizon "
        "observability Gramian and use its eigenvalues to assess observability."
    )
    answer = (
        f"### 1. Observability Gramian Equation\n"
        f"The infinite-horizon observability Gramian $W_o$ solves $A^T W_o + W_o A + C^T C = 0$.\n\n"
        f"### 2. Solution Matrix\n"
        f"$$W_o = {mat(W)}$$\n\n"
        f"### 3. Eigenvalue Analysis\n"
        f"Eigenvalues: $\\lambda(W_o) = \\{{{', '.join(n(x) for x in eig)}\\}}.\n"
        f"Therefore the pair $(A, C)$ is **{'observable' if np.all(eig > 1e-10) else 'not observable'}**."
    )
    return make_record(
        record_id=f"observability_gramian_{index:05d}",
        domain="linear_systems",
        family="continuous_observability_gramian",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"observability_gramian_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "observability_gramian",
            "A": A.tolist(),
            "C": C.tolist(),
            "W": W.tolist(),
            "eigenvalues": eig.tolist(),
        },
        source_refs=SOURCE_LINEAR,
        verifier="verify_observability_gramian",
    )


def first_order_frequency(rng: np.random.Generator, index: int) -> dict:
    del rng
    gain = [0.5, 1.0, 2.0, 5.0][index % 4]
    tau = [0.1, 0.25, 0.5, 1.0, 2.0][(index * 3) % 5]
    omega = 0.15 + 0.075 * index
    x = tau * omega
    magnitude = gain / math.sqrt(1 + x * x)
    phase = -math.degrees(math.atan(x))
    db = 20 * math.log10(magnitude)
    prompt = (
        f"{PROMPT_OPENERS[index % 4]} Evaluate G(j omega) for G(s) = {n(gain)}/({n(tau)}s + 1) at "
        f"omega = {n(omega)} rad/s. Report magnitude, dB magnitude, and phase."
    )
    answer = (
        f"### 1. Frequency Response Formula\n"
        f"For first-order transfer function $G(s) = \\frac{{{n(gain)}}}{{{n(tau)}s + 1}}$, substituting $s = j\\omega$ at $\\omega = {n(omega)}$ rad/s:\n"
        f"$$G(j{n(omega)}) = \\frac{{{n(gain)}}}{{1 + j ({n(tau)} \\times {n(omega)})}} = \\frac{{{n(gain)}}}{{1 + j {n(x)}}}$$\n\n"
        f"### 2. Magnitude and Decibels\n"
        f"- Linear magnitude: $|G| = \\frac{{{n(gain)}}}{{\\sqrt{{1 + ({n(tau)} \\times {n(omega)})^2}}}} = \\frac{{{n(gain)}}}{{\\sqrt{{1 + {n(x*x)}}}}} = {n(magnitude)}$\n"
        f"- Logarithmic magnitude: $20\\log_{{10}}|G| = 20\\log_{{10}}({n(magnitude)}) = {n(db)}\\text{{ dB}}$\n\n"
        f"### 3. Phase\n"
        f"$$\\angle G = -\\arctan(\\tau \\omega) = -\\arctan({n(x)}) = {n(phase)}^\\circ$$\n\n"
        f"Result: |G| = {n(magnitude)}, 20log10|G| = {n(db)} dB, and phase = {n(phase)} degrees."
    )
    return make_record(
        record_id=f"first_order_frequency_{index:05d}",
        domain="classical_control",
        family="first_order_frequency_response_with_db",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"first_order_frequency_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "first_order_frequency",
            "gain": gain,
            "tau": tau,
            "omega": omega,
            "magnitude": magnitude,
            "magnitude_db": db,
            "phase_deg": phase,
        },
        source_refs=SOURCE_FEEDBACK,
        verifier="verify_first_order_frequency",
    )


FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, int], dict]], ...] = (
    ("continuous_triangular_eigenvalue_stability", lambda r, i: triangular_stability(r, i, False)),
    ("discrete_triangular_eigenvalue_stability", lambda r, i: triangular_stability(r, i, True)),
    ("diagonal_state_transition_exponential", diagonal_transition),
    ("jordan_block_state_transition", jordan_transition),
    ("diagonal_exact_zoh_discretization", diagonal_zoh),
    ("double_integrator_exact_zoh", double_integrator_zoh),
    ("observability_rank_numeric_3x3", observability_rank),
    ("pbh_observability_diagonal_modes", pbh_observability),
    ("proposed_state_feedback_stability_check", state_feedback_check),
    ("observer_error_dynamics_stability_check", observer_error_check),
    ("continuous_diagonal_lyapunov_equation", continuous_lyapunov),
    ("discrete_diagonal_lyapunov_equation", discrete_lyapunov),
    ("continuous_controllability_gramian", controllability_gramian),
    ("continuous_observability_gramian", observability_gramian),
    ("first_order_frequency_response_with_db", first_order_frequency),
)


def generate_linear_v1(count_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for _, generator in FAMILIES:
        for index in range(1, count_per_family + 1):
            records.append(generator(rng, index))
    return records
