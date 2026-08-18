"""Safety, aerospace, stochastic, fault-tolerant, and learning-control families with Chain of Thought (CoT)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from controlai_data.schema import make_record


OPENERS = (
    "Provide a detailed step-by-step engineering calculation with intermediate derivations.",
    "Solve this numerical control problem step-by-step without omitting intermediate math.",
    "State the governing equations, compute the requested values, and verify the conclusion.",
    "Give a rigorous control engineering derivation with verified numbers.",
)


def number(value: float) -> str:
    if abs(value) < 5e-12:
        value = 0.0
    return f"{value:.10g}"


def record(
    *,
    index: int,
    prefix: str,
    domain: str,
    family: str,
    difficulty: str,
    prompt: str,
    answer: str,
    ground_truth: dict,
    source_refs: list[str],
    verifier: str,
) -> dict:
    return make_record(
        record_id=f"{prefix}_{index:05d}",
        domain=domain,
        family=family,
        task_type="numerical_reasoning",
        difficulty=difficulty,
        template_id=f"{prefix}_prompt_{index % 4}",
        prompt=prompt,
        answer=answer,
        ground_truth=ground_truth,
        source_refs=source_refs,
        verifier=verifier,
    )


def nonlinear_dynamic_inversion(_: np.random.Generator, index: int) -> dict:
    a = -0.7 + 0.018 * index
    b = 0.8 + 0.012 * index
    x = -1.2 + 0.041 * index
    reference = 0.35 - 0.006 * index
    gain = 1.1 + 0.017 * index
    virtual = -gain * (x - reference)
    control = (virtual - a * x) / b
    
    prompt = (
        f"{OPENERS[index % 4]} For scalar affine system x_dot={number(a)}x+{number(b)}u, "
        f"use exact nonlinear dynamic inversion so that x_dot=v=-{number(gain)}(x-r). "
        f"At x={number(x)}, r={number(reference)}, compute the virtual control v and the required control input u."
    )
    answer = (
        f"### 1. Virtual Control Law Definition\n"
        f"To enforce the target error dynamics $\\dot{{e}} = \\dot{{x}} - \\dot{{r}} = -k(x - r)$ with setpoint $r = {number(reference)}$:\n"
        f"$$v = -k (x - r) = -{number(gain)} \\times ({number(x)} - {number(reference)}) = -{number(gain)} \\times ({number(x - reference)}) = {number(virtual)}$$\n\n"
        f"### 2. Exact Dynamic Inversion Algebraic Solution\n"
        f"Equating the open-loop dynamics $\\dot{{x}} = a x + b u$ to the synthetic virtual input $v$:\n"
        f"$$a x + b u = v \\implies u = \\frac{{v - a x}}{{b}}$$\n"
        f"$$u = \\frac{{{number(virtual)} - ({number(a)})({number(x)})}}{{{number(b)}}} = \\frac{{{number(virtual)} - ({number(a * x)})}}{{{number(b)}}} = {number(control)}$$\n\n"
        f"### 3. Direct Closed-Loop Verification\n"
        f"$$\\dot{{x}} = ({number(a)})({number(x)}) + ({number(b)})({number(control)}) = {number(a * x)} + {number(b * control)} = {number(a * x + b * control)} = v$$\n"
        f"**Summary:** Virtual control v = {number(virtual)}, control input u = {number(control)}."
    )
    return record(
        index=index,
        prefix="ndi_scalar",
        domain="aerospace_control",
        family="scalar_affine_nonlinear_dynamic_inversion",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "scalar_dynamic_inversion",
            "a": a,
            "b": b,
            "x": x,
            "reference": reference,
            "gain": gain,
            "virtual": virtual,
            "u": control,
        },
        source_refs=["slotine_li_applied_nonlinear_control", "ndi_control_literature"],
        verifier="analytic_scalar_dynamic_inversion",
    )


def gain_schedule(_: np.random.Generator, index: int) -> dict:
    low = np.array([0.8 + 0.01 * index, 0.25 + 0.004 * index])
    high = np.array([2.0 + 0.014 * index, 0.75 + 0.006 * index])
    sigma_low = 10.0
    sigma_high = 30.0 + 0.1 * index
    sigma = 11.0 + (index % 17) * 0.9
    weight = (sigma - sigma_low) / (sigma_high - sigma_low)
    scheduled = (1 - weight) * low + weight * high
    
    prompt = (
        f"{OPENERS[index % 4]} Linearly schedule gain vector K between K_low={low.tolist()} at "
        f"sigma={number(sigma_low)} and K_high={high.tolist()} at sigma={number(sigma_high)}. "
        f"Find K at operating point sigma={number(sigma)}; do not extrapolate or claim global stability."
    )
    answer = (
        f"### 1. Interpolation Weight Calculation\n"
        f"$$\\mu = \\frac{{\\sigma - \\sigma_{{low}}}}{{\\sigma_{{high}} - \\sigma_{{low}}}} = \\frac{{{number(sigma)} - {number(sigma_low)}}}{{{number(sigma_high)} - {number(sigma_low)}}} = {number(weight)}$$\n\n"
        f"### 2. Gain Vector Linear Interpolation\n"
        f"$$K(\\sigma) = (1 - \\mu) K_{{low}} + \\mu K_{{high}} = (1 - {number(weight)}) {low.tolist()} + {number(weight)} {high.tolist()} = {scheduled.tolist()}$$\n\n"
        f"### 3. Engineering Context\n"
        f"The scheduled gain is $K = {scheduled.tolist()}$. This gain scheduling is valid for slow parameter variation; it does not constitute a global nonlinear stability certificate."
    )
    return record(
        index=index,
        prefix="gain_schedule",
        domain="aerospace_control",
        family="linear_gain_schedule_interpolation",
        difficulty="intermediate",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "gain_schedule_interpolation",
            "low": low.tolist(),
            "high": high.tolist(),
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "sigma": sigma,
            "weight": weight,
            "scheduled": scheduled.tolist(),
        },
        source_refs=["gain_scheduling_control_literature"],
        verifier="analytic_linear_interpolation",
    )


def cbf_filter(_: np.random.Generator, index: int) -> dict:
    x_min = -0.4 + 0.005 * index
    x = x_min + 0.08 + 0.012 * (index % 9)
    alpha = 0.8 + 0.025 * index
    u_nom = -1.1 + 0.033 * index
    h = x - x_min
    lower = -alpha * h
    safe = max(u_nom, lower)
    active = u_nom < lower
    
    prompt = (
        f"{OPENERS[index % 4]} A single integrator x_dot=u must satisfy safety set x>={number(x_min)}. "
        f"With barrier function h(x)=x-x_min, enforce the forward-invariance CBF condition h_dot+{number(alpha)}h>=0 "
        f"while minimally modifying nominal command u_nom={number(u_nom)}. At current state x={number(x)}, compute the filtered control input u."
    )
    answer = (
        f"### 1. Control Barrier Function (CBF) Condition\n"
        f"For safety barrier $h(x) = x - x_{{min}} = {number(x)} - ({number(x_min)}) = {number(h)}$:\n"
        f"$$\\dot{{h}}(x, u) + \\alpha h(x) \\ge 0 \\implies u + {number(alpha)} ({number(h)}) \\ge 0 \\implies u \\ge -{number(alpha)} \\times {number(h)} = {number(lower)}$$\n\n"
        f"### 2. Quadratic Program Safety Filter Projection\n"
        f"$$\\min_u \\frac{{1}}{{2}} (u - u_{{nom}})^2 \\quad \\text{{s.t.}} \\quad u \\ge {number(lower)}$$\n"
        f"The analytical scalar projection is:\n"
        f"$$u^* = \\max(u_{{nom}}, -\\alpha h) = \\max({number(u_nom)}, {number(lower)}) = {number(safe)}$$\n\n"
        f"### 3. Verification\n"
        f"- Constraint status: **{'Active (intervened to enforce safety)' if active else 'Inactive (nominal command is already safe)'}**\n"
        f"- Residual safety derivative: $\\dot{{h}} + \\alpha h = {number(safe)} + {number(alpha * h)} = {number(safe + alpha * h)} \\ge 0$\n\n"
        f"**Summary:** Filtered control u = {number(safe)}."
    )
    return record(
        index=index,
        prefix="cbf_scalar",
        domain="safety_critical_control",
        family="scalar_control_barrier_safety_filter",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "scalar_cbf_filter",
            "x_min": x_min,
            "x": x,
            "alpha": alpha,
            "u_nom": u_nom,
            "h": h,
            "lower_bound": lower,
            "u_safe": safe,
            "active": active,
        },
        source_refs=["ames_control_barrier_functions"],
        verifier="analytic_scalar_cbf_projection",
    )


def scalar_stationary_covariance(_: np.random.Generator, index: int) -> dict:
    a = 0.2 + 0.009 * index
    q = 0.05 + 0.003 * index
    p = q / (1 - a * a)
    prompt = (
        f"{OPENERS[index % 4]} For discrete stochastic system x_(k+1)={number(a)}x_k+w_k with E[w_k^2]="
        f"{number(q)}, compute the stationary variance P if it exists and verify the Lyapunov equation."
    )
    answer = (
        f"### 1. Existence of Stationary Second Moment\n"
        f"For $x_{{k+1}} = a x_k + w_k$, a stationary variance exists if and only if $|a| < 1$.\n"
        f"Here $|a| = {number(abs(a))} < 1$, so the state process is stationary.\n\n"
        f"### 2. Discrete Lyapunov Equation for Covariance\n"
        f"$$P = a^2 P + Q \\implies (1 - a^2) P = Q \\implies P = \\frac{{Q}}{{1 - a^2}}$$\n"
        f"$$P = \\frac{{{number(q)}}}{{1 - ({number(a)})^2}} = \\frac{{{number(q)}}}{{1 - {number(a * a)}}} = \\frac{{{number(q)}}}{{{number(1 - a * a)}}} = {number(p)}$$\n\n"
        f"### 3. Verification\n"
        f"$$a^2 P + Q = ({number(a * a)})({number(p)}) + {number(q)} = {number(a * a * p + q)} = P$$\n"
        f"**Summary:** Stationary variance P = {number(p)}."
    )
    return record(
        index=index,
        prefix="stationary_covariance",
        domain="stochastic_control",
        family="scalar_stationary_process_covariance",
        difficulty="intermediate",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "scalar_stationary_covariance",
            "a": a,
            "q": q,
            "P": p,
            "exists": abs(a) < 1,
        },
        source_refs=["anderson_moore_optimal_filtering"],
        verifier="scalar_discrete_lyapunov",
    )


def fault_allocation(_: np.random.Generator, index: int) -> dict:
    effectiveness = np.array([1.0, 0.75 + 0.003 * index, 0.45 + 0.002 * index])
    desired = -1.0 + 0.047 * index
    b_norm_sq = float(effectiveness @ effectiveness)
    allocation = effectiveness * desired / b_norm_sq
    achieved = float(effectiveness @ allocation)
    
    prompt = (
        f"{OPENERS[index % 4]} Three redundant actuators have effectiveness row "
        f"B={effectiveness.tolist()}. Find the unconstrained minimum-2-norm control command u "
        f"that produces desired virtual control tau={number(desired)}, and verify Bu=tau."
    )
    answer = (
        f"### 1. Minimum 2-Norm Control Allocation Formulation\n"
        f"$$\\min_u \\frac{{1}}{{2}} \\|u\\|_2^2 \\quad \\text{{s.t.}} \\quad B u = \\tau$$\n"
        f"For full-row-rank matrix $B$, the unique minimum-norm solution via Moore-Penrose pseudoinverse is:\n"
        f"$$u^* = B^T (B B^T)^{{-1}} \\tau = \\frac{{\\tau}}{{B B^T}} B^T$$\n\n"
        f"### 2. Evaluation\n"
        f"- Denominator: $B B^T = \\|B\\|_2^2 = 1.0^2 + {number(effectiveness[1])}^2 + {number(effectiveness[2])}^2 = {number(b_norm_sq)}$\n"
        f"- Multiplier: $\\frac{{\\tau}}{{\\|B\\|_2^2}} = \\frac{{{number(desired)}}}{{{number(b_norm_sq)}}} = {number(desired / b_norm_sq)}$\n"
        f"- Allocated actuator commands:\n"
        f"  $$u^* = {allocation.tolist()}$$\n\n"
        f"### 3. Virtual Control Verification\n"
        f"$$B u^* = {effectiveness.tolist()} \\cdot {allocation.tolist()} = {number(achieved)}$$\n"
        f"Residual: $|B u^* - \\tau| = {number(abs(achieved - desired))} \\approx 0$.\n\n"
        f"**Summary:** Allocated command u = {allocation.tolist()}."
    )
    return record(
        index=index,
        prefix="fault_allocation",
        domain="fault_tolerant_control",
        family="minimum_norm_redundant_control_allocation",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "minimum_norm_allocation",
            "B": effectiveness.tolist(),
            "desired": desired,
            "u": allocation.tolist(),
            "achieved": achieved,
        },
        source_refs=["fault_tolerant_control_allocation_literature"],
        verifier="pseudoinverse_allocation",
    )


def jump_scalar(_: np.random.Generator, index: int) -> dict:
    probability = 0.2 + 0.006 * (index % 30)
    a1 = 0.45 + 0.003 * index
    a2 = 1.02 + 0.002 * index
    factor = probability * a1 * a1 + (1 - probability) * a2 * a2
    stable = factor < 1
    
    prompt = (
        f"{OPENERS[index % 4]} An i.i.d. scalar jump linear system x_(k+1)=a_k x_k uses multiplier a1={number(a1)} "
        f"with probability p={number(probability)} and a2={number(a2)} with probability 1-p={number(1 - probability)}. "
        "Compute the expected squared multiplier E[a^2] and decide mean-square stability."
    )
    answer = (
        f"### 1. Second-Moment Dynamics\n"
        f"For the i.i.d. scalar jump system $x_{{k+1}} = a_k x_k$:\n"
        f"$$\\mathbb{{E}}[x_{{k+1}}^2] = \\mathbb{{E}}[a_k^2] \\mathbb{{E}}[x_k^2]$$\n"
        f"The system is mean-square stable if and only if $\\mathbb{{E}}[a^2] < 1$.\n\n"
        f"### 2. Expected Squared Multiplier Calculation\n"
        f"$$\\mathbb{{E}}[a^2] = p a_1^2 + (1 - p) a_2^2$$\n"
        f"$$\\mathbb{{E}}[a^2] = ({number(probability)}) ({number(a1)})^2 + ({number(1 - probability)}) ({number(a2)})^2$$\n"
        f"$$\\mathbb{{E}}[a^2] = ({number(probability)}) ({number(a1 * a1)}) + ({number(1 - probability)}) ({number(a2 * a2)}) = {number(factor)}$$\n\n"
        f"### 3. Stability Conclusion\n"
        f"Since $\\mathbb{{E}}[a^2] = {number(factor)}$ {'< 1' if stable else '>= 1'}, the system is **{'mean-square stable' if stable else 'not certified mean-square stable'}**."
    )
    return record(
        index=index,
        prefix="jump_scalar",
        domain="stochastic_control",
        family="iid_scalar_jump_mean_square_test",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "iid_jump_mean_square",
            "p": probability,
            "a1": a1,
            "a2": a2,
            "factor": factor,
            "stable": stable,
        },
        source_refs=["markov_jump_linear_systems_literature"],
        verifier="expected_square_multiplier",
    )


def event_trigger(_: np.random.Generator, index: int) -> dict:
    state = -1.4 + 0.05 * index
    held = state + (-0.3 + 0.011 * index)
    sigma = 0.12 + 0.002 * index
    error = held - state
    threshold = sigma * abs(state)
    trigger = abs(error) >= threshold
    prompt = (
        f"{OPENERS[index % 4]} An event-triggered control mechanism broadcasts an update when |x_hat-x|>=sigma|x|. "
        f"Given current state x={number(state)}, last transmitted state x_hat={number(held)}, and threshold parameter sigma={number(sigma)}, "
        "compute both sides of the condition and decide whether an event is triggered."
    )
    answer = (
        f"### 1. State Error and Threshold Evaluation\n"
        f"- State discrepancy: $|\\hat{{x}} - x| = |{number(held)} - ({number(state)})| = |{number(error)}| = {number(abs(error))}$\n"
        f"- State-dependent threshold: $\\sigma |x| = {number(sigma)} \\times |{number(state)}| = {number(threshold)}$\n\n"
        f"### 2. Trigger Condition Check\n"
        f"Condition $|\\hat{{x}} - x| \\ge \\sigma |x|$ evaluates to ${number(abs(error))} \\ge {number(threshold)}$, which is **{trigger}**.\n\n"
        f"**Summary:** Trigger is {str(trigger).lower()}."
    )
    return record(
        index=index,
        prefix="event_trigger",
        domain="networked_control",
        family="relative_state_error_event_trigger",
        difficulty="intermediate",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "relative_event_trigger",
            "x": state,
            "x_hat": held,
            "sigma": sigma,
            "error_abs": abs(error),
            "threshold": threshold,
            "trigger": trigger,
        },
        source_refs=["event_triggered_control_literature"],
        verifier="analytic_event_threshold",
    )


def tube_tightening(_: np.random.Generator, index: int) -> dict:
    bound = 2.0 + 0.025 * index
    error = 0.15 + 0.004 * index
    tightened = bound - error
    prompt = (
        f"{OPENERS[index % 4]} In scalar tube MPC with state decomposition x=z+e, the true state constraint is "
        f"|x|<={number(bound)}, and the robustly invariant error tube is |e|<={number(error)}. "
        "Compute the Pontryagin-tightened nominal state constraint interval for z."
    )
    answer = (
        f"### 1. Pontryagin Set Difference $\\mathbb{{Z}} = \\mathbb{{X}} \\ominus \\mathbb{{E}}$\n"
        f"For state constraint set $\\mathbb{{X}} = [-{number(bound)}, {number(bound)}]$ and error tube $\\mathbb{{E}} = [-{number(error)}, {number(error)}]$:\n"
        f"$$z \\in \\mathbb{{X}} \\ominus \\mathbb{{E}} \\iff |z| \\le X_{{\\max}} - E_{{\\max}}$$\n\n"
        f"### 2. Tightened Bound Calculation\n"
        f"$$z_{{\\max}} = {number(bound)} - {number(error)} = {number(tightened)}$$\n"
        f"The nominal state $z$ must satisfy $|z| \\le {number(tightened)}$, i.e., $z \\in [{number(-tightened)}, {number(tightened)}]$.\n\n"
        f"**Summary:** Tightened interval is [{number(-tightened)}, {number(tightened)}]."
    )
    return record(
        index=index,
        prefix="tube_tightening",
        domain="mpc",
        family="symmetric_tube_mpc_constraint_tightening",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "symmetric_tube_tightening",
            "bound": bound,
            "error": error,
            "tightened": tightened,
            "interval": [-tightened, tightened],
            "nonempty": tightened >= 0,
        },
        source_refs=["rawlings_mayne_diehl_mpc_2e"],
        verifier="interval_pontryagin_difference",
    )


def discounted_return(_: np.random.Generator, index: int) -> dict:
    rewards = np.array([1.0 + 0.01 * index, -0.25 + 0.004 * index, 0.6 - 0.003 * index, 0.2 + 0.002 * index])
    gamma = 0.82 + 0.0015 * index
    terms = rewards * gamma ** np.arange(len(rewards))
    total = float(terms.sum())
    prompt = (
        f"{OPENERS[index % 4]} For reward sequence r={rewards.tolist()} and discount factor gamma="
        f"{number(gamma)}, compute the finite discounted return G_0 = sum_(k=0)^3 gamma^k r_k."
    )
    answer = (
        f"### 1. Discounted Return Formula\n"
        f"$$G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\gamma^3 r_3$$\n\n"
        f"### 2. Step-by-Step Discounted Terms\n"
        f"- $k=0$: $r_0 = {number(terms[0])}$\n"
        f"- $k=1$: $\\gamma r_1 = ({number(gamma)})({number(rewards[1])}) = {number(terms[1])}$\n"
        f"- $k=2$: $\\gamma^2 r_2 = ({number(gamma**2)})({number(rewards[2])}) = {number(terms[2])}$\n"
        f"- $k=3$: $\\gamma^3 r_3 = ({number(gamma**3)})({number(rewards[3])}) = {number(terms[3])}$\n\n"
        f"### 3. Total Return\n"
        f"$$G_0 = {number(terms[0])} + ({number(terms[1])}) + {number(terms[2])} + {number(terms[3])} = {number(total)}$$\n\n"
        f"**Summary:** G_0 = {number(total)}."
    )
    return record(
        index=index,
        prefix="rl_return",
        domain="reinforcement_learning_control",
        family="finite_discounted_control_return",
        difficulty="foundation",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "finite_discounted_return",
            "rewards": rewards.tolist(),
            "gamma": gamma,
            "terms": terms.tolist(),
            "return": total,
        },
        source_refs=["sutton_barto_reinforcement_learning_2e"],
        verifier="finite_discounted_sum",
    )


def nonlinear_feedforward(_: np.random.Generator, index: int) -> dict:
    a = 0.5 + 0.012 * index
    b = 0.9 + 0.008 * index
    reference = -0.8 + 0.026 * index
    equilibrium_input = a * reference ** 3 / b
    residual = -a * reference ** 3 + b * equilibrium_input
    prompt = (
        f"{OPENERS[index % 4]} For nonlinear system x_dot=-{number(a)}x^3+{number(b)}u, find the "
        f"constant feedforward control input u_eq that makes state x=r={number(reference)} an equilibrium. "
        "Verify the residual; do not claim stability."
    )
    answer = (
        f"### 1. Equilibrium Condition\n"
        f"At equilibrium $x = r = {number(reference)}$, $\\dot{{x}} = 0$:\n"
        f"$$\\dot{{x}} = -{number(a)} r^3 + {number(b)} u_{{eq}} = 0 \\implies u_{{eq}} = \\frac{{{number(a)} r^3}}{{{number(b)}}}$$\n\n"
        f"### 2. Numerical Computation\n"
        f"$$u_{{eq}} = \\frac{{{number(a)} \\times ({number(reference)})^3}}{{{number(b)}}} = \\frac{{{number(a * reference**3)}}}{{{number(b)}}} = {number(equilibrium_input)}$$\n\n"
        f"### 3. Residual Verification\n"
        f"$$\\dot{{x}} = -{number(a)}({number(reference)})^3 + {number(b)}({number(equilibrium_input)}) = {number(residual)}$$\n"
        f"**Summary:** u_eq = {number(equilibrium_input)}."
    )
    return record(
        index=index,
        prefix="nonlinear_feedforward",
        domain="nonlinear_control",
        family="nonlinear_equilibrium_feedforward_input",
        difficulty="intermediate",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "nonlinear_equilibrium_feedforward",
            "a": a,
            "b": b,
            "reference": reference,
            "u_eq": equilibrium_input,
            "residual": residual,
        },
        source_refs=["khalil_nonlinear_systems"],
        verifier="analytic_equilibrium_substitution",
    )


def robust_safe_interval(_: np.random.Generator, index: int) -> dict:
    a = 0.82 + 0.002 * index
    b = 0.65 + 0.004 * index
    state = -1.0 + 0.031 * index
    lower_state = -2.2 - 0.005 * index
    upper_state = 2.0 + 0.006 * index
    disturbance = 0.08 + 0.0015 * index
    nominal = -1.1 + 0.037 * index
    lower_u = (lower_state + disturbance - a * state) / b
    upper_u = (upper_state - disturbance - a * state) / b
    safe_u = float(np.clip(nominal, lower_u, upper_u))
    prompt = (
        f"{OPENERS[index % 4]} For uncertain discrete system x_next={number(a)}x+{number(b)}u+w with bounded disturbance "
        f"|w|<={number(disturbance)}, require x_next in [{number(lower_state)}, {number(upper_state)}] for all possible w. "
        f"At current state x={number(state)}, derive the robust admissible u interval and project nominal command u_nom={number(nominal)} onto it."
    )
    answer = (
        f"### 1. Robust Constraint Inversion under Worst-Case Disturbance\n"
        f"Since $x_{{next}} = a x + b u + w$, the safety condition $x_{{min}} \\le x_{{next}} \\le x_{{max}}$ requires:\n"
        f"- Worst-case lower bound ($w = -w_{{max}}$): $a x + b u - w_{{max}} \\ge x_{{min}} \\implies u \\ge \\frac{{x_{{min}} + w_{{max}} - a x}}{{b}} = {number(lower_u)}$\n"
        f"- Worst-case upper bound ($w = +w_{{max}}$): $a x + b u + w_{{max}} \\le x_{{max}} \\implies u \\le \\frac{{x_{{max}} - w_{{max}} - a x}}{{b}} = {number(upper_u)}$\n\n"
        f"### 2. Admissible Control Interval and Projection\n"
        f"Admissible interval: $u \\in [{number(lower_u)}, {number(upper_u)}]$.\n"
        f"Projecting $u_{{nom}} = {number(nominal)}$ onto this interval yields:\n"
        f"$$u^* = \\text{{clip}}({number(nominal)}, {number(lower_u)}, {number(upper_u)}) = {number(safe_u)}$$\n\n"
        f"**Summary:** Admissible interval = [{number(lower_u)}, {number(upper_u)}], safe control u = {number(safe_u)}."
    )
    return record(
        index=index,
        prefix="robust_safe_interval",
        domain="safety_critical_control",
        family="robust_one_step_safe_input_interval",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "robust_safe_input_interval",
            "a": a,
            "b": b,
            "x": state,
            "x_min": lower_state,
            "x_max": upper_state,
            "w_max": disturbance,
            "u_nom": nominal,
            "u_lower": lower_u,
            "u_upper": upper_u,
            "u_safe": safe_u,
        },
        source_refs=["ames_control_barrier_functions", "rawlings_mayne_diehl_mpc_2e"],
        verifier="robust_interval_projection",
    )


def actuator_loss_isolation(_: np.random.Generator, index: int) -> dict:
    effectiveness = np.array([1.0, 0.8 + 0.002 * index, 0.55 + 0.0015 * index])
    command = np.array([0.4 + 0.011 * index, -0.7 + 0.008 * index, 0.9 - 0.006 * index])
    loss_factor = 0.35 + 0.004 * (index % 20)
    failed = (index - 1) % 3
    candidate_outputs = []
    for candidate in range(3):
        candidate_effectiveness = effectiveness.copy()
        candidate_effectiveness[candidate] *= loss_factor
        candidate_outputs.append(float(candidate_effectiveness @ command))
    measured = candidate_outputs[failed]
    residuals = np.abs(np.asarray(candidate_outputs) - measured)
    identified = int(np.argmin(residuals))
    
    prompt = (
        f"{OPENERS[index % 4]} Nominal actuator effectiveness is B={effectiveness.tolist()} "
        f"and commanded input is u={command.tolist()}. Exactly one actuator retains effectiveness factor "
        f"lambda={number(loss_factor)} while the others remain nominal. The measured virtual control is tau={number(measured)}. "
        "Compute each single-fault hypothesis prediction/residual and isolate the failed actuator."
    )
    answer = (
        f"### 1. Single-Fault Hypothesis Testing\n"
        f"For each actuator $i \\in \\{{1, 2, 3\\}}$, compute predicted virtual control $\\tau_i = B^{{(i)}} u$ with $B^{{(i)}}_i = \\lambda B_i$:\n"
        f"- Hypothesis 1 (Actuator 1 degraded): $\\tau_1 = {number(candidate_outputs[0])}$, residual $r_1 = |\\tau_1 - \\tau| = {number(residuals[0])}$\n"
        f"- Hypothesis 2 (Actuator 2 degraded): $\\tau_2 = {number(candidate_outputs[1])}$, residual $r_2 = |\\tau_2 - \\tau| = {number(residuals[1])}$\n"
        f"- Hypothesis 3 (Actuator 3 degraded): $\\tau_3 = {number(candidate_outputs[2])}$, residual $r_3 = |\\tau_3 - \\tau| = {number(residuals[2])}$\n\n"
        f"### 2. Fault Isolation Decision\n"
        f"The minimum residual is $r_{{{identified + 1}}} = {number(residuals[identified])} \\approx 0$.\n\n"
        f"**Summary:** Actuator {identified + 1} is isolated as the degraded unit."
    )
    return record(
        index=index,
        prefix="actuator_loss",
        domain="fault_tolerant_control",
        family="single_actuator_loss_residual_isolation",
        difficulty="advanced",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "single_actuator_loss_isolation",
            "B": effectiveness.tolist(),
            "u": command.tolist(),
            "loss_factor": loss_factor,
            "measured": measured,
            "candidate_outputs": candidate_outputs,
            "residuals": residuals.tolist(),
            "failed_index": failed,
            "identified_index": identified,
        },
        source_refs=["fault_tolerant_control_allocation_literature"],
        verifier="single_fault_hypothesis_residuals",
    )


FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, int], dict]], ...] = (
    ("scalar_affine_nonlinear_dynamic_inversion", nonlinear_dynamic_inversion),
    ("linear_gain_schedule_interpolation", gain_schedule),
    ("scalar_control_barrier_safety_filter", cbf_filter),
    ("scalar_stationary_process_covariance", scalar_stationary_covariance),
    ("minimum_norm_redundant_control_allocation", fault_allocation),
    ("iid_scalar_jump_mean_square_test", jump_scalar),
    ("relative_state_error_event_trigger", event_trigger),
    ("symmetric_tube_mpc_constraint_tightening", tube_tightening),
    ("finite_discounted_control_return", discounted_return),
    ("nonlinear_equilibrium_feedforward_input", nonlinear_feedforward),
    ("robust_one_step_safe_input_interval", robust_safe_interval),
    ("single_actuator_loss_residual_isolation", actuator_loss_isolation),
)


def generate_extended_v1(count_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        generator(rng, index)
        for _, generator in FAMILIES
        for index in range(1, count_per_family + 1)
    ]
