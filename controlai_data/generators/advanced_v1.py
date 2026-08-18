"""Family-diverse, solver-backed advanced, nonlinear, robust, and adaptive SFT generators with Chain of Thought (CoT)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import linalg

from controlai_data.schema import make_record

OPENERS = (
    "Provide a rigorous engineering analysis for the following advanced control problem.",
    "Compute the requested quantities and state the decisive criterion.",
    "Work this nonlinear/robust control problem step-by-step with verified values.",
    "Derive the requested objects from the stated model; show intermediate equations.",
)
NONLINEAR = ["astrom_murray_feedback_systems_1e"]
ROBUST = ["boyd_lmi_system_control", "mit_ocw_6_245_multivariable_control"]
ADAPTIVE = ["astrom_murray_feedback_systems_1e"]
IDENTIFICATION = ["stanford_ee263_course_reader"]
NETWORKED = ["boyd_lmi_system_control"]


def n(x: float) -> str:
    if abs(x) < 5e-11:
        x = 0.0
    return f"{x:.6g}"


def mat(x: np.ndarray) -> str:
    return repr(np.asarray(x, dtype=float).tolist())


def roots(x: np.ndarray) -> list[list[float]]:
    x = np.asarray(x, dtype=complex)
    x = x[np.lexsort((x.imag, x.real))]
    return [[float(v.real), float(v.imag)] for v in x]


def cubic_equilibria(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.4 + 0.035 * index
    b = 0.7 + 0.025 * index
    r = math.sqrt(a / b)
    eq = [-r, 0.0, r]
    derivatives = [a - 3 * b * x * x for x in eq]
    labels = ["locally asymptotically stable" if d < 0 else "unstable" for d in derivatives]
    prompt = f"{OPENERS[index%4]} For x_dot={n(a)}x-{n(b)}x^3, find every equilibrium and classify each by linearization."
    eval_steps = "\n".join(
        f"- At $x = {n(x)}$: $f'({n(x)}) = {n(a)} - 3({n(b)})({n(x)})^2 = {n(d)}$ ({'< 0: stable' if d < 0 else '> 0: unstable'})"
        for x, d in zip(eq, derivatives)
    )
    answer = (
        f"### 1. Equilibrium Points Calculation\n"
        f"Setting $\\dot{{x}} = f(x) = {n(a)}x - {n(b)}x^3 = x({n(a)} - {n(b)}x^2) = 0$:\n"
        f"$$x_1 = -\\sqrt{{{n(a)}/{n(b)}}} = {n(-r)}, \\quad x_2 = 0, \\quad x_3 = \\sqrt{{{n(a)}/{n(b)}}} = {n(r)}$$\n\n"
        f"### 2. Linearization and Stability\n"
        f"The Jacobian derivative is $f'(x) = {n(a)} - 3({n(b)}) x^2 = {n(a)} - {n(3*b)} x^2$:\n"
        f"{eval_steps}\n\n"
        f"### 3. Conclusion\n"
        f"The equilibria are {list(map(float, eq))}. Their classifications are: {labels}."
    )
    return make_record(
        record_id=f"cubic_equilibria_{index:05d}",
        domain="nonlinear_control",
        family="scalar_cubic_multiple_equilibria",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"cubic_equilibria_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "cubic_equilibria",
            "a": a,
            "b": b,
            "equilibria": eq,
            "derivatives": derivatives,
            "stable": [d < 0 for d in derivatives],
        },
        source_refs=NONLINEAR,
        verifier="verify_cubic_equilibria",
    )


def nonlinear_jacobian(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.5 + 0.04 * index
    b = 0.8 + 0.03 * index
    J = np.array([[-a, 0], [1, -b]], float)
    poles = np.linalg.eigvals(J)
    prompt = (
        f"{OPENERS[index%4]} Linearize x1_dot=-{n(a)}x1+x2^2, x2_dot=-{n(b)}x2+sin(x1) at the origin. "
        "Report the Jacobian, eigenvalues, and local conclusion."
    )
    answer = (
        f"### 1. Jacobian Matrix at Origin $(0, 0)$\n"
        f"$$J = \\begin{{bmatrix}} \\frac{{\\partial f_1}}{{\\partial x_1}} & \\frac{{\\partial f_1}}{{\\partial x_2}} \\\\ \\frac{{\\partial f_2}}{{\\partial x_1}} & \\frac{{\\partial f_2}}{{\\partial x_2}} \\end{{bmatrix}}_{{(0,0)}} = \\begin{{bmatrix}} -{n(a)} & 2 x_2 \\\\ \\cos(x_1) & -{n(b)} \\end{{bmatrix}}_{{(0,0)}} = \\begin{{bmatrix}} -{n(a)} & 0 \\\\ 1 & -{n(b)} \\end{{bmatrix}} = {mat(J)}$$\n\n"
        f"### 2. Eigenvalues and Local Stability\n"
        f"Since $J$ is lower triangular, its eigenvalues are the diagonal entries:\n"
        f"$$\\lambda(J) = \\{{-{n(a)}, -{n(b)}\\}} = \\{{{', '.join(n(float(x.real)) for x in poles)}\\}}$$\n\n"
        f"### 3. Conclusion\n"
        f"Both eigenvalues have strictly negative real parts ($\\text{{Re}}(\\lambda_i) < 0$), so the origin is **locally exponentially stable**."
    )
    return make_record(
        record_id=f"nonlinear_jacobian_{index:05d}",
        domain="nonlinear_control",
        family="nonlinear_jacobian_local_stability",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"nonlinear_jacobian_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "nonlinear_jacobian",
            "a": a,
            "b": b,
            "J": J.tolist(),
            "poles": roots(poles),
            "locally_stable": True,
        },
        source_refs=NONLINEAR,
        verifier="verify_nonlinear_jacobian",
    )


def scalar_lyapunov(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.2 + 0.025 * index
    b = 0.4 + 0.03 * index
    prompt = (
        f"{OPENERS[index%4]} For x_dot=-{n(a)}x-{n(b)}x^3, use V=x^2/2 to compute V_dot and give the strongest global stability conclusion."
    )
    answer = (
        f"### 1. Lyapunov Candidate Function\n"
        f"$V(x) = \\frac{{1}}{{2}} x^2$ is positive definite ($V(x) > 0, \\forall x \\ne 0$, $V(0) = 0$) and radially unbounded ($V(x) \\to \\infty$ as $|x| \\to \\infty$).\n\n"
        f"### 2. Time Derivative Along Trajectories\n"
        f"$$\\dot{{V}}(x) = x \\dot{{x}} = x (-{n(a)}x - {n(b)}x^3) = -{n(a)}x^2 - {n(b)}x^4$$\n"
        f"Since $a = {n(a)} > 0$ and $b = {n(b)} > 0$, $\\dot{{V}}(x) < 0$ strictly for all $x \\ne 0$.\n\n"
        f"### 3. Conclusion\n"
        f"By Barbashin-Krasovskii theorem, the origin is **globally asymptotically stable** (and locally exponentially stable due to the linear term)."
    )
    return make_record(
        record_id=f"scalar_lyapunov_{index:05d}",
        domain="nonlinear_control",
        family="scalar_linear_cubic_lyapunov",
        task_type="derivation",
        difficulty="foundation",
        template_id=f"scalar_lyapunov_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "scalar_linear_cubic_lyapunov",
            "a": a,
            "b": b,
            "global_asymptotic": True,
            "local_exponential": True,
        },
        source_refs=NONLINEAR,
        verifier="verify_scalar_linear_cubic_lyapunov",
    )


def quadratic_lyapunov_2d(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.5 + 0.03 * index
    b = 1.0 + 0.04 * index
    p1 = 1.0 + 0.02 * index
    p2 = 0.7 + 0.015 * index
    A = np.diag([-a, -b])
    P = np.diag([p1, p2])
    Q = -(A.T @ P + P @ A)
    qe = np.linalg.eigvalsh(Q)
    prompt = (
        f"{OPENERS[index%4]} For x_dot=Ax with A={mat(A)}, test V=x^T P x using P={mat(P)}. "
        "Compute -V_dot=x^T Qx and verify definiteness."
    )
    answer = (
        f"### 1. Lyapunov Derivative Matrix $Q$\n"
        f"For $V(x) = x^T P x$, $\\dot{{V}}(x) = x^T (A^T P + P A) x = -x^T Q x$, where $Q = -(A^T P + P A)$:\n"
        f"$$Q = -\\begin{{bmatrix}} -2 a p_1 & 0 \\\\ 0 & -2 b p_2 \\end{{bmatrix}} = \\begin{{bmatrix}} 2({n(a)})({n(p1)}) & 0 \\\\ 0 & 2({n(b)})({n(p2)}) \\end{{bmatrix}} = {mat(Q)}$$\n\n"
        f"### 2. Definiteness Assessment\n"
        f"Eigenvalues of $Q$: $\\lambda(Q) = {list(map(float, qe))}$. Both are strictly positive, so $Q > 0$.\n\n"
        f"### 3. Conclusion\n"
        f"Since $P > 0$ and $Q > 0$, $\\dot{{V}} = -x^T Q x < 0$, proving the origin is **globally exponentially stable**."
    )
    return make_record(
        record_id=f"quadratic_lyapunov_2d_{index:05d}",
        domain="nonlinear_control",
        family="quadratic_lyapunov_matrix_derivative",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"quadratic_lyapunov_2d_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "quadratic_lyapunov",
            "A": A.tolist(),
            "P": P.tolist(),
            "Q": Q.tolist(),
            "Q_eigenvalues": qe.tolist(),
        },
        source_refs=NONLINEAR,
        verifier="verify_quadratic_lyapunov",
    )


def feedback_linearization(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.7 + 0.04 * index
    c0 = 0.3 + 0.025 * index
    b = 1.0 + 0.03 * index
    k1 = 1.5 + 0.05 * index
    k2 = 1.0 + 0.04 * index
    poles = np.roots([1, k2, k1])
    prompt = (
        f"{OPENERS[index%4]} For x1_dot=x2, x2_dot=-{n(a)}sin(x1)-{n(c0)}x2+{n(b)}u, y=x1, "
        f"derive an exact feedback-linearizing regulator giving y_ddot+{n(k2)}y_dot+{n(k1)}y=0."
    )
    answer = (
        f"### 1. Relative Degree and Input-Output Differentiation\n"
        f"- $\\dot{{y}} = \\dot{{x}}_1 = x_2$\n"
        f"- $\\ddot{{y}} = \\dot{{x}}_2 = -{n(a)} \\sin(x_1) - {n(c0)} x_2 + {n(b)} u$\n"
        f"Since $u$ appears explicitly in $\\ddot{{y}}$ and $b = {n(b)} \\ne 0$, the relative degree is $r = 2$ (full state).\n\n"
        f"### 2. Feedback Linearizing Control Law\n"
        f"Set $\\ddot{{y}} = v = -{n(k1)} y - {n(k2)} \\dot{{y}} = -{n(k1)} x_1 - {n(k2)} x_2$:\n"
        f"$$-{n(a)} \\sin(x_1) - {n(c0)} x_2 + {n(b)} u = -{n(k1)} x_1 - {n(k2)} x_2$$\n"
        f"$$u = \\frac{{{n(a)} \\sin(x_1) + {n(c0)} x_2 - {n(k1)} x_1 - {n(k2)} x_2}}{{{n(b)}}}$$\n\n"
        f"### 3. Closed-Loop Linear Error Dynamics\n"
        f"The resulting closed-loop error dynamics $\\ddot{{y}} + {n(k2)} \\dot{{y}} + {n(k1)} y = 0$ has poles:\n"
        f"$$\\text{{Poles}} = \\{{{', '.join(f'{x.real:.6g}{x.imag:+.6g}j' for x in poles)}\\}}$$"
    )
    return make_record(
        record_id=f"feedback_linearization_{index:05d}",
        domain="nonlinear_control",
        family="second_order_exact_feedback_linearization",
        task_type="derivation",
        difficulty="advanced",
        template_id=f"feedback_linearization_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "feedback_linearization",
            "a": a,
            "c": c0,
            "b": b,
            "k1": k1,
            "k2": k2,
            "closed_poles": roots(poles),
            "relative_degree": 2,
        },
        source_refs=NONLINEAR,
        verifier="verify_feedback_linearization",
    )


def zero_dynamics(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.5 + 0.035 * index
    c0 = 0.2 + 0.02 * index
    prompt = (
        f"{OPENERS[index%4]} Consider x1_dot=x2, x2_dot=u, x3_dot=-{n(a)}x3+{n(c0)}x1, y=x1. "
        "Find the relative degree and zero dynamics, then classify minimum phase."
    )
    answer = (
        f"### 1. Relative Degree\n"
        f"Differentiating output $y = x_1$:\n"
        f"- $\\dot{{y}} = x_2$\n"
        f"- $\\ddot{{y}} = u$\n"
        f"The control input $u$ appears in the 2nd derivative, so relative degree $r = 2$.\n\n"
        f"### 2. Zero Dynamics Derivation\n"
        f"Constraining output identically to zero: $y(t) = 0 \\implies x_1(t) = 0, \\dot{{y}}(t) = x_2(t) = 0, \\ddot{{y}}(t) = u(t) = 0$.\n"
        f"The remaining unobservable internal dynamics for state $x_3$ is:\n"
        f"$$\\dot{{x}}_3 = -{n(a)} x_3 + {n(c0)}(0) = -{n(a)} x_3$$\n\n"
        f"### 3. Minimum Phase Classification\n"
        f"The zero dynamics eigenvalue is $\\lambda = -{n(a)} < 0$ (strictly in LHP). Therefore the system is **minimum phase**."
    )
    return make_record(
        record_id=f"zero_dynamics_{index:05d}",
        domain="nonlinear_control",
        family="relative_degree_and_zero_dynamics",
        task_type="derivation",
        difficulty="advanced",
        template_id=f"zero_dynamics_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "zero_dynamics",
            "a": a,
            "c": c0,
            "relative_degree": 2,
            "zero_pole": -a,
            "minimum_phase": True,
        },
        source_refs=NONLINEAR,
        verifier="verify_zero_dynamics",
    )


def sliding_reachability(rng: np.random.Generator, index: int) -> dict:
    del rng
    dmax = 0.2 + 0.025 * index
    k = dmax + 0.3 + 0.01 * index
    eta = k - dmax
    prompt = (
        f"{OPENERS[index%4]} For s_dot=d(t)+u with |d(t)|<={n(dmax)}, use u=-{n(k)}sign(s). "
        "Establish a worst-case reaching inequality and state whether k is sufficient."
    )
    answer = (
        f"### 1. Sliding Mode Lyapunov Reachability Condition\n"
        f"Using Lyapunov candidate $V(s) = \\frac{{1}}{{2}} s^2$ with sliding manifold $s=0$:\n"
        f"$$\\dot{{V}} = s \\dot{{s}} = s (d(t) - {n(k)} \\text{{sgn}}(s)) = s d(t) - {n(k)} |s|$$\n\n"
        f"### 2. Worst-Case Bounding\n"
        f"Using $|s d(t)| \\le |s| d_{{\\max}} = {n(dmax)} |s|$:\n"
        f"$$\\dot{{V}} \\le |s| ({n(dmax)} - {n(k)}) = -{n(eta)} |s| = -\\eta \\sqrt{{2 V}}$$\n"
        f"Equivalently, $\\frac{{d|s|}}{{dt}} \\le -{n(eta)}$.\n\n"
        f"### 3. Conclusion\n"
        f"The manifold is reached in finite time $t_{{reach}} \\le \\frac{{|s(0)|}}{{{n(eta)}}}$. Since $k = {n(k)} > d_{{\\max}} = {n(dmax)}$, the gain $k$ is **sufficient**."
    )
    return make_record(
        record_id=f"sliding_reachability_{index:05d}",
        domain="nonlinear_control",
        family="scalar_sliding_mode_reaching_bound",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"sliding_reachability_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "sliding_reachability",
            "dmax": dmax,
            "k": k,
            "eta": eta,
            "sufficient": True,
        },
        source_refs=NONLINEAR,
        verifier="verify_sliding_reachability",
    )


def small_gain(rng: np.random.Generator, index: int) -> dict:
    del rng
    g1 = 0.25 + 0.018 * index
    g2 = 0.35 + 0.014 * index
    product = g1 * g2
    certified = product < 1
    prompt = (
        f"{OPENERS[index%4]} Two stable operators in feedback satisfy ||G1||<={n(g1)} and ||G2||<={n(g2)}. "
        "Apply the small-gain theorem and report the product and certification result."
    )
    answer = (
        f"### 1. Small Gain Theorem Criterion\n"
        f"For feedback interconnection of stable operators $G_1$ and $G_2$, closed-loop input-output stability is guaranteed if the loop gain product satisfies:\n"
        f"$$\\|G_1\\| \\cdot \\|G_2\\| < 1$$\n\n"
        f"### 2. Norm Product Evaluation\n"
        f"$$\\|G_1\\| \\cdot \\|G_2\\| = {n(g1)} \\times {n(g2)} = {n(product)}$$\n\n"
        f"### 3. Conclusion\n"
        f"Since the norm product is {n(product)} {'< 1' if certified else '>= 1'}, the small-gain theorem **{'certifies input-output stability' if certified else 'does not certify stability'}**."
    )
    return make_record(
        record_id=f"small_gain_{index:05d}",
        domain="robust_control",
        family="small_gain_norm_product_test",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"small_gain_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "small_gain",
            "g1": g1,
            "g2": g2,
            "product": product,
            "certified": certified,
        },
        source_refs=ROBUST,
        verifier="verify_small_gain",
    )


def multiplicative_uncertainty(rng: np.random.Generator, index: int) -> dict:
    del rng
    scale = 0.5 + 0.025 * index
    W = np.array([0.2, 0.45, 0.7, 0.9]) * scale
    T = np.array([0.8, 0.65, 0.4, 0.2]) * (1 + 0.005 * index)
    products = W * T
    peak = float(np.max(products))
    certified = peak < 1
    prompt = (
        f"{OPENERS[index%4]} For output-multiplicative uncertainty bounded at four frequencies by |W|={W.tolist()}, "
        f"nominal |T|={T.tolist()}, apply the sampled robust-stability check max |WT|<1."
    )
    answer = (
        f"### 1. Robust Stability Condition for Multiplicative Uncertainty\n"
        f"Robust stability requires $|W(j\\omega) T(j\\omega)| < 1$ for all frequencies $\\omega$.\n\n"
        f"### 2. Pointwise Evaluation\n"
        f"The pointwise products $|W| \\cdot |T|$ across the 4 frequency points are:\n"
        f"$${products.tolist()}$$\n"
        f"The peak value is $\\max |W T| = {n(peak)}$.\n\n"
        f"### 3. Conclusion\n"
        f"Since the sampled peak is {n(peak)} {'< 1' if certified else '>= 1'}, the robust-stability condition **{'is satisfied' if certified else 'is violated'}** on this frequency grid."
    )
    return make_record(
        record_id=f"multiplicative_uncertainty_{index:05d}",
        domain="robust_control",
        family="multiplicative_uncertainty_weighted_T_test",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"multiplicative_uncertainty_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "weighted_peak",
            "weight": W.tolist(),
            "response": T.tolist(),
            "products": products.tolist(),
            "peak": peak,
            "certified": certified,
            "strict_bound": 1.0,
        },
        source_refs=ROBUST,
        verifier="verify_weighted_peak",
    )


def additive_uncertainty(rng: np.random.Generator, index: int) -> dict:
    del rng
    Wa = np.array([0.1, 0.2, 0.35, 0.6]) * (0.8 + 0.02 * index)
    KS = np.array([0.7, 0.8, 0.9, 1.0]) * (0.9 + 0.004 * index)
    products = Wa * KS
    peak = float(np.max(products))
    certified = peak < 1
    prompt = (
        f"{OPENERS[index%4]} An additive plant uncertainty satisfies |Delta_a|<=|W_a|. On a frequency grid, "
        f"|W_a|={Wa.tolist()} and |KS|={KS.tolist()}. Check max |W_a K S|<1 and state the limitation."
    )
    answer = (
        f"### 1. Robust Stability Condition for Additive Uncertainty\n"
        f"For additive perturbation $|\\Delta_a| \\le |W_a|$, robust stability requires $|W_a(j\\omega) K(j\\omega) S(j\\omega)| < 1$.\n\n"
        f"### 2. Pointwise Evaluation\n"
        f"Pointwise products: {products.tolist()}\n"
        f"Peak product: $\\max |W_a KS| = {n(peak)}$.\n\n"
        f"### 3. Conclusion\n"
        f"The grid condition **{'passes' if certified else 'fails'}** the sufficient robust stability inequality."
    )
    return make_record(
        record_id=f"additive_uncertainty_{index:05d}",
        domain="robust_control",
        family="additive_uncertainty_weighted_KS_test",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"additive_uncertainty_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "weighted_peak",
            "weight": Wa.tolist(),
            "response": KS.tolist(),
            "products": products.tolist(),
            "peak": peak,
            "certified": certified,
            "strict_bound": 1.0,
        },
        source_refs=ROBUST,
        verifier="verify_weighted_peak",
    )


def hinf_first_order(rng: np.random.Generator, index: int) -> dict:
    del rng
    gain = (-1 if index % 7 == 0 else 1) * (0.4 + 0.06 * index)
    tau = 0.2 + 0.03 * index
    hinf = abs(gain)
    prompt = (
        f"{OPENERS[index%4]} Compute the continuous-time H-infinity norm of stable G(s)={n(gain)}/({n(tau)}s+1) "
        "and identify the frequency where the supremum occurs."
    )
    answer = (
        f"### 1. H-Infinity Norm Definition\n"
        f"$$\\|G(s)\\|_\\infty = \\sup_{{\\omega \\in \\mathbb{{R}}}} |G(j\\omega)|$$\n\n"
        f"### 2. Frequency Response Magnitude Analysis\n"
        f"$$|G(j\\omega)| = \\frac{{|{n(gain)}|}}{{\\sqrt{{1 + ({n(tau)} \\omega)^2}}}}$$\n"
        f"Since the denominator is strictly increasing with $|\\omega|$, the supremum occurs at DC ($\\omega = 0$ rad/s):\n"
        f"$$\\|G\\|_\\infty = |G(j0)| = |{n(gain)}| = {n(hinf)}$$\n\n"
        f"**Summary:** ||G||_infinity = {n(hinf)}, attained at omega = 0 rad/s."
    )
    return make_record(
        record_id=f"hinf_first_order_{index:05d}",
        domain="robust_control",
        family="first_order_hinfinity_norm",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"hinf_first_order_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "hinf_first_order",
            "gain": gain,
            "tau": tau,
            "hinf": hinf,
            "peak_frequency": 0.0,
        },
        source_refs=ROBUST,
        verifier="verify_hinf_first_order",
    )


def kharitonov_cubic(rng: np.random.Generator, index: int) -> dict:
    del rng
    lows = np.array([1.0 + 0.02 * index, 2.0 + 0.03 * index, 3.0 + 0.04 * index, 1.0])
    highs = lows + np.array([0.3, 0.4, 0.5, 0.2])
    polys = np.array([
        [lows[0], lows[1], highs[2], highs[3]],
        [highs[0], highs[1], lows[2], lows[3]],
        [highs[0], lows[1], lows[2], highs[3]],
        [lows[0], highs[1], highs[2], lows[3]],
    ])
    pole_sets = [np.roots(poly[::-1]) for poly in polys]
    stable_flags = [bool(np.all(np.real(p) < 0)) for p in pole_sets]
    robust = all(stable_flags)
    prompt = (
        f"{OPENERS[index%4]} A real cubic interval polynomial has ascending-power coefficient bounds a-={lows.tolist()}, "
        f"a+={highs.tolist()}. Form the four Kharitonov polynomials and test their Hurwitz stability."
    )
    
    k_steps = []
    for idx, (poly, poles, st) in enumerate(zip(polys, pole_sets, stable_flags), 1):
        desc = poly[::-1]
        hurwitz_test = desc[1] * desc[2] - desc[0] * desc[3]
        k_steps.append(
            f"- $K_{idx}(s) = {n(desc[0])}s^3 + {n(desc[1])}s^2 + {n(desc[2])}s + {n(desc[3])}$: "
            f"Hurwitz condition $a_2 a_1 - a_3 a_0 = {n(desc[1])}\\times{n(desc[2])} - {n(desc[0])}\\times{n(desc[3])} = {n(hurwitz_test)} "
            f"({'> 0: Hurwitz' if st else '<= 0: Unstable'})"
        )
    k_text = "\n".join(k_steps)
    
    answer = (
        f"### 1. Kharitonov Polynomial Formulation\n"
        f"For interval polynomial $p(s) = [a_0^-, a_0^+] + [a_1^-, a_1^+]s + [a_2^-, a_2^+]s^2 + [a_3^-, a_3^+]s^3$, "
        f"Kharitonov's theorem states the entire family is Hurwitz stable iff the 4 extreme polynomials are Hurwitz:\n\n"
        f"{k_text}\n\n"
        f"### 2. Conclusion\n"
        f"The four ascending coefficient vectors are {polys.tolist()}. Their Hurwitz flags are {stable_flags}; "
        f"therefore Kharitonov's theorem **{'certifies the whole interval family' if robust else 'does not certify the interval family as robustly Hurwitz'}**."
    )
    return make_record(
        record_id=f"kharitonov_cubic_{index:05d}",
        domain="robust_control",
        family="kharitonov_cubic_interval_stability",
        task_type="numerical",
        difficulty="advanced",
        template_id=f"kharitonov_cubic_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "kharitonov_cubic",
            "lower": lows.tolist(),
            "upper": highs.tolist(),
            "polynomials": polys.tolist(),
            "poles": [roots(x) for x in pole_sets],
            "stable_flags": stable_flags,
            "robustly_hurwitz": robust,
        },
        source_refs=ROBUST,
        verifier="verify_kharitonov_cubic",
    )


def sensitivity_peak(rng: np.random.Generator, index: int) -> dict:
    del rng
    real = np.array([1.5, 0.8, 0.2, -0.3]) * (1 + 0.01 * index)
    imag = np.array([-0.2, -0.6, -1.0, -0.4]) * (1 + 0.006 * index)
    L = real + 1j * imag
    S = 1 / (1 + L)
    mags = np.abs(S)
    peak = float(np.max(mags))
    idx = int(np.argmax(mags))
    prompt = (
        f"{OPENERS[index%4]} Loop samples are L={[[float(x.real),float(x.imag)] for x in L]}. "
        "Compute sampled |S|=|1/(1+L)|, its peak, and the peak sample index."
    )
    answer = (
        f"### 1. Sensitivity Calculation\n"
        f"$$S(j\\omega) = \\frac{{1}}{{1 + L(j\\omega)}}$$\n"
        f"Sampled magnitudes $|S|$: {mags.tolist()}\n\n"
        f"### 2. Peak Sensitivity\n"
        f"Peak magnitude: **{n(peak)}** at zero-based index **{idx}**."
    )
    return make_record(
        record_id=f"sensitivity_peak_{index:05d}",
        domain="robust_control",
        family="sampled_sensitivity_peak",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"sensitivity_peak_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "sampled_sensitivity_peak",
            "L": [[float(x.real), float(x.imag)] for x in L],
            "magnitudes": mags.tolist(),
            "peak": peak,
            "peak_index": idx,
        },
        source_refs=ROBUST,
        verifier="verify_sampled_sensitivity_peak",
    )


def pe_sincos(rng: np.random.Generator, index: int) -> dict:
    del rng
    omega = 0.5 + 0.05 * index
    T = 2 * math.pi / omega
    gram = (math.pi / omega) * np.eye(2)
    prompt = (
        f"{OPENERS[index%4]} For phi(t)=[sin({n(omega)}t), cos({n(omega)}t)]^T, integrate phi phi^T over one period and state a persistent-excitation lower bound."
    )
    answer = (
        f"### 1. Persistent Excitation (PE) Gramian Integration\n"
        f"Over one fundamental period $T = \\frac{{2\\pi}}{{\\omega}} = {n(T)}$ s:\n"
        f"$$\\int_0^T \\phi(t) \\phi^T(t) dt = \\int_0^T \\begin{{bmatrix}} \\sin^2(\\omega t) & \\sin(\\omega t)\\cos(\\omega t) \\\\ \\sin(\\omega t)\\cos(\\omega t) & \\cos^2(\\omega t) \\end{{bmatrix}} dt = \\begin{{bmatrix}} T/2 & 0 \\\\ 0 & T/2 \\end{{bmatrix}} = \\frac{{\\pi}}{{\\omega}} I = {mat(gram)}$$\n\n"
        f"### 2. Conclusion\n"
        f"The regressor is persistently exciting with period $T = {n(T)}$ and lower bound $\\alpha = {n(math.pi/omega)}$."
    )
    return make_record(
        record_id=f"pe_sincos_{index:05d}",
        domain="adaptive_control",
        family="sinusoidal_regressor_exact_pe_gramian",
        task_type="derivation",
        difficulty="intermediate",
        template_id=f"pe_sincos_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "pe_sincos",
            "omega": omega,
            "period": T,
            "gramian": gram.tolist(),
            "alpha_max": math.pi / omega,
        },
        source_refs=ADAPTIVE,
        verifier="verify_pe_sincos",
    )


def gradient_identifier(rng: np.random.Generator, index: int) -> dict:
    del rng
    theta = np.array([0.2 + 0.02 * index, -0.5 + 0.015 * index])
    phi = np.array([1.0, 0.3 + 0.01 * index])
    y = 0.8 + 0.025 * index
    gamma = 0.4 + 0.01 * index
    dt = 0.05 + 0.001 * index
    error = float(y - phi @ theta)
    next_theta = theta + dt * gamma * phi * error
    prompt = (
        f"{OPENERS[index%4]} Apply one Euler step of theta_dot=gamma phi(y-phi^T theta) using theta={theta.tolist()}, "
        f"phi={phi.tolist()}, y={n(y)}, gamma={n(gamma)}, dt={n(dt)}."
    )
    answer = (
        f"### 1. Parameter Estimation Error\n"
        f"$$e(t) = y - \\phi^T \\theta = {n(y)} - ({n(float(phi @ theta))}) = {n(error)}$$\n\n"
        f"### 2. Gradient Update Step (Euler Discretization)\n"
        f"$$\\theta_{{k+1}} = \\theta_k + \\Delta t \\cdot \\gamma \\phi_k e_k = {theta.tolist()} + {n(dt)} \\times {n(gamma)} \\times {phi.tolist()} \\times {n(error)} = {next_theta.tolist()}$$\n\n"
        f"**Summary:** theta_next = {next_theta.tolist()}."
    )
    return make_record(
        record_id=f"gradient_identifier_{index:05d}",
        domain="adaptive_control",
        family="gradient_parameter_update_euler",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"gradient_identifier_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "gradient_identifier",
            "theta": theta.tolist(),
            "phi": phi.tolist(),
            "y": y,
            "gamma": gamma,
            "dt": dt,
            "error": error,
            "theta_next": next_theta.tolist(),
        },
        source_refs=ADAPTIVE,
        verifier="verify_gradient_identifier",
    )


def normalized_identifier(rng: np.random.Generator, index: int) -> dict:
    del rng
    theta = np.array([0.1 + 0.03 * index, 0.4 + 0.02 * index])
    phi = np.array([1.0, 0.6 + 0.02 * index])
    y = 0.5 + 0.04 * index
    gamma = 0.3 + 0.01 * index
    dt = 0.04 + 0.002 * index
    denom = 1.0 + float(phi @ phi)
    error = float(y - phi @ theta)
    next_theta = theta + dt * (gamma * phi * error / denom)
    prompt = (
        f"{OPENERS[index%4]} Perform one Euler step of the normalized gradient identifier "
        f"theta_dot=gamma phi(y-phi^T theta)/(1+||phi||^2) with theta={theta.tolist()}, phi={phi.tolist()}, y={n(y)}, gamma={n(gamma)}, dt={n(dt)}."
    )
    answer = (
        f"### 1. Normalized Estimation Error\n"
        f"- Output error: $e = y - \\phi^T \\theta = {n(error)}$\n"
        f"- Normalization factor: $1 + \\|\\phi\\|^2 = 1 + {n(float(phi @ phi))} = {n(denom)}$\n\n"
        f"### 2. Parameter Update\n"
        f"$$\\theta_{{k+1}} = \\theta_k + \\Delta t \\frac{{\\gamma \\phi e}}{{1 + \\|\\phi\\|^2}} = {next_theta.tolist()}$$\n\n"
        f"**Summary:** theta_next = {next_theta.tolist()}."
    )
    return make_record(
        record_id=f"normalized_identifier_{index:05d}",
        domain="adaptive_control",
        family="normalized_gradient_parameter_update",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"normalized_identifier_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "normalized_identifier",
            "theta": theta.tolist(),
            "phi": phi.tolist(),
            "y": y,
            "gamma": gamma,
            "dt": dt,
            "denom": denom,
            "error": error,
            "theta_next": next_theta.tolist(),
        },
        source_refs=ADAPTIVE,
        verifier="verify_normalized_identifier",
    )


def regression_rank(rng: np.random.Generator, index: int) -> dict:
    del rng
    phi = np.array([[1.0, 0.5 + 0.05 * index], [1.0, 1.0 + 0.05 * index], [1.0, 1.5 + 0.05 * index]])
    G = phi.T @ phi
    rank = int(np.linalg.matrix_rank(G))
    cond = float(np.linalg.cond(G))
    prompt = (
        f"{OPENERS[index%4]} For regressor matrix Phi={mat(phi)}, compute the Gram matrix G=Phi^T Phi, "
        "its rank, and condition number; confirm parameter identifiability."
    )
    answer = (
        f"### 1. Regressor Gram Matrix\n"
        f"$$G = \\Phi^T \\Phi = {mat(G)}$$\n\n"
        f"### 2. Rank and Conditioning\n"
        f"- $\\text{{rank}}(G) = {rank}$ out of 2 (full rank).\n"
        f"- Condition number: $\\kappa(G) = {n(cond)}$.\n\n"
        f"### 3. Conclusion\n"
        f"Since $G$ is full rank, the parameters are **uniquely identifiable** by ordinary least squares."
    )
    return make_record(
        record_id=f"regression_rank_{index:05d}",
        domain="system_identification",
        family="regressor_matrix_identifiability_rank",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"regression_rank_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "regression_rank",
            "phi": phi.tolist(),
            "G": G.tolist(),
            "rank": rank,
            "cond": cond,
            "identifiable": rank == 2,
        },
        source_refs=IDENTIFICATION,
        verifier="verify_regression_rank",
    )


def arx_least_squares(rng: np.random.Generator, index: int) -> dict:
    del rng
    Phi = np.array([[-0.5, 1.0 + 0.02 * index], [-0.8, 1.2 + 0.02 * index], [-0.9, 1.1 + 0.02 * index]])
    Y = np.array([[0.2 + 0.03 * index], [0.35 + 0.02 * index], [0.4 + 0.02 * index]])
    theta = np.linalg.solve(Phi.T @ Phi, Phi.T @ Y)
    residual = Y - Phi @ theta
    res_norm = float(np.linalg.norm(residual))
    prompt = (
        f"{OPENERS[index%4]} Solve the batch least-squares ARX identification problem for Phi={mat(Phi)} "
        f"and Y={mat(Y)}. Report parameter estimate theta_hat=(Phi^T Phi)^-1 Phi^T Y and residual 2-norm."
    )
    answer = (
        f"### 1. Least Squares Normal Equations\n"
        f"$$\\hat{{\\theta}} = (\\Phi^T \\Phi)^{{-1}} \\Phi^T Y = {mat(theta)}$$\n\n"
        f"### 2. Residual Vector and Norm\n"
        f"$$r = Y - \\Phi \\hat{{\\theta}} = {mat(residual)}$$\n"
        f"$$\\|r\\|_2 = {n(res_norm)}$$\n\n"
        f"**Summary:** theta_hat = {mat(theta)}, residual norm = {n(res_norm)}."
    )
    return make_record(
        record_id=f"arx_least_squares_{index:05d}",
        domain="system_identification",
        family="noise_free_arx_least_squares",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"arx_least_squares_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "arx_least_squares",
            "Phi": Phi.tolist(),
            "Y": Y.tolist(),
            "theta": theta.tolist(),
            "residual_norm": res_norm,
        },
        source_refs=IDENTIFICATION,
        verifier="verify_arx_least_squares",
    )


def fopdt_step(rng: np.random.Generator, index: int) -> dict:
    del rng
    K = 1.2 + 0.05 * index
    theta = 0.5 + 0.04 * index
    tau = 1.0 + 0.08 * index
    y_final = K
    t_28 = theta + 0.356 * tau
    t_63 = theta + tau
    prompt = (
        f"{OPENERS[index%4]} For a First-Order Plus Dead Time (FOPDT) model with K={n(K)}, delay theta={n(theta)} s, "
        f"tau={n(tau)} s under unit step, find final value, t_28 (28.3% output), and t_63 (63.2% output)."
    )
    answer = (
        f"### 1. FOPDT Step Response Formulas\n"
        f"- Steady-state final value: $y_{{final}} = K = {n(y_final)}$\n"
        f"- Time to 28.3% response: $t_{{28}} = \\theta + 0.356 \\tau = {n(theta)} + 0.356({n(tau)}) = {n(t_28)}\\text{{ s}}$\n"
        f"- Time to 63.2% response: $t_{{63}} = \\theta + \\tau = {n(theta)} + {n(tau)} = {n(t_63)}\\text{{ s}}$\n\n"
        f"**Summary:** y_final = {n(y_final)}, t_28 = {n(t_28)} s, t_63 = {n(t_63)} s."
    )
    return make_record(
        record_id=f"fopdt_step_{index:05d}",
        domain="system_identification",
        family="fopdt_two_point_step_identification",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"fopdt_step_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "fopdt_step",
            "K": K,
            "theta": theta,
            "tau": tau,
            "y_final": y_final,
            "t_28": t_28,
            "t_63": t_63,
        },
        source_refs=IDENTIFICATION,
        verifier="verify_fopdt_step",
    )


def residual_autocorrelation(rng: np.random.Generator, index: int) -> dict:
    del rng
    e = np.array([0.7 + 0.05 * index, 0.4 + 0.03 * index, -0.2, -0.5 - 0.02 * index, 0.1])
    n_pts = len(e)
    r0 = float(np.sum(e * e) / n_pts)
    r1 = float(np.sum(e[:-1] * e[1:]) / n_pts)
    rho1 = r1 / r0 if r0 > 1e-12 else 0.0
    prompt = (
        f"{OPENERS[index%4]} For identification residual sequence e={e.tolist()}, "
        "compute sample variance r(0), lag-1 autocovariance r(1), and normalized lag-1 correlation rho(1)=r(1)/r(0)."
    )
    answer = (
        f"### 1. Autocovariance and Correlation Calculations\n"
        f"- Sample variance $r(0) = \\frac{{1}}{{N}} \\sum e_k^2 = {n(r0)}$\n"
        f"- Lag-1 autocovariance $r(1) = \\frac{{1}}{{N}} \\sum e_k e_{{k+1}} = {n(r1)}$\n"
        f"- Normalized lag-1 autocorrelation $\\rho(1) = \\frac{{r(1)}}{{r(0)}} = {n(rho1)}$\n\n"
        f"**Summary:** r(0) = {n(r0)}, r(1) = {n(r1)}, rho(1) = {n(rho1)}."
    )
    return make_record(
        record_id=f"residual_autocorrelation_{index:05d}",
        domain="system_identification",
        family="residual_normalized_autocorrelation",
        task_type="numerical",
        difficulty="foundation",
        template_id=f"residual_autocorrelation_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "residual_autocorrelation",
            "e": e.tolist(),
            "r0": r0,
            "r1": r1,
            "rho1": rho1,
        },
        source_refs=IDENTIFICATION,
        verifier="verify_residual_autocorrelation",
    )


def path_consensus(rng: np.random.Generator, index: int) -> dict:
    del rng
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    eig = np.sort(np.linalg.eigvalsh(L))
    lambda2 = float(eig[1])
    prompt = (
        f"{OPENERS[index%4]} For a 3-node path graph with Laplacian L={mat(L)}, compute its eigenvalues, "
        "identify the algebraic connectivity lambda2, and state consensus convergence rate."
    )
    answer = (
        f"### 1. Graph Laplacian Eigenvalues\n"
        f"$$\\lambda(L) = {list(map(float, eig))}$$\n\n"
        f"### 2. Algebraic Connectivity (Fiedler Eigenvalue)\n"
        f"$$\\lambda_2(L) = {n(lambda2)}$$\n\n"
        f"### 3. Conclusion\n"
        f"Since the graph is connected, $\\lambda_1 = 0$ and $\\lambda_2 = {n(lambda2)} > 0$. "
        f"The continuous consensus dynamics $\\dot{{x}} = -L x$ converges exponentially to the average consensus at asymptotic rate $\\lambda_2 = {n(lambda2)}$."
    )
    return make_record(
        record_id=f"path_consensus_{index:05d}",
        domain="networked_control",
        family="path_graph_laplacian_algebraic_connectivity",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"path_consensus_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "path_consensus",
            "L": L.tolist(),
            "eigenvalues": eig.tolist(),
            "lambda2": lambda2,
        },
        source_refs=NETWORKED,
        verifier="verify_path_consensus",
    )


def discrete_consensus(rng: np.random.Generator, index: int) -> dict:
    del rng
    eps = 0.2 + 0.02 * (index % 5)
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    P = np.eye(3) - eps * L
    eig = np.sort(np.linalg.eigvalsh(P))[::-1]
    sec = float(eig[1])
    prompt = (
        f"{OPENERS[index%4]} For step size epsilon={n(eps)} and 3-node path Laplacian L={mat(L)}, "
        "form Perron matrix P=I-epsilon L, find its eigenvalues, and report the second largest eigenvalue module."
    )
    answer = (
        f"### 1. Perron Matrix Construction\n"
        f"$$P = I - \\epsilon L = {mat(P)}$$\n\n"
        f"### 2. Eigenvalue Spectrum\n"
        f"$$\\lambda(P) = {list(map(float, eig))}$$\n\n"
        f"### 3. Convergence Factor\n"
        f"The second largest eigenvalue module is $\\lambda_2(P) = {n(sec)} < 1$, ensuring geometric discrete consensus convergence."
    )
    return make_record(
        record_id=f"discrete_consensus_{index:05d}",
        domain="networked_control",
        family="discrete_cycle_consensus_step_size",
        task_type="numerical",
        difficulty="intermediate",
        template_id=f"discrete_consensus_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "discrete_consensus",
            "eps": eps,
            "L": L.tolist(),
            "P": P.tolist(),
            "eigenvalues": eig.tolist(),
            "second_eigenvalue": sec,
        },
        source_refs=NETWORKED,
        verifier="verify_discrete_consensus",
    )


FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, int], dict]], ...] = (
    ("scalar_cubic_multiple_equilibria", cubic_equilibria),
    ("nonlinear_jacobian_local_stability", nonlinear_jacobian),
    ("scalar_linear_cubic_lyapunov", scalar_lyapunov),
    ("quadratic_lyapunov_matrix_derivative", quadratic_lyapunov_2d),
    ("second_order_exact_feedback_linearization", feedback_linearization),
    ("relative_degree_and_zero_dynamics", zero_dynamics),
    ("scalar_sliding_mode_reaching_bound", sliding_reachability),
    ("small_gain_norm_product_test", small_gain),
    ("multiplicative_uncertainty_weighted_T_test", multiplicative_uncertainty),
    ("additive_uncertainty_weighted_KS_test", additive_uncertainty),
    ("first_order_hinfinity_norm", hinf_first_order),
    ("kharitonov_cubic_interval_stability", kharitonov_cubic),
    ("sampled_sensitivity_peak", sensitivity_peak),
    ("sinusoidal_regressor_exact_pe_gramian", pe_sincos),
    ("gradient_parameter_update_euler", gradient_identifier),
    ("normalized_gradient_parameter_update", normalized_identifier),
    ("regressor_matrix_identifiability_rank", regression_rank),
    ("noise_free_arx_least_squares", arx_least_squares),
    ("fopdt_two_point_step_identification", fopdt_step),
    ("residual_normalized_autocorrelation", residual_autocorrelation),
    ("path_graph_laplacian_algebraic_connectivity", path_consensus),
    ("discrete_cycle_consensus_step_size", discrete_consensus),
)


def generate_advanced_v1(count_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        generator(rng, index)
        for _, generator in FAMILIES
        for index in range(1, count_per_family + 1)
    ]
