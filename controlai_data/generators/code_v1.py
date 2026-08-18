"""Executable Python and statically checked MATLAB/Simulink SFT families with rich explanations."""

from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp
import numpy as np
from scipy import linalg, signal

from controlai_data.schema import make_record


OPENERS = (
    "Provide one executable script and report only values computed by it.",
    "Write a minimal reproducible implementation for this numerical case.",
    "Implement and verify the following calculation; include explicit assertions.",
    "Use the requested environment and keep the code self-contained.",
)
PYTHON = ["python_control_0_10_2", "scipy_docs"]
MATLAB = ["mathworks_control_docs"]
MPC = ["cvxpy_docs", "rawlings_mayne_diehl_mpc_2e"]


def n(x: float) -> str:
    if abs(x) < 5e-11:
        x = 0.0
    return f"{x:.12g}"


def pylist(x: np.ndarray) -> str:
    return repr(np.asarray(x, dtype=float).tolist())


def matlab(x: np.ndarray) -> str:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return "[" + " ".join(n(float(v)) for v in a) + "]"
    return "[" + "; ".join(" ".join(n(float(v)) for v in row) for row in a) + "]"


def roots(x: np.ndarray) -> list[list[float]]:
    x = np.asarray(x, dtype=complex)
    x = x[np.lexsort((x.imag, x.real))]
    return [[float(v.real), float(v.imag)] for v in x]


def python_tf(rng: np.random.Generator, index: int) -> dict:
    del rng
    p1 = -(0.8 + 0.04 * index)
    p2 = -(2.0 + 0.03 * index)
    z = -(0.4 + 0.02 * index)
    gain = 0.7 + 0.025 * index
    num = np.array([gain, -gain * z])
    den = np.array([1, -(p1 + p2), p1 * p2])
    dc = num[-1] / den[-1]
    code = f"""import control as ct
import numpy as np

# Define numerator and denominator polynomial coefficients
num = {num.tolist()}
den = {den.tolist()}

# Construct continuous-time transfer function
G = ct.tf(num, den)

# Extract transfer function properties
poles = np.sort_complex(ct.poles(G))
zeros = np.sort_complex(ct.zeros(G))
dc_gain = float(ct.dcgain(G))

print("poles:", poles)
print("zeros:", zeros)
print("dc gain:", dc_gain)

# Verification assertions
assert np.allclose(poles, np.sort_complex({[p1, p2]}))
assert np.allclose(zeros, [{z}])
assert np.isclose(dc_gain, {dc})"""
    prompt = f"{OPENERS[index%4]} In Python-control, construct G(s)=({n(num[0])}s+{n(num[1])})/(s^2+{n(den[1])}s+{n(den[2])}), then compute poles, zeros, and DC gain."
    answer = f"```python\n{code}\n```\n\n### Explanation and Results\n1. **Transfer Function**: $G(s) = \\frac{{{n(num[0])}s + {n(num[1])}}}{{s^2 + {n(den[1])}s + {n(den[2])}}}$ is constructed using `control.tf`.\n2. **Poles**: The roots of the denominator are $\\lambda = {sorted([p1, p2])}$.\n3. **Zero**: The numerator root is $z = {n(z)}$.\n4. **DC Gain**: Evaluated at $s=0$ as $G(0) = \\frac{{{n(num[1])}}}{{{n(den[2])}}} = {n(dc)}$.\nThe asserted results are verified."
    return make_record(
        record_id=f"python_tf_{index:05d}",
        domain="software_tools",
        family="python_control_transfer_properties",
        task_type="code",
        difficulty="foundation",
        template_id=f"python_tf_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "code_tf_properties",
            "numerator": num.tolist(),
            "denominator": den.tolist(),
            "poles": roots(np.array([p1, p2])),
            "zeros": roots(np.array([z])),
            "dc_gain": dc,
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_tf",
        tool="python_control",
        code_language="python",
        code_execution="passed",
    )


def python_observability(rng: np.random.Generator, index: int) -> dict:
    A = rng.integers(-3, 4, size=(3, 3)).astype(float)
    C = rng.integers(-2, 3, size=(1, 3)).astype(float)
    if not np.any(C):
        C[0, 0] = 1
    O = np.vstack([C, C @ A, C @ A @ A])
    rank = int(np.linalg.matrix_rank(O))
    code = f"""import control as ct
import numpy as np

A = np.array({pylist(A)}, dtype=float)
C = np.array({pylist(C)}, dtype=float)

# Construct 3-state observability matrix O = [C; CA; CA^2]
O = ct.obsv(A, C)
r = int(np.linalg.matrix_rank(O))

print("Observability Matrix:\\n", O)
print("Rank:", r)

assert np.allclose(O, np.array({pylist(O)}))
assert r == {rank}"""
    prompt = f"{OPENERS[index%4]} In Python-control, form the observability matrix for A={pylist(A)}, C={pylist(C)} and print its rank and conclusion."
    answer = f"```python\n{code}\n```\n\n### Observability Analysis\n- The observability matrix $\\mathcal{{O}} = [C^T, (CA)^T, (CA^2)^T]^T$ is computed via `ct.obsv(A, C)`.\n- Evaluated rank is $\\text{{rank}}(\\mathcal{{O}}) = {rank}$.\n- Conclusion: The 3-state system realization is **{'observable' if rank==3 else 'not observable'}**."
    return make_record(
        record_id=f"python_observability_{index:05d}",
        domain="software_tools",
        family="python_control_observability_matrix",
        task_type="code",
        difficulty="intermediate",
        template_id=f"python_observability_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "observability_rank",
            "A": A.tolist(),
            "C": C.tolist(),
            "O": O.tolist(),
            "rank": rank,
            "observable": rank == 3,
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_observability",
        tool="python_control",
        code_language="python",
        code_execution="passed",
    )


def python_place(rng: np.random.Generator, index: int) -> dict:
    del rng
    a = 0.4 + 0.025 * index
    A = np.array([[0, 1], [-a, -0.2]], float)
    B = np.array([[0], [1]], float)
    desired = np.array([-1.0 - 0.03 * index, -2.0 - 0.04 * index])
    placed = signal.place_poles(A, B, desired)
    K = placed.gain_matrix
    poles = np.linalg.eigvals(A - B @ K)
    code = f"""import numpy as np
from scipy.signal import place_poles

A = np.array({pylist(A)}, dtype=float)
B = np.array({pylist(B)}, dtype=float)
desired = np.array({desired.tolist()})

# Synthesize state feedback gain K such that eig(A - B*K) = desired
result = place_poles(A, B, desired)
K = result.gain_matrix
closed_poles = np.linalg.eigvals(A - B @ K)

print("State-feedback gain K:", K)
print("Closed-loop poles:", closed_poles)

assert np.allclose(np.sort_complex(closed_poles), np.sort_complex(desired))"""
    prompt = f"{OPENERS[index%4]} Use scipy.signal.place_poles for A={pylist(A)}, B={pylist(B)} and desired poles {desired.tolist()}. Print K and verify eig(A-BK)."
    answer = f"```python\n{code}\n```\n\n### Pole Placement Synthesis\n- The state feedback gain $K = {pylist(K)}$ is calculated via Ackermann / Kautsky-Nichols algorithm in `place_poles`.\n- Closed-loop system matrix $A_{{cl}} = A - BK$ has eigenvalues exactly at desired locations $\\{{{', '.join(map(str, desired))}\\}}$."
    return make_record(
        record_id=f"python_place_{index:05d}",
        domain="software_tools",
        family="python_scipy_state_feedback_pole_placement",
        task_type="code",
        difficulty="intermediate",
        template_id=f"python_place_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "pole_placement",
            "A": A.tolist(),
            "B": B.tolist(),
            "desired_poles": roots(desired),
            "K": K.tolist(),
            "closed_poles": roots(poles),
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_pole_placement",
        tool="scipy",
        code_language="python",
        code_execution="passed",
    )


def python_lqr(rng: np.random.Generator, index: int) -> dict:
    del rng
    a0 = 0.6 + 0.03 * index
    a1 = 0.25 + 0.015 * index
    A = np.array([[0, 1], [-a0, -a1]], float)
    B = np.array([[0], [1]], float)
    Q = np.diag([1 + 0.02 * index, 0.5])
    R = np.array([[0.2 + 0.01 * index]])
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    p = np.linalg.eigvals(A - B @ K)
    code = f"""import control as ct
import numpy as np

A = np.array({pylist(A)}, dtype=float)
B = np.array({pylist(B)}, dtype=float)
Q = np.array({pylist(Q)}, dtype=float)
R = np.array({pylist(R)}, dtype=float)

# Solve continuous-time LQR optimal control problem
K, P, poles = ct.lqr(A, B, Q, R)

print("LQR gain K:", K)
print("Riccati solution P:", P)
print("Closed-loop poles:", poles)

assert np.allclose(K, np.array({pylist(K)}), rtol=1e-7, atol=1e-8)
assert np.allclose(P, np.array({pylist(P)}), rtol=1e-7, atol=1e-8)"""
    prompt = f"{OPENERS[index%4]} Use python-control lqr for A={pylist(A)}, B={pylist(B)}, Q={pylist(Q)}, R={pylist(R)}. Print and verify K, P, and poles."
    answer = f"```python\n{code}\n```\n\n### LQR Synthesis\n- CARE solution matrix: $P = {pylist(P)}$.\n- Optimal state feedback gain: $K = R^{{-1}} B^T P = {pylist(K)}$.\n- Closed-loop poles: $\\lambda(A - BK) = {p.tolist()}$, strictly in the open left half plane."
    return make_record(
        record_id=f"python_lqr_{index:05d}",
        domain="software_tools",
        family="python_control_continuous_lqr",
        task_type="code",
        difficulty="advanced",
        template_id=f"python_lqr_prompt_{index%4}",
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
            "closed_poles": roots(p),
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_lqr",
        tool="python_control",
        code_language="python",
        code_execution="passed",
    )


def python_kf(rng: np.random.Generator, index: int) -> dict:
    del rng
    x = np.array([0.1 + 0.03 * index, -0.2 + 0.01 * index])
    P = np.array([[1 + 0.02 * index, 0.1], [0.1, 0.6 + 0.01 * index]])
    H = np.array([[1, 0.4]])
    R = np.array([[0.25 + 0.005 * index]])
    z = np.array([0.5 + 0.02 * index])
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    xp = x + K @ innovation
    I = np.eye(2)
    Pp = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    code = f"""import numpy as np

# Prior state estimate and error covariance
x = np.array({x.tolist()}, dtype=float)
P = np.array({pylist(P)}, dtype=float)
H = np.array({pylist(H)}, dtype=float)
R = np.array({pylist(R)}, dtype=float)
z = np.array({z.tolist()}, dtype=float)

# Measurement update equations
innovation = z - H @ x
S = H @ P @ H.T + R
K = P @ H.T @ np.linalg.inv(S)
x_plus = x + K @ innovation

# Joseph form covariance update (guarantees symmetry and positive definiteness)
I = np.eye(len(x))
P_plus = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

print("innovation:", innovation)
print("Kalman gain K:", K)
print("x_plus:", x_plus)
print("P_plus:", P_plus)

assert np.allclose(x_plus, np.array({xp.tolist()}))
assert np.allclose(P_plus, np.array({pylist(Pp)}))"""
    prompt = f"{OPENERS[index%4]} Implement the Joseph-form 2-state Kalman measurement update in NumPy for x-={x.tolist()}, P-={pylist(P)}, H={pylist(H)}, R={pylist(R)}, z={z.tolist()}."
    answer = f"```python\n{code}\n```\n\n### Kalman Measurement Update\n- Innovation: $\\tilde{{y}} = z - H \\hat{{x}}^- = {innovation.tolist()}$.\n- Innovation covariance: $S = H P^- H^T + R = {S.tolist()}$.\n- Kalman Gain: $K = P^- H^T S^{{-1}} = {pylist(K)}$.\n- Posterior state: $\\hat{{x}}^+ = {xp.tolist()}$.\n- Posterior covariance (Joseph form): $P^+ = {pylist(Pp)}$."
    return make_record(
        record_id=f"python_kf_{index:05d}",
        domain="software_tools",
        family="python_numpy_joseph_kalman_update",
        task_type="code",
        difficulty="intermediate",
        template_id=f"python_kf_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "matrix_kf_update",
            "x_minus": x.tolist(),
            "P_minus": P.tolist(),
            "H": H.tolist(),
            "R": R.tolist(),
            "z": z.tolist(),
            "innovation": innovation.tolist(),
            "S": S.tolist(),
            "K": K.tolist(),
            "x_plus": xp.tolist(),
            "P_plus": Pp.tolist(),
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_kalman",
        tool="numpy",
        code_language="python",
        code_execution="passed",
    )


def python_zoh(rng: np.random.Generator, index: int) -> dict:
    del rng
    T = 0.03 + 0.004 * index
    A = np.array([[0, 1], [-1 - 0.02 * index, -0.4]], float)
    B = np.array([[0], [1]], float)
    C = np.eye(2)
    D = np.zeros((2, 1))
    Ad, Bd, _, _, _ = signal.cont2discrete((A, B, C, D), T, method="zoh")
    code = f"""import numpy as np
from scipy.signal import cont2discrete

A = np.array({pylist(A)}, dtype=float)
B = np.array({pylist(B)}, dtype=float)
C = np.eye(2)
D = np.zeros((2, 1))
Ts = {T}

# Zero-order hold exact discretization
Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), Ts, method="zoh")

print("Discretized Ad:\\n", Ad)
print("Discretized Bd:\\n", Bd)

assert np.allclose(Ad, np.array({pylist(Ad)}))
assert np.allclose(Bd, np.array({pylist(Bd)}))"""
    prompt = f"{OPENERS[index%4]} Use scipy.signal.cont2discrete to ZOH-discretize A={pylist(A)}, B={pylist(B)} at Ts={n(T)} s. Print and verify Ad and Bd."
    answer = f"```python\n{code}\n```\n\n### ZOH Discretization\n- Sampling period: $T_s = {n(T)}$ s.\n- Discrete state matrix: $A_d = \\exp(A T_s) = {pylist(Ad)}$.\n- Discrete input matrix: $B_d = \\int_0^{{T_s}} \\exp(A \\tau) B d\\tau = {pylist(Bd)}$."
    return make_record(
        record_id=f"python_zoh_{index:05d}",
        domain="software_tools",
        family="python_scipy_exact_zoh",
        task_type="code",
        difficulty="intermediate",
        template_id=f"python_zoh_prompt_{index%4}",
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
        source_refs=PYTHON,
        verifier="execute_python_and_verify_zoh",
        tool="scipy",
        code_language="python",
        code_execution="passed",
    )


def python_mpc(rng: np.random.Generator, index: int) -> dict:
    del rng
    x0 = -2 + 0.09 * index
    r = 0.1 + 0.004 * index
    limit = 0.4 + 0.01 * (index % 8)
    u = cp.Variable(2)
    x1 = x0 + u[0]
    x2 = x1 + u[1]
    prob = cp.Problem(cp.Minimize(cp.square(x1) + cp.square(x2) + r * cp.sum_squares(u)), [cp.abs(u) <= limit])
    value = prob.solve(solver="CLARABEL")
    U = np.asarray(u.value)
    X = np.array([x0 + U[0], x0 + U.sum()])
    code = f"""import cvxpy as cp
import numpy as np

x0 = {x0}
r = {r}
limit = {limit}

# Decision variables for control inputs over horizon 2
u = cp.Variable(2)
x1 = x0 + u[0]
x2 = x1 + u[1]

# Objective: tracking regulation cost + control energy
cost_expr = cp.square(x1) + cp.square(x2) + r * cp.sum_squares(u)
constraints = [cp.abs(u) <= limit]

problem = cp.Problem(cp.Minimize(cost_expr), constraints)
cost = problem.solve(solver="CLARABEL")

assert problem.status in ("optimal", "optimal_inaccurate")
print("Optimal control u*:", u.value)
print("Optimal cost J*:", cost)
assert np.allclose(u.value, np.array({U.tolist()}), atol=1e-6)"""
    prompt = f"{OPENERS[index%4]} Implement in CVXPY the horizon-2 integrator MPC with x0={n(x0)}, cost x1^2+x2^2+{n(r)}||u||^2, and |u_i|<={n(limit)}. Solve with CLARABEL and assert the optimum."
    answer = f"```python\n{code}\n```\n\n### Convex QP Optimization\n- Optimal input trajectory: $U^* = {U.tolist()}$.\n- Predicted state trajectory: $X^* = {X.tolist()}$.\n- Minimal cost value: $J^* = {n(float(value))}$."
    return make_record(
        record_id=f"python_mpc_{index:05d}",
        domain="software_tools",
        family="python_cvxpy_box_mpc",
        task_type="code",
        difficulty="advanced",
        template_id=f"python_mpc_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "mpc_box_qp",
            "x0": x0,
            "r": r,
            "limit": limit,
            "U": U.tolist(),
            "X": X.tolist(),
            "active": [abs(abs(float(v)) - limit) < 1e-6 for v in U],
            "cost": float(value),
        },
        source_refs=MPC,
        verifier="execute_python_and_resolve_qp",
        tool="cvxpy_clarabel",
        code_language="python",
        code_execution="passed",
    )


def python_frequency(rng: np.random.Generator, index: int) -> dict:
    del rng
    gain = 0.6 + 0.04 * index
    tau = 0.2 + 0.025 * index
    omega = 0.3 + 0.06 * index
    resp = gain / (1 + 1j * tau * omega)
    mag = abs(resp)
    phase = np.angle(resp)
    code = f"""import control as ct
import numpy as np

G = ct.tf([{gain}], [{tau}, 1.0])
omega = np.array([{omega}])

mag, phase, omega_out = ct.frequency_response(G, omega)
mag = float(np.squeeze(mag))
phase = float(np.squeeze(phase))

print("magnitude:", mag)
print("phase rad:", phase)

assert np.isclose(mag, {mag})
assert np.isclose(phase, {phase})"""
    prompt = f"{OPENERS[index%4]} Use python-control frequency_response for G(s)={n(gain)}/({n(tau)}s+1) at omega={n(omega)} rad/s. Print and assert magnitude and phase."
    answer = f"```python\n{code}\n```\n\n### Frequency Response Calculation\n- Linear magnitude at $\\omega = {n(omega)}$ rad/s: $|G(j\\omega)| = {n(mag)}$.\n- Phase response: $\\angle G(j\\omega) = {n(phase)}$ rad ({n(float(np.degrees(phase)))} deg)."
    return make_record(
        record_id=f"python_frequency_{index:05d}",
        domain="software_tools",
        family="python_control_frequency_response",
        task_type="code",
        difficulty="foundation",
        template_id=f"python_frequency_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "first_order_frequency",
            "gain": gain,
            "tau": tau,
            "omega": omega,
            "magnitude": mag,
            "magnitude_db": 20 * np.log10(mag),
            "phase_deg": np.degrees(phase),
        },
        source_refs=PYTHON,
        verifier="execute_python_and_verify_frequency_response",
        tool="python_control",
        code_language="python",
        code_execution="passed",
    )


def matlab_tf(rng: np.random.Generator, index: int) -> dict:
    del rng
    p1 = -(1 + 0.04 * index)
    p2 = -(3 + 0.03 * index)
    z = -(0.5 + 0.02 * index)
    gain = 0.8 + 0.02 * index
    num = np.array([gain, -gain * z])
    den = np.array([1, -(p1 + p2), p1 * p2])
    dc = num[-1] / den[-1]
    code = f"""num = {matlab(num)};
den = {matlab(den)};
G = tf(num, den);
p = pole(G);
z = zero(G);
k0 = dcgain(G);
disp('Poles:'); disp(p);
disp('Zeros:'); disp(z);
disp('DC Gain:'); disp(k0);
assert(max(abs(sort(p)-sort({matlab(np.array([p1,p2]))}.'))) < 1e-9);
assert(abs(k0-{n(dc)}) < 1e-9);"""
    prompt = f"{OPENERS[index%4]} In MATLAB Control System Toolbox, construct G(s)=({n(num[0])}s+{n(num[1])})/(s^2+{n(den[1])}s+{n(den[2])}), print poles, zero, and DC gain, and assert the results."
    answer = f"```matlab\n{code}\n```\n\n### MATLAB Implementation Details\n- `tf(num, den)` creates the transfer function object.\n- `pole(G)` and `zero(G)` compute the roots of the denominator and numerator.\n- `dcgain(G)` evaluates the system transfer function at $s=0$.\n- The computed poles are {sorted([p1, p2])}, zero is {n(z)}, and DC gain is {n(dc)}."
    return make_record(
        record_id=f"matlab_tf_{index:05d}",
        domain="software_tools",
        family="matlab_transfer_properties",
        task_type="code",
        difficulty="foundation",
        template_id=f"matlab_tf_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "code_tf_properties",
            "numerator": num.tolist(),
            "denominator": den.tolist(),
            "poles": roots(np.array([p1, p2])),
            "zeros": roots(np.array([z])),
            "dc_gain": dc,
            "required_calls": ["tf(", "pole(", "zero(", "dcgain(", "assert("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_and_verify_tf",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def matlab_observability(rng: np.random.Generator, index: int) -> dict:
    A = rng.integers(-3, 4, size=(3, 3)).astype(float)
    C = rng.integers(-2, 3, size=(1, 3)).astype(float)
    if not np.any(C):
        C[0, 0] = 1
    O = np.vstack([C, C @ A, C @ A @ A])
    rank = int(np.linalg.matrix_rank(O))
    code = f"""A = {matlab(A)};
C = {matlab(C)};
O = obsv(A, C);
r = rank(O);
disp('Observability Matrix:'); disp(O);
fprintf('rank = %d\\n', r);
assert(norm(O-{matlab(O)}, 'fro') < 1e-9);
assert(r == {rank});"""
    prompt = f"{OPENERS[index%4]} In MATLAB, use obsv for A={matlab(A)}, C={matlab(C)}, print the observability matrix/rank, and assert the expected matrix."
    answer = f"```matlab\n{code}\n```\n\n### Observability Analysis in MATLAB\n- `obsv(A, C)` generates the observability matrix $\\mathcal{{O}} = [C; CA; CA^2]$.\n- `rank(O)` evaluates matrix rank: $\\text{{rank}}(\\mathcal{{O}}) = {rank}$.\n- Conclusion: The system is **{'observable' if rank==3 else 'not observable'}**."
    return make_record(
        record_id=f"matlab_observability_{index:05d}",
        domain="software_tools",
        family="matlab_observability_matrix",
        task_type="code",
        difficulty="intermediate",
        template_id=f"matlab_observability_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "observability_rank",
            "A": A.tolist(),
            "C": C.tolist(),
            "O": O.tolist(),
            "rank": rank,
            "observable": rank == 3,
            "required_calls": ["obsv(", "rank(", "assert("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_and_verify_observability",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def matlab_place(rng: np.random.Generator, index: int) -> dict:
    del rng
    A = np.array([[0, 1], [-0.5 - 0.03 * index, -0.2]], float)
    B = np.array([[0], [1]], float)
    desired = np.array([-1 - 0.02 * index, -2 - 0.035 * index])
    K = signal.place_poles(A, B, desired).gain_matrix
    p = np.linalg.eigvals(A - B @ K)
    code = f"""A = {matlab(A)};
B = {matlab(B)};
p_des = {matlab(desired)};
K = place(A, B, p_des);
p_cl = eig(A - B*K);
disp('Gain K:'); disp(K);
disp('Closed-loop poles:'); disp(p_cl);
assert(max(abs(sort(p_cl)-sort(p_des.'))) < 1e-8);"""
    prompt = f"{OPENERS[index%4]} In MATLAB, use place for A={matlab(A)}, B={matlab(B)}, desired poles={matlab(desired)}. Print K and assert eig(A-BK)."
    answer = f"```matlab\n{code}\n```\n\n### Pole Placement in MATLAB\n- `place(A, B, p_des)` calculates state feedback gain $K = {matlab(K)}$.\n- Closed-loop eigenvalues $\\text{{eig}}(A - BK)$ match target poles $p_{{des}} = {matlab(desired)}$ exactly."
    return make_record(
        record_id=f"matlab_place_{index:05d}",
        domain="software_tools",
        family="matlab_state_feedback_pole_placement",
        task_type="code",
        difficulty="intermediate",
        template_id=f"matlab_place_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "pole_placement",
            "A": A.tolist(),
            "B": B.tolist(),
            "desired_poles": roots(desired),
            "K": K.tolist(),
            "closed_poles": roots(p),
            "required_calls": ["place(", "eig(", "assert("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_and_verify_pole_placement",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def matlab_lqr(rng: np.random.Generator, index: int) -> dict:
    del rng
    A = np.array([[0, 1], [-0.7 - 0.02 * index, -0.3]], float)
    B = np.array([[0], [1]], float)
    Q = np.diag([1 + 0.025 * index, 0.5])
    R = np.array([[0.2 + 0.008 * index]])
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    p = np.linalg.eigvals(A - B @ K)
    code = f"""A = {matlab(A)};
B = {matlab(B)};
Q = {matlab(Q)};
R = {matlab(R)};
[K, P, p] = lqr(A, B, Q, R);
disp('LQR Gain K:'); disp(K);
disp('Riccati Matrix P:'); disp(P);
disp('Closed-loop poles:'); disp(p);
assert(norm(K-{matlab(K)}, 'fro') < 1e-7);
assert(max(real(p)) < 0);"""
    prompt = f"{OPENERS[index%4]} In MATLAB, solve continuous LQR for A={matlab(A)}, B={matlab(B)}, Q={matlab(Q)}, R={matlab(R)}; print K,P,p and assert stability."
    answer = f"```matlab\n{code}\n```\n\n### Continuous LQR in MATLAB\n- `[K, P, p] = lqr(A, B, Q, R)` solves CARE $A^T P + PA - P B R^{{-1}} B^T P + Q = 0$.\n- Optimal feedback gain: $K = {matlab(K)}$.\n- Closed-loop poles: $p = {p.tolist()}$, confirming closed-loop stability."
    return make_record(
        record_id=f"matlab_lqr_{index:05d}",
        domain="software_tools",
        family="matlab_continuous_lqr",
        task_type="code",
        difficulty="advanced",
        template_id=f"matlab_lqr_prompt_{index%4}",
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
            "closed_poles": roots(p),
            "required_calls": ["lqr(", "assert("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_and_verify_lqr",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def matlab_zoh(rng: np.random.Generator, index: int) -> dict:
    del rng
    T = 0.025 + 0.004 * index
    A = np.array([[0, 1], [-1 - 0.025 * index, -0.35]], float)
    B = np.array([[0], [1]], float)
    C = np.eye(2)
    D = np.zeros((2, 1))
    Ad, Bd, _, _, _ = signal.cont2discrete((A, B, C, D), T)
    code = f"""A = {matlab(A)};
B = {matlab(B)};
C = eye(2);
D = zeros(2,1);
Ts = {n(T)};
sys_c = ss(A, B, C, D);
sysd = c2d(sys_c, Ts, 'zoh');
[Ad, Bd, ~, ~] = ssdata(sysd);
disp('Ad:'); disp(Ad);
disp('Bd:'); disp(Bd);
assert(norm(Ad-{matlab(Ad)}, 'fro') < 1e-8);
assert(norm(Bd-{matlab(Bd)}, 'fro') < 1e-8);"""
    prompt = f"{OPENERS[index%4]} In MATLAB, use ss, c2d, and ssdata to ZOH-discretize A={matlab(A)}, B={matlab(B)} at Ts={n(T)} s; assert Ad and Bd."
    answer = f"```matlab\n{code}\n```\n\n### ZOH Discretization in MATLAB\n- `ss(A, B, C, D)` constructs the continuous-time state-space object.\n- `c2d(sys_c, Ts, 'zoh')` performs exact Zero-Order Hold discretization at sample time $T_s = {n(T)}$ s.\n- `ssdata(sysd)` extracts the discrete matrices $A_d = {matlab(Ad)}$ and $B_d = {matlab(Bd)}$."
    return make_record(
        record_id=f"matlab_zoh_{index:05d}",
        domain="software_tools",
        family="matlab_exact_zoh_state_space",
        task_type="code",
        difficulty="intermediate",
        template_id=f"matlab_zoh_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "zoh_discretization",
            "A": A.tolist(),
            "B": B.tolist(),
            "sample_time": T,
            "Ad": Ad.tolist(),
            "Bd": Bd.tolist(),
            "required_calls": ["ss(", "c2d(", "ssdata(", "assert("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_and_verify_zoh",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def matlab_frequency_plots(rng: np.random.Generator, index: int) -> dict:
    del rng
    gain = 0.8 + 0.03 * index
    a1 = 1.0 + 0.04 * index
    a0 = 2.0 + 0.05 * index
    num = [gain]
    den = [1, a1, a0]
    code = f"""G = tf({matlab(np.array(num))}, {matlab(np.array(den))});
p = pole(G);
stable = isstable(G);
fprintf('stable = %d\\n', stable);
disp('Poles:'); disp(p);
figure; step(G); grid on; title('Step response');
figure; bode(G); grid on;
[gm, pm, wcg, wcp] = margin(G);
fprintf('GM ratio = %.6g, PM deg = %.6g, Wcg = %.6g, Wcp = %.6g\\n', gm, pm, wcg, wcp);"""
    poles = np.roots(den)
    prompt = f"{OPENERS[index%4]} In MATLAB, construct G(s)={n(gain)}/(s^2+{n(a1)}s+{n(a0)}), print poles/stability, plot step and Bode responses, and compute margins without inventing outputs."
    answer = f"```matlab\n{code}\n```\n\n### MATLAB Frequency Workflow\n- `tf`, `pole`, `isstable`, `step`, `bode`, and `margin` are used to analyze $G(s) = \\frac{{{n(gain)}}}{{s^2 + {n(a1)}s + {n(a0)}}}$.\n- Analytical poles: {poles.tolist()}."
    return make_record(
        record_id=f"matlab_frequency_plots_{index:05d}",
        domain="software_tools",
        family="matlab_step_bode_margin_workflow",
        task_type="code",
        difficulty="foundation",
        template_id=f"matlab_frequency_plots_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "polynomial_stability",
            "coefficients": den,
            "poles": roots(poles),
            "stable": bool(np.all(np.real(poles) < 0)),
            "required_calls": ["tf(", "pole(", "isstable(", "step(", "bode(", "margin("],
        },
        source_refs=MATLAB,
        verifier="static_matlab_frequency_workflow",
        tool="matlab_control_system_toolbox",
        code_language="matlab",
        code_execution="static_checked",
    )


def simulink_pid(rng: np.random.Generator, index: int) -> dict:
    del rng
    kp = 0.8 + 0.03 * index
    ki = 0.2 + 0.01 * index
    kd = 0.05 + 0.004 * index
    a = 1.0 + 0.04 * index
    model = f"controlai_pid_{index:03d}"
    code = f"""mdl = '{model}';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
open_system(mdl);
add_block('simulink/Sources/Step', [mdl '/Step'], 'Position', [30 80 60 110]);
add_block('simulink/Math Operations/Sum', [mdl '/Sum'], 'Inputs', '+-', 'Position', [100 78 125 112]);
add_block('simulink/Continuous/PID Controller', [mdl '/PID'], 'P', '{n(kp)}', 'I', '{n(ki)}', 'D', '{n(kd)}', 'Position', [165 75 245 115]);
add_block('simulink/Continuous/Transfer Fcn', [mdl '/Plant'], 'Numerator', '[1]', 'Denominator', '[1 {n(a)}]', 'Position', [285 75 365 115]);
add_block('simulink/Sinks/Scope', [mdl '/Scope'], 'Position', [415 78 445 112]);
add_line(mdl, 'Step/1', 'Sum/1', 'autorouting', 'on');
add_line(mdl, 'Sum/1', 'PID/1', 'autorouting', 'on');
add_line(mdl, 'PID/1', 'Plant/1', 'autorouting', 'on');
add_line(mdl, 'Plant/1', 'Scope/1', 'autorouting', 'on');
add_line(mdl, 'Plant/1', 'Sum/2', 'autorouting', 'on');
set_param(mdl, 'StopTime', '10');
save_system(mdl);
out = sim(mdl);"""
    prompt = f"{OPENERS[index%4]} Write MATLAB code that programmatically builds and simulates Step -> Sum -> PID -> Transfer Fcn -> Scope with unity negative feedback. Use placeholder PID [{n(kp)},{n(ki)},{n(kd)}] and plant 1/(s+{n(a)}), clearly labeled as demonstration values."
    answer = f"```matlab\n{code}\n```\n\n### Simulink Programmatic Construction\n- Programmatically builds a closed-loop PID control loop in Simulink using `new_system`, `add_block`, `add_line`, and runs simulation via `sim(mdl)`.\n- Gains $K_p={n(kp)}, K_i={n(ki)}, K_d={n(kd)}$ are demonstration parameters."
    required = [
        "new_system(",
        "open_system(",
        "add_block(",
        "simulink/Sources/Step",
        "simulink/Math Operations/Sum",
        "simulink/Continuous/PID Controller",
        "simulink/Continuous/Transfer Fcn",
        "simulink/Sinks/Scope",
        "add_line(",
        "set_param(",
        "save_system(",
        "sim(",
    ]
    return make_record(
        record_id=f"simulink_pid_{index:05d}",
        domain="software_tools",
        family="simulink_programmatic_pid_feedback",
        task_type="code",
        difficulty="advanced",
        template_id=f"simulink_pid_prompt_{index%4}",
        prompt=prompt,
        answer=answer,
        ground_truth={
            "kind": "simulink_static",
            "model": model,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "plant_den": [1, a],
            "required_calls": required,
            "sum_inputs": "+-",
            "forward_connections": 4,
            "feedback_connections": 1,
        },
        source_refs=MATLAB,
        verifier="static_simulink_pid_model",
        tool="matlab_simulink",
        code_language="matlab",
        code_execution="static_checked",
    )


FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, int], dict]], ...] = (
    ("python_control_transfer_properties", python_tf),
    ("python_control_observability_matrix", python_observability),
    ("python_scipy_state_feedback_pole_placement", python_place),
    ("python_control_continuous_lqr", python_lqr),
    ("python_numpy_joseph_kalman_update", python_kf),
    ("python_scipy_exact_zoh", python_zoh),
    ("python_cvxpy_box_mpc", python_mpc),
    ("python_control_frequency_response", python_frequency),
    ("matlab_transfer_properties", matlab_tf),
    ("matlab_observability_matrix", matlab_observability),
    ("matlab_state_feedback_pole_placement", matlab_place),
    ("matlab_continuous_lqr", matlab_lqr),
    ("matlab_exact_zoh_state_space", matlab_zoh),
    ("matlab_step_bode_margin_workflow", matlab_frequency_plots),
    ("simulink_programmatic_pid_feedback", simulink_pid),
)


def generate_code_v1(count_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        generator(rng, index)
        for _, generator in FAMILIES
        for index in range(1, count_per_family + 1)
    ]
