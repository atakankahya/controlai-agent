#!/usr/bin/env python3
"""Independently validate ControlAI SFT v1 records and split invariants."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import cvxpy as cp
from scipy import linalg, signal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).casefold().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def similarity_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", " <NUM> ", text)
    return re.sub(r"(?:\s*<num>\s*,?){3,}", " <NUMSEQ> ", text)


def near_duplicate_errors(
    left_name: str,
    left: list[tuple[str, str]],
    right_name: str,
    right: list[tuple[str, str]],
    threshold: float = 0.90,
) -> list[str]:
    if not left or not right:
        return []
    texts = [similarity_text(text) for _, text in left + right]
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=100_000
    ).fit_transform(texts)
    left_matrix = matrix[: len(left)]
    right_matrix = matrix[len(left) :]
    distances, indices = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
        left_matrix
    ).kneighbors(right_matrix)
    errors = []
    for right_index, (distance, nearest) in enumerate(zip(distances[:, 0], indices[:, 0])):
        similarity = 1.0 - float(distance)
        if similarity >= threshold:
            errors.append(
                f"near-duplicate prompt across {left_name}/{right_name} "
                f"({left[int(nearest)][0]} vs {right[right_index][0]}, cosine={similarity:.3f})"
            )
    return errors


def solution_digest(ground_truth: dict[str, Any]) -> str:
    payload = json.dumps(ground_truth, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def roots(pairs: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(real, imag) for real, imag in pairs])


def same_roots(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.shape != right.shape:
        return False
    left = left[np.lexsort((left.imag, left.real))]
    right = right[np.lexsort((right.imag, right.real))]
    return bool(np.allclose(left, right, rtol=1e-7, atol=1e-8))


def load(path: Path) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{number}: invalid JSON ({exc.msg})")
                continue
            row["_location"] = f"{path}:{number}"
            rows.append(row)
    return rows, errors


def extract_code(answer: str, language: str) -> str:
    match = re.search(
        rf"```{re.escape(language)}\s*\n(?P<code>.*?)```",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"missing fenced {language} code block")
    return match.group("code").strip()


def execute_python(code: str) -> None:
    """Compile and execute our generated, self-checking Python examples."""
    compiled = compile(code, "<generated-sft-example>", "exec")
    namespace = {"__name__": "__controlai_validation__"}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compiled, namespace, namespace)


def static_check_matlab(code: str, ground_truth: dict[str, Any]) -> None:
    """Reject missing required APIs and common hallucinated MATLAB spellings."""
    for required in ground_truth.get("required_calls", []):
        if required.casefold() not in code.casefold():
            raise ValueError(f"missing required MATLAB/Simulink API: {required}")
    banned = (
        "newSystem(",
        "addblock(",
        "addline(",
        "setmdl(",
        "dimpoles(",
        "systems(",
    )
    for token in banned:
        if token.casefold() in code.casefold():
            raise ValueError(f"hallucinated MATLAB/Simulink API: {token}")
    if code.count("(") != code.count(")"):
        raise ValueError("unbalanced parentheses in MATLAB code")
    if code.count("[") != code.count("]"):
        raise ValueError("unbalanced brackets in MATLAB code")


def check_underspecified_answer(answer: str, ground_truth: dict[str, Any]) -> None:
    lowered = re.sub(r"\s+", " ", answer.casefold())
    for term in ground_truth["required_answer_terms"]:
        if term.casefold() not in lowered:
            raise ValueError(f"underspecified answer omits required term: {term}")
    if ground_truth.get("forbid_numeric_controller_coefficients"):
        forbidden = re.compile(
            r"\b(?:k_?p|k_?i|k_?d|controller coefficients?|kalman gain)\s*(?:=|:)\s*[-+]?\d",
            flags=re.IGNORECASE,
        )
        if forbidden.search(answer):
            raise ValueError("underspecified answer fabricates a numerical design value")


def verify_ground_truth(gt: dict[str, Any]) -> bool:
    kind = gt["kind"]
    if kind == "code_tf_properties":
        numerator = np.asarray(gt["numerator"], dtype=float)
        denominator = np.asarray(gt["denominator"], dtype=float)
        poles = np.roots(denominator)
        zeros = np.roots(numerator)
        dc_gain = numerator[-1] / denominator[-1]
        return bool(
            same_roots(poles, roots(gt["poles"]))
            and same_roots(zeros, roots(gt["zeros"]))
            and np.isclose(dc_gain, gt["dc_gain"])
        )
    if kind == "pole_placement":
        A = np.asarray(gt["A"], dtype=float)
        B = np.asarray(gt["B"], dtype=float)
        desired = roots(gt["desired_poles"])
        placed = signal.place_poles(A, B, desired).gain_matrix
        closed = np.linalg.eigvals(A - B @ np.asarray(gt["K"], dtype=float))
        return bool(
            np.allclose(placed, gt["K"], rtol=1e-7, atol=1e-8)
            and same_roots(closed, roots(gt["closed_poles"]))
            and same_roots(closed, desired)
        )
    if kind == "simulink_static":
        return bool(
            str(gt["model"]).startswith("controlai_pid_")
            and gt["kp"] > 0
            and gt["ki"] >= 0
            and gt["kd"] >= 0
            and len(gt["plant_den"]) == 2
            and gt["plant_den"][0] == 1
            and gt["plant_den"][1] > 0
            and gt["sum_inputs"] == "+-"
            and gt["forward_connections"] == 4
            and gt["feedback_connections"] == 1
        )
    if kind == "scalar_dynamic_inversion":
        virtual = -float(gt["gain"]) * (float(gt["x"]) - float(gt["reference"]))
        control = (virtual - float(gt["a"]) * float(gt["x"])) / float(gt["b"])
        return bool(np.isclose(virtual, gt["virtual"]) and np.isclose(control, gt["u"])
                    and np.isclose(float(gt["a"])*float(gt["x"])+float(gt["b"])*control, virtual))
    if kind == "gain_schedule_interpolation":
        weight = (float(gt["sigma"])-float(gt["sigma_low"])) / (float(gt["sigma_high"])-float(gt["sigma_low"]))
        scheduled = (1-weight)*np.asarray(gt["low"], float)+weight*np.asarray(gt["high"], float)
        return bool(0 <= weight <= 1 and np.isclose(weight, gt["weight"]) and np.allclose(scheduled, gt["scheduled"]))
    if kind == "scalar_cbf_filter":
        h = float(gt["x"])-float(gt["x_min"]); lower = -float(gt["alpha"])*h
        safe = max(float(gt["u_nom"]), lower)
        return bool(np.isclose(h, gt["h"]) and np.isclose(lower, gt["lower_bound"])
                    and np.isclose(safe, gt["u_safe"]) and (float(gt["u_nom"]) < lower)==gt["active"]
                    and safe+float(gt["alpha"])*h >= -1e-10)
    if kind == "scalar_stationary_covariance":
        a=float(gt["a"]); q=float(gt["q"]); exists=abs(a)<1
        p=q/(1-a*a) if exists else np.nan
        return bool(exists==gt["exists"] and np.isclose(p,gt["P"]) and np.isclose(a*a*p+q,p))
    if kind == "minimum_norm_allocation":
        B=np.asarray(gt["B"],float).reshape(1,-1); desired=float(gt["desired"])
        u=B.T@np.linalg.solve(B@B.T,np.array([desired])); achieved=float((B@u)[0])
        return bool(np.allclose(u,gt["u"]) and np.isclose(achieved,gt["achieved"]) and np.isclose(achieved,desired))
    if kind == "iid_jump_mean_square":
        factor=float(gt["p"])*float(gt["a1"])**2+(1-float(gt["p"]))*float(gt["a2"])**2
        return bool(np.isclose(factor,gt["factor"]) and (factor<1)==gt["stable"])
    if kind == "relative_event_trigger":
        error=abs(float(gt["x_hat"])-float(gt["x"])); threshold=float(gt["sigma"])*abs(float(gt["x"])); trigger=error>=threshold
        return bool(np.isclose(error,gt["error_abs"]) and np.isclose(threshold,gt["threshold"]) and trigger==gt["trigger"])
    if kind == "symmetric_tube_tightening":
        tightened=float(gt["bound"])-float(gt["error"]); interval=[-tightened,tightened]
        return bool(np.isclose(tightened,gt["tightened"]) and np.allclose(interval,gt["interval"]) and (tightened>=0)==gt["nonempty"])
    if kind == "finite_discounted_return":
        rewards=np.asarray(gt["rewards"],float); terms=rewards*float(gt["gamma"])**np.arange(len(rewards))
        return bool(np.allclose(terms,gt["terms"]) and np.isclose(terms.sum(),gt["return"]))
    if kind == "nonlinear_equilibrium_feedforward":
        u=float(gt["a"])*float(gt["reference"])**3/float(gt["b"])
        residual=-float(gt["a"])*float(gt["reference"])**3+float(gt["b"])*u
        return bool(np.isclose(u,gt["u_eq"]) and np.isclose(residual,gt["residual"]) and np.isclose(residual,0))
    if kind == "robust_safe_input_interval":
        a,b,x=map(float,(gt["a"],gt["b"],gt["x"])); w=float(gt["w_max"])
        lower=(float(gt["x_min"])+w-a*x)/b; upper=(float(gt["x_max"])-w-a*x)/b
        safe=float(np.clip(float(gt["u_nom"]),lower,upper))
        return bool(b>0 and lower<=upper and np.isclose(lower,gt["u_lower"])
                    and np.isclose(upper,gt["u_upper"]) and np.isclose(safe,gt["u_safe"])
                    and a*x+b*safe-w>=float(gt["x_min"])-1e-10
                    and a*x+b*safe+w<=float(gt["x_max"])+1e-10)
    if kind == "single_actuator_loss_isolation":
        B=np.asarray(gt["B"],float);u=np.asarray(gt["u"],float);factor=float(gt["loss_factor"])
        predictions=[]
        for candidate in range(len(B)):
            trial=B.copy();trial[candidate]*=factor;predictions.append(float(trial@u))
        residuals=np.abs(np.asarray(predictions)-float(gt["measured"]));identified=int(np.argmin(residuals))
        return bool(np.allclose(predictions,gt["candidate_outputs"]) and np.allclose(residuals,gt["residuals"])
                    and identified==gt["identified_index"]==gt["failed_index"])
    if kind == "underspecified_requirements":
        return bool(gt["can_compute_final_design"] is False
                    and gt["forbid_numeric_controller_coefficients"] is True
                    and len(gt["required_answer_terms"]) >= 5)
    if kind == "eigenvalue_stability":
        poles = np.linalg.eigvals(np.asarray(gt["A"], dtype=float))
        stable = (
            np.all(np.abs(poles) < 1)
            if gt["time_domain"] == "discrete"
            else np.all(np.real(poles) < 0)
        )
        return same_roots(poles, roots(gt["eigenvalues"])) and bool(stable) == gt["stable"]
    if kind == "matrix_exponential":
        actual = linalg.expm(np.asarray(gt["A"], dtype=float) * float(gt["time"]))
        return bool(np.allclose(actual, gt["Phi"], rtol=1e-8, atol=1e-9))
    if kind == "zoh_discretization":
        A = np.asarray(gt["A"], dtype=float)
        B = np.asarray(gt["B"], dtype=float)
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))
        Ad, Bd, _, _, _ = signal.cont2discrete(
            (A, B, C, D), float(gt["sample_time"]), method="zoh"
        )
        return bool(np.allclose(Ad, gt["Ad"]) and np.allclose(Bd, gt["Bd"]))
    if kind == "observability_rank":
        A = np.asarray(gt["A"], dtype=float)
        C = np.asarray(gt["C"], dtype=float)
        O = np.vstack([C @ np.linalg.matrix_power(A, power) for power in range(A.shape[0])])
        rank = int(np.linalg.matrix_rank(O))
        return bool(
            np.allclose(O, gt["O"])
            and rank == gt["rank"]
            and (rank == A.shape[0]) == gt["observable"]
        )
    if kind == "pbh_observability":
        A = np.asarray(gt["A"], dtype=float)
        C = np.asarray(gt["C"], dtype=float)
        ranks = [
            int(np.linalg.matrix_rank(np.vstack([value * np.eye(A.shape[0]) - A, C])))
            for value in gt["eigenvalues"]
        ]
        return ranks == gt["pbh_ranks"] and all(x == A.shape[0] for x in ranks) == gt["observable"]
    if kind == "state_feedback":
        A = np.asarray(gt["A"], dtype=float)
        Acl = A - np.asarray(gt["B"], dtype=float) @ np.asarray(gt["K"], dtype=float)
        poles = np.linalg.eigvals(Acl)
        return bool(
            np.allclose(Acl, gt["Acl"])
            and same_roots(poles, roots(gt["eigenvalues"]))
            and bool(np.all(np.real(poles) < 0)) == gt["stable"]
        )
    if kind == "observer_error":
        A = np.asarray(gt["A"], dtype=float)
        Ae = A - np.asarray(gt["L"], dtype=float) @ np.asarray(gt["C"], dtype=float)
        poles = np.linalg.eigvals(Ae)
        return bool(
            np.allclose(Ae, gt["Ae"])
            and same_roots(poles, roots(gt["eigenvalues"]))
            and bool(np.all(np.real(poles) < 0)) == gt["stable"]
        )
    if kind == "continuous_lyapunov":
        A = np.asarray(gt["A"], dtype=float)
        Q = np.asarray(gt["Q"], dtype=float)
        P = linalg.solve_continuous_lyapunov(A.T, -Q)
        return bool(np.allclose(P, gt["P"]) and np.allclose(A.T @ P + P @ A + Q, 0, atol=1e-8))
    if kind == "discrete_lyapunov":
        A = np.asarray(gt["A"], dtype=float)
        Q = np.asarray(gt["Q"], dtype=float)
        P = linalg.solve_discrete_lyapunov(A.T, Q)
        return bool(np.allclose(P, gt["P"]) and np.allclose(A.T @ P @ A - P + Q, 0, atol=1e-8))
    if kind == "controllability_gramian":
        A = np.asarray(gt["A"], dtype=float)
        B = np.asarray(gt["B"], dtype=float)
        W = linalg.solve_continuous_lyapunov(A, -(B @ B.T))
        return bool(np.allclose(W, gt["W"]) and np.allclose(np.linalg.eigvalsh(W), gt["eigenvalues"]))
    if kind == "observability_gramian":
        A = np.asarray(gt["A"], dtype=float)
        C = np.asarray(gt["C"], dtype=float)
        W = linalg.solve_continuous_lyapunov(A.T, -(C.T @ C))
        return bool(np.allclose(W, gt["W"]) and np.allclose(np.linalg.eigvalsh(W), gt["eigenvalues"]))
    if kind == "first_order_frequency":
        x = float(gt["tau"]) * float(gt["omega"])
        magnitude = float(gt["gain"]) / np.sqrt(1 + x * x)
        return bool(
            np.isclose(magnitude, gt["magnitude"])
            and np.isclose(20 * np.log10(magnitude), gt["magnitude_db"])
            and np.isclose(-np.degrees(np.arctan(x)), gt["phase_deg"])
        )
    if kind == "first_order_step":
        final = float(gt["gain"]) * float(gt["step"])
        tau = float(gt["tau"])
        return bool(
            np.isclose(final, gt["final"])
            and np.isclose(final * (1 - np.exp(-1)), gt["at_tau"])
            and np.isclose(-tau * np.log(0.02), gt["settling_2"])
        )
    if kind == "second_order_transient":
        zeta, wn = float(gt["zeta"]), float(gt["omega_n"])
        wd = wn * np.sqrt(1 - zeta * zeta)
        poles = np.asarray([-zeta * wn + 1j * wd, -zeta * wn - 1j * wd])
        return bool(
            np.isclose(wd, gt["omega_d"])
            and np.isclose(np.exp(-np.pi * zeta / np.sqrt(1-zeta*zeta))*100, gt["overshoot_percent"])
            and np.isclose(np.pi / wd, gt["peak_time"])
            and np.isclose(-np.log(0.02)/(zeta*wn), gt["settling_2"])
            and same_roots(poles, roots(gt["poles"]))
        )
    if kind == "closed_loop_tf":
        pnum = np.asarray(gt["plant_num"], dtype=float)
        pden = np.asarray(gt["plant_den"], dtype=float)
        gain = float(gt["gain"])
        closed_num = gain * pnum
        padded = np.pad(closed_num, (len(pden)-len(closed_num), 0))
        closed_den = pden + padded
        poles = np.roots(closed_den)
        return bool(
            np.allclose(closed_num, gt["closed_num"])
            and np.allclose(closed_den, gt["closed_den"])
            and same_roots(poles, roots(gt["poles"]))
            and bool(np.all(np.real(poles)<0)) == gt["stable"]
        )
    if kind == "sensitivity_complex":
        L = complex(*gt["L"]); S = 1/(1+L); T=L/(1+L)
        return bool(
            np.allclose([S.real,S.imag], gt["S"])
            and np.allclose([T.real,T.imag], gt["T"])
            and np.isclose(abs(S), gt["S_mag"])
            and np.isclose(abs(T), gt["T_mag"])
            and np.isclose(S+T, 1)
        )
    if kind == "lead_parameters":
        phi=np.radians(float(gt["phi_deg"])); wm=float(gt["omega_m"])
        alpha=(1-np.sin(phi))/(1+np.sin(phi)); tau=1/(wm*np.sqrt(alpha))
        return bool(
            np.isclose(alpha,gt["alpha"]) and np.isclose(tau,gt["tau"])
            and np.isclose(-1/tau,gt["zero"]) and np.isclose(-1/(alpha*tau),gt["pole"])
        )
    if kind == "polynomial_stability":
        poles=np.roots(gt["coefficients"])
        return same_roots(poles,roots(gt["poles"])) and bool(np.all(np.real(poles)<0))==gt["stable"]
    if kind == "routh_quartic":
        _,a3,a2,a1,a0=map(float,gt["coefficients"])
        b1=(a3*a2-a1)/a3; c1=(b1*a1-a3*a0)/b1
        first=[1.0,a3,b1,c1,a0]; poles=np.roots(gt["coefficients"])
        return bool(np.allclose(first,gt["first_column"]) and same_roots(poles,roots(gt["poles"])) and (all(x>0 for x in first)==gt["stable"]))
    if kind == "root_locus_real_axis":
        count=sum(float(x)>float(gt["point"]) for x in gt["poles"]+gt["zeros"])
        return count==gt["right_count"] and (count%2==1)==gt["on_locus"]
    if kind == "crossover_margins":
        gm=1/float(gt["magnitude_at_phase_crossover"]); pm=180+float(gt["phase_at_gain_crossover_deg"])
        return bool(np.isclose(gm,gt["gain_margin"]) and np.isclose(20*np.log10(gm),gt["gain_margin_db"]) and np.isclose(pm,gt["phase_margin_deg"]))
    if kind == "scalar_kf_predict":
        xm=gt["a"]*gt["x"]+gt["b"]*gt["u"]; pm=gt["a"]**2*gt["P"]+gt["Q"]
        return bool(np.isclose(xm,gt["x_minus"]) and np.isclose(pm,gt["P_minus"]))
    if kind == "matrix_kf_update":
        x=np.asarray(gt["x_minus"],float); P=np.asarray(gt["P_minus"],float); H=np.asarray(gt["H"],float); R=np.asarray(gt["R"],float); z=np.asarray(gt["z"],float)
        innovation=z-H@x; S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S); xp=x+K@innovation; I=np.eye(len(x)); Pp=(I-K@H)@P@(I-K@H).T+K@R@K.T
        return bool(np.allclose(innovation,gt["innovation"]) and np.allclose(S,gt["S"]) and np.allclose(K,gt["K"]) and np.allclose(xp,gt["x_plus"]) and np.allclose(Pp,gt["P_plus"]))
    if kind == "covariance_propagation":
        A=np.asarray(gt["A"],float); P=np.asarray(gt["P"],float); Q=np.asarray(gt["Q"],float)
        return bool(np.allclose(A@P@A.T+Q,gt["P_next"]))
    if kind == "continuous_lqr":
        A=np.asarray(gt["A"],float); B=np.asarray(gt["B"],float); Q=np.asarray(gt["Q"],float); R=np.asarray(gt["R"],float)
        P=linalg.solve_continuous_are(A,B,Q,R); K=np.linalg.solve(R,B.T@P); poles=np.linalg.eigvals(A-B@K)
        return bool(np.allclose(P,gt["P"]) and np.allclose(K,gt["K"]) and same_roots(poles,roots(gt["closed_poles"])))
    if kind == "discrete_lqr":
        A=np.asarray(gt["A"],float); B=np.asarray(gt["B"],float); Q=np.asarray(gt["Q"],float); R=np.asarray(gt["R"],float)
        P=linalg.solve_discrete_are(A,B,Q,R); K=np.linalg.solve(R+B.T@P@B,B.T@P@A); poles=np.linalg.eigvals(A-B@K)
        return bool(np.allclose(P,gt["P"]) and np.allclose(K,gt["K"]) and same_roots(poles,roots(gt["closed_poles"])))
    if kind == "finite_horizon_lqr":
        a,b,q,r,qf=map(float,(gt["a"],gt["b"],gt["q"],gt["r"],gt["qf"])); horizon=int(gt["horizon"])
        P=[0.0]*(horizon+1); K=[0.0]*horizon; P[horizon]=qf
        for k in range(horizon-1,-1,-1):
            den=r+b*b*P[k+1]; K[k]=b*P[k+1]*a/den; P[k]=q+a*a*P[k+1]-(a*b*P[k+1])**2/den
        return bool(np.allclose(K,gt["K"]) and np.allclose(P,gt["P"]))
    if kind == "mpc_prediction":
        a,b,h=float(gt["a"]),float(gt["b"]),int(gt["horizon"]); F=np.array([[a**i] for i in range(1,h+1)]); G=np.zeros((h,h))
        for row in range(h):
            for col in range(row+1): G[row,col]=a**(row-col)*b
        return bool(np.allclose(F,gt["F"]) and np.allclose(G,gt["G"]))
    if kind == "mpc_unconstrained":
        a,b,x0,r=map(float,(gt["a"],gt["b"],gt["x0"],gt["r"])); F=np.array([[a],[a*a]]); G=np.array([[b,0],[a*b,b]]); U=-np.linalg.solve(G.T@G+r*np.eye(2),G.T@(F[:,0]*x0)); X=F[:,0]*x0+G@U; cost=X@X+r*(U@U)
        return bool(np.allclose(F,gt["F"]) and np.allclose(G,gt["G"]) and np.allclose(U,gt["U"]) and np.allclose(X,gt["X"]) and np.isclose(cost,gt["cost"]))
    if kind == "mpc_box_qp":
        x0,r,limit=map(float,(gt["x0"],gt["r"],gt["limit"])); u=cp.Variable(2); x1=x0+u[0]; x2=x1+u[1]; problem=cp.Problem(cp.Minimize(cp.square(x1)+cp.square(x2)+r*cp.sum_squares(u)),[u>=-limit,u<=limit]); value=problem.solve(solver="CLARABEL"); U=np.asarray(u.value); X=np.array([x0+U[0],x0+U[0]+U[1]])
        # CLARABEL's equivalent abs() and two-inequality formulations can differ
        # by a few solver-tolerance units near an active bound.
        return bool(problem.status in {"optimal","optimal_inaccurate"} and np.allclose(U,gt["U"],rtol=1e-7,atol=1e-5) and np.allclose(X,gt["X"],rtol=1e-7,atol=1e-5) and np.isclose(value,gt["cost"],rtol=1e-7,atol=1e-7))
    if kind == "cubic_equilibria":
        a,b=float(gt["a"]),float(gt["b"]); radius=np.sqrt(a/b); eq=np.array([-radius,0,radius]); deriv=a-3*b*eq**2
        return bool(np.allclose(eq,gt["equilibria"]) and np.allclose(deriv,gt["derivatives"]) and np.array_equal(deriv<0,gt["stable"]))
    if kind == "nonlinear_jacobian":
        J=np.array([[-float(gt["a"]),0],[1,-float(gt["b"])]],float); poles=np.linalg.eigvals(J)
        return bool(np.allclose(J,gt["J"]) and same_roots(poles,roots(gt["poles"])) and bool(np.all(np.real(poles)<0))==gt["locally_stable"])
    if kind == "scalar_linear_cubic_lyapunov":
        return bool(float(gt["a"])>0 and float(gt["b"])>0 and gt["global_asymptotic"] and gt["local_exponential"])
    if kind == "quadratic_lyapunov":
        A=np.asarray(gt["A"],float);P=np.asarray(gt["P"],float);Q=-(A.T@P+P@A);eig=np.linalg.eigvalsh(Q)
        return bool(np.allclose(Q,gt["Q"]) and np.allclose(eig,gt["Q_eigenvalues"]) and np.all(eig>0))
    if kind == "feedback_linearization":
        poles=np.roots([1,float(gt["k2"]),float(gt["k1"])])
        return bool(float(gt["b"])!=0 and gt["relative_degree"]==2 and same_roots(poles,roots(gt["closed_poles"])))
    if kind == "zero_dynamics":
        a=float(gt["a"])
        return bool(gt["relative_degree"]==2 and np.isclose(gt["zero_pole"],-a) and gt["minimum_phase"]==(a>0))
    if kind == "sliding_reachability":
        eta=float(gt["k"])-float(gt["dmax"])
        return bool(np.isclose(eta,gt["eta"]) and gt["sufficient"]==(eta>0))
    if kind == "small_gain":
        product=float(gt["g1"])*float(gt["g2"])
        return bool(np.isclose(product,gt["product"]) and gt["certified"]==(product<1))
    if kind == "weighted_peak":
        products=np.asarray(gt["weight"],float)*np.asarray(gt["response"],float);peak=float(np.max(products))
        return bool(np.allclose(products,gt["products"]) and np.isclose(peak,gt["peak"]) and gt["certified"]==(peak<float(gt["strict_bound"])))
    if kind == "hinf_first_order":
        hinf=abs(float(gt["gain"]))
        return bool(float(gt["tau"])>0 and np.isclose(hinf,gt["hinf"]) and np.isclose(gt["peak_frequency"],0))
    if kind == "kharitonov_cubic":
        lo=np.asarray(gt["lower"],float);hi=np.asarray(gt["upper"],float);polys=np.array([[lo[0],lo[1],hi[2],hi[3]],[hi[0],hi[1],lo[2],lo[3]],[hi[0],lo[1],lo[2],hi[3]],[lo[0],hi[1],hi[2],lo[3]]]);pole_sets=[np.roots(p[::-1]) for p in polys];flags=[bool(np.all(np.real(p)<0)) for p in pole_sets]
        return bool(np.allclose(polys,gt["polynomials"]) and all(same_roots(p,roots(g)) for p,g in zip(pole_sets,gt["poles"])) and flags==gt["stable_flags"] and all(flags)==gt["robustly_hurwitz"])
    if kind == "sampled_sensitivity_peak":
        L=np.asarray([complex(*x) for x in gt["L"]]);mags=np.abs(1/(1+L));idx=int(np.argmax(mags));peak=float(mags[idx])
        return bool(np.allclose(mags,gt["magnitudes"]) and idx==gt["peak_index"] and np.isclose(peak,gt["peak"]))
    if kind == "pe_sincos":
        omega=float(gt["omega"]);period=2*np.pi/omega;gram=np.pi/omega*np.eye(2)
        return bool(np.isclose(period,gt["period"]) and np.allclose(gram,gt["gramian"]) and np.isclose(np.pi/omega,gt["alpha_max"]))
    if kind == "gradient_identifier":
        theta=np.asarray(gt["theta"],float);phi=np.asarray(gt["phi"],float);error=float(gt["y"])-float(phi@theta);next_theta=theta+float(gt["dt"])*float(gt["gamma"])*phi*error
        return bool(np.isclose(error,gt["error"]) and np.allclose(next_theta,gt["theta_next"]))
    if kind == "normalized_identifier":
        theta=np.asarray(gt["theta"],float);phi=np.asarray(gt["phi"],float);error=float(gt["y"])-float(phi@theta);den=1+float(phi@phi);next_theta=theta+float(gt["gamma"])*phi*error/den
        return bool(np.isclose(error,gt["error"]) and np.isclose(den,gt["denominator"]) and np.allclose(next_theta,gt["theta_next"]))
    if kind == "regression_rank":
        Phi=np.asarray(gt["Phi"],float);rank=int(np.linalg.matrix_rank(Phi));gram=Phi.T@Phi;eig=np.linalg.eigvalsh(gram)
        return bool(rank==gt["rank"] and np.allclose(gram,gt["gramian"]) and np.allclose(eig,gt["eigenvalues"]) and (rank==Phi.shape[1])==gt["identifiable"])
    if kind == "arx_least_squares":
        u=np.asarray(gt["u"],float);y=np.asarray(gt["y"],float);Phi=np.column_stack([y[:-1],u[:-1]]);theta=np.linalg.lstsq(Phi,y[1:],rcond=None)[0];residual=y[1:]-Phi@theta
        return bool(np.allclose(Phi,gt["Phi"]) and np.allclose(theta,gt["theta"]) and int(np.linalg.matrix_rank(Phi))==gt["rank"] and np.isclose(np.linalg.norm(residual),gt["residual_norm"]))
    if kind == "fopdt_step":
        gain=float(gt["delta_y"])/float(gt["delta_u"]);tau=float(gt["t63"])-float(gt["delay"])
        return bool(np.isclose(gain,gt["gain"]) and np.isclose(tau,gt["tau"]) and tau>0)
    if kind == "residual_autocorrelation":
        residual=np.asarray(gt["residual"],float);den=float(residual@residual);acf=[float(residual[lag:]@residual[:-lag]/den) for lag in range(1,int(gt["lags"])+1)]
        return bool(np.allclose(acf,gt["acf"]))
    if kind == "consensus_laplacian":
        L=np.asarray(gt["L"],float);eig=np.linalg.eigvalsh(L);connected=eig[1]>1e-10
        return bool(np.allclose(L,L.T) and np.allclose(L.sum(axis=1),0) and np.allclose(eig,gt["eigenvalues"]) and connected==gt["connected"] and np.isclose(eig[1],gt["lambda2"]))
    if kind == "discrete_consensus":
        L=np.asarray(gt["L"],float);eig=np.linalg.eigvalsh(L);closed=1-float(gt["alpha"])*eig;stable=bool(np.all(np.abs(closed[1:])<1))
        return bool(np.allclose(eig,gt["laplacian_eigenvalues"]) and np.allclose(closed,gt["closed_eigenvalues"]) and stable==gt["stable_disagreement"])
    raise ValueError(f"unknown ground-truth kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("data/training/sft_v1"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/v0.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    train, errors = load(args.dataset / "train.jsonl")
    valid, valid_errors = load(args.dataset / "valid.jsonl")
    errors.extend(valid_errors)
    all_rows = train + valid
    benchmark = [json.loads(line) for line in args.benchmark.open() if line.strip()]
    benchmark_families = {row["family"] for row in benchmark}
    benchmark_prompts = {normalized_hash(row["prompt"]) for row in benchmark}
    seen_ids, seen_prompts = {}, {}
    families = {"train": set(), "valid": set()}
    template_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {"train": Counter(), "valid": Counter()}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    for split, rows in (("train", train), ("valid", valid)):
        for row in rows:
            location = row["_location"]
            messages = row.get("messages")
            if not isinstance(messages, list) or [x.get("role") for x in messages] != ["system", "user", "assistant"]:
                errors.append(f"{location}: expected system/user/assistant messages")
                continue
            if any(not isinstance(x.get("content"), str) or not x["content"].strip() for x in messages):
                errors.append(f"{location}: message content must be non-empty text")
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                errors.append(f"{location}: missing metadata")
                continue
            for field in (
                "id", "domain", "family", "split_group", "task_type", "difficulty",
                "template_id", "generator_version", "source_refs", "verification",
                "ground_truth", "solution_sha256", "status",
            ):
                if metadata.get(field) in (None, "", []):
                    errors.append(f"{location}: missing metadata.{field}")
            record_id = metadata.get("id")
            task_counts[split][metadata.get("task_type", "missing")] += 1
            if record_id in seen_ids:
                errors.append(f"{location}: duplicate id also at {seen_ids[record_id]}")
            seen_ids[record_id] = location
            prompt_hash = normalized_hash(messages[1]["content"])
            if prompt_hash in seen_prompts:
                errors.append(f"{location}: duplicate prompt also at {seen_prompts[prompt_hash]}")
            seen_prompts[prompt_hash] = location
            if prompt_hash in benchmark_prompts:
                errors.append(f"{location}: exact benchmark prompt leakage")
            family = metadata.get("family")
            families[split].add(family)
            if split == "train" and family in benchmark_families:
                errors.append(f"{location}: benchmark family leakage: {family}")
            template_counts[metadata.get("template_id", "missing")] += 1
            gt = metadata.get("ground_truth")
            if isinstance(gt, dict):
                kind_counts[gt.get("kind", "missing")] += 1
                if solution_digest(gt) != metadata.get("solution_sha256"):
                    errors.append(f"{location}: solution digest mismatch")
                try:
                    if not verify_ground_truth(gt):
                        errors.append(f"{location}: independent ground-truth check failed")
                except Exception as exc:
                    errors.append(f"{location}: verifier raised {type(exc).__name__}: {exc}")
            if metadata.get("task_type") == "code":
                language = metadata.get("code_language")
                try:
                    code = extract_code(messages[2]["content"], language)
                    if language == "python":
                        execute_python(code)
                        if metadata.get("code_execution") != "passed":
                            raise ValueError("executed Python must have code_execution=passed")
                    elif language == "matlab":
                        static_check_matlab(code, gt)
                        if metadata.get("code_execution") != "static_checked":
                            raise ValueError("MATLAB must have code_execution=static_checked")
                    else:
                        raise ValueError(f"unsupported code language {language!r}")
                except Exception as exc:
                    errors.append(f"{location}: code check raised {type(exc).__name__}: {exc}")
            if metadata.get("task_type") == "underspecified" and isinstance(gt, dict):
                try:
                    check_underspecified_answer(messages[2]["content"], gt)
                except Exception as exc:
                    errors.append(f"{location}: underspecification check raised {type(exc).__name__}: {exc}")
            token_count = len(tokenizer.apply_chat_template(messages, return_dict=False))
            if token_count > 1024:
                errors.append(f"{location}: {token_count} tokens exceeds 1024")

    overlap = families["train"] & families["valid"]
    if overlap:
        errors.append(f"family split leakage: {sorted(overlap)}")
    required_train_tasks = {"concept", "derivation", "numerical", "code", "critique", "design", "underspecified"}
    missing_tasks = required_train_tasks - set(task_counts["train"])
    if missing_tasks:
        errors.append(f"train split missing task types: {sorted(missing_tasks)}")
    underspecified_fraction = task_counts["train"]["underspecified"] / len(train) if train else 0
    if underspecified_fraction < 0.05:
        errors.append(f"underspecified train fraction {underspecified_fraction:.2%} is below 5%")
    prompt_splits = {
        "train": [(row["metadata"]["id"], row["messages"][1]["content"]) for row in train],
        "valid": [(row["metadata"]["id"], row["messages"][1]["content"]) for row in valid],
        "benchmark": [(row["id"], row["prompt"]) for row in benchmark],
    }
    errors.extend(near_duplicate_errors("train", prompt_splits["train"], "valid", prompt_splits["valid"]))
    errors.extend(near_duplicate_errors("train", prompt_splits["train"], "benchmark", prompt_splits["benchmark"]))
    errors.extend(near_duplicate_errors("valid", prompt_splits["valid"], "benchmark", prompt_splits["benchmark"]))
    max_template = max(template_counts.values(), default=0)
    concentration = max_template / len(all_rows) if all_rows else 0
    if concentration > 0.02:
        errors.append(f"template concentration {concentration:.2%} exceeds 2%")

    print(f"records: {len(all_rows):,}")
    print(f"train/valid: {len(train):,}/{len(valid):,}")
    print(f"train families: {len(families['train'])}")
    print(f"valid families: {len(families['valid'])}")
    print(f"ground-truth kinds: {dict(sorted(kind_counts.items()))}")
    print(f"train task types: {dict(sorted(task_counts['train'].items()))}")
    print(f"max declared-template concentration: {concentration:.2%}")
    if errors:
        print("validation failed:", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"- ... {len(errors)-100} additional errors", file=sys.stderr)
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
