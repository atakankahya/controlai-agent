"""Independent verification and residual calculation layer for ControlAI tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import linalg


class ControlVerifier:
    """Computes mathematical residuals, stability checks, and invariant certificates."""

    @staticmethod
    def verify_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray, K: np.ndarray) -> dict[str, Any]:
        """Verify Continuous Algebraic Riccati Equation: A^T P + P A - P B R^-1 B^T P + Q = 0."""
        R_inv = np.linalg.inv(R)
        riccati_lhs = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
        residual_norm = float(np.max(np.abs(riccati_lhs)))
        p_eigs = np.linalg.eigvalsh(P)
        p_is_pos_def = bool(np.all(p_eigs > 1e-10))
        
        A_cl = A - B @ K
        cl_poles = np.linalg.eigvals(A_cl)
        is_stable = bool(np.all(np.real(cl_poles) < 0.0))
        
        return {
            "riccati_residual_max": residual_norm,
            "P_positive_definite": p_is_pos_def,
            "closed_loop_stable": is_stable,
            "verification_passed": residual_norm < 1e-5 and p_is_pos_def and is_stable,
        }

    @staticmethod
    def verify_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray, K: np.ndarray) -> dict[str, Any]:
        """Verify Discrete Algebraic Riccati Equation: P = A^T P A - (A^T P B)(R + B^T P B)^-1 (B^T P A) + Q."""
        mid = np.linalg.inv(R + B.T @ P @ B)
        dare_rhs = A.T @ P @ A - (A.T @ P @ B) @ mid @ (B.T @ P @ A) + Q
        residual_norm = float(np.max(np.abs(P - dare_rhs)))
        p_eigs = np.linalg.eigvalsh(P)
        p_is_pos_def = bool(np.all(p_eigs > 1e-10))
        
        A_cl = A - B @ K
        cl_poles = np.linalg.eigvals(A_cl)
        is_stable = bool(np.all(np.abs(cl_poles) < 1.0))
        
        return {
            "riccati_residual_max": residual_norm,
            "P_positive_definite": p_is_pos_def,
            "closed_loop_stable": is_stable,
            "verification_passed": residual_norm < 1e-5 and p_is_pos_def and is_stable,
        }

    @staticmethod
    def verify_pole_placement(A: np.ndarray, B: np.ndarray, K: np.ndarray, target_poles: list[float]) -> dict[str, Any]:
        """Verify closed-loop eigenvalues match target poles."""
        A_cl = A - B @ K
        actual_poles = np.linalg.eigvals(A_cl)
        sorted_actual = np.sort_complex(actual_poles)
        sorted_target = np.sort_complex(np.array(target_poles, dtype=complex))
        error = float(np.max(np.abs(sorted_actual - sorted_target)))
        return {
            "pole_placement_error_max": error,
            "verification_passed": error < 1e-4,
        }

    @staticmethod
    def verify_zoh(A: np.ndarray, B: np.ndarray, Ts: float, Ad: np.ndarray, Bd: np.ndarray) -> dict[str, Any]:
        """Cross-check ZOH discretization with block matrix exponential method."""
        n = A.shape[0]
        m = B.shape[1] if B.ndim > 1 else 1
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        eM = linalg.expm(M * Ts)
        Ad_cross = eM[:n, :n]
        Bd_cross = eM[:n, n:]
        
        ad_diff = float(np.max(np.abs(Ad - Ad_cross)))
        bd_diff = float(np.max(np.abs(Bd - Bd_cross)))
        passed = ad_diff < 1e-7 and bd_diff < 1e-7
        return {
            "Ad_cross_validation_diff": ad_diff,
            "Bd_cross_validation_diff": bd_diff,
            "verification_passed": passed,
        }

    @staticmethod
    def verify_cbf(x: float, u_safe: float, alpha: float, x_min: float, u_nom: float) -> dict[str, Any]:
        """Verify Control Barrier Function forward invariance inequality."""
        h = x - x_min
        lower = -alpha * h
        h_dot = u_safe
        residual = h_dot + alpha * h
        passed = residual >= -1e-7 and (u_safe == u_nom or abs(u_safe - lower) < 1e-7)
        return {
            "barrier_h": h,
            "cbf_inequality_residual": float(residual),
            "is_forward_invariant": residual >= -1e-7,
            "verification_passed": passed,
        }

    @staticmethod
    def verify_allocation(B: np.ndarray, u: np.ndarray, desired_tau: float) -> dict[str, Any]:
        """Verify actuator control allocation equality."""
        achieved = float(B @ u)
        residual = abs(achieved - desired_tau)
        return {
            "achieved_virtual_control": achieved,
            "allocation_residual": residual,
            "verification_passed": residual < 1e-6,
        }


verifier = ControlVerifier()
