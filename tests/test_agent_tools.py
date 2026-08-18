"""Comprehensive unit tests for ControlAI deterministic tools, JSON Schema validation, and verifiers."""

from __future__ import annotations

import unittest
import numpy as np

from controlai_agent.registry import registry
import controlai_agent.tools  # Register all tools


class TestControlAITools(unittest.TestCase):

    def test_exact_zoh(self) -> None:
        result = registry.execute("exact_zoh", {
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "Ts": 0.05,
        })
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["verification"]["verification_passed"])
        Ad = np.array(result["Ad"])
        self.assertEqual(Ad.shape, (2, 2))
        self.assertAlmostEqual(Ad[0, 0], 0.997621, places=4)

    def test_continuous_lqr(self) -> None:
        result = registry.execute("continuous_lqr", {
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "Q": [[10, 0], [0, 1]],
            "R": [[1]],
        })
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_stable"])
        self.assertTrue(result["verification"]["verification_passed"])
        self.assertLess(result["verification"]["riccati_residual_max"], 1e-10)

    def test_cbf_safety_filter(self) -> None:
        result = registry.execute("cbf_safety_filter", {
            "x": -0.3,
            "u_nom": -2.0,
            "alpha": 1.2,
            "x_min": -0.5,
        })
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(result["u_safe"], -0.24, places=6)
        self.assertTrue(result["is_constraint_active"])
        self.assertTrue(result["verification"]["verification_passed"])

    def test_kharitonov_stability(self) -> None:
        result = registry.execute("kharitonov_stability_test", {
            "lower_bounds": [1.0, 2.0, 3.0, 1.0],
            "upper_bounds": [1.3, 2.4, 3.5, 1.2],
        })
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_robustly_hurwitz"])
        self.assertEqual(len(result["kharitonov_polynomials_ascending"]), 4)

    def test_minimum_norm_allocation(self) -> None:
        result = registry.execute("minimum_norm_control_allocation", {
            "B": [1.0, 2.0, 3.0],
            "desired_tau": 14.0,
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["u"], [1.0, 2.0, 3.0])
        self.assertTrue(result["verification"]["verification_passed"])

    def test_mpc_solve_qp(self) -> None:
        result = registry.execute("mpc_solve_qp", {
            "A": [[1.0, 0.1], [0.0, 1.0]],
            "B": [[0.0], [0.1]],
            "Q": [[1.0, 0.0], [0.0, 0.1]],
            "R": [[0.01]],
            "x0": [2.0, 0.0],
            "horizon": 10,
            "u_limit": 5.0,
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["first_control_move"]), 1)
        self.assertEqual(len(result["predicted_state_trajectory"]), 2)

    def test_pid_tune_fopdt(self) -> None:
        result = registry.execute("pid_tune_fopdt", {
            "K_plant": 2.0,
            "T_tau": 5.0,
            "L_delay": 1.0,
            "tuning_objective": "setpoint_tracking_0_overshoot",
        })
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["Kp"], 0)
        self.assertGreater(result["Ki"], 0)
        self.assertGreater(result["Kd"], 0)

    def test_kalman_measurement_update(self) -> None:
        result = registry.execute("kalman_measurement_update", {
            "x_minus": [1.0, 0.5],
            "P_minus": [[1.0, 0.0], [0.0, 1.0]],
            "H": [[1.0, 0.0]],
            "R": [[0.1]],
            "z": [1.2],
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["x_plus_updated"]), 2)
        self.assertEqual(len(result["kalman_gain_K"]), 2)

    def test_step_response_simulation(self) -> None:
        result = registry.execute("simulate_step_response", {
            "numerator": [1.0],
            "denominator": [1.0, 2.0, 1.0],
            "sim_time": 5.0,
            "plot_filename": "test_step.png",
        })
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(result["final_value"], 1.0, places=1)
        self.assertIn("plot_artifact_path", result)

    def test_json_schema_validation_failure(self) -> None:
        # Pass string instead of number to exact_zoh
        result = registry.execute("exact_zoh", {
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "Ts": "not_a_number",
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "SchemaValidationError")

    def test_missing_required_parameter(self) -> None:
        result = registry.execute("continuous_lqr", {
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            # Missing Q and R
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "SchemaValidationError")


if __name__ == "__main__":
    unittest.main()
