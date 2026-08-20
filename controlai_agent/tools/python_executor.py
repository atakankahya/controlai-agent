"""Jupyter-style deterministic Python execution tool for Control-LLM."""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import linalg, signal
try:
    import control as ct
except ImportError:
    ct = None

from controlai_agent.registry import registry

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@registry.register(
    name="execute_python_code",
    description=(
        "Execute Python code for control engineering, numerical simulations, differential equations, "
        "optimization, and signal plotting (similar to a Jupyter Notebook cell). Can use numpy (np), "
        "scipy (scipy), scipy.signal (signal), scipy.linalg (linalg), control (ct, control), and "
        "matplotlib.pyplot (plt). Captures stdout and any generated Matplotlib figures. "
        "IMPORTANT -- the `control` package (ct) takes POSITIONAL arguments only, never num=/den= "
        "or sys1=/sys2= keywords (they raise 'Needs 1, 2, or 3 arguments'): "
        "ct.tf(num, den) not ct.tf(num=num, den=den); "
        "ct.feedback(sys1, sys2=1, sign=-1) for a closed loop; "
        "ct.series(sys1, sys2) and ct.parallel(sys1, sys2); "
        "ct.step_response(sys, T=t_array) returns (T, yout); "
        "ct.poles(sys) and ct.zeros(sys) for pole/zero locations; "
        "ct.bode(sys) / ct.bode_plot(sys) is PLOT-ONLY and does not return (mag, phase, omega) arrays "
        "-- for numeric Bode data use resp = ct.frequency_response(sys, omega); "
        "resp.magnitude, resp.phase (radians), resp.omega. "
        "Every other registered tool (continuous_lqr, discrete_lqr, place_state_feedback, "
        "stability_margins, exact_zoh, etc.) is ALSO directly callable here by its exact name with "
        "its normal arguments -- e.g. `result = place_state_feedback(A=A, B=B, desired_poles=poles)`. "
        "Each returns a dict just like the standalone tool call does, so pull out the field you need, "
        "e.g. `K = np.array(result['K'])`, before using it in further computation."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Valid Python code to execute. Can use numpy, scipy, control, matplotlib.pyplot, etc.",
            },
        },
        "required": ["code"],
    },
)
def execute_python_code(code: str) -> dict[str, Any]:
    """Execute Python code in an isolated namespace and capture stdout + plots."""
    # Setup execution environment
    stdout_capture = io.StringIO()
    plt.close("all")  # Reset existing figures

    # Configure plot defaults for dark/clean aesthetic
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = "#151b23"
    plt.rcParams["axes.facecolor"] = "#0f1217"
    plt.rcParams["axes.edgecolor"] = "#30363d"
    plt.rcParams["axes.labelcolor"] = "#8b949e"
    plt.rcParams["xtick.color"] = "#8b949e"
    plt.rcParams["ytick.color"] = "#8b949e"
    plt.rcParams["grid.color"] = "#30363d"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    exec_globals: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "linalg": linalg,
        "signal": signal,
        "plt": plt,
        "matplotlib": matplotlib,
        "ct": ct,
        "control": ct,
    }
    # Every other registered deterministic tool (continuous_lqr, place_state_feedback,
    # stability_margins, ...) is also callable directly by name here, with its
    # normal keyword arguments and dict return value -- the model otherwise
    # reasonably expects a tool it knows to be usable in code it writes, not
    # only through the separate tool-call protocol, and hits a NameError.
    exec_globals.update(registry.get_callables(exclude={"execute_python_code"}))

    saved_plots = []

    # Intercept any direct plt.savefig / fig.savefig calls so they are redirected into outputs/plots
    orig_plt_savefig = plt.savefig
    orig_fig_savefig = matplotlib.figure.Figure.savefig

    def safe_plt_savefig(fname, *args, **kwargs):
        fname_name = Path(fname).name if fname else f"plot_{int(time.time()*1000)}.png"
        target_path = PLOTS_DIR / fname_name
        saved_plots.append(str(target_path))
        return orig_plt_savefig(str(target_path), *args, **kwargs)

    def safe_fig_savefig(self, fname, *args, **kwargs):
        fname_name = Path(fname).name if fname else f"plot_{int(time.time()*1000)}.png"
        target_path = PLOTS_DIR / fname_name
        saved_plots.append(str(target_path))
        return orig_fig_savefig(self, str(target_path), *args, **kwargs)

    plt.savefig = safe_plt_savefig
    matplotlib.figure.Figure.savefig = safe_fig_savefig

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            exec(code, exec_globals)

        stdout_val = stdout_capture.getvalue().strip()

        # Check if any matplotlib figures remain unsaved
        fig_nums = plt.get_fignums()
        if fig_nums:
            for num in fig_nums:
                fig = plt.figure(num)
                hash_val = hashlib.md5(f"{code}_{num}_{time.time()}".encode()).hexdigest()[:8]
                out_path = PLOTS_DIR / f"plot_exec_{hash_val}.png"
                orig_fig_savefig(fig, str(out_path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
                saved_plots.append(str(out_path))
            plt.close("all")

        result = {
            "status": "success",
            "stdout": stdout_val if stdout_val else "Code executed successfully.",
            "executed_code": code,
        }
        if saved_plots:
            result["plot_path"] = saved_plots[0]
            result["all_plots"] = list(dict.fromkeys(saved_plots))

        return result

    except Exception as exc:
        plt.close("all")
        return {
            "status": "error",
            "error": str(exc),
            "stdout": stdout_capture.getvalue().strip(),
            "executed_code": code,
        }
    finally:
        plt.savefig = orig_plt_savefig
        matplotlib.figure.Figure.savefig = orig_fig_savefig
