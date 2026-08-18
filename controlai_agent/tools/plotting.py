"""General mathematical function and signal plotting tools."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controlai_agent.registry import registry

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


SAFE_MATH_ENV: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinc": np.sinc,
    "arcsin": np.arcsin,
    "asin": np.arcsin,
    "arccos": np.arccos,
    "acos": np.arccos,
    "arctan": np.arctan,
    "atan": np.arctan,
    "arctan2": np.arctan2,
    "atan2": np.arctan2,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "heaviside": lambda x: np.heaviside(x, 1.0),
    "step": lambda x: (x >= 0).astype(float),
    "sign": np.sign,
    "rad2deg": np.rad2deg,
    "deg2rad": np.deg2rad,
}


@registry.register(
    name="plot_math_expression",
    description="Plot any mathematical function, curve, or signal f(t) or f(x) (e.g. 'sin(t)', 'exp(-t)*cos(2*t)', 'sin(x)', 't**2 - 3*t') and save the plot figure.",
    parameters_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression in terms of variable 't' or 'x', e.g. 'sin(t)', 'sin(x)', 'exp(-0.5*t)*sin(3*t)'.",
            },
            "t_start": {
                "type": "number",
                "description": "Start of range (default -10.0 for x, or 0.0 for t).",
            },
            "t_end": {
                "type": "number",
                "description": "End of range (default 10.0).",
            },
            "title": {
                "type": "string",
                "description": "Optional title for the plot.",
            },
            "xlabel": {
                "type": "string",
                "description": "Optional label for horizontal axis (default 't' or 'x').",
            },
            "ylabel": {
                "type": "string",
                "description": "Optional label for vertical axis (default 'f(t)' or 'y').",
            },
        },
        "required": ["expression"],
    },
)
def plot_math_expression(
    expression: str,
    t_start: float | None = None,
    t_end: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> dict[str, Any]:
    # Normalize expression (replace ^ with **)
    clean_expr = expression.replace("^", "**")
    
    # Detect independent variable (t or x)
    var_name = "x" if "x" in clean_expr and "t" not in clean_expr else "t"
    
    # Default range
    if t_start is None:
        t_start = -2 * np.pi if var_name == "x" else 0.0
    if t_end is None:
        t_end = 2 * np.pi if var_name == "x" else 10.0

    t_vals = np.linspace(t_start, t_end, 600)

    # Evaluate safely
    eval_env = dict(SAFE_MATH_ENV)
    for v in ["x", "t", "alpha", "theta", "phi", "omega", "u", "y", "s", "rad"]:
        eval_env[v] = t_vals

    try:
        y_vals = eval(clean_expr, {"__builtins__": {}}, eval_env)
        # Handle scalar constant output
        if np.isscalar(y_vals):
            y_vals = np.full_like(t_vals, float(y_vals))
        else:
            y_vals = np.asarray(y_vals, dtype=float)
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to evaluate expression '{expression}': {exc}",
        }

    # Plot styling
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    fig.patch.set_facecolor("#151b23")
    ax.set_facecolor("#0f1217")

    line_color = "#58a6ff"
    ax.plot(t_vals, y_vals, color=line_color, linewidth=2, label=f"${expression}$")
    ax.axhline(0, color="#484f58", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="#484f58", linestyle="--", linewidth=0.8, alpha=0.7)

    plot_title = title or f"Plot of $f({var_name}) = {expression}$"
    plot_xlabel = xlabel or var_name
    plot_ylabel = ylabel or f"f({var_name})"

    ax.set_title(plot_title, color="#f0f6fc", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel(plot_xlabel, color="#8b949e", fontsize=10)
    ax.set_ylabel(plot_ylabel, color="#8b949e", fontsize=10)
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, color="#8b949e")

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.legend(loc="best", facecolor="#151b23", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)
    plt.tight_layout()

    # Save unique plot
    hash_str = hashlib.md5(f"{clean_expr}_{t_start}_{t_end}_{time.time()}".encode()).hexdigest()[:8]
    out_file = PLOTS_DIR / f"plot_{hash_str}.png"
    plt.savefig(str(out_file), facecolor=fig.get_facecolor(), edgecolor="none", dpi=120)
    plt.close(fig)

    return {
        "status": "success",
        "plot_path": str(out_file),
        "expression": expression,
        "range": [float(t_start), float(t_end)],
    }
