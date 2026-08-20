"""Deterministic linear-algebra tool: the basic matrix operations every other
analysis builds on. Direct requests like "A times B", "invert this matrix", or
"eigenvalues of A" must never depend on model-written freehand code -- they get
one schema-validated tool call."""

from __future__ import annotations

from typing import Any

import numpy as np

from controlai_agent.registry import registry


@registry.register(
    name="matrix_arithmetic",
    description=(
        "Exact matrix arithmetic on one or two numeric matrices. THE tool for any direct matrix "
        "computation the user states explicitly: 'A times B' / 'A*B' (operation='multiply'), "
        "addition, subtraction, transpose, inverse, determinant, rank, trace, eigenvalues, or "
        "elementwise multiply. Use this instead of writing Python code for plain matrix math. "
        "This is NOT a controller-design tool: a request to multiply or invert matrices is a "
        "linear-algebra question, not an LQR/pole-placement problem."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "multiply",
                    "add",
                    "subtract",
                    "elementwise_multiply",
                    "transpose",
                    "inverse",
                    "determinant",
                    "rank",
                    "trace",
                    "eigenvalues",
                ],
                "description": "Which operation to perform. Binary operations use matrix_a (op) matrix_b in that order.",
            },
            "matrix_a": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "First matrix (2D, row-major). For unary operations this is the only operand.",
            },
            "matrix_b": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Second matrix, required for multiply / add / subtract / elementwise_multiply.",
            },
        },
        "required": ["operation", "matrix_a"],
    },
)
def matrix_arithmetic(
    operation: str,
    matrix_a: list[list[float]],
    matrix_b: list[list[float]] | None = None,
) -> dict[str, Any]:
    A = np.array(matrix_a, dtype=float)
    binary_ops = {"multiply", "add", "subtract", "elementwise_multiply"}

    if operation in binary_ops:
        if matrix_b is None:
            return {"status": "error", "error": f"operation '{operation}' requires matrix_b."}
        B = np.array(matrix_b, dtype=float)
        if operation == "multiply":
            if A.shape[1] != B.shape[0]:
                return {
                    "status": "error",
                    "error": f"Cannot multiply: matrix_a is {A.shape[0]}x{A.shape[1]} but matrix_b is {B.shape[0]}x{B.shape[1]} (inner dimensions must match).",
                }
            result = A @ B
        elif operation == "elementwise_multiply":
            if A.shape != B.shape:
                return {"status": "error", "error": f"Elementwise multiply needs equal shapes, got {A.shape} and {B.shape}."}
            result = A * B
        else:
            if A.shape != B.shape:
                return {"status": "error", "error": f"'{operation}' needs equal shapes, got {A.shape} and {B.shape}."}
            result = A + B if operation == "add" else A - B
        return {
            "operation": operation,
            "shape_a": list(A.shape),
            "shape_b": list(B.shape),
            "result": result.tolist(),
            "result_shape": list(result.shape),
        }

    # Unary operations
    out: dict[str, Any] = {"operation": operation, "shape_a": list(A.shape)}
    if operation == "transpose":
        out["result"] = A.T.tolist()
    elif operation == "rank":
        out["result"] = int(np.linalg.matrix_rank(A))
    elif operation in ("inverse", "determinant", "trace", "eigenvalues"):
        if A.shape[0] != A.shape[1]:
            return {"status": "error", "error": f"'{operation}' requires a square matrix, got {A.shape[0]}x{A.shape[1]}."}
        if operation == "inverse":
            det = float(np.linalg.det(A))
            if abs(det) < 1e-12:
                return {"status": "error", "error": f"Matrix is singular (determinant = {det:g}); no inverse exists."}
            out["result"] = np.linalg.inv(A).tolist()
            out["determinant"] = det
        elif operation == "determinant":
            out["result"] = float(np.linalg.det(A))
        elif operation == "trace":
            out["result"] = float(np.trace(A))
        else:  # eigenvalues
            eig = np.linalg.eigvals(A)
            out["result"] = [[float(v.real), float(v.imag)] for v in eig]
            out["spectral_radius"] = float(np.max(np.abs(eig)))
    return out
