"""ControlAI Agent: Typed tool registry with JSON Schema validation and verifier execution."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import jsonschema

# Every tool computes internally at full double precision -- this only
# affects what gets reported back. 6 significant figures is well past any
# real sensor/actuator precision, so nothing engineering-relevant is lost,
# while `K = [1.7416573867739407, 0.6719633404417155]` in a chat answer
# clearly is: raw float64 repr in prose reads as noise, not rigor.
RESULT_SIGNIFICANT_FIGURES = 6


def _round_significant(x: float, sig: int = RESULT_SIGNIFICANT_FIGURES) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, digits)


def _round_floats(obj: Any, sig: int = RESULT_SIGNIFICANT_FIGURES) -> Any:
    """Recursively round every float in a tool result to `sig` significant
    figures, leaving ints, bools, strings, and structure untouched."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _round_significant(obj, sig)
    if isinstance(obj, dict):
        return {k: _round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_round_floats(v, sig) for v in obj)
    return obj


def _parse_stringified_array(raw: str) -> Any:
    """Best-effort parse of a numeric array the model wrote as a JSON string.

    The model sometimes emits `"numerator": "[10]"` or even
    `"denominator": "[1 6 5 0]"` -- a string containing array-shaped text,
    including MATLAB/Numpy space-separated form, instead of an actual JSON
    array. Schema validation correctly rejects that as type "string" where
    "array" is required, and a perfectly usable numeric tool call is lost
    over pure formatting. Recover the intended array where unambiguous.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        spaced = re.sub(r"(?<=[\d\.\]])\s+(?=[\-\d\.\[])", ", ", stripped)
        try:
            return json.loads(spaced)
        except json.JSONDecodeError:
            pass
    return raw


def _coerce_array_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively repair string-typed values against `"type": "array"` schema
    properties (including nested arrays, e.g. matrix parameters) before
    validation, so a stringified array no longer fails a tool call outright.
    """

    def _coerce(value: Any, node: dict[str, Any]) -> Any:
        node_type = node.get("type")
        if node_type == "array" and isinstance(value, str):
            value = _parse_stringified_array(value)
        if node_type == "array" and isinstance(value, list) and "items" in node:
            return [_coerce(v, node["items"]) for v in value]
        return value

    props = schema.get("properties", {})
    return {
        key: (_coerce(val, props[key]) if key in props else val)
        for key, val in arguments.items()
    }


class ToolRegistry:
    """Registry for deterministic mathematical control tools with strict JSON Schema validation."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._param_schemas: dict[str, dict[str, Any]] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
    ) -> Callable:
        def decorator(func: Callable[..., dict[str, Any]]) -> Callable:
            self._tools[name] = func
            self._descriptions[name] = description
            self._param_schemas[name] = parameters_schema
            self._schemas[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters_schema,
                },
            }
            return func

        return decorator

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    def get_callables(self, exclude: set[str] = frozenset()) -> dict[str, Callable[..., dict[str, Any]]]:
        """Name -> underlying function for every registered tool except `exclude`.

        Used to expose the deterministic tools as plain callables inside the
        execute_python_code sandbox, since the model naturally expects a tool
        it knows by name (e.g. place_state_feedback) to be usable directly in
        code it writes, not only through the separate tool-call protocol.
        """
        return {name: fn for name, fn in self._tools.items() if name not in exclude}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {
                "status": "error",
                "error": f"Tool '{name}' is not registered. Available tools: {sorted(self._tools.keys())}",
            }

        # 1. Strict JSON Schema Validation (after repairing stringified arrays)
        param_schema = self._param_schemas[name]
        arguments = _coerce_array_arguments(arguments, param_schema)
        try:
            jsonschema.validate(instance=arguments, schema=param_schema)
        except jsonschema.ValidationError as schema_err:
            return {
                "status": "error",
                "error_type": "SchemaValidationError",
                "error": f"Invalid arguments for tool '{name}': {schema_err.message} (at path: {list(schema_err.path)})",
                "expected_schema": param_schema,
            }

        # 2. Execution & Deterministic Calculation
        try:
            func = self._tools[name]
            result = func(**arguments)
            if "status" not in result:
                result["status"] = "success"
            return _round_floats(result)
        except Exception as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": f"Execution error in '{name}': {str(exc)}",
            }


registry = ToolRegistry()
