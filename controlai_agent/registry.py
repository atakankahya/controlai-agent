"""ControlAI Agent: Typed tool registry with JSON Schema validation and verifier execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import jsonschema


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

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {
                "status": "error",
                "error": f"Tool '{name}' is not registered. Available tools: {sorted(self._tools.keys())}",
            }

        # 1. Strict JSON Schema Validation
        param_schema = self._param_schemas[name]
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
            return result
        except Exception as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": f"Execution error in '{name}': {str(exc)}",
            }


registry = ToolRegistry()
