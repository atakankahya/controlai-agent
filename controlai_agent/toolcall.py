"""Parsing of Qwen-style `<tool_call>` blocks out of a model response.

The previous implementation carried several hundred lines of regex repair --
unbalanced-brace closing, space-separated MATLAB array rewriting, LaTeX
backslash unescaping. Almost all of it existed to compensate for a fine-tuned
adapter that emitted malformed JSON; the base model emits well-formed calls, so
what remains here is a small, readable tolerance margin rather than a repair
pipeline. Argument-level coercion (stringified arrays and the like) already
happens in `registry.execute`, so it is deliberately not duplicated here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
# Some checkpoints emit a bare JSON object with no wrapping tags at all.
BARE_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _close_json(raw: str) -> str:
    """Append the brackets needed to balance a truncated JSON object.

    Generation can hit the token limit mid-call. Closing the structure recovers
    a usable call instead of discarding one that was merely cut short.
    """
    in_string = escaped = False
    stack: list[str] = []
    for ch in raw:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                stack.pop()
    return raw + ('"' if in_string else "") + "".join("}" if c == "{" else "]" for c in reversed(stack))


def _loads(raw: str) -> dict[str, Any] | None:
    for candidate in (raw, _close_json(raw)):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("name"):
            return parsed
    return None


def parse(text: str) -> tuple[list[ToolCall], str]:
    """Split a response into its tool calls and its prose.

    Returns `(calls, prose)`. An empty `calls` list means the response was a
    direct answer.
    """
    calls: list[ToolCall] = []
    for block in TOOL_CALL_RE.findall(text):
        parsed = _loads(block.strip())
        if parsed:
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            if isinstance(args, str):
                args = _loads(args) or {}
            calls.append(ToolCall(name=str(parsed["name"]), arguments=args if isinstance(args, dict) else {}))

    prose = TOOL_CALL_RE.sub("", text)
    if not calls:
        for block in BARE_CALL_RE.findall(prose):
            parsed = _loads(block)
            if parsed:
                args = parsed.get("arguments") or {}
                calls.append(ToolCall(name=str(parsed["name"]), arguments=args if isinstance(args, dict) else {}))
                prose = prose.replace(block, "")

    prose = re.sub(r"<think>.*?</think>", "", prose, flags=re.DOTALL)
    prose = re.sub(r"</?(?:think|tool_call|tool_response)>", "", prose)
    return calls, prose.strip()


_MAX_ARRAY_LEN = 400
_MAX_IDENTICAL_RUN = 12


def degenerate_reason(value: Any, depth: int = 0) -> str | None:
    """Detect runaway repetition in a tool argument.

    A decoding loop can produce a several-hundred-element array of the same
    number, which is never a real system and wastes a step plus a large slice
    of context if it is executed. The thresholds are set well above any
    plausible hand-written matrix so ordinary inputs are never touched.
    """
    if depth > 4 or not isinstance(value, list):
        return None
    if len(value) > _MAX_ARRAY_LEN:
        return f"has {len(value)} elements"
    run = 1
    for prev, cur in zip(value, value[1:]):
        run = run + 1 if prev == cur and isinstance(cur, (int, float)) else 1
        if run >= _MAX_IDENTICAL_RUN:
            return f"repeats the value {cur} {run} times in a row"
    for item in value:
        reason = degenerate_reason(item, depth + 1)
        if reason:
            return reason
    return None
