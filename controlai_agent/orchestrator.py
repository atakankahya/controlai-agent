"""Control-LLM Agent Orchestrator: Multi-step tool calling, execution, grounding, and streaming synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import os
import sys

try:
    from mlx_lm import generate as mlx_generate, load as mlx_load
    from mlx_lm.sample_utils import make_logits_processors
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from transformers import AutoModelForCausalLM, AutoTokenizer

from controlai_agent.prompts import CONTROLAI_SYSTEM_PROMPT
from controlai_agent.registry import ToolRegistry, registry
import controlai_agent.tools  # noqa: F401 (ensure all tools are registered)
from controlai_rag.index import get_shared_index


@dataclass
class ToolExecutionTrace:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResult:
    final_response: str
    tool_traces: list[ToolExecutionTrace] = field(default_factory=list)
    total_steps: int = 0
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    is_grounded: bool = True
    plots: list[str] = field(default_factory=list)


def sanitize_json_escapes(raw: str) -> str:
    """Escape unescaped backslashes in JSON strings (e.g. \\dot, \\omega, \\mu in math/code)."""
    return re.sub(r"\\(?![/\"\\bfnrtu])", r"\\\\", raw)


def fix_space_separated_arrays(json_str: str) -> str:
    """Convert MATLAB/Numpy space-separated lists inside brackets [1 2 3] to [1, 2, 3]."""
    def _fix_brackets(match: re.Match) -> str:
        content = match.group(1).strip()
        # Replace spaces between numbers with commas
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", content)
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", fixed)
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", fixed)
        return f"[{fixed}]"
    return re.sub(r"\[\s*([0-9eE\.\-+\s]+?)\s*\]", _fix_brackets, json_str)


def close_unbalanced_json(raw: str) -> str:
    """Append the closing braces/brackets a truncated JSON object is missing.

    Small quantized models routinely drop the final `}` of a tool call (emitting
    `{"name": ..., "arguments": {...}` with the outer object left open). Without
    repair the call fails to parse, the tool never runs, and the turn collapses
    into an empty answer -- so balance the delimiters and let it parse.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()

    repaired = raw
    if in_string:
        repaired += '"'
    # A generation cut off by the token budget mid-array/object usually stops
    # right after a comma (about to write the next element) -- a dangling
    # trailing comma is invalid JSON even once the brackets below are
    # balanced ("[1, 0, 0,]" still fails to parse), so drop it first.
    if not in_string:
        repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def parse_flexible_json(raw_str: str) -> dict[str, Any] | None:
    """Parse JSON with fallback to escape sanitization, array fixing, and non-strict control characters."""
    raw_str = raw_str.strip()
    try:
        obj = json.loads(raw_str, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        sanitized = sanitize_json_escapes(raw_str)
        obj = json.loads(sanitized, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        fixed_arrays = fix_space_separated_arrays(sanitize_json_escapes(raw_str))
        obj = json.loads(fixed_arrays, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Last resort: the object is well-formed but truncated (a dropped closing
    # brace), so balance the delimiters and retry each variant.
    for candidate in (
        raw_str,
        sanitize_json_escapes(raw_str),
        fix_space_separated_arrays(sanitize_json_escapes(raw_str)),
    ):
        try:
            obj = json.loads(close_unbalanced_json(candidate), strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


# A doubled backslash is an accidental JSON-style escape when it precedes:
#   - a command name       \\zeta  \\frac  \\sum
#   - a delimiter escape   \\|  \\{  \\}  \\(  \\)  \\]
#   - \\[ that does NOT begin a spacing argument such as a genuine "\\[2pt]"
# A real LaTeX row break inside matrix/aligned is followed by whitespace, a
# newline, or a digit-led spacing option -- never by any of the above -- so
# these rewrites leave legitimate line breaks untouched.
_LATEX_DOUBLE_BACKSLASH_RE = re.compile(r"\\\\(?=[a-zA-Z|{}()\]])|\\\\(?=\[\s*[^\d\s])")


def _fix_doubled_latex_backslashes(text: str) -> str:
    """Collapse an accidental JSON-escape-style double backslash before a LaTeX
    command or delimiter (\\\\zeta, \\\\|) down to the single backslash KaTeX
    expects (\\zeta, \\|).

    The base model was heavily trained on JSON-argument function calling, where
    a literal backslash must be written as `\\\\` inside a JSON string. That
    habit leaks into plain-text math even outside of a tool call. Left alone,
    `\\\\|x - x_f\\\\|^2` renders as a line break followed by a stray pipe,
    tearing a norm across two lines instead of drawing it.
    """
    return _LATEX_DOUBLE_BACKSLASH_RE.sub(lambda m: "\\", text)


def _extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse tool calls from <tool_call>, markdown code blocks, or raw JSON robustly."""
    calls: list[dict[str, Any]] = []

    # 1. Standard <tool_call> tags
    for match in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text):
        obj = parse_flexible_json(match.group(1))
        if obj and "name" in obj:
            calls.append(obj)

    # 2. Markdown json blocks with tool call schema
    if not calls:
        for match in re.finditer(r"```(?:json)?\s*(\{\s*[\"']name[\"']\s*:[\s\S]*?\})\s*```", text):
            obj = parse_flexible_json(match.group(1))
            if obj and "name" in obj:
                calls.append(obj)

    # 3. Raw JSON object containing "name" and "arguments" / "parameters"
    if not calls and ('"name"' in text or "'name'" in text):
        match = re.search(r"(\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*\})", text)
        if not match:
            # Generation was truncated mid-JSON (e.g. a repetition loop hit the
            # token budget before the array closed), so there's no closing "}"
            # to anchor on -- fall back to matching to the end of the string so
            # close_unbalanced_json can still repair it and _degenerate_array_reason
            # can refuse it, instead of the raw truncated JSON leaking into the chat.
            match = re.search(r"(\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*)", text)
        if match:
            obj = parse_flexible_json(match.group(1))
            if obj and "name" in obj:
                calls.append(obj)

    # Clean pre-tool thought / raw JSON artifacts
    cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*```", "", cleaned)
    cleaned = re.sub(r"\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*\}", "", cleaned)
    # Same truncated-JSON fallback as above, so the leftover raw JSON is
    # stripped from the visible text even when it never closed.
    cleaned = re.sub(r"\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*", "", cleaned)
    # An orphaned <tool_call> tag with no matching close (the model started a
    # call, then abandoned it mid-generation for plain text) survives the
    # paired regex above -- strip any leftover tag so it never reaches the UI.
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = _fix_doubled_latex_backslashes(cleaned)
    return calls, cleaned


# Applied across every inference backend below. Without it, a small quantized
# model under low-temperature decoding has no defense against falling into a
# token-repetition loop once it starts (observed in production as
# routh_hurwitz_analysis coefficient arrays of 300+ repeated zeros): it burns
# the entire max_tokens budget on garbage, which is both the direct cause of
# the degenerate-array tool-call failures and a major source of latency.
REPETITION_PENALTY = 1.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Our own fine-tuned model's repo -- every GGUF/Ollama/CUDA backend must load
# ITS tokenizer and chat template (never a generic base model's), since the
# tool-call format, special tokens, and vocab are specific to this fine-tune.
CONTROLAI_HF_REPO = "atakankahya/ControlAI-Agent"

# How many times a single tool may be invoked within one user turn. Two allows
# a legitimate retry with corrected arguments after an error, while stopping
# the model from spending its whole step budget re-running the same lookup.
MAX_CALLS_PER_TOOL = 2


# ---------------------------------------------------------------------------
# Parameter provenance enforcement
#
# Prompt rules against inventing parameters demonstrably do not hold on a 4B
# model: given "A times A" after an earlier LQR conversation, it fabricated a
# brand-new B, resized Q, and ran a confident LQR synthesis -- three separate
# prompt formulations failed to stop it. So faithfulness is enforced in code:
# every 2D matrix handed to a tool must be traceable to the user's messages or
# a prior tool result in this conversation, or the call is refused before it
# executes and the model is told to ask the user for the missing value.
# ---------------------------------------------------------------------------

# Parameters exempt from the guard:
#  - C/D: conventional output-selector / feedthrough matrices (entries 0/1),
#    routinely and legitimately chosen by the designer, and harmless.
#  - desired_poles: pole locations are design choices derived from specs
#    ("settling time under 2s"), not data the user must dictate literally.
_PROVENANCE_EXEMPT_PARAMS = {"C", "D", "desired_poles"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _extract_arrays_from_text(text: str) -> list[list]:
    """Find every parseable bracketed numeric array (1D or 2D) in free text."""
    arrays: list[list] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            i += 1
            continue
        candidate = text[i : j + 1]
        # Third variant: users mix separators freely ("[3 0, 1]"), which the
        # comma-only fixer can't handle -- insert a comma between any two
        # adjacent number tokens regardless of other separators present.
        mixed_fixed = re.sub(r"(?<=[\d.])\s+(?=[-\d.])", ", ", candidate)
        for variant in (candidate, fix_space_separated_arrays(candidate), mixed_fixed):
            try:
                parsed = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and parsed:
                arrays.append(parsed)
                break
        # Only skip past this bracket char, not the whole span: inner arrays
        # of a 2D matrix should also be collected individually (rows are
        # legitimate 1D vectors in their own right).
        i += 1
    return arrays


def _collect_numeric_leaves(obj: Any, arrays: list[list], numbers: set[float]) -> None:
    """Harvest nested numeric lists and scalars from a parsed JSON object."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        numbers.add(float(obj))
        return
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            arrays.append(list(obj))
        elif obj and all(isinstance(v, list) for v in obj):
            arrays.append(obj)
        for v in obj:
            _collect_numeric_leaves(v, arrays, numbers)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numeric_leaves(v, arrays, numbers)


def _as_2d_float_array(value: Any):
    """Return value as a 2D float ndarray, or None if it isn't one."""
    import numpy as _np

    try:
        arr = _np.array(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1 and arr.size > 0:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


class ParameterProvenance:
    """Everything numeric the conversation has actually provided so far."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        import numpy as _np

        self._np = _np
        self.arrays: list[Any] = []
        self.current_numbers: set[float] = set()
        all_user_texts: list[str] = []
        current_user_text = ""

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                all_user_texts.append(content)
                current_user_text = content
            elif role == "tool" and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                found_arrays: list[list] = []
                found_numbers: set[float] = set()
                _collect_numeric_leaves(parsed, found_arrays, found_numbers)
                for a in found_arrays:
                    arr = _as_2d_float_array(a)
                    if arr is not None:
                        self.arrays.append(arr)
                # Scalars computed by tools (gains, margins) are legitimate
                # inputs for later steps.
                self.current_numbers |= found_numbers

        for text in all_user_texts:
            for a in _extract_arrays_from_text(text):
                arr = _as_2d_float_array(a)
                if arr is not None:
                    self.arrays.append(arr)
        # Loose scalars only count from the CURRENT user message: numbers from
        # an earlier, different problem are exactly what must not silently
        # seed a new Q or R.
        for m in _NUMBER_RE.finditer(current_user_text):
            try:
                self.current_numbers.add(float(m.group()))
            except ValueError:
                pass

    def _matches_known_array(self, arr) -> bool:
        np_ = self._np
        for known in self.arrays:
            for cand in (known, known.T):
                if cand.shape == arr.shape and np_.allclose(cand, arr, rtol=1e-6, atol=1e-9):
                    return True
        return False

    def _number_provided(self, x: float) -> bool:
        if x in (0.0, 1.0, -1.0):
            return True
        return any(abs(x - n) <= 1e-9 * max(1.0, abs(n)) for n in self.current_numbers)

    def verify(self, value: Any) -> bool:
        """True if this matrix is traceable to the conversation."""
        np_ = self._np
        arr = _as_2d_float_array(value)
        if arr is None:
            return True  # not a matrix -- out of scope for this guard

        if self._matches_known_array(arr):
            return True

        # 1x1 "matrix" wrapping a scalar the user stated (R=1 -> [[1]]).
        if arr.shape == (1, 1):
            return self._number_provided(float(arr[0, 0]))

        # Identity / zero matrices are structural, not data.
        if arr.shape[0] == arr.shape[1]:
            if np_.allclose(arr, np_.eye(arr.shape[0])) or np_.allclose(arr, 0.0):
                return True
            # diag(...) built from numbers in the current message, e.g. the
            # user wrote "Q=diag([10, 1])" in prose rather than as an array.
            if np_.allclose(arr, np_.diag(np_.diag(arr))):
                return all(self._number_provided(float(d)) for d in np_.diag(arr))

        return False


# Real control-engineering coefficient/numerator/denominator arrays are
# essentially always short -- a 10th-order polynomial (already an extreme
# hand-solved case) has 11 coefficients. A flat numeric array cannot be
# provenance-traced the way a matrix can (a legitimately *expanded* factored
# polynomial, e.g. (s+1)(s+2)(s+3) -> [1,6,11,6], never appears verbatim in
# the user's text, so a strict "must match the conversation" rule would
# wrongly block correct derivations). Instead this catches the actual
# observed failure mode directly: a runaway repetition loop, where a small
# model gets stuck emitting the same value and the token budget cuts it off
# mid-array -- e.g. 300+ elements of mostly zeros for routh_hurwitz_analysis
# on a question that never gave it a polynomial at all.
_MAX_SANE_1D_ARRAY_LEN = 25
_MAX_IDENTICAL_RUN = 6


def _degenerate_array_reason(values: list) -> str | None:
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    if len(values) > _MAX_SANE_1D_ARRAY_LEN:
        return f"has {len(values)} elements, far beyond any real control-engineering array of this kind"
    run = 1
    for i in range(1, len(values)):
        run = run + 1 if values[i] == values[i - 1] else 1
        if run >= _MAX_IDENTICAL_RUN:
            return f"repeats the value {values[i]!r} {run}+ times in a row -- a runaway generation loop, not real data"
    return None


def _check_parameter_provenance(
    tool_args: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[str, Any, str] | None:
    """Return (param_name, value, reason) for the first bad matrix/array, else None."""
    provenance = ParameterProvenance(messages)
    for param, value in tool_args.items():
        if param in _PROVENANCE_EXEMPT_PARAMS:
            continue
        if not (isinstance(value, list) and value):
            continue
        if isinstance(value[0], list):
            if not provenance.verify(value):
                return param, value, "was not provided by the user in this conversation and did not come from any prior tool result"
        else:
            reason = _degenerate_array_reason(value)
            if reason:
                return param, value, reason
    return None


# ---------------------------------------------------------------------------
# RAG fast path
#
# The system prompt already tells the model "for definitional questions,
# answer from the retrieved reference passages, do not run a numeric solver"
# -- but that instruction demonstrably does not hold on a 4B model either: in
# production, "what is Routh-Hurwitz, explain with an example" (fully
# answerable from the Nise/Ogata passages already injected into the system
# prompt by _get_grounded_instruction) still drove the model through 4 tool
# calls (a redundant re-search plus three unrelated numeric tools) and 5
# sequential full-length generations before it produced an answer. Since
# prompting alone can't be trusted here any more than it can for parameter
# provenance above, this is enforced the same way: in code.
#
# A real computational request in this domain essentially always names a
# concrete number (a coefficient, a gain, a frequency) or an explicit
# computational verb ("plot", "design", "simulate"); a pure "what is X" /
# "explain X" / "how does X work" question has neither. This heuristic only
# ever widens back to the full tool loop on a false negative (a computational
# question with no digits and none of these verbs) -- it never narrows
# correctness, since the loop it skips still runs whenever this returns True.
_COMPUTATION_KEYWORDS = (
    "plot", "simulate", "simulation", "compute", "calculate", "design",
    "solve", "gain", "matrix", "matrices", "pole", "place", "locus", "bode",
    "nyquist", "margin", "response", "transfer function", "eigen",
    "controllab", "observab", "lyapunov", "kalman", "mpc", "invert",
    "determinant", "transpose", "multiply", "rank", "code", "script",
)


def _needs_tools(user_prompt: str) -> bool:
    """True if the question plausibly needs a numeric tool rather than being
    answerable straight from grounded reference text."""
    if any(ch.isdigit() for ch in user_prompt):
        return True
    lower = user_prompt.lower()
    return any(kw in lower for kw in _COMPUTATION_KEYWORDS)


def _fabrication_refusal_note(refusals: list[str]) -> str:
    """Extra synthesis instruction appended when a tool call was refused for
    inventing a parameter.

    Without this, the model's forced final-answer turn happily hallucinates a
    plausible-looking numeric replacement anyway: observed directly with
    sft_v2 on "design an LQR controller for A = [[0, 1], [-2, -3]]" (no B, Q,
    R given) -- the tool call was correctly REFUSED by the provenance guard,
    but the final prose still confidently printed a fully invented gain
    $K = [1.83, 1.83]$, silently routing around its own refusal. The system
    prompt already forbids this (rule 6), but that alone doesn't hold on a 4B
    model any more than the provenance rules did in prose form -- so the
    refusal is restated directly in the synthesis turn itself, where it can't
    be missed.
    """
    if not refusals:
        return ""
    return (
        "\n\nIMPORTANT: at least one tool call above was REFUSED for inventing a parameter you were never "
        "given (see the REFUSED error message(s) in the tool results above for exactly which one and why). "
        "This means you do NOT have enough information to compute a numeric answer for that part of the "
        "request. Do NOT invent a substitute number, gain, matrix, or result to answer anyway -- state "
        "plainly what is missing and ask the user for it instead of guessing."
    )


class ControlAIAgent:
    """Universal Control Engineering Agent supporting GGUF, Ollama C++, Apple MLX, and PyTorch."""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        adapter_path: str | None = None,
        tool_registry: ToolRegistry = registry,
        max_tool_steps: int = 4,
    ) -> None:
        self.model_path = model_path
        
        # Auto-detect trained LoRA adapter if none explicitly provided
        default_adapter = PROJECT_ROOT / "adapters" / "controlai_qwen3_4b_sft_v2"
        if adapter_path is None and default_adapter.exists():
            adapter_path = str(default_adapter)

        self.adapter_path = adapter_path
        self.registry = tool_registry
        self.max_tool_steps = max_tool_steps

        # Detect platform & backend
        self.is_ollama = str(model_path).startswith("ollama")
        self.is_gguf = str(model_path).endswith(".gguf") or "gguf" in str(model_path).lower()
        self.is_mlx = HAS_MLX and not self.is_ollama and not self.is_gguf and not str(model_path).startswith("Qwen/") and not os.environ.get("FORCE_TRANSFORMERS")

        if self.is_ollama:
            self.ollama_model = model_path.split(":", 1)[1] if ":" in str(model_path) else "controlai"
            self.hf_tokenizer = AutoTokenizer.from_pretrained(CONTROLAI_HF_REPO, trust_remote_code=True)
        elif self.is_gguf:
            if not HAS_LLAMA_CPP:
                raise ImportError("llama-cpp-python is required to run GGUF models. Install it with: pip install llama-cpp-python")
            self.llama_model = llama_cpp.Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,  # Offload all layers to Metal / CUDA GPU
                n_ctx=16384,
                verbose=False,
            )
            # A local .gguf path carries no tokenizer/chat-template of its own --
            # always load ours (never a generic base model's), same as the
            # auto-downloaded deployment path below.
            local_fused = PROJECT_ROOT / "models" / "controlai_fused"
            tokenizer_source = str(local_fused) if local_fused.exists() else CONTROLAI_HF_REPO
            self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        elif self.is_mlx:
            if adapter_path:
                self.model, self.mlx_tokenizer = mlx_load(model_path, adapter_path=adapter_path)
            else:
                self.model, self.mlx_tokenizer = mlx_load(model_path)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Universal Linux / Cloud / HuggingFace Spaces backend: fast C++ GGUF
            # of OUR OWN fine-tuned ControlAI model (never a generic base model).
            #
            # Q8_0, not Q4_K_M: measured directly against this exact model on the
            # "simulate a step response" prompt (3 runs each). Q4_K_M mis-routed
            # tool calls in 3/3 runs (reached for continuous_lqr/bode_analysis
            # instead of simulate_step_response, then hallucinated inconsistent
            # overshoot values -- 8.6%, 43.1%, 25.4%, none near the true ~9.5%).
            # Q6_K, Q8_0, and full f16 all called the correct tool with
            # consistent, tool-verified numbers in 3/3 runs apiece, matching
            # local MLX behavior -- Q8_0 was picked among those three because it
            # was both the fastest of the three in this test (8-19s vs Q6_K's
            # 11-25s and f16's 11-22s) and, at 8.5 bits/weight, close enough to
            # f16 to be considered near-lossless, at roughly half f16's size.
            gguf_repo = os.environ.get("CONTROLAI_GGUF_REPO", CONTROLAI_HF_REPO)
            gguf_filename = os.environ.get("CONTROLAI_GGUF_FILENAME", "*controlai-q8_0.gguf")

            self.llama_model = None
            if HAS_LLAMA_CPP:
                try:
                    threads = min(os.cpu_count() or 2, 4)
                    print(f"Loading high-speed GGUF Q4_K_M ControlAI model from {gguf_repo} (threads: {threads})...")
                    self.llama_model = llama_cpp.Llama.from_pretrained(
                        repo_id=gguf_repo,
                        filename=gguf_filename,
                        n_ctx=16384,
                        n_threads=threads,
                        verbose=False,
                    )
                    self.is_gguf = True
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(CONTROLAI_HF_REPO, trust_remote_code=True)
                    print("GGUF Q4_K_M ControlAI model loaded successfully via llama_cpp.")
                except Exception as exc:
                    print(f"Notice: llama_cpp GGUF auto-load failed, falling back to PyTorch: {exc}")
                    self.is_gguf = False

            if not self.is_gguf:
                # PyTorch fallback on CUDA or if the GGUF engine is unavailable.
                # Always our own fine-tuned model, never a generic base model.
                import torch
                num_threads = min(os.cpu_count() or 2, 4)
                try:
                    torch.set_num_threads(num_threads)
                except Exception:
                    pass

                hf_id = CONTROLAI_HF_REPO if "mlx" in str(model_path) or str(model_path).startswith("Qwen/") else model_path
                print(f"Loading PyTorch model: {hf_id} (threads: {num_threads})...")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                has_cuda = torch.cuda.is_available()
                dtype = torch.bfloat16 if has_cuda else torch.float32
                # SDPA is a large, free speedup over the "eager" attention
                # default -- meaningful for the long tool-schema prompt prefix.
                attn_impl = "sdpa" if has_cuda else None
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                )
                if has_cuda:
                    # device_map="auto" infers available VRAM at load time; on
                    # ZeroGPU no physical GPU is attached to the process yet at
                    # this point (that only happens inside a @spaces.GPU call),
                    # so it can silently offload layers to CPU. An explicit
                    # move avoids that and any ambiguity about what actually
                    # ran where.
                    self.model = self.model.to("cuda")
                # The fused ControlAI model already has the LoRA weights merged in;
                # only apply a separate adapter when loading a plain base model.
                if hf_id != CONTROLAI_HF_REPO and adapter_path and Path(adapter_path).exists():
                    try:
                        from peft import PeftModel
                        self.model = PeftModel.from_pretrained(self.model, adapter_path)
                        print(f"Loaded PEFT LoRA adapter from: {adapter_path}")
                    except Exception as exc:
                        print(f"Warning: Could not load LoRA adapter in PyTorch: {exc}")
                print(f"PyTorch model loaded successfully. Model device: {next(self.model.parameters()).device}")

        # Initialize local offline RAG index
        try:
            self.rag_index = get_shared_index()
        except Exception:
            self.rag_index = None

    def _generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Universal text generation handling Ollama C++, GGUF llama_cpp, MLX, and PyTorch."""
        if self.is_ollama:
            res = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": max_tokens, "repeat_penalty": REPETITION_PENALTY},
            )
            return res.get("response", "").strip()
        elif self.is_gguf:
            # llama_cpp defaults repeat_penalty to 1.0 (fully off) when unset --
            # it does NOT inherit any sane default, so this must be passed explicitly.
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
                temperature=0.2,
                repeat_penalty=REPETITION_PENALTY,
            )
            return output["choices"][0]["text"].strip()
        elif self.is_mlx:
            return mlx_generate(
                self.model,
                self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                logits_processors=make_logits_processors(repetition_penalty=REPETITION_PENALTY),
                verbose=False,
            ).strip()
        else:
            import time as _time
            import torch
            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            eos_ids = [
                tid
                for tid in (self.hf_tokenizer.eos_token_id, self.hf_tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if tid is not None and tid >= 0
            ] or None
            t0 = _time.time()
            print(f"[_generate] starting: prompt_tokens={prompt_tokens} max_new_tokens={max_tokens} eos_ids={eos_ids}")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=REPETITION_PENALTY,
                    eos_token_id=eos_ids,
                    pad_token_id=self.hf_tokenizer.pad_token_id or self.hf_tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][prompt_tokens:]
            elapsed = _time.time() - t0
            print(f"[_generate] done: generated_tokens={len(new_tokens)} elapsed={elapsed:.1f}s ({len(new_tokens)/max(elapsed,0.001):.1f} tok/s)")
            return self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _get_grounded_instruction(self, user_prompt: str, base_instruction: str) -> str:
        """Retrieve relevant textbook theorems and inject grounding into system instructions."""
        if not self.rag_index or not self.rag_index.chunks:
            return base_instruction

        try:
            rag_hits = self.rag_index.search(user_prompt, top_k=6)
            high_rel = [h for h in rag_hits if h.get("score", 0) > 2.5]
            if not high_rel:
                return base_instruction

            ref_texts = []
            for h in high_rel[:4]:
                # Cite the cleaned display name, never the raw indexed filename.
                label = h.get("source_name") or h.get("filename", "Reference")
                page = h.get("page")
                page_str = f", p. {page}" if page else ""
                clean_chunk = " ".join(h.get("text", "").split())[:900].strip()
                ref_texts.append(f"[{label}{page_str}]\n{clean_chunk}")

            return (
                base_instruction
                + "\n\n### Grounded Reference Context from Canonical Control Literature:\n"
                + "\n\n".join(ref_texts)
                + "\n\nThese passages were retrieved from the local library for THIS question. When the "
                "question asks what a specific author or textbook says, answer directly from these "
                "passages -- that is already the reference lookup, so do not call a numeric solver tool "
                "to answer a conceptual question. Where a passage conflicts with your own recollection, "
                "TRUST THE PASSAGE. Cite using exactly the bracketed label shown above (for example "
                "[Nise, Control Systems Engineering, p. 583]); never invent a citation and never print a "
                "raw filename, file extension, or course code."
            )
        except Exception as exc:
            print(f"[_get_grounded_instruction] FAILED, continuing ungrounded: {type(exc).__name__}: {exc}")
            return base_instruction

    def _direct_answer(
        self,
        user_prompt: str,
        system_instruction: str,
        history: list[dict[str, Any]] | None = None,
        max_tokens: int = 1400,
    ) -> str:
        """Answer the question with no tools exposed at all.

        This is the recovery path for when the tool loop fails to produce usable
        prose -- the model spent its steps on tool calls, or its synthesis turn
        came back empty or as yet another tool call. Re-asking with tools=None
        and without the tool-result transcript removes whatever was derailing it
        (and shortens the prompt considerably), which reliably yields a real
        answer instead of a canned "analysis complete" placeholder. The grounded
        system instruction is kept, so retrieved reference passages are still
        available to answer from.
        """
        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if history:
            for item in history:
                r = item.get("role")
                c = item.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_prompt})

        try:
            rendered = self.hf_tokenizer.apply_chat_template(
                messages, tools=None, tokenize=False, add_generation_prompt=True
            )
            _, cleaned = _extract_tool_calls(self._generate(rendered, max_tokens=max_tokens))
            return cleaned
        except Exception as exc:
            # This is the last line of defense before the user sees the canned
            # placeholder -- if it fails, that failure must leave a trace
            # instead of vanishing, or a real bug here is undiagnosable.
            print(f"[_direct_answer] FAILED for prompt {user_prompt[:80]!r}: {type(exc).__name__}: {exc}")
            return ""

    def _execute_with_provenance(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a tool only after every matrix argument is traced to the
        conversation; refuse fabricated inputs before they execute."""
        try:
            fabricated = _check_parameter_provenance(tool_args, messages)
        except Exception:
            fabricated = None  # the guard must never take down a legitimate call
        if fabricated is not None:
            param, value, reason = fabricated
            # Never echo a runaway array (possibly hundreds of elements) back
            # into the context -- it wastes the token budget and risks
            # priming the exact same repetition pathology again.
            shown = value if len(json.dumps(value)) < 200 else f"[{len(value)}-element array, truncated]"
            return {
                "status": "error",
                "error_type": "FabricatedParameter",
                "error": (
                    f"REFUSED: the value passed as '{param}' ({shown}) {reason}. Inventing or "
                    f"malformed parameter values is forbidden. Do NOT retry this tool with another "
                    f"guessed or partially-repeated '{param}' -- if you don't actually have a concrete "
                    f"value for it, this tool cannot be used for this question. Answer from your own "
                    f"knowledge or the retrieved reference passages instead, or tell the user what's "
                    f"missing."
                ),
            }
        return self.registry.execute(tool_name, tool_args)

    def run(
        self,
        user_prompt: str,
        system_instruction: str = CONTROLAI_SYSTEM_PROMPT,
        history: list[dict[str, Any]] | None = None,
        max_tokens_per_step: int = 2500,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute a complete agent interaction loop synchronously."""
        messages: list[dict[str, Any]] = []
        effective_sys = self._get_grounded_instruction(user_prompt, system_instruction)

        # RAG fast path: a conceptual/definitional question that already has
        # strong grounded passages needs one generation, not a multi-step tool
        # loop. See the _needs_tools docstring for why this is safe.
        if effective_sys != system_instruction and not _needs_tools(user_prompt):
            fast_answer = self._direct_answer(user_prompt, effective_sys, history)
            if fast_answer:
                return AgentResult(
                    final_response=fast_answer,
                    tool_traces=[],
                    total_steps=1,
                    raw_messages=[{"role": "system", "content": effective_sys}, {"role": "user", "content": user_prompt}],
                    is_grounded=True,
                    plots=[],
                )

        if effective_sys:
            messages.append({"role": "system", "content": effective_sys})

        if history:
            for item in history:
                r = item.get("role")
                c = item.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": user_prompt})

        tools_schema = self.registry.get_tool_schemas()
        traces: list[ToolExecutionTrace] = []
        plots: list[str] = []
        called_signatures: set[str] = set()
        tool_call_counts: dict[str, int] = {}
        fabrication_refusals: list[str] = []

        for step in range(1, self.max_tool_steps + 1):
            rendered_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tools=tools_schema,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_output = self._generate(rendered_prompt, max_tokens=max_tokens_per_step)

            tool_calls, pre_text = _extract_tool_calls(model_output)

            if not tool_calls and not fabrication_refusals:
                # The model chose to stop calling tools but produced no usable
                # text either (typically right after a tool error it has no
                # good way to recover from in-context). Retry once with no
                # tools and no error-laden transcript rather than returning
                # nothing.
                final_response = pre_text or self._direct_answer(user_prompt, effective_sys, history)
                if not final_response:
                    final_response = "The computational analysis has been completed as detailed above."
                return AgentResult(
                    final_response=final_response,
                    tool_traces=traces,
                    total_steps=step,
                    raw_messages=messages,
                    is_grounded=True,
                    plots=plots,
                )

            if not tool_calls:
                # A fabrication refusal happened earlier this turn -- pre_text
                # here is exactly the kind of confident, ungrounded prose that
                # refusal was meant to prevent (observed directly: a blocked
                # LQR call still produced a fully invented gain in this same
                # spot). Don't trust it at face value; fall through to the
                # shared forced-synthesis turn below instead of returning,
                # since that turn explicitly instructs against inventing a
                # substitute value.
                break

            # Drop repeats. An exact-signature check alone is not enough: the
            # model will re-search with a lightly reworded query ("... MPC",
            # "... MPC algorithm", "... MPC definition"), exhausting the step
            # budget on near-identical lookups and leaving nothing for the
            # answer. Cap how many times any single tool may run per turn.
            new_calls = []
            for call in tool_calls:
                name = call.get("name")
                sig = f"{name}:{json.dumps(call.get('arguments', {}), sort_keys=True)}"
                if sig in called_signatures:
                    continue
                if tool_call_counts.get(name, 0) >= MAX_CALLS_PER_TOOL:
                    continue
                called_signatures.add(sig)
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                new_calls.append(call)

            if not new_calls:
                break

            messages.append({"role": "assistant", "content": model_output})

            for call_data in new_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})

                tool_result = self._execute_with_provenance(tool_name, tool_args, messages)
                traces.append(ToolExecutionTrace(tool_name=tool_name, arguments=tool_args, result=tool_result))
                if tool_result.get("error_type") == "FabricatedParameter":
                    fabrication_refusals.append(tool_result.get("error", ""))

                if "plot_path" in tool_result:
                    p_path = Path(tool_result["plot_path"])
                    if p_path.exists():
                        plots.append(f"/plots/{p_path.name}")

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Final synthesis after tool execution
        messages.append({
            "role": "user",
            "content": (
                "Now answer the user's most recent question directly and completely, in LaTeX-formatted "
                "prose. If the tool results above are relevant to that question, incorporate them; if the "
                "question is conceptual, definitional, or about what a source says and the tool results "
                "above don't actually address it, answer the question from your own knowledge instead of "
                "describing the tool results. Do not call any more tools and do not output JSON or "
                "tool-call tags -- write the final answer now."
            ) + _fabrication_refusal_note(fabrication_refusals),
        })
        forced_prompt = self.hf_tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        final_output = self._generate(forced_prompt, max_tokens=max_tokens_per_step)

        _, clean_final = _extract_tool_calls(final_output)
        # Never fall back to the raw final_output: if the model's closing turn
        # degenerates into another (unparseable) tool-call attempt, clean_final
        # is stripped down to "" and showing final_output would leak a raw
        # <tool_call>{...}</tool_call> JSON blob straight into the chat.
        final_response = clean_final or self._direct_answer(user_prompt, effective_sys, history)
        if not final_response:
            # _direct_answer reliably produces real text when tested in
            # isolation, so an empty result here is most likely a transient
            # failure (resource contention, a stray exception) rather than a
            # repeatable one -- one retry clears most of those before giving
            # up and showing the placeholder.
            final_response = self._direct_answer(user_prompt, effective_sys, history)
        if not final_response:
            final_response = "The computational analysis has been completed as detailed above."
        return AgentResult(
            final_response=final_response,
            tool_traces=traces,
            total_steps=len(traces) + 1,
            raw_messages=messages,
            is_grounded=True,
            plots=plots,
        )

    def run_stream(
        self,
        user_prompt: str,
        system_instruction: str = CONTROLAI_SYSTEM_PROMPT,
        history: list[dict[str, Any]] | None = None,
        max_tokens_per_step: int = 2500,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream token-by-token generation and tool execution events with zero JSON leakage."""
        messages: list[dict[str, Any]] = []
        effective_sys = self._get_grounded_instruction(user_prompt, system_instruction)

        # RAG fast path: a conceptual/definitional question that already has
        # strong grounded passages needs one generation, not a multi-step tool
        # loop. See the _needs_tools docstring for why this is safe.
        if effective_sys != system_instruction and not _needs_tools(user_prompt):
            fast_answer = self._direct_answer(user_prompt, effective_sys, history)
            if fast_answer:
                words = re.split(r"(\s+)", fast_answer)
                for w in words:
                    if w:
                        yield {"type": "token", "content": w}
                yield {
                    "type": "done",
                    "response": fast_answer,
                    "traces": [],
                    "plots": [],
                    "thoughts": [],
                }
                return

        if effective_sys:
            messages.append({"role": "system", "content": effective_sys})

        if history:
            for item in history:
                r = item.get("role")
                c = item.get("content", "")
                if r in ("user", "assistant") and c:
                    # Strip any legacy JSON tool call artifacts from previous chat sessions
                    c_clean = re.sub(r"\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*\}", "", c).strip()
                    c_clean = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", c_clean).strip()
                    if c_clean:
                        messages.append({"role": r, "content": c_clean})

        messages.append({"role": "user", "content": user_prompt})

        tools_schema = self.registry.get_tool_schemas()
        traces: list[dict[str, Any]] = []
        plots: list[str] = []
        thoughts: list[str] = []
        called_signatures: set[str] = set()
        tool_call_counts: dict[str, int] = {}
        fabrication_refusals: list[str] = []

        # Dynamic thought generation - only show thoughts when tools or derivations occur
        for step in range(1, self.max_tool_steps + 1):
            rendered_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tools=tools_schema,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_output = self._generate(rendered_prompt, max_tokens=max_tokens_per_step)

            tool_calls, pre_text = _extract_tool_calls(model_output)

            if not tool_calls and not fabrication_refusals:
                # Direct final answer without tools -> stream tokens directly
                clean_output = pre_text or self._direct_answer(user_prompt, effective_sys, history)
                if not clean_output:
                    clean_output = "The computational analysis has been completed as detailed above."
                words = re.split(r"(\s+)", clean_output)
                for w in words:
                    if w:
                        yield {"type": "token", "content": w}

                yield {
                    "type": "done",
                    "response": clean_output,
                    "traces": traces,
                    "plots": plots,
                    "thoughts": thoughts,
                }
                return

            if not tool_calls:
                # A fabrication refusal happened earlier this turn -- pre_text
                # here is exactly the kind of confident, ungrounded prose that
                # refusal was meant to prevent. Don't trust it at face value;
                # fall through to the shared forced-synthesis turn below,
                # which explicitly instructs against inventing a substitute
                # value, instead of streaming it straight to the user.
                break

            if pre_text:
                thoughts.append(pre_text)
                yield {"type": "thought", "content": pre_text}

            # Drop repeats. An exact-signature check alone is not enough: the
            # model will re-search with a lightly reworded query ("... MPC",
            # "... MPC algorithm", "... MPC definition"), exhausting the step
            # budget on near-identical lookups and leaving nothing for the
            # answer. Cap how many times any single tool may run per turn.
            new_calls = []
            for call in tool_calls:
                name = call.get("name")
                sig = f"{name}:{json.dumps(call.get('arguments', {}), sort_keys=True)}"
                if sig in called_signatures:
                    continue
                if tool_call_counts.get(name, 0) >= MAX_CALLS_PER_TOOL:
                    continue
                called_signatures.add(sig)
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                new_calls.append(call)

            if not new_calls:
                break

            messages.append({"role": "assistant", "content": model_output})

            for call_data in new_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})

                t_start_msg = f"Executing tool: {tool_name} with parameters: {json.dumps(tool_args, ensure_ascii=False)}"
                thoughts.append(t_start_msg)
                yield {"type": "thought", "content": t_start_msg}
                yield {"type": "tool_start", "tool": tool_name, "args": tool_args}

                tool_result = self._execute_with_provenance(tool_name, tool_args, messages)
                trace_item = {
                    "tool": tool_name,
                    "args": tool_args,
                    "status": tool_result.get("status", "success"),
                    "residual": tool_result.get("residual"),
                }
                traces.append(trace_item)
                if tool_result.get("error_type") == "FabricatedParameter":
                    fabrication_refusals.append(tool_result.get("error", ""))

                if "plot_path" in tool_result:
                    p_path = Path(tool_result["plot_path"])
                    if p_path.exists():
                        plot_url = f"/plots/{p_path.name}"
                        plots.append(plot_url)
                        yield {"type": "plot", "url": plot_url}

                yield {"type": "tool_end", "trace": trace_item}
                t_end_msg = f"Tool {tool_name} returned status: {trace_item['status']}"
                thoughts.append(t_end_msg)
                yield {"type": "thought", "content": t_end_msg}

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Final Synthesis phase
        synth_thought = "Synthesizing verified engineering response and LaTeX formulations..."
        thoughts.append(synth_thought)
        yield {"type": "thought", "content": synth_thought}

        messages.append({
            "role": "user",
            "content": (
                "Now answer the user's most recent question directly and completely, in LaTeX-formatted "
                "prose. If the tool results above are relevant to that question, incorporate them; if the "
                "question is conceptual, definitional, or about what a source says and the tool results "
                "above don't actually address it, answer the question from your own knowledge instead of "
                "describing the tool results. Do not call any more tools and do not output JSON or "
                "tool-call tags -- write the final answer now."
            ) + _fabrication_refusal_note(fabrication_refusals),
        })

        forced_prompt = self.hf_tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )

        final_output = self._generate(forced_prompt, max_tokens=max_tokens_per_step)

        # If model generated another tool call during final synthesis, execute it!
        synth_tool_calls, clean_synth = _extract_tool_calls(final_output)
        if synth_tool_calls:
            for call_data in synth_tool_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})
                t_res = self._execute_with_provenance(tool_name, tool_args, messages)
                if "plot_path" in t_res:
                    p_path = Path(t_res["plot_path"])
                    if p_path.exists():
                        plot_url = f"/plots/{p_path.name}"
                        plots.append(plot_url)
                        yield {"type": "plot", "url": plot_url}
                messages.append({"role": "tool", "name": tool_name, "content": json.dumps(t_res, ensure_ascii=False)})

            # Re-generate synthesis after executing the tool
            re_prompt = self.hf_tokenizer.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True)
            final_output = self._generate(re_prompt, max_tokens=max_tokens_per_step)
            _, clean_synth = _extract_tool_calls(final_output)

        # Never fall back to the raw final_output here: if the model's closing
        # turn degenerates into another (unparseable) tool-call attempt,
        # clean_synth is stripped down to "" and showing final_output would
        # leak a raw <tool_call>{...}</tool_call> JSON blob into the chat.
        final_text = clean_synth
        # Strip any lingering raw json
        final_text = re.sub(r"\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*\}", "", final_text).strip()
        if not final_text:
            # A fresh, tool-free, minimal-context generation reliably produces
            # real text even when the tool-heavy synthesis turn degenerated --
            # try it (twice, for a transient failure) before the placeholder.
            final_text = self._direct_answer(user_prompt, effective_sys, history)
            if not final_text:
                final_text = self._direct_answer(user_prompt, effective_sys, history)
        if not final_text:
            final_text = "The computational analysis and simulation have been executed successfully as detailed above."

        # Stream words smoothly
        words = re.split(r"(\s+)", final_text)
        for w in words:
            if w:
                yield {"type": "token", "content": w}

        yield {
            "type": "done",
            "response": final_text,
            "traces": traces,
            "plots": plots,
            "thoughts": thoughts,
        }
