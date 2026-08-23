"""The ControlAI agent loop.

Replaces the previous `orchestrator.py`. The differences that matter:

*   **It streams.** Text reaches the caller as the model produces it. The old
    loop blocked for a full generation and then re-emitted the finished string
    word by word, which looked like streaming but meant the user waited for the
    entire answer before seeing anything.
*   **It reuses the KV cache** across tool steps via `LocalEngine`, so the
    multi-thousand-token tool-schema prefix is prefilled once per process
    rather than once per step.
*   **It trusts the model with parameters.** The old loop ran a "provenance"
    check that refused any matrix it could not trace back to the user's text.
    That blocked the most useful thing the agent does -- working an example the
    user asked for -- so it is gone. What remains is schema validation and
    execution in `registry.execute`, which are real guarantees, plus a
    repetition guard for genuinely degenerate output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterable, Sequence

from controlai_agent import toolcall
from controlai_agent.engine import LocalEngine, SamplingConfig
from controlai_agent.prompts import RETRIEVAL_PREAMBLE, SYNTHESIS_NUDGE, SYSTEM_PROMPT
from controlai_agent.registry import registry
from controlai_agent.toolcall import ToolCall

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_HISTORY_TOKENS = 8000
# Two rounds of tools answers essentially every real question (design, then
# simulate). The old default of four mostly bought extra latency and gave a
# stuck model more room to loop.
MAX_TOOL_STEPS = 2
MAX_CALLS_PER_TOOL = 2

THINKING_MODE = os.environ.get("CONTROLAI_THINKING", "auto").lower()
THINK_BUDGET = int(os.environ.get("CONTROLAI_THINK_BUDGET", "512"))

# Questions that are about a concept rather than a specific system. Used only
# to decide whether to spend thinking tokens -- never to block a tool call.
_CONCEPTUAL_RE = re.compile(
    r"\b(why|explain|what is|what are|difference between|compare|derive|"
    r"derivation|prove|proof|intuition|when should|trade-?off|meaning of)\b",
    re.IGNORECASE,
)


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.result.get("status", "success"))


@dataclass
class AgentResult:
    answer: str
    traces: list[ToolTrace] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class _StreamGate:
    """Emits streamed text while withholding anything from `marker` onward.

    The model decides between answering and calling a tool by what it emits
    first, and that decision is only visible partway through a token. This
    releases text as soon as it cannot be the start of `marker`, so prose
    streams with no perceptible delay while a tool call never leaks into the
    chat.
    """

    def __init__(self, marker: str = "<tool_call>") -> None:
        self.marker = marker
        self._pending = ""
        self.suppressed = False

    def feed(self, text: str) -> str:
        if self.suppressed:
            return ""
        self._pending += text
        idx = self._pending.find(self.marker)
        if idx != -1:
            out, self._pending, self.suppressed = self._pending[:idx], "", True
            return out
        # Hold back only a possible partial marker at the very end.
        hold = 0
        for n in range(min(len(self.marker) - 1, len(self._pending)), 0, -1):
            if self._pending.endswith(self.marker[:n]):
                hold = n
                break
        out, self._pending = (self._pending[:-hold], self._pending[-hold:]) if hold else (self._pending, "")
        return out

    def flush(self) -> str:
        if self.suppressed:
            return ""
        out, self._pending = self._pending, ""
        return out


class ControlAgent:
    """Control-engineering agent over a local model and deterministic tools."""

    def __init__(
        self,
        engine: LocalEngine | None = None,
        tool_registry=registry,
        retriever: Any | None = None,
        max_tool_steps: int = MAX_TOOL_STEPS,
        thinking: str = THINKING_MODE,
        think_budget: int = THINK_BUDGET,
    ) -> None:
        import controlai_agent.tools  # noqa: F401  -- registers every tool

        self.engine = engine or LocalEngine()
        self.registry = tool_registry
        self.max_tool_steps = max_tool_steps
        self.thinking = thinking
        self.think_budget = think_budget
        self.tool_schemas = self.registry.get_tool_schemas()

        if retriever is None:
            try:
                from controlai_rag.retriever import get_retriever

                retriever = get_retriever()
            except Exception as exc:  # retrieval is an enhancement, not a dependency
                print(f"[agent] retrieval unavailable ({type(exc).__name__}: {exc}); continuing without it")
        self.retriever = retriever

        self._prewarm()

    # ------------------------------------------------------------- setup

    def _prewarm(self) -> None:
        """Prefill the fixed system-prompt-plus-tool-schema prefix.

        Everything after it in a real prompt is conversation, so this is the
        one part of every request that is byte-identical every time. Paying for
        it at startup is what makes the first question feel instant.
        """
        prefix = self.engine.render(
            [{"role": "system", "content": SYSTEM_PROMPT}], tools=self.tool_schemas
        )
        # Cut at the end of the system block: the generation prompt that
        # `render` appends belongs to the user's turn, not to the prefix.
        anchor = prefix.rfind("<|im_end|>")
        if anchor != -1:
            prefix = prefix[: anchor + len("<|im_end|>\n")]
        n = self.engine.prewarm(prefix)
        print(f"[agent] prewarmed {n} prefix tokens ({self.engine.model_id})")

    # --------------------------------------------------------- prompting

    def _retrieve(self, question: str) -> tuple[str, list[str]]:
        if self.retriever is None:
            return "", []
        try:
            hits = self.retriever.search(question, top_k=4)
        except Exception as exc:
            print(f"[agent] retrieval failed ({type(exc).__name__}: {exc})")
            return "", []
        if not hits:
            return "", []
        blocks, labels = [], []
        for hit in hits:
            label = hit.get("label") or hit.get("source_name") or "Reference"
            page = hit.get("page")
            label = f"{label}, p. {page}" if page else label
            text = " ".join(str(hit.get("text", "")).split())[:800]
            blocks.append(f"[{label}]\n{text}")
            labels.append(label)
        return RETRIEVAL_PREAMBLE + "\n\n" + "\n\n".join(blocks), labels

    def _build_messages(
        self, question: str, history: Sequence[dict[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self._truncate(history or [])

        context, sources = self._retrieve(question)
        # Retrieved passages ride along with the user's turn rather than being
        # spliced into the system prompt. That keeps the cached prefix
        # byte-stable across questions, which is worth more than the tidier
        # placement.
        content = f"{context}\n\n---\n\n{question}" if context else question
        messages.append({"role": "user", "content": content})
        return messages, sources

    def _truncate(self, history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the oldest turns until the history fits the token budget.

        The web client resends the whole conversation every request and trims
        nothing, so the server has to.
        """
        kept: list[dict[str, Any]] = []
        used = 0
        for item in reversed(list(history)):
            role, content = item.get("role"), (item.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            cost = self.engine.count_tokens(content) + 8
            if used + cost > MAX_HISTORY_TOKENS:
                break
            kept.append({"role": role, "content": content})
            used += cost
        return list(reversed(kept))

    def _wants_thinking(self, question: str) -> bool:
        if self.thinking == "on":
            return True
        if self.thinking == "off":
            return False
        # "auto": reasoning earns its latency on conceptual questions, which is
        # where this model is weakest and where no solver can help it.
        return bool(_CONCEPTUAL_RE.search(question))

    # ------------------------------------------------------------- tools

    def _execute(self, call: ToolCall) -> dict[str, Any]:
        for key, value in call.arguments.items():
            reason = toolcall.degenerate_reason(value)
            if reason:
                return {
                    "status": "error",
                    "error_type": "DegenerateArgument",
                    "error": (
                        f"The value passed as '{key}' {reason}, which is not a real system. "
                        f"Re-read the question and pass the actual values, or say what is missing."
                    ),
                }
        return self.registry.execute(call.name, call.arguments)

    # ----------------------------------------------------------- running

    def stream(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1536,
    ) -> Generator[dict[str, Any], None, None]:
        """Run one turn, yielding events as they happen.

        Event types: `thinking`, `text`, `tool_start`, `tool_end`, `plot`,
        `done`.
        """
        messages, sources = self._build_messages(question, history)
        traces: list[ToolTrace] = []
        plots: list[str] = []
        answer_parts: list[str] = []
        call_counts: dict[str, int] = {}
        seen: set[str] = set()
        # Accumulated across every generation in the turn: the engine's own
        # stats only describe its most recent call, which for a tool-using
        # question is the short synthesis pass and badly understates the work.
        totals = {"prompt_tokens": 0, "cached_tokens": 0, "generated_tokens": 0, "prefill_seconds": 0.0, "decode_seconds": 0.0}

        # Decided once, before the loop. Deciding it per-pass looked equivalent
        # but was not: a conceptual question is answered on the *first* pass,
        # which is never the final pass, so reasoning was silently never
        # enabled for exactly the questions "auto" exists to help.
        think_turn = self._wants_thinking(question)

        for step in range(self.max_tool_steps + 1):
            final_pass = step == self.max_tool_steps
            tools = None if final_pass else self.tool_schemas
            # Once a solver has produced the number, the number is the answer;
            # reasoning over it only adds latency.
            think = think_turn and not traces

            if final_pass and traces:
                messages.append({"role": "user", "content": SYNTHESIS_NUDGE})

            prompt = self.engine.render(messages, tools=tools, enable_thinking=think)
            gate = _StreamGate()
            raw: list[str] = []

            for chunk in self.engine.stream(
                prompt,
                sampling=self.engine.sampling.with_(max_tokens=max_tokens),
                stop=("</tool_call>",) if tools else (),
                think_budget=self.think_budget if think else None,
            ):
                if chunk.thinking:
                    yield {"type": "thinking", "text": chunk.text}
                    continue
                raw.append(chunk.text)
                visible = gate.feed(chunk.text)
                if visible:
                    answer_parts.append(visible)
                    yield {"type": "text", "text": visible}

            tail = gate.flush()
            if tail:
                answer_parts.append(tail)
                yield {"type": "text", "text": tail}

            stats = self.engine.last_stats
            totals["prompt_tokens"] += stats.prompt_tokens
            totals["cached_tokens"] += stats.cached_tokens
            totals["generated_tokens"] += stats.generated_tokens
            totals["prefill_seconds"] += stats.prefill_seconds
            totals["decode_seconds"] += stats.decode_seconds

            output = "".join(raw)
            calls, _ = toolcall.parse(output)

            if not calls:
                break

            # A tool call is arriving, so whatever prose preceded it was
            # narration ("Let me compute that"), not the answer. Drop it from
            # the answer text; the user already saw it stream past.
            answer_parts.clear()
            messages.append({"role": "assistant", "content": output})

            for call in calls:
                if call_counts.get(call.name, 0) >= MAX_CALLS_PER_TOOL:
                    continue
                signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                if signature in seen:
                    continue
                seen.add(signature)
                call_counts[call.name] = call_counts.get(call.name, 0) + 1

                yield {"type": "tool_start", "tool": call.name, "arguments": call.arguments}
                result = self._execute(call)
                traces.append(ToolTrace(call.name, call.arguments, result))
                yield {
                    "type": "tool_end",
                    "tool": call.name,
                    "status": result.get("status", "success"),
                    "result": result,
                }

                plot_path = result.get("plot_path")
                if plot_path and Path(plot_path).exists():
                    url = f"/plots/{Path(plot_path).name}"
                    plots.append(url)
                    yield {"type": "plot", "url": url}

                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = self._recover(question, messages)
            if answer:
                yield {"type": "text", "text": answer}

        yield {
            "type": "done",
            "answer": answer,
            "traces": [{"tool": t.name, "arguments": t.arguments, "status": t.status} for t in traces],
            "plots": plots,
            "sources": sources,
            "stats": {
                "prompt_tokens": totals["prompt_tokens"],
                "cached_tokens": totals["cached_tokens"],
                "generated_tokens": totals["generated_tokens"],
                "prefill_seconds": round(totals["prefill_seconds"], 3),
                "decode_tps": round(
                    totals["generated_tokens"] / totals["decode_seconds"], 1
                ) if totals["decode_seconds"] else 0.0,
            },
        }

    def _recover(self, question: str, messages: list[dict[str, Any]]) -> str:
        """Last resort when the loop produced no prose.

        Re-asks with the tool results kept but the tool schemas withdrawn, so
        the model has nothing to answer with except words.
        """
        messages = messages + [{"role": "user", "content": SYNTHESIS_NUDGE}]
        prompt = self.engine.render(messages, tools=None, enable_thinking=False)
        _, prose = toolcall.parse(self.engine.generate(prompt))
        return prose.strip()

    def run(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1536,
    ) -> AgentResult:
        """Blocking variant of `stream`."""
        result = AgentResult(answer="")
        traces: list[ToolTrace] = []
        for event in self.stream(question, history, max_tokens):
            if event["type"] == "tool_end":
                traces.append(ToolTrace(event["tool"], {}, event["result"]))
            elif event["type"] == "done":
                result = AgentResult(
                    answer=event["answer"],
                    traces=traces,
                    plots=event["plots"],
                    sources=event["sources"],
                    stats=event["stats"],
                )
        return result
