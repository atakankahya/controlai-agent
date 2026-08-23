"""Remote inference engine: the same contract as `LocalEngine`, over an HTTP API.

Why this exists: ZeroGPU is built around "the Gradio app *is* the Space", and
ControlAI is a FastAPI app with its own console. Seven distinct incompatibilities
came out of trying to bridge that (they are catalogued in CLAUDE.md); the last is
structural rather than a bug. This sidesteps the whole category. The Space runs
on plain CPU hardware and keeps everything that makes ControlAI what it is --
all 29 deterministic solvers, the verifier, the 80,370-chunk hybrid retriever --
running locally in-process, where they cost milliseconds. Only token generation
leaves the machine, to a real GPU behind an API.

That is a deliberate trade and it is not free: the demo is no longer
self-contained, which is the local app's whole point. The Space page says so.
Nothing about `engine.py` or the local Apple Silicon path changes.

**The tool schemas go through the model's own chat template locally, and the
result is handed over as an ordinary system message** -- never as a `tools=`
argument. That matters: a provider given `tools=` applies its own template and
returns structured `tool_calls` objects, which the agent does not speak. Doing it
this way, the model emits `<tool_call>` as ordinary text and `toolcall.parse` and
`_StreamGate` see the same input they see locally.

Every provider serving Qwen3 offers `conversational` only, not raw
text-generation, so `render()` returns a *message list* rather than a prompt
string. The agent treats that value as opaque -- it renders and passes it
straight to `stream()` -- so nothing downstream cares which it is.
"""

from __future__ import annotations

import os
import time
from typing import Any, Generator, Iterable, Sequence

from controlai_agent.engine import Chunk, SamplingConfig, Stats

DEFAULT_API_MODEL = os.environ.get("CONTROLAI_MODEL_API", "Qwen/Qwen3-14B")
# "auto" lets the Hub pick whichever provider currently serves the model.
DEFAULT_PROVIDER = os.environ.get("CONTROLAI_PROVIDER", "auto")


class RemoteEngine:
    """Streaming generation against a hosted model, over huggingface_hub."""

    def __init__(
        self,
        model_id: str = DEFAULT_API_MODEL,
        adapter_path: str | None = None,
        sampling: SamplingConfig | None = None,
        provider: str = DEFAULT_PROVIDER,
        token: str | None = None,
    ) -> None:
        from huggingface_hub import InferenceClient
        from transformers import AutoTokenizer

        if adapter_path:
            raise ValueError(
                "RemoteEngine cannot load a LoRA adapter: the weights are not local. "
                "Serve a merged model, or use LocalEngine."
            )

        self.model_id = model_id
        self.adapter_path = None
        self.sampling = sampling or SamplingConfig()
        self.provider = provider

        t0 = time.time()
        # Tokenizer only -- no weights. This is what renders the chat template and
        # counts tokens for history truncation, so it must be the served model's.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.client = InferenceClient(
            model=model_id,
            provider=provider,
            token=token or os.environ.get("HF_TOKEN"),
            timeout=300,
        )
        self.load_seconds = time.time() - t0
        self.last_stats = Stats()
        self.supports_thinking = self._probe_thinking_support()
        print(f"[remote] {model_id} via provider={provider} ({self.load_seconds:.1f}s)")

    # ------------------------------------------------------------------ setup

    def _probe_thinking_support(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return True
        except (TypeError, ValueError):
            return False

    def render(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> list[dict[str, Any]]:
        """Messages ready for a chat-completions call, tool block folded in.

        The tool block is not hand-written: the model's own template renders it,
        and it is lifted back out of the rendered system turn. That keeps the
        wording identical to the local path even if the template changes.
        """
        out = [dict(m) for m in messages]
        if tools:
            rendered = self.tokenizer.apply_chat_template(
                list(messages), tools=list(tools), tokenize=False, add_generation_prompt=True
            )
            head, tail = "<|im_start|>system\n", "<|im_end|>"
            start = rendered.find(head)
            end = rendered.find(tail, start) if start >= 0 else -1
            if start >= 0 and end > start:
                system = rendered[start + len(head):end]
                out = [m for m in out if m.get("role") != "system"]
                out.insert(0, {"role": "system", "content": system})

        if not enable_thinking and self.supports_thinking:
            # Qwen3's documented soft switch. `chat_template_kwargs` would be the
            # direct equivalent, but it is an `extra_body` passthrough that not
            # every provider forwards; this is in the prompt and always arrives.
            for m in reversed(out):
                if m.get("role") == "user":
                    m["content"] = f"{m['content']} /no_think"
                    break
        return out

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------ cache

    def reset_cache(self) -> None:
        """No-op. The KV cache lives on the provider's side, out of reach."""

    def prewarm(self, text: str) -> int:
        """No-op beyond reporting size.

        `LocalEngine.prewarm` prefills the shared prefix into a persistent cache.
        There is no cache to prefill here, and the provider's own prefix caching
        is not ours to manage, so this only reports what the prefix costs.
        """
        return self.count_tokens(text)

    # ------------------------------------------------------------- generation

    def _request(self, messages: list[dict[str, Any]], cfg: SamplingConfig,
                 stop: list[str], budget: int):
        return self.client.chat_completion(
            messages=messages,
            stream=True,
            max_tokens=max(budget, 1),
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p if 0 < cfg.top_p < 1 else None,
            # Natively supported here, so this matches engine.py exactly rather
            # than approximating it -- and a flat repetition_penalty stays out,
            # for the reason engine.py gives.
            presence_penalty=cfg.presence_penalty or None,
            stop=stop or None,
        )

    def stream(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> Generator[Chunk, None, None]:
        """Yield output chunks as they arrive. See `LocalEngine.stream`."""
        cfg = sampling or self.sampling
        convo = prompt if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) else [
            {"role": "user", "content": prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)}
        ]
        stops = [s for s in stop if s]

        t0 = time.time()
        self.last_stats = Stats(
            prompt_tokens=sum(self.count_tokens(str(m.get("content", ""))) for m in convo)
        )
        first_token_at: float | None = None

        emitted = 0
        thinking = False
        think_tokens = 0
        seen = ""
        # Runs at most twice: once normally, and again if the reasoning budget
        # was overrun. `LocalEngine` closes an overrunning <think> block by
        # injecting the closing token mid-stream; there is no mid-stream here, so
        # the equivalent is to stop, hand back what was generated with `</think>`
        # appended, and let a continuation produce the answer. Same transcript,
        # one extra round trip.
        for attempt in (0, 1):
            overran = False
            for event in self._request(convo, cfg, stops, cfg.max_tokens - emitted):
                try:
                    piece = event.choices[0].delta.content
                except (AttributeError, IndexError, TypeError):
                    piece = None
                if not piece:
                    continue
                if first_token_at is None:
                    first_token_at = time.time()
                    self.last_stats.prefill_seconds = first_token_at - t0

                seen += piece
                if "<think>" in piece:
                    thinking, think_tokens = True, 0
                elif "</think>" in piece:
                    thinking = False
                elif thinking:
                    think_tokens += 1

                emitted += 1
                yield Chunk(
                    text=piece,
                    token=-1,  # the API returns text, not ids
                    thinking=thinking,
                    tool_call="<tool_call>" in piece,
                )

                if thinking and think_budget and think_tokens >= think_budget:
                    overran = True
                    break
                if emitted >= cfg.max_tokens:
                    break

            if not (overran and attempt == 0):
                break
            yield Chunk(text="</think>", token=-1, thinking=False)
            convo = convo + [{"role": "assistant", "content": seen + "</think>"}]
            thinking = False

        self.last_stats.generated_tokens = emitted
        self.last_stats.decode_seconds = time.time() - (first_token_at or t0)

    def generate(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> str:
        return "".join(c.text for c in self.stream(prompt, sampling, stop, think_budget))
