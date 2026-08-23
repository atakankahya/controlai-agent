"""Local inference engine: MLX on Apple Silicon, with a prefix-reusing prompt cache.

Design notes that matter for anyone changing this file:

*   **One persistent KV cache per engine, reused across every generation.**
    The agent's prompt is dominated by a fixed prefix -- the system prompt plus
    the JSON schemas of ~28 tools, which together are several thousand tokens.
    Re-prefilling that on every tool step is what made the previous
    implementation feel slow (measured: 6.4 s of prefill per step, five to six
    steps per question). Here the cache is kept between calls and only the
    tokens that actually differ from what the cache already holds are fed to
    the model, so the fixed prefix is prefilled exactly once per process.

*   **Generated tokens stay in the cache too.** A tool-calling turn is
    append-only: prompt, then the assistant's tool call, then the tool result,
    then more assistant text. Tracking generated tokens alongside prompt tokens
    means continuing that turn costs only the tool-result tokens.

*   **Streaming is real.** `stream()` yields text as the model produces it.
    Nothing here buffers a whole response and re-emits it word by word.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = os.environ.get("CONTROLAI_MODEL", "mlx-community/Qwen3-14B-4bit")
# The project's own LoRA adapters are deliberately NOT loaded by default. Both
# of them regressed the behaviour they were meant to improve: `behavior_v1`
# emits a spurious empty `<tool_call></tool_call>` as its first output on
# essentially every prompt (so no tool ever runs), and `sft_v2` generates empty
# output when no tools are exposed and string-typed numbers when they are.
# Set CONTROLAI_ADAPTER=<path> to load one anyway for A/B work.
DEFAULT_ADAPTER = os.environ.get("CONTROLAI_ADAPTER") or None


@dataclass
class SamplingConfig:
    """Decoding parameters. Defaults follow Qwen3's own non-thinking recipe."""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Qwen3 recommends presence_penalty over the blunt repetition_penalty that
    # the previous implementation applied at 1.15 across the board. A flat
    # repetition penalty is actively harmful for this workload: it penalises
    # the repeated structural tokens that matrices and JSON are made of
    # (`[`, `0`, `,`) exactly when the model is emitting a tool call.
    presence_penalty: float = 0.5
    max_tokens: int = 1024

    def with_(self, **kw: Any) -> "SamplingConfig":
        merged = {**self.__dict__, **{k: v for k, v in kw.items() if v is not None}}
        return SamplingConfig(**merged)


@dataclass
class Chunk:
    """One streamed piece of model output."""

    text: str
    token: int
    thinking: bool = False
    tool_call: bool = False


@dataclass
class Stats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def decode_tps(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds else 0.0


class LocalEngine:
    """Streaming text generation against a locally held MLX model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        adapter_path: str | None = DEFAULT_ADAPTER,
        sampling: SamplingConfig | None = None,
        max_cache_tokens: int = 32768,
    ) -> None:
        from mlx_lm import load

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.sampling = sampling or SamplingConfig()
        self.max_cache_tokens = max_cache_tokens

        t0 = time.time()
        if adapter_path:
            self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_id)
        self.load_seconds = time.time() - t0

        self._cache: list[Any] | None = None
        self._cache_tokens: list[int] = []
        self.last_stats = Stats()

        # Resolved once: whether this checkpoint's chat template understands
        # Qwen3-style `enable_thinking`, and the ids of the think delimiters.
        self.supports_thinking = self._probe_thinking_support()
        # `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are each a
        # single special token in the Qwen3 vocabulary. Watching for the token
        # id rather than matching the rendered string is exact: it cannot be
        # defeated by a tag split across two streamed chunks, and it costs one
        # integer comparison per token instead of a substring scan.
        self._think_open = self._single_token("<think>")
        self._think_close = self._single_token("</think>")
        self._tool_open = self._single_token("<tool_call>")
        self._tool_close = self._single_token("</tool_call>")

    # ------------------------------------------------------------------ setup

    def _token_ids(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def _single_token(self, text: str) -> int | None:
        """The id of `text` if the tokenizer represents it as one token."""
        ids = self._token_ids(text)
        return ids[0] if len(ids) == 1 else None

    def _probe_thinking_support(self) -> bool:
        probe = [{"role": "user", "content": "hi"}]
        try:
            on = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            off = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:
            return False
        return on != off

    # -------------------------------------------------------------- rendering

    def render(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if self.supports_thinking:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------ cache

    def _align_cache(self, tokens: list[int]) -> list[int]:
        """Point the persistent cache at the longest prefix of `tokens` it
        already holds, and return the tokens that still need prefilling."""
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        reusable = 0
        if self._cache is not None:
            limit = min(len(self._cache_tokens), len(tokens))
            while reusable < limit and self._cache_tokens[reusable] == tokens[reusable]:
                reusable += 1

        # A cache that cannot be trimmed back to the divergence point is worse
        # than no cache: it would silently condition generation on stale
        # tokens. Rebuild instead.
        if self._cache is not None and reusable < len(self._cache_tokens):
            if can_trim_prompt_cache(self._cache):
                trim_prompt_cache(self._cache, len(self._cache_tokens) - reusable)
            else:
                self._cache, reusable = None, 0

        if self._cache is None or reusable == 0:
            self._cache = make_prompt_cache(self.model)
            self._cache_tokens = []
            reusable = 0

        # MLX must be fed at least one token; an exact cache hit therefore
        # rewinds by one and replays the final token.
        if reusable == len(tokens) and reusable > 0:
            from mlx_lm.models.cache import trim_prompt_cache as _trim

            _trim(self._cache, 1)
            reusable -= 1

        self._cache_tokens = list(tokens[:reusable])
        self.last_stats.cached_tokens = reusable
        return list(tokens[reusable:])

    def reset_cache(self) -> None:
        self._cache = None
        self._cache_tokens = []

    def prewarm(self, text: str) -> int:
        """Prefill a prompt prefix so the first real question doesn't pay for it.

        Returns the number of tokens now resident in the cache. Called at
        startup with the system-prompt-plus-tool-schemas prefix, which turns
        first-question latency from a multi-second prefill into a cache hit.
        """
        import mlx.core as mx
        from mlx_lm import stream_generate

        tokens = self.encode(text)
        to_feed = self._align_cache(tokens)
        if to_feed:
            self.model(mx.array(to_feed)[None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
        self._cache_tokens = list(tokens)

        # Prefilling alone leaves the single-token decode kernels uncompiled,
        # so the first real question still paid several seconds of Metal
        # warm-up. Generate and discard one token against a throwaway cache to
        # force that compilation now, without disturbing the prefix cache.
        from mlx_lm.models.cache import make_prompt_cache

        scratch = make_prompt_cache(self.model)
        for _ in stream_generate(
            self.model, self.tokenizer, [tokens[-1]], max_tokens=1, prompt_cache=scratch
        ):
            break
        return len(tokens)

    # ------------------------------------------------------------- generation

    def stream(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> Generator[Chunk, None, None]:
        """Yield output chunks as they are generated.

        `stop` sequences end generation as soon as they appear (the sequence
        itself is emitted, since tool-call parsing wants the closing tag).
        `think_budget` caps how many tokens may be spent inside a `<think>`
        block: on overrun the block is closed by hand and the model is made to
        answer, which bounds worst-case latency on a reasoning model.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        cfg = sampling or self.sampling
        tokens = self.encode(prompt) if isinstance(prompt, str) else list(prompt)

        t0 = time.time()
        to_feed = self._align_cache(tokens)
        self.last_stats = Stats(
            prompt_tokens=len(tokens),
            cached_tokens=len(tokens) - len(to_feed),
        )

        sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        logits_processors = (
            make_logits_processors(presence_penalty=cfg.presence_penalty)
            if cfg.presence_penalty
            else None
        )

        stop = tuple(s for s in stop if s)
        stop_ids = {self._tool_close} if "</tool_call>" in stop and self._tool_close else set()
        text_stops = tuple(s for s in stop if not (s == "</tool_call>" and self._tool_close))
        # Only the tail can contain a partial stop sequence, so matching over a
        # bounded window keeps this O(1) per token instead of rescanning the
        # whole response.
        window = max((len(s) for s in text_stops), default=0) + 8

        emitted: list[int] = []
        tail = ""
        budget_left = think_budget
        in_think = False
        in_tool_call = False
        first_token_at: float | None = None
        remaining = cfg.max_tokens

        while remaining > 0:
            forced_close = False
            for resp in stream_generate(
                self.model,
                self.tokenizer,
                to_feed,
                max_tokens=remaining,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self._cache,
            ):
                if first_token_at is None:
                    first_token_at = time.time()
                    self.last_stats.prefill_seconds = first_token_at - t0
                emitted.append(resp.token)
                self._cache_tokens.append(resp.token)
                remaining -= 1
                text = resp.text

                # Exact, token-id state transitions. The tags themselves are
                # never emitted -- the caller gets the content and the flags.
                if resp.token == self._think_open:
                    in_think = True
                    continue
                if resp.token == self._think_close:
                    in_think = False
                    continue
                if resp.token == self._tool_open:
                    # Unlike the think tags, this one is emitted: the caller
                    # parses the `<tool_call>...</tool_call>` block out of the
                    # raw text. The flag lets it suppress the same text from
                    # the user-visible stream.
                    in_tool_call = True
                if text and not in_think:
                    tail = (tail + text)[-window:] if window else ""
                    # Fallback for a checkpoint whose think tags are not single
                    # tokens; harmless when the ids above already matched.
                    if self._think_open is None and "<think>" in tail:
                        in_think = True
                    if self._think_close is None and "</think>" in tail:
                        in_think = False

                yield Chunk(text=text, token=resp.token, thinking=in_think, tool_call=in_tool_call)

                if in_think and budget_left is not None:
                    budget_left -= 1
                    if budget_left <= 0:
                        forced_close = True
                        break

                if resp.token in stop_ids or (text_stops and any(s in tail for s in text_stops)):
                    remaining = 0
                    break
            else:
                remaining = 0

            if not forced_close:
                break

            # Overran the thinking budget: close the block ourselves and let
            # the same cache continue straight into the answer.
            closer = "\n</think>\n\n"
            closer_ids = self._token_ids(closer)
            self._cache_tokens.extend(closer_ids)
            to_feed = closer_ids
            # Deliberately not yielded: this is a control action on the model,
            # not model output. Emitting it put a bare "</think>" at the top of
            # the answer whenever the budget was reached.
            in_think = False
            budget_left = None

        now = time.time()
        self.last_stats.generated_tokens = len(emitted)
        self.last_stats.decode_seconds = now - (first_token_at or now)
        if len(self._cache_tokens) > self.max_cache_tokens:
            self.reset_cache()

    def generate(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> str:
        return "".join(
            c.text for c in self.stream(prompt, sampling, stop, think_budget)
        )
