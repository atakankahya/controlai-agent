"""CUDA inference engine: the same contract as `LocalEngine`, on transformers.

`LocalEngine` is MLX, so it runs on Apple Silicon and nowhere else. The public
demo Space runs on Linux/NVIDIA (ZeroGPU), which needs a second path. This is
that path and nothing more -- it is not a supported way to run ControlAI locally,
and `agent.py`, `registry.py` and everything under `tools/` are untouched by it.
The seam is `ControlAgent(engine=...)`.

It mirrors `LocalEngine`'s design decisions rather than reaching for
`model.generate`:

*   **Prefix-reusing KV cache.** `DynamicCache.crop()` trims the cache to the
    longest prefix the incoming prompt shares with it, so the ~8k-token
    system-prompt-plus-tool-schema prefix is prefilled once per process, not
    once per tool step. `model.generate` cannot express that.
*   **A hand-written decode loop.** `think_budget` closes an overrunning
    `<think>` block by *injecting* the closing token, which no `generate`
    callback can do, and `presence_penalty` (not `repetition_penalty`) is
    applied for the reason spelled out in `engine.py`: a flat repetition penalty
    punishes the `[`, `0`, `,` that matrices and JSON are made of.

**Loading is bf16 and moves to the GPU with an explicit `.to("cuda")`. Do not
reintroduce `device_map` or bitsandbytes, and do not construct this class lazily
at request time.** All three break on ZeroGPU, which is the only place this file
runs, and all three fail the same way:

    RuntimeError: Low-level CUDA init (`torch._C._cuda_init`) reached. This
    means ZeroGPU's PyTorch CUDA emulation mode did not intercept a CUDA
    operation in your code.

ZeroGPU patches torch during the import of the Space's entry module and attaches
real hardware only inside a `@spaces.GPU` call. Only CUDA operations inside that
import window are intercepted, so **where this object is constructed matters as
much as how**: `app_space.py` builds it at module scope for exactly that reason.
`device_map` fails on top of that, because it routes transformers through
`caching_allocator_warmup`, which calls `torch.empty(..., device="cuda")`
directly. bitsandbytes in turn *requires* `device_map`, so 4-bit quantisation is
unavailable here — which is why the model has to be small enough in bf16.

That is why the model is Qwen3-8B rather than the 14B run locally: bf16 8B is
~16GB to download against ~28GB, and a Space rebuild re-downloads from scratch.
"""

from __future__ import annotations

import os
import time
from typing import Any, Generator, Iterable, Sequence

from controlai_agent.engine import Chunk, SamplingConfig, Stats


def dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """`{"dtype": ...}` or `{"torch_dtype": ...}`, whichever this release takes.

    transformers renamed the argument in 4.56 and the old spelling is gone in
    recent releases. Pinning below 4.56 to keep using it is what broke the Space
    build: the platform force-installs gradio 6.x, which requires
    huggingface-hub >= 1.16, while every transformers < 4.56 requires < 1.0.
    Detecting the spelling costs two lines and pins nothing.
    """
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return {key: dtype}

# Smaller than the 14B run locally, deliberately: see the module docstring.
DEFAULT_TORCH_MODEL = os.environ.get("CONTROLAI_MODEL_TORCH", "Qwen/Qwen3-8B")


class TorchEngine:
    """Streaming generation against a CUDA-resident transformers model."""

    def __init__(
        self,
        model_id: str = DEFAULT_TORCH_MODEL,
        adapter_path: str | None = None,
        sampling: SamplingConfig | None = None,
        max_cache_tokens: int = 32768,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.sampling = sampling or SamplingConfig()
        self.max_cache_tokens = max_cache_tokens

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # No device_map and no quantization_config -- see the module docstring.
        # Load to CPU, then move with .to(), which ZeroGPU's emulation intercepts.
        # The CPU branch exists so the decode loop can be exercised on a small
        # model off a GPU box; it is far too slow to actually serve.
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, **dtype_kwarg(torch.bfloat16 if cuda else torch.float32)
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.to("cuda" if cuda else "cpu")
        self.model.eval()
        self.load_seconds = time.time() - t0
        # Verifiable in the Space logs: if this says cpu on the Space, the .to()
        # did not take and every request will be minutes rather than seconds.
        print(f"[torch] {model_id} on {next(self.model.parameters()).device} "
              f"in {self.load_seconds:.1f}s")

        self._cache: Any | None = None
        self._cache_tokens: list[int] = []
        self.last_stats = Stats()

        self.supports_thinking = self._probe_thinking_support()
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
        ids = self._token_ids(text)
        return ids[0] if len(ids) == 1 else None

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
    ) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
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

    def reset_cache(self) -> None:
        self._cache = None
        self._cache_tokens = []

    def _align_cache(self, tokens: list[int]) -> list[int]:
        """Trim the cache to the longest prefix it shares with `tokens`.

        Returns the suffix that still has to be fed to the model. Mirrors
        `LocalEngine._align_cache`; see that docstring for why this exists.
        """
        from transformers import DynamicCache

        if self._cache is None or not self._cache_tokens:
            self._cache = DynamicCache()
            self._cache_tokens = []
            return list(tokens)

        shared = 0
        for a, b in zip(self._cache_tokens, tokens):
            if a != b:
                break
            shared += 1

        # Never keep the whole prompt: the model needs at least one token to
        # run forward on, or there are no logits to sample from.
        if shared >= len(tokens):
            shared = len(tokens) - 1
        if shared > self.max_cache_tokens:
            shared = 0

        if shared == 0:
            self._cache = DynamicCache()
            self._cache_tokens = []
            return list(tokens)

        if shared < len(self._cache_tokens):
            # crop() is what makes prefix reuse possible. It has moved around
            # between transformers releases and this file no longer pins a
            # version, so losing it costs speed, not correctness: fall back to
            # re-prefilling the whole prompt.
            if not hasattr(self._cache, "crop"):
                self._cache = DynamicCache()
                self._cache_tokens = []
                return list(tokens)
            self._cache.crop(shared)
        self._cache_tokens = list(tokens[:shared])
        return list(tokens[shared:])

    def prewarm(self, text: str) -> int:
        """Prefill a prompt prefix so the first real question doesn't pay for it."""
        import torch

        tokens = self.encode(text)
        to_feed = self._align_cache(tokens)
        if to_feed:
            with torch.inference_mode():
                self.model(
                    input_ids=torch.tensor([to_feed], device=self.model.device),
                    past_key_values=self._cache,
                    use_cache=True,
                )
        self._cache_tokens = list(tokens)
        return len(tokens)

    # ------------------------------------------------------------- generation

    def _sample(self, logits: Any, cfg: SamplingConfig, seen: set[int]) -> int:
        import torch

        logits = logits.float()
        if cfg.presence_penalty and seen:
            idx = torch.tensor(sorted(seen), device=logits.device)
            logits[idx] -= cfg.presence_penalty
        if cfg.temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / cfg.temperature

        if cfg.top_k and cfg.top_k > 0:
            kth = torch.topk(logits, min(cfg.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        if cfg.top_p and 0 < cfg.top_p < 1:
            ordered, order = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(ordered, dim=-1)
            # Keep the first token that crosses top_p, so the mask is never
            # empty even when one token already carries more than top_p mass.
            drop = cumulative - ordered > cfg.top_p
            ordered[drop] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, ordered)
            probs = probs / probs.sum()

        return int(torch.multinomial(probs, 1).item())

    def stream(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> Generator[Chunk, None, None]:
        """Yield output chunks as they are generated. See `LocalEngine.stream`."""
        import torch

        cfg = sampling or self.sampling
        tokens = self.encode(prompt) if isinstance(prompt, str) else list(prompt)

        t0 = time.time()
        to_feed = self._align_cache(tokens)
        self.last_stats = Stats(
            prompt_tokens=len(tokens),
            cached_tokens=len(tokens) - len(to_feed),
        )

        stop = tuple(s for s in stop if s)
        stop_ids = {self._tool_close} if "</tool_call>" in stop and self._tool_close else set()
        text_stops = tuple(s for s in stop if not (s == "</tool_call>" and self._tool_close))
        window = max((len(s) for s in text_stops), default=0) + 8

        eos_ids = {self.tokenizer.eos_token_id}
        for extra in ("<|im_end|>", "<|endoftext|>"):
            tid = self._single_token(extra)
            if tid is not None:
                eos_ids.add(tid)
        eos_ids.discard(None)

        emitted: list[int] = []
        seen: set[int] = set()
        tail = ""
        thinking = False
        think_tokens = 0
        prefill_done = False

        with torch.inference_mode():
            step_input = to_feed
            while len(emitted) < cfg.max_tokens:
                out = self.model(
                    input_ids=torch.tensor([step_input], device=self.model.device),
                    past_key_values=self._cache,
                    use_cache=True,
                )
                self._cache = out.past_key_values
                self._cache_tokens.extend(step_input)
                if not prefill_done:
                    self.last_stats.prefill_seconds = time.time() - t0
                    t1 = time.time()
                    prefill_done = True

                token = self._sample(out.logits[0, -1, :], cfg, seen)
                if token in eos_ids:
                    break

                seen.add(token)
                emitted.append(token)

                if token == self._think_open:
                    thinking, think_tokens = True, 0
                elif token == self._think_close:
                    thinking = False
                elif thinking:
                    think_tokens += 1

                text = self.tokenizer.decode([token], skip_special_tokens=False)
                yield Chunk(
                    text=text,
                    token=token,
                    thinking=thinking,
                    tool_call=token == self._tool_open,
                )

                step_input = [token]

                # Overran the reasoning budget: close the block by hand and make
                # the model answer. Bounds worst-case latency on a reasoning model.
                if (
                    thinking
                    and think_budget
                    and think_tokens >= think_budget
                    and self._think_close is not None
                ):
                    thinking = False
                    emitted.append(self._think_close)
                    yield Chunk(text="</think>", token=self._think_close, thinking=False)
                    step_input = [token, self._think_close]

                if token in stop_ids:
                    break
                if text_stops:
                    tail = (tail + text)[-window:]
                    if any(s in tail for s in text_stops):
                        break

        self.last_stats.generated_tokens = len(emitted)
        self.last_stats.decode_seconds = time.time() - (t1 if prefill_done else t0)

    def generate(
        self,
        prompt: str | list[int],
        sampling: SamplingConfig | None = None,
        stop: Iterable[str] = (),
        think_budget: int | None = None,
    ) -> str:
        return "".join(c.text for c in self.stream(prompt, sampling, stop, think_budget))
