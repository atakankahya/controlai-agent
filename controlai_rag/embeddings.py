"""Local dense embeddings for the control-engineering corpus.

Uses Qwen3-Embedding-0.6B under MLX -- small, fast on Apple Silicon, and
already present in the local Hugging Face cache, so retrieval stays fully
offline. The model is a causal backbone whose sentence embedding is the final
hidden state at the last position; queries take an instruction prefix while
documents do not, which is the recipe the model was trained with.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("CONTROLAI_EMBED_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
# The Space runs on Linux, where MLX does not exist, so the same embedder has a
# transformers path. It is the same weights unquantised; see `Embedder._backend`.
TORCH_MODEL_ID = os.environ.get("CONTROLAI_EMBED_MODEL_TORCH", "Qwen/Qwen3-Embedding-0.6B")
MAX_TOKENS = 512
# Qwen3-Embedding pools the hidden state at the final position, and it was
# trained with an explicit end-of-text token in that position. Omitting it is
# not a small detail: measured on this corpus, the margin between relevant
# passages and junk went from +0.150 without it to +0.322 with it, and
# retrieval for "Routh-Hurwitz table construction" went from returning a
# book index page to returning the actual Routh-Hurwitz section.
# `tokenizer.eos_token_id` on this checkpoint is <|im_end|>, which is the chat
# terminator, not this one -- <|im_end|> scored +0.185. Pin the right token.
EOS_TOKEN = "<|endoftext|>"
QUERY_INSTRUCTION = (
    "Instruct: Given a control engineering question, retrieve textbook passages "
    "that explain the underlying theory.\nQuery: "
)


class Embedder:
    """Lazily-loaded sentence embedder producing L2-normalised float32 vectors."""

    def __init__(self, model_id: str | None = None, backend: str | None = None) -> None:
        # "mlx" locally, "torch" on the Space. The vectors in embeddings.npz were
        # produced by the MLX 4-bit checkpoint; the bf16 transformers weights are
        # the same model, so the two agree closely but not bit-exactly. If
        # retrieval on the Space looks over- or under-eager, MIN_COSINE is the
        # knob (CONTROLAI_MIN_COSINE), not this.
        self._backend = (backend or os.environ.get("CONTROLAI_BACKEND", "mlx")).lower()
        if self._backend != "torch":
            self._backend = "mlx"
        self.model_id = model_id or (TORCH_MODEL_ID if self._backend == "torch" else MODEL_ID)
        self._model = None
        self._tokenizer = None
        self._eos_id: int | None = None
        self._pad_id: int | None = None

    def _ensure_loaded(self) -> None:
        """Load the model and tokenizer, or leave the object exactly as it was.

        Everything is built into locals and committed to `self` only once all of
        it succeeded. An earlier version assigned `self._model` from
        `from_pretrained` and then called `.to("cuda")`, which on ZeroGPU raises:
        `self._model` was left set, `_eos_id` was never reached, and the next
        call short-circuited on `self._model is not None` and appended None as a
        token id -- surfacing much later as
        `RuntimeError: Could not infer dtype of NoneType`, nowhere near the
        actual failure. A half-loaded embedder must not look like a loaded one.
        """
        if self._model is not None:
            return

        if self._backend == "torch":
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # transformers renamed torch_dtype -> dtype in 4.56, and nothing
            # here pins a version. See engine_torch.dtype_kwarg.
            version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            key = "dtype" if version >= (4, 56) else "torch_dtype"
            cuda = torch.cuda.is_available()
            model = AutoModel.from_pretrained(
                self.model_id, **{key: torch.float16 if cuda else torch.float32}
            )
            model = model.to("cuda" if cuda else "cpu")
            model.eval()
        else:
            from mlx_lm import load

            model, tokenizer = load(self.model_id)

        ids = tokenizer.encode(EOS_TOKEN)
        eos_id = ids[-1] if ids else tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError(
                f"{self.model_id}: could not resolve an id for {EOS_TOKEN!r}, "
                "which last-token pooling depends on"
            )

        self._tokenizer = tokenizer
        self._eos_id = eos_id
        self._pad_id = tokenizer.pad_token_id or eos_id
        self._model = model   # last: this is what _ensure_loaded() checks

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        if self._backend == "torch":
            return int(self._model.config.hidden_size)
        return int(self._model.args.hidden_size)

    def _tokens_for(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)[: MAX_TOKENS - 1] + [self._eos_id]

    def _encode_one(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        if self._backend == "torch":
            return self._encode_batch([self._tokens_for(text)])[0]

        import mlx.core as mx

        ids = self._tokens_for(text)
        # `model.model` is the backbone; calling `model` itself would project
        # through the language-model head and give logits, not an embedding.
        hidden = self._model.model(mx.array(ids)[None])
        vector = hidden[0, -1].astype(mx.float32)
        vector = vector / (mx.linalg.norm(vector) + 1e-9)
        return np.array(vector, copy=True)

    def _encode_batch(self, batch: list[list[int]]) -> np.ndarray:
        """Embed a batch of already-tokenised inputs.

        Sequences are right-padded to the longest in the batch and pooled at
        each sequence's own final position. Right-padding is safe here
        precisely because the backbone is causal: position i attends only to
        positions <= i, so tokens appended after the real end cannot influence
        the hidden state being pooled.
        """
        self._ensure_loaded()
        if self._backend == "torch":
            return self._encode_batch_torch(batch)

        import mlx.core as mx

        lengths = [len(ids) for ids in batch]
        width = max(lengths)
        pad = self._pad_id
        padded = mx.array([ids + [pad] * (width - len(ids)) for ids in batch])
        hidden = self._model.model(padded)
        picked = mx.stack([hidden[i, n - 1] for i, n in enumerate(lengths)]).astype(mx.float32)
        picked = picked / (mx.linalg.norm(picked, axis=-1, keepdims=True) + 1e-9)
        return np.array(picked, copy=True)

    def _encode_batch_torch(self, batch: list[list[int]]) -> np.ndarray:
        """`_encode_batch` on transformers. Same right-padding and same pooling.

        An explicit attention mask is passed even though right-padding a causal
        backbone is already safe, because transformers otherwise warns on every
        call and the mask costs nothing.
        """
        import torch

        lengths = [len(ids) for ids in batch]
        width = max(lengths)
        pad = self._pad_id
        device = self._model.device
        ids = torch.tensor(
            [row + [pad] * (width - len(row)) for row in batch], device=device
        )
        mask = torch.zeros_like(ids)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1

        with torch.inference_mode():
            hidden = self._model(input_ids=ids, attention_mask=mask).last_hidden_state
        picked = torch.stack(
            [hidden[i, n - 1] for i, n in enumerate(lengths)]
        ).float()
        picked = picked / (picked.norm(dim=-1, keepdim=True) + 1e-9)
        return picked.cpu().numpy().astype(np.float32)

    def encode_documents(
        self,
        texts: list[str],
        progress_every: int = 2000,
        batch_tokens: int = 16384,
    ) -> np.ndarray:
        """Embed a corpus, batching by token budget rather than by count.

        Sorting by length before batching keeps padding waste low; the original
        order is restored before returning. One chunk at a time was ~13 minutes
        per 10k chunks, which does not scale to a corpus of 80k.
        """
        self._ensure_loaded()
        tokenised = [self._tokens_for(t) for t in texts]
        order = sorted(range(len(tokenised)), key=lambda i: len(tokenised[i]))

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        batch: list[int] = []
        done = 0

        def flush(batch: list[int]) -> None:
            nonlocal done
            if not batch:
                return
            vectors = self._encode_batch([tokenised[i] for i in batch])
            for slot, i in enumerate(batch):
                out[i] = vectors[slot]
            done += len(batch)
            if progress_every and done % progress_every < len(batch):
                print(f"  embedded {done}/{len(texts)}", flush=True)

        for i in order:
            # The cost of a batch is (rows x longest row), so cap on that
            # product rather than on row count.
            if batch and (len(batch) + 1) * len(tokenised[i]) > batch_tokens:
                flush(batch)
                batch = []
            batch.append(i)
        flush(batch)
        return out

    def encode_query(self, query: str) -> np.ndarray:
        self._ensure_loaded()
        return self._encode_one(QUERY_INSTRUCTION + query)


_shared: Embedder | None = None


def get_embedder() -> Embedder:
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
