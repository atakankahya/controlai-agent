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

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._eos_id: int | None = None
        self._pad_id: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_id)
            ids = self._tokenizer.encode(EOS_TOKEN)
            self._eos_id = ids[-1] if ids else self._tokenizer.eos_token_id
            self._pad_id = self._tokenizer.pad_token_id or self._eos_id

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.args.hidden_size)

    def _tokens_for(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)[: MAX_TOKENS - 1] + [self._eos_id]

    def _encode_one(self, text: str) -> np.ndarray:
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
        import mlx.core as mx

        lengths = [len(ids) for ids in batch]
        width = max(lengths)
        pad = self._pad_id
        padded = mx.array([ids + [pad] * (width - len(ids)) for ids in batch])
        hidden = self._model.model(padded)
        picked = mx.stack([hidden[i, n - 1] for i, n in enumerate(lengths)]).astype(mx.float32)
        picked = picked / (mx.linalg.norm(picked, axis=-1, keepdims=True) + 1e-9)
        return np.array(picked, copy=True)

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
