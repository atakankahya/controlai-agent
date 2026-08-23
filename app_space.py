"""Hugging Face Space entry point: FastAPI, CPU hardware, hosted generation.

`./run.sh` runs `app.py` directly; this exists only for the Space.

**This deliberately does not use ZeroGPU.** Seven separate incompatibilities came
out of trying to, and the last one is structural rather than a bug: ZeroGPU is
built around "the Gradio app *is* the Space", scheduling GPU workers by forking
the Gradio server process, while ControlAI is a FastAPI app with its own console
that merely borrows the process. The findings are catalogued in CLAUDE.md so the
attempt is not repeated blind, and `engine_torch.py` is kept for anyone running
this on an ordinary CUDA box, where all of it works.

What runs where: everything that makes ControlAI what it is stays in-process on
the Space's CPU -- all 29 deterministic solvers, the verifier, the full
80,370-chunk hybrid retriever. Those cost milliseconds. Only token generation
leaves, to a GPU behind an API, via `RemoteEngine`.

The Space needs `HF_TOKEN` as a secret, with **two** permissions: read access to
the private dataset repo holding the index, and "Make calls to Inference
Providers". Without the first, retrieval is silently disabled; without the
second, generation fails with 403.
"""

from __future__ import annotations

import os

# Must precede the `app` import: it decides which engine gets built. Assigned,
# not setdefault-ed, because a stale Space variable must not select something
# else -- CONTROLAI_BACKEND=pytorch, left from the old orchestrator, once did.
os.environ["CONTROLAI_BACKEND"] = "api"

import uvicorn

import app as app_module
from app import app


def _fetch_index() -> None:
    """Pull the retrieval index in before the agent starts."""
    from controlai_rag.fetch_index import fetch

    if not os.environ.get("HF_TOKEN"):
        print("[space] HF_TOKEN not set -- skipping index fetch, retrieval disabled")
        return
    try:
        fetch()
        print("[space] retrieval index ready")
    except Exception as exc:  # noqa: BLE001 - a missing index must not stop boot
        print(f"[space] could not fetch the index ({exc}); retrieval disabled")


def main() -> None:
    _fetch_index()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)),
                log_level="info")


if __name__ == "__main__":
    main()
