"""Hugging Face Space entry point: the same FastAPI app, on ZeroGPU.

Not used for local runs -- `./run.sh` runs `app.py` directly. This module exists
because the Space's platform imposes requirements that have nothing to do with
ControlAI, and each of the four below was learned by having the Space fail:

1.  A `@spaces.GPU`-decorated function is only recognised by the platform's
    startup validation if it is wired to an actual Gradio event handler. A
    function called only from a FastAPI route fails startup with "No @spaces.GPU
    function detected". Hence the hidden probe button.
2.  That function must be a module-level `def`. Nested inside a `with
    gr.Blocks():` block, the same detection fails.
3.  Do not `gr.mount_gradio_app()` the probe into the FastAPI app and then run
    uvicorn on the same port -- Gradio's own server setup collides with it
    ("address already in use"). The probe launches on its own port; FastAPI is
    the only thing bound to the public one.
4.  Never pass the model-holding object as an *argument* to a `@spaces.GPU`
    function. ZeroGPU marshals arguments across a process boundary and will try
    to share the model's CUDA tensors, failing with `_share_cuda_: only
    available on CUDA` after producing no output at all -- which reads exactly
    like "just slow". Reach the agent through the module-level global instead.

The retrieval index is not in the repo (see CLAUDE.md), so it is pulled from the
private Hub dataset repo at startup using the Space's HF_TOKEN secret. Without
it the retriever has no corpus and every answer falls back to model knowledge.
"""

from __future__ import annotations

import os

# Assigned, not setdefault: this module is the CUDA entry point by definition,
# and a stale Space variable must not be able to select something else. One did
# -- CONTROLAI_BACKEND=pytorch, left over from the old orchestrator -- and the
# Space booted into the MLX engine and died on `import mlx_lm`.
os.environ["CONTROLAI_BACKEND"] = "torch"
# Qwen3-14B in 4-bit NF4. Override with CONTROLAI_MODEL_TORCH in Space settings.

import gradio as gr
import spaces
import uvicorn

import app as app_module
from app import app


def _fetch_index() -> None:
    """Pull the retrieval index into data/rag_index/ before the agent loads."""
    from controlai_rag.fetch_index import fetch

    if not os.environ.get("HF_TOKEN"):
        print("[space] HF_TOKEN not set -- skipping index fetch, retrieval disabled")
        return
    try:
        fetch()
        print("[space] retrieval index ready")
    except Exception as exc:  # noqa: BLE001 - a missing index must not stop boot
        print(f"[space] could not fetch the index ({exc}); retrieval disabled")


def _build_agent_at_import() -> None:
    """Construct the agent *now*, while this module is still being imported.

    ZeroGPU installs its CUDA emulation by patching torch during the import of
    the Space's entry module, and only operations that happen inside that window
    are intercepted. `app.py` normally builds the model in FastAPI's `lifespan`,
    on a ThreadPoolExecutor worker -- long after import, on another thread, where
    the patching does not apply. `.to("cuda")` there reaches the real
    `torch._C._cuda_init` and raises:

        RuntimeError: Low-level CUDA init (`torch._C._cuda_init`) reached.

    Building here and handing the finished agent to `app.py` keeps the load
    inside the window. `lifespan` then finds `_agent` already set and its
    `get_agent()` is a no-op.
    """
    from controlai_agent.agent import ControlAgent
    from controlai_agent.engine_torch import TorchEngine
    from controlai_rag.embeddings import get_embedder

    print("[space] building agent at import scope (ZeroGPU CUDA window)")
    app_module._agent = ControlAgent(engine=TorchEngine())

    # The retrieval embedder is a second model, and it loads lazily on first
    # query -- which is a request, outside the import window, so it hit exactly
    # the same _cuda_init wall and every answer came back with
    # "[agent] retrieval failed". Force it onto the GPU here too. Embedding one
    # throwaway string is what actually triggers the load.
    get_embedder().encode_query("warmup")
    print("[space] agent and embedder ready")


_fetch_index()          # the agent prewarms retrieval, so the index must precede it
_build_agent_at_import()


@spaces.GPU(duration=120)
def _gpu_probe(text: str) -> str:
    """Satisfies ZeroGPU's startup validation, and warms the model on first use.

    Takes only a plain string. The agent is reached through the module global --
    see note 4 above; passing it in is what hangs the whole Space.
    """
    return app_module.get_agent().run(text).answer if text else "ready"


with gr.Blocks() as _gpu_demo:
    _probe_in = gr.Textbox(visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn = gr.Button("probe", visible=False)
    _probe_btn.click(fn=_gpu_probe, inputs=_probe_in, outputs=_probe_out)


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    # Separate port, non-blocking: see note 3.
    _gpu_demo.launch(
        server_name="0.0.0.0",
        server_port=port + 1,
        prevent_thread_lock=True,
        share=False,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
