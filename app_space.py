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

# Must be set before app.py is imported: it decides which engine gets built.
os.environ.setdefault("CONTROLAI_BACKEND", "torch")
# Qwen3-14B in 4-bit NF4. Override with CONTROLAI_MODEL_TORCH in Space settings.

import gradio as gr
import spaces
import uvicorn

from app import app, get_agent


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


@spaces.GPU(duration=120)
def _gpu_probe(text: str) -> str:
    """Satisfies ZeroGPU's startup validation, and warms the model on first use.

    Takes only a plain string. The agent is reached through the module global --
    see note 4 above; passing it in is what hangs the whole Space.
    """
    agent = get_agent()
    return agent.run(text).answer if text else "ready"


with gr.Blocks() as _gpu_demo:
    _probe_in = gr.Textbox(visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn = gr.Button("probe", visible=False)
    _probe_btn.click(fn=_gpu_probe, inputs=_probe_in, outputs=_probe_out)


def main() -> None:
    _fetch_index()
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
