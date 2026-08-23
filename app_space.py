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
3.  The probe must be `launch()`ed, on its own port, and **not** merely mounted.
    Mounting it with `gr.mount_gradio_app` was tried: the Space fails to start
    with "No @spaces.GPU function detected during startup", so mounting does not
    register the handler the way the platform's validation looks for. Do not
    mount it into the FastAPI app and *also* run uvicorn on the same port,
    though -- that is two servers and collides with "address already in use".
    Launch with `ssr_mode=False`: Gradio 6 defaults to SSR, which spawns a Node
    subprocess into the very process ZeroGPU then `fork()`s its CUDA worker
    from, and a fork parent holding a subprocess is a plausible cause of the
    worker's "No CUDA GPUs are available".
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
from typing import Iterator

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

# Filled in at import, before and after the model load, so the parent's CUDA
# state can be compared against the worker's. See /api/gpudiag.
_parent_state: dict = {}


def _torch_state(label: str) -> dict:
    import os as _os

    import torch

    state = {
        "where": label,
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": _os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ZERO_GPU_PATCH_TORCH": _os.environ.get("ZERO_GPU_PATCH_TORCH"),
        "pid": _os.getpid(),
    }
    # is_initialized() is the one that matters: if the parent has really
    # initialised CUDA before ZeroGPU forks its worker, the child cannot use the
    # GPU and reports "No CUDA GPUs are available" -- which is our exact error.
    for name, fn in (
        ("cuda.is_initialized", lambda: torch.cuda.is_initialized()),
        ("cuda.is_available", lambda: torch.cuda.is_available()),
        ("cuda.device_count", lambda: torch.cuda.device_count()),
    ):
        try:
            state[name] = fn()
        except Exception as exc:  # noqa: BLE001 - the message is the datum
            state[name] = f"{type(exc).__name__}: {exc}"
    return state


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

    _parent_state["before_model_load"] = _torch_state("parent-before-load")
    print("[space] building agent at import scope (ZeroGPU CUDA window)")
    app_module._agent = ControlAgent(engine=TorchEngine())

    # The retrieval embedder is a second model, and it loads lazily on first
    # query -- which is a request, outside the import window, so it hit exactly
    # the same _cuda_init wall and every answer came back with
    # "[agent] retrieval failed". Force it onto the GPU here too. Embedding one
    # throwaway string is what actually triggers the load.
    get_embedder().encode_query("warmup")
    # Route every turn through the GPU-decorated generator above.
    app_module.stream_hook = _gpu_stream
    _parent_state["after_model_load"] = _torch_state("parent-after-load")
    print(f"[space] parent CUDA state after load: {_parent_state['after_model_load']}")
    print("[space] agent and embedder ready, GPU stream hook installed")


# One allocation per user turn, not per engine call. A turn is up to
# MAX_TOOL_STEPS generations plus the tool executions between them, and they all
# mutate the same KV cache; splitting them across separate @spaces.GPU calls
# would put that shared state on the far side of a process boundary each time.
# 300s is the ceiling for a turn that actually calls two tools.
@spaces.GPU(duration=300)
def _gpu_stream(message: str, history: list) -> Iterator[dict]:
    """Run one whole agent turn with real hardware attached.

    Everything about ControlAI's inference has to happen inside a call like this
    one. ZeroGPU creates the model under CUDA *emulation* at import and packs its
    tensors; only a @spaces.GPU call materialises them on a real device. Running
    the forward passes anywhere else does not fail loudly -- it reads
    unmaterialised tensors and returns fluent-looking multilingual noise at a few
    minutes per turn, which is exactly what the Space did before this existed.

    `message` and `history` are plain data. The agent is reached through the
    module global and never passed in: ZeroGPU marshals arguments across a
    process boundary and would try to share the model's CUDA tensors, hanging
    with no output at all.
    """
    yield from app_module.get_agent().stream(message, history)


@spaces.GPU(duration=60)
def _gpu_diagnostics() -> dict:
    """Report CUDA state from inside the ZeroGPU worker, where it fails."""
    import subprocess

    state = _torch_state("gpu-worker")
    try:
        state["nvidia-smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip() or "(no output)"
    except Exception as exc:  # noqa: BLE001
        state["nvidia-smi"] = f"{type(exc).__name__}: {exc}"
    return state


@app.get("/api/gpudiag")
def gpudiag() -> dict:
    """Compare parent-process CUDA state with the @spaces.GPU worker's.

    A plain `def`, so Starlette runs it on its own threadpool -- the same
    context the real inference path uses.
    """
    try:
        worker = _gpu_diagnostics()
    except Exception as exc:  # noqa: BLE001
        worker = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "parent_at_import": _parent_state,
        "parent_now": _torch_state("parent-now"),
        "worker": worker,
    }


@spaces.GPU(duration=60)
def _gpu_probe(text: str) -> str:
    """Satisfies ZeroGPU's startup validation.

    A @spaces.GPU function is only detected if it is wired to a real Gradio event
    handler, and _gpu_stream is driven by FastAPI rather than by Gradio, so it
    cannot serve that purpose itself. This one exists to be wired to the hidden
    button below.
    """
    return "ready"


# Both run at import, in this order: the index first (the agent touches retrieval
# as it starts), then the agent -- which installs _gpu_stream, so it has to come
# after that function exists.
_fetch_index()
_build_agent_at_import()


with gr.Blocks() as _gpu_demo:
    _probe_in = gr.Textbox(visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn = gr.Button("probe", visible=False)
    _probe_btn.click(fn=_gpu_probe, inputs=_probe_in, outputs=_probe_out)


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    # Launched, not mounted: mounting fails startup validation ("No @spaces.GPU
    # function detected"). Separate port, non-blocking, and ssr_mode=False so no
    # Node subprocess lands in the process ZeroGPU forks its CUDA worker from.
    _gpu_demo.launch(
        server_name="0.0.0.0",
        server_port=port + 1,
        prevent_thread_lock=True,
        share=False,
        ssr_mode=False,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
