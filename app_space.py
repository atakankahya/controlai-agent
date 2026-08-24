"""Hugging Face Space entry point: a Gradio app on ZeroGPU.

`./run.sh` runs `app.py` -- FastAPI plus the `web/` console -- and that is the
real product. This file is a *different front end over the same agent*, and it
exists because of one hard platform constraint:

**ZeroGPU assumes the Gradio app is the Space.** It schedules GPU workers by
forking the server process, and its startup validation looks for a `@spaces.GPU`
function wired to a Gradio event handler. An earlier version of this file kept
FastAPI on the public port and ran Gradio as a hidden side-car; the GPU was
scheduled and acquired and the forked worker still died in `torch.init()`, while
the platform probed the public port for `/api/predict` and got 404. Six other
incompatibilities were fixed before that one; all seven are in CLAUDE.md.

So here Gradio owns the port and the agent runs inside `@spaces.GPU`. What is
lost is the `web/` console, on the Space only. What is kept is everything that
matters: the same `ControlAgent`, the same 29 deterministic solvers, the same
verifier, the same 80,370-chunk hybrid retriever, the same model.

Two ordering rules, both learned the hard way:
*   The model is built at **import scope**. ZeroGPU patches torch during the
    entry module's import and only intercepts CUDA inside that window; building
    it later reaches real CUDA init and raises.
*   The agent is reached from inside the GPU function through a **module
    global**, never passed as an argument. ZeroGPU marshals arguments across a
    process boundary and would try to share the model's CUDA tensors, hanging
    with no output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

# Must precede the `controlai_agent` imports below.
os.environ["CONTROLAI_BACKEND"] = "torch"

import gradio as gr
import spaces

PLOTS_DIR = Path("outputs/plots")

AGENT = None


def _fetch_index() -> None:
    from controlai_rag.fetch_index import fetch

    if not os.environ.get("HF_TOKEN"):
        print("[space] HF_TOKEN not set -- skipping index fetch, retrieval disabled")
        return
    try:
        fetch()
        print("[space] retrieval index ready")
    except Exception as exc:  # noqa: BLE001 - a missing index must not stop boot
        print(f"[space] could not fetch the index ({exc}); retrieval disabled")


def _build() -> None:
    """Build the agent while ZeroGPU is still watching for CUDA calls."""
    global AGENT
    from controlai_agent.agent import ControlAgent
    from controlai_agent.engine_torch import TorchEngine
    from controlai_rag.embeddings import get_embedder

    print("[space] building agent at import scope (ZeroGPU CUDA window)")
    AGENT = ControlAgent(engine=TorchEngine())
    # The retrieval embedder is a second model and loads lazily on first query --
    # a request, outside the window. Embedding one string forces it in here too.
    get_embedder().encode_query("warmup")
    print("[space] agent and embedder ready")


@spaces.GPU(duration=300)
def respond(message: str, history: list) -> Iterator[str]:
    """One agent turn, streamed. Runs with real hardware attached.

    Wraps the whole turn rather than each generation: a turn is several
    generations sharing one KV cache, and splitting them across separate
    @spaces.GPU calls would put that shared state across a process boundary on
    every tool step.
    """
    if AGENT is None:  # pragma: no cover - import always builds it
        yield "Agent failed to start; check the Space logs."
        return

    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]

    answer, tools, sources, plots = "", [], [], []
    for event in AGENT.stream(message, turns):
        kind = event["type"]
        if kind == "text":
            answer += event["text"]
            yield answer
        elif kind == "tool_end":
            tools.append(event.get("tool"))
            # Show tool activity while the model is still thinking, so a
            # multi-step turn does not look like a hang.
            yield answer + f"\n\n*running `{event.get('tool')}`…*"
        elif kind == "done":
            answer = event["answer"]
            sources = event.get("sources") or []
            plots = event.get("plots") or []

    footer = ""
    for plot in plots:
        name = Path(plot).name
        footer += f"\n\n![{name}](/gradio_api/file={PLOTS_DIR / name})"
    if tools:
        footer += "\n\n---\n*Computed with: " + ", ".join(f"`{t}`" for t in dict.fromkeys(tools)) + "*"
    if sources:
        footer += "\n\n*Sources: " + "; ".join(str(s) for s in sources[:4]) + "*"
    yield answer + footer


with gr.Blocks(title="ControlAI", fill_height=True) as demo:
    gr.Markdown(
        "# ControlAI\n"
        "Control-systems assistant. Every number in an answer comes from a deterministic "
        "solver — SciPy/LAPACK/CVXPY behind a validated tool registry — never from the "
        "model's own arithmetic, and conceptual answers are grounded in a local "
        "control-theory corpus.\n\n"
        "*This hosted demo runs on Hugging Face's hardware, so the offline guarantee of a "
        "local install does not apply here — don't enter anything confidential. The full "
        "app, with its own console, runs on Apple Silicon: see the repository.*"
    )
    # No `type=` argument: Gradio 6 dropped it, messages is the only format now.
    # History therefore arrives as [{"role":..., "content":...}], which is what
    # `respond` expects and what ControlAgent.stream takes.
    gr.ChatInterface(
        fn=respond,
        examples=[
            "Design an LQR for A=[[0,1],[-2,-3]], B=[[0],[1]], Q=eye(2), R=1.",
            "What is the phase margin of G(s) = 10/(s(s+1)(s+5))?",
            "Explain the Bode sensitivity integral and what it implies for loop shaping.",
            "Place the poles of A=[[0,1],[0,0]], B=[[0],[1]] at -2 and -3.",
        ],
        cache_examples=False,
    )


_fetch_index()
_build()

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(allowed_paths=[str(PLOTS_DIR)])
