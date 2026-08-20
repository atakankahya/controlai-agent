"""ControlAI Web Application Backend (FastAPI)."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.orchestrator import HAS_MLX, ControlAIAgent
from controlai_rag.chunker import chunk_document
from controlai_rag.document_loader import load_single_file
from controlai_rag.index import get_shared_index

# MLX keeps its compute stream in thread-local state: the model must be
# loaded on the exact same OS thread that later runs generation, or MLX
# raises "There is no Stream(cpu, 0) in current thread." Routing every agent
# call through this single dedicated worker thread keeps MLX on one
# consistent thread for the whole process lifetime.
#
# This must NEVER be used for the CUDA/ZeroGPU path: ZeroGPU's `spaces`
# library only intercepts CUDA calls within the exact context it manages
# (the main thread during startup; the request-handling context for
# @spaces.GPU calls). Moving a CUDA-touching call into this manually-created
# thread let a real `torch._C._cuda_init()` leak through outside any
# @spaces.GPU context and crashed the Space on startup:
#   "Low-level CUDA init reached. ZeroGPU's PyTorch CUDA emulation mode
#    did not intercept a CUDA operation in your code."
# HAS_MLX is only ever True on the machine that actually has mlx_lm
# installed (Apple Silicon) -- never on a HF Spaces Linux/CUDA container --
# so gating on it keeps MLX's fix local while restoring the CUDA/GGUF path
# to calling directly on whatever thread FastAPI/spaces already controls,
# exactly as it worked before.
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="controlai-inference") if HAS_MLX else None
# llama.cpp's Llama object is not safe for concurrent generation calls from
# multiple threads; on the non-MLX path (no dedicated executor serializing
# things for us) a plain lock does that job instead.
inference_lock = threading.Lock()


async def _run_inference(fn, *args):
    """Run an inference call on the MLX-safe dedicated thread if MLX is in
    play, otherwise directly (correct for CUDA/ZeroGPU and GGUF)."""
    if inference_executor is not None:
        return await asyncio.get_event_loop().run_in_executor(inference_executor, fn, *args)
    with inference_lock:
        return fn(*args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading ControlAI Core Engine on startup...")
    await _run_inference(get_agent)
    print("ControlAI Core Engine is online and ready for traffic.")
    yield


app = FastAPI(title="ControlAI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directories
STATIC_DIR = PROJECT_ROOT / "web"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
USER_DOCS_DIR = PROJECT_ROOT / "data" / "user_docs"
UPLOADED_DOCS_DIR = USER_DOCS_DIR / "user_uploaded"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Mount plots for web rendering
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Agent
agent_instance: ControlAIAgent | None = None


def get_agent() -> ControlAIAgent:
    global agent_instance
    if agent_instance is None:
        print("Initializing ControlAI Core Engine...")
        agent_instance = ControlAIAgent()
        print("ControlAI Core Engine initialized.")
    return agent_instance


# On ZeroGPU Spaces, actual CUDA work may only happen inside a function
# decorated with @spaces.GPU (it requests physical GPU time for the call and
# releases it afterward). Outside of a ZeroGPU Space this decorator is a
# harmless no-op, so it's safe to always wrap these.
@spaces.GPU(duration=300)
def _run_stream_on_gpu(message: str, history: list[dict[str, str]]):
    yield from get_agent().run_stream(message, history=history)


@spaces.GPU(duration=300)
def _run_on_gpu(message: str, history: list[dict[str, str]]):
    return get_agent().run(message, history=history, verbose=False)


# ZeroGPU's startup check statically looks for a @spaces.GPU function wired
# to a Gradio event handler -- must be a module-level function referenced by
# name, not one defined inline inside a `with gr.Blocks():` block.
@spaces.GPU(duration=300)
def _zerogpu_registration_probe(message: str) -> str:
    result = get_agent().run(message or "hi", history=[], verbose=False)
    return result.final_response


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    response: str
    tool_traces: list[dict[str, Any]] = []
    plots: list[str] = []
    elapsed_seconds: float


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(
            content=index_file.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
        )
    return HTMLResponse("<h1>ControlAI Web UI initializing...</h1>")


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    rag_index_path = PROJECT_ROOT / "data" / "rag_index"
    chunk_count = 0
    if rag_index_path.exists():
        chunks_file = rag_index_path / "chunks.json"
        if chunks_file.exists():
            try:
                data = json.loads(chunks_file.read_text(encoding="utf-8"))
                chunk_count = len(data)
            except Exception:
                pass

    return {
        "system": "ControlAI",
        "status": "ready",
        "indexed_chunks": chunk_count,
    }


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    categories: dict[str, list[str]] = {}
    if USER_DOCS_DIR.exists():
        for item in USER_DOCS_DIR.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                cat_name = item.name.replace("_", " ").title()
                files = [f.name for f in item.rglob("*.pdf") if not f.name.startswith(".")]
                if files:
                    categories[cat_name] = files[:10]  # Show up to 10 per category
    return {"categories": categories}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")

    target_path = UPLOADED_DOCS_DIR / filename
    with target_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest directly into RAG index
    try:
        pages = load_single_file(target_path)
        new_chunks = []
        for p in pages:
            chunks = chunk_document(p)
            new_chunks.extend(chunks)

        # Mutating the shared index makes the upload live for the running
        # agent immediately -- a fresh instance would only persist to disk and
        # stay invisible until restart.
        index = get_shared_index()
        index.add_chunks(new_chunks)

        return {
            "status": "success",
            "filename": filename,
            "pages_parsed": len(pages),
            "chunks_added": len(new_chunks),
            "total_indexed_chunks": len(index.chunks),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(exc)}")


from fastapi.responses import StreamingResponse


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    message = req.message.strip()
    history = req.history

    if inference_executor is not None:
        # MLX path: a persistent dedicated thread is required (see the
        # inference_executor comment above), so bridge it into the async
        # response via a queue -- Starlette's own rotating threadpool would
        # violate MLX's single-thread requirement.
        event_queue: queue.Queue = queue.Queue()
        _DONE = object()

        def _produce() -> None:
            try:
                with inference_lock:
                    for event in _run_stream_on_gpu(message, history):
                        event_queue.put(event)
            except Exception as exc:
                event_queue.put({"type": "error", "error": str(exc)})
            finally:
                event_queue.put(_DONE)

        async def event_generator():
            loop = asyncio.get_event_loop()
            loop.run_in_executor(inference_executor, _produce)
            while True:
                # Draining the queue never touches MLX, so this can safely
                # run on the default threadpool.
                event = await loop.run_in_executor(None, event_queue.get)
                if event is _DONE:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    else:
        # CUDA/ZeroGPU/GGUF path: a plain sync generator handed directly to
        # StreamingResponse, exactly as this ran before the MLX fix existed.
        # Starlette wraps this in its own threadpool (iterate_in_threadpool),
        # which -- unlike a manually created ThreadPoolExecutor -- is a
        # context ZeroGPU's CUDA interception correctly recognizes.
        def event_generator():
            try:
                with inference_lock:
                    for event in _run_stream_on_gpu(message, history):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    t0 = time.time()
    try:
        result = await _run_inference(_run_on_gpu, req.message.strip(), req.history)
        elapsed = time.time() - t0

        # Collect tool traces
        traces = []
        plots = []
        for t in result.tool_traces:
            trace_data = {
                "tool": t.tool_name,
                "args": t.arguments,
                "status": t.result.get("status", "success"),
                "residual": t.result.get("residual"),
            }
            traces.append(trace_data)

            # Check if plot was generated by any tool
            if "plot_path" in t.result:
                p_path = Path(t.result["plot_path"])
                if p_path.exists():
                    plots.append(f"/plots/{p_path.name}")

        return ChatResponse(
            response=result.final_response,
            tool_traces=traces,
            plots=plots,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"Chat execution error: {exc}")
        return ChatResponse(
            response=f"Computation error during inference: {str(exc)}",
            tool_traces=[],
            plots=[],
            elapsed_seconds=round(elapsed, 2),
        )


# A minimal Gradio Blocks app exists purely so this Space is recognized as a
# Gradio SDK app -- required for ZeroGPU hardware -- and so the @spaces.GPU
# registration probe below is wired to a real Gradio event handler ("No
# @spaces.GPU function detected" otherwise). It is launched on its own
# internal port rather than mounted into our FastAPI app: mounting it and
# then binding our own uvicorn to the same public port collided with
# Gradio's own server startup ("address already in use"). The real UI and
# API are still served entirely by our own FastAPI routes above, on the
# public port.
_gpu_demo = gr.Blocks()
with _gpu_demo:
    gr.Markdown("ControlAI is running. Visit the Space's root URL for the full app.")
    _probe_in = gr.Textbox(visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn = gr.Button(visible=False)
    _probe_btn.click(fn=_zerogpu_registration_probe, inputs=_probe_in, outputs=_probe_out)


def main() -> None:
    import uvicorn

    if os.environ.get("SPACE_ID"):
        port = int(os.environ.get("PORT", 7860))
        _gpu_demo.launch(server_name="0.0.0.0", server_port=port + 1, prevent_thread_lock=True, show_error=False, quiet=True)
        print(f"ControlAI Web UI running on Hugging Face Spaces (port {port})...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        return

    import threading
    import webbrowser

    def _open_browser() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()
    print("ControlAI Web UI running at: http://127.0.0.1:8000 (Opening browser...)")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
