"""ControlAI web server.

Local-only by design. The previous version carried a Hugging Face Spaces
deployment inside it -- Gradio blocks that existed purely to satisfy a ZeroGPU
SDK check, `@spaces.GPU` decorators, a CUDA/GGUF branch, and a threading split
whose two halves each existed to work around the other environment. None of it
ran on this machine, and all of it had to be reasoned about on every change.
What is left is a FastAPI app serving one MLX-backed agent.

MLX keeps its compute stream in thread-local state: the model must be used from
the same OS thread that loaded it, or it raises "There is no Stream(gpu, 0) in
current thread." Every call therefore goes through one dedicated worker thread.
"""

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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.agent import ControlAgent
from controlai_rag.chunker import chunk_document
from controlai_rag.document_loader import load_single_file
from controlai_rag.index import get_shared_index

STATIC_DIR = PROJECT_ROOT / "web"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
UPLOADS_DIR = PROJECT_ROOT / "data" / "user_docs" / "user_uploaded"
for directory in (STATIC_DIR, PLOTS_DIR, UPLOADS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# One thread, for the lifetime of the process: see the module docstring.
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="controlai")
# The agent holds a single KV cache that every turn mutates, so turns must not
# interleave even though they all land on the same thread.
inference_lock = threading.Lock()

_agent: ControlAgent | None = None


def _make_engine():
    """The engine for this process. MLX unless the Space asked for CUDA.

    This is the *only* backend branch left in the serving path, and it exists
    for one reason: the public demo Space runs on Linux/NVIDIA, where MLX does
    not exist. `TorchEngine` is imported lazily so a normal Apple Silicon run
    never needs torch installed. Everything downstream -- the agent loop, the
    registry, every tool -- is identical either way.
    """
    if os.environ.get("CONTROLAI_BACKEND", "mlx").lower() == "torch":
        from controlai_agent.engine_torch import TorchEngine

        return TorchEngine()
    return None  # ControlAgent's own default, LocalEngine


def get_agent() -> ControlAgent:
    global _agent
    if _agent is None:
        _agent = ControlAgent(engine=_make_engine())
    return _agent


async def _on_inference_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(inference_executor, fn, *args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    print("Loading ControlAI…")
    await _on_inference_thread(get_agent)
    print(f"ControlAI ready in {time.time() - started:.1f}s -> http://127.0.0.1:8000")
    yield


app = FastAPI(title="ControlAI", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    response: str
    tool_traces: list[dict[str, Any]] = []
    plots: list[str] = []
    elapsed_seconds: float


def _to_wire_events(message: str, history: list[dict[str, str]]):
    """Translate agent events into the shape the browser client consumes.

    The client speaks `thought`/`token`/`tool_end`/`plot`/`done`; the agent
    speaks `thinking`/`text`/... Doing the mapping here keeps the wire format
    stable for the existing UI while the agent's own vocabulary stays clean.
    """
    thoughts: list[str] = []
    for event in get_agent().stream(message, history):
        kind = event["type"]
        if kind == "text":
            yield {"type": "token", "content": event["text"]}
        elif kind == "thinking":
            yield {"type": "thought", "content": event["text"]}
        elif kind == "tool_start":
            note = f"Calling {event['tool']} with {json.dumps(event['arguments'], ensure_ascii=False, default=str)[:300]}"
            thoughts.append(note)
            yield {"type": "thought", "content": note}
            yield {"type": "tool_start", "tool": event["tool"], "args": event["arguments"]}
        elif kind == "tool_end":
            trace = {"tool": event["tool"], "status": event["status"]}
            note = f"{event['tool']} -> {event['status']}"
            thoughts.append(note)
            yield {"type": "thought", "content": note}
            yield {"type": "tool_end", "trace": trace}
        elif kind == "plot":
            yield {"type": "plot", "url": event["url"]}
        elif kind == "done":
            yield {
                "type": "done",
                "response": event["answer"],
                "traces": event["traces"],
                "plots": event["plots"],
                "sources": event["sources"],
                "thoughts": thoughts,
                "stats": event["stats"],
            }


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>ControlAI: web/index.html is missing</h1>", status_code=500)
    return HTMLResponse(
        index_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/status")
async def status() -> dict[str, Any]:
    index = get_shared_index()
    agent = _agent
    payload: dict[str, Any] = {
        "system": "ControlAI",
        "status": "ready" if agent else "loading",
        "indexed_chunks": len(index.chunks),
    }
    if agent:
        payload.update(
            model=agent.engine.model_id,
            adapter=agent.engine.adapter_path,
            thinking=agent.thinking,
            dense_retrieval=bool(agent.retriever and agent.retriever.has_dense),
        )
    return payload


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    categories: dict[str, list[str]] = {}
    root = UPLOADS_DIR.parent
    if root.exists():
        for item in sorted(root.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                files = [f.name for f in item.rglob("*.pdf") if not f.name.startswith(".")]
                if files:
                    categories[item.name.replace("_", " ").title()] = files[:10]
    return {"categories": categories}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if Path(file.filename).suffix.lower() not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")

    target = UPLOADS_DIR / Path(file.filename).name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    def _ingest() -> dict[str, Any]:
        pages = load_single_file(target)
        new_chunks = [c for page in pages for c in chunk_document(page)]
        index = get_shared_index()
        index.add_chunks(new_chunks)
        # Embedding happens on the inference thread because it uses MLX too.
        embedded = 0
        agent = get_agent()
        if agent.retriever is not None:
            embedded = agent.retriever.add_chunks(
                [c for c in index.chunks if c["chunk_id"] in {n["chunk_id"] for n in new_chunks}]
            )
        return {
            "status": "success",
            "filename": target.name,
            "pages_parsed": len(pages),
            "chunks_added": len(new_chunks),
            "chunks_embedded": embedded,
            "total_indexed_chunks": len(index.chunks),
        }

    try:
        with inference_lock:
            return await _on_inference_thread(_ingest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    events: queue.Queue = queue.Queue()
    sentinel = object()

    def produce() -> None:
        try:
            with inference_lock:
                for event in _to_wire_events(message, req.history):
                    events.put(event)
        except Exception as exc:  # surface the failure in the chat, don't hang
            print(f"[chat] {type(exc).__name__}: {exc}")
            events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(sentinel)

    async def relay():
        loop = asyncio.get_running_loop()
        loop.run_in_executor(inference_executor, produce)
        while True:
            # Draining the queue never touches MLX, so the default threadpool
            # is fine here and keeps the single inference thread free.
            event = await loop.run_in_executor(None, events.get)
            if event is sentinel:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    started = time.time()

    def _run():
        with inference_lock:
            return get_agent().run(message, req.history)

    try:
        result = await _on_inference_thread(_run)
    except Exception as exc:
        print(f"[chat] {type(exc).__name__}: {exc}")
        return ChatResponse(
            response=f"Inference failed: {exc}",
            elapsed_seconds=round(time.time() - started, 2),
        )

    return ChatResponse(
        response=result.answer,
        tool_traces=[{"tool": t.name, "status": t.status} for t in result.traces],
        plots=result.plots,
        elapsed_seconds=round(time.time() - started, 2),
    )


def main() -> None:
    import threading as _threading
    import webbrowser

    import uvicorn

    def open_browser() -> None:
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    _threading.Thread(target=open_browser, daemon=True).start()
    # No reload: the model load is far too expensive to repeat on every file
    # save, and reload would fork a second copy of it.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
