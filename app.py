"""ControlAI Web Application Backend (FastAPI)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.orchestrator import ControlAIAgent
from controlai_rag.chunker import chunk_document
from controlai_rag.document_loader import load_single_file
from controlai_rag.index import ControlRAGIndex

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading ControlAI Core Engine on startup...")
    get_agent()
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

        index_dir = PROJECT_ROOT / "data" / "rag_index"
        index = ControlRAGIndex(index_dir)
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

    agent = get_agent()

    def event_generator():
        try:
            for event in agent.run_stream(req.message.strip(), history=req.history):
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
        agent = get_agent()
        result = agent.run(req.message.strip(), history=req.history, verbose=False)
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


def main() -> None:
    import threading
    import webbrowser
    import uvicorn

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
