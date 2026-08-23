# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

ControlAI is an offline control-systems engineering assistant. A locally held model
(`mlx-community/Qwen3-14B-4bit` by default, via MLX) answers control questions, and every number it
states comes from a deterministic solver — SciPy/LAPACK/CVXPY behind a validated tool registry —
never from the model's own arithmetic. A BM25 + dense hybrid retriever over a local control-theory
corpus grounds conceptual answers. A FastAPI + vanilla-JS console and a terminal CLI are the two
front ends. Nothing leaves the machine.

Apple Silicon only for local use. There is no GGUF or Ollama path — an earlier version carried
both plus CUDA and a Spaces deployment, and maintaining four backends for one machine was most of
the complexity in the codebase.

One CUDA path came back, narrowly, for the public demo Space: see **Deployment** below. It is a
second `Engine` implementation behind the same contract, selected by one env var. It does not touch
the agent loop, the registry, or any tool.

## Commands

```bash
./run.sh                      # web console at http://127.0.0.1:8000
./run.sh --cli                # interactive terminal chat
./run.sh --cli "design an LQR for A=[[0,1],[-2,-3]], B=[[0],[1]], Q=eye(2), R=1"
./run.sh --fetch-index        # download the prebuilt index from the private Hub dataset repo
./run.sh --build-index        # rebuild the dense retrieval index (~45 min, checkpointed/resumable)
./run.sh --ingest-corpus      # merge data/processed/ chunks into the index
./run.sh --calibrate          # re-measure MIN_COSINE against the current corpus

python -m unittest discover tests -v          # unittest, not pytest -- pytest is not installed
python -m unittest tests.test_agent_tools     # deterministic numerics
python -m unittest tests.test_agent_core      # parsing, stream gating, truncation

pip install -r requirements.txt               # runtime
pip install -r requirements-training.txt      # dataset generation / ground truth
pip install -r requirements-corpus.txt        # corpus crawling and extraction
```

Environment: `CONTROLAI_MODEL`, `CONTROLAI_ADAPTER`, `CONTROLAI_THINKING` (`off|auto|on`),
`CONTROLAI_THINK_BUDGET`, `CONTROLAI_EMBED_MODEL`.

There is no lint or format command configured — don't invent one.

## Architecture

### Request flow
`web/` (SSE console) or `cli.py` → `app.py` (`/api/chat`, `/api/chat/stream`) →
`ControlAgent` (`controlai_agent/agent.py`) → `LocalEngine` (`controlai_agent/engine.py`, MLX) and
`registry.execute` (`controlai_agent/registry.py`) → tools in `controlai_agent/tools/*.py`.

### `engine.py` — inference
One persistent KV cache lives for the process. `_align_cache` finds the longest prefix of the
incoming prompt that the cache already holds, trims to that point, and feeds only the remainder, so
the fixed system-prompt-plus-tool-schema prefix (~8k tokens) is prefilled once at startup by
`prewarm()` rather than once per tool step. Generated tokens are tracked in the cache too, which is
what makes continuing a tool-calling turn nearly free. Measured effect: first-token latency fell
from ~6.4 s per step to ~0.7 s.

`stream()` yields tokens as they are produced. `think_budget` caps tokens spent inside a `<think>`
block and closes it by hand on overrun, bounding worst-case latency on a reasoning model.

Sampling uses `presence_penalty`, not a flat `repetition_penalty`. A blanket repetition penalty
punishes the repeated structural tokens that matrices and JSON are made of (`[`, `0`, `,`) exactly
when the model is emitting a tool call.

### `agent.py` — the loop
A short tool-calling loop (`MAX_TOOL_STEPS = 2`, `MAX_CALLS_PER_TOOL = 2`) that streams throughout.
`_StreamGate` releases text as soon as it cannot be the start of `<tool_call>`, so prose appears
immediately while a tool call never leaks into the chat. Retrieved passages are attached to the
*user* turn, never spliced into the system prompt, because that keeps the cached prefix
byte-identical across questions.

**There is deliberately no parameter-provenance check.** An earlier version refused any matrix it
could not trace back to the user's message. It blocked the single most useful thing the assistant
does — working an example that the user asked for — so it was removed. Schema validation and
sandboxed execution in `registry.execute` are the real guarantees; `toolcall.degenerate_reason`
catches genuine decoding loops with thresholds set well above any hand-written matrix.

### `prompts.py`
506 tokens, down from 2,587. States what to do rather than what not to do. The previous prompt's
"never invent a parameter" clause taught the model to refuse worked examples; the current one
explicitly authorises choosing an illustrative system when the user asks for a demonstration.

### Tool registry (`registry.py` + `controlai_agent/tools/`)
Tools are plain functions decorated with `@registry.register(name, description, parameters_schema)`
across `linear.py`, `frequency.py`, `synthesis.py`, `estimation.py`, `nonlinear.py`, `robust.py`,
`allocation.py`, `simulation.py`, `matrix_ops.py`, `plotting.py`, `python_executor.py`, `rag.py`.
`registry.execute()` coerces stringified arrays, validates against JSON Schema, runs the function,
and rounds every float to 6 significant figures. `controlai_agent/verifier.py` cross-checks results
with independent invariants — Riccati residual, pole-placement error, CBF forward invariance.

**Prefer putting a fact in a tool result over instructing the model to derive it.** Two observed
failures were fixed that way rather than by prompt wording, and it is the pattern to reach for
first:
- `continuous_lqr`, `discrete_lqr` and `place_state_feedback` return `closed_loop_A`. Given a
  correct gain, the model wrote $A-BK$ for a double integrator as `[[-6,1],[-5,-6]]` instead of
  `[[0,1],[-6,-5]]` — right gain, wrong write-up.
- `bode_analysis` returns the gain/phase margins as well. Asked for a phase margin the model
  reached for it instead of `stability_margins`, got back sampled curves, and correctly but
  uselessly concluded the margin "cannot be determined".

### Retrieval (`controlai_rag/`)

Three data bugs were found here and are worth knowing about, because each one silently degraded
retrieval rather than failing:
- **Chunk ids were not unique.** `chunk_document` is called once per page and restarted its counter
  each time, so every page's first chunk was `<file>_c0000` — 154 distinct ids across 9,976 chunks.
  Anything keyed on chunk_id resolved to the wrong row. Fixed in `chunker.py` (ids are now
  page-qualified); `scripts/repair_chunk_ids.py` migrated the existing index in place.
- **Embeddings were pooled without an EOS token.** Qwen3-Embedding pools the last position and was
  trained with `<|endoftext|>` there. Omitting it dropped the margin between relevant and irrelevant
  passages from +0.32 to +0.15. Note the checkpoint's own `eos_token_id` is `<|im_end|>`, which is
  the wrong token here — `controlai_rag/embeddings.py` pins the right one.
- **71.5% of the Nise textbook was mojibake.** Its PDF has a broken symbol-font ToUnicode map, so
  every extractor returns `L½ f ðtÞ/C138 ¼FðsÞ` for `L[f(t)] = F(s)`. `controlai_rag/textfix.py`
  reverses the substitution (it is deterministic); `scripts/repair_corpus_text.py` migrated the
  index. `document_loader` now applies it on ingest.

`document_loader.py` → `chunker.py` → `index.py` (BM25, persisted to `data/rag_index/`) and
`embeddings.py` (Qwen3-Embedding-0.6B under MLX, last-token pooling, instruction-prefixed queries).

**The index holds 80,370 chunks.** 9,976 come from `data/user_docs/` (course notes, Nise, Ogata);
the other 70,394 were bridged in from `data/processed/*_chunks/` by
`scripts/ingest_processed_corpus.py`. That bridge did not exist before: the `scripts/` pipeline
(`raw → extracted → processed`) fed *training dataset generation* only, while `ControlRAGIndex` read
`user_docs` alone. Retrieval was therefore seeing 12% of the corpus, and none of the canonical
texts — Doyle/Francis/Tannenbaum, Åström & Murray, Rawlings/Mayne/Diehl, Sontag, Liberzon, Boyd,
Söderström & Stoica — which were downloaded, extracted and chunked on disk, unread. Re-run the
bridge after adding anything to `data/processed/`.

**The index is not in git.** `chunks.json` (187MB), `embeddings.npz` (144MB) and `bm25.pkl`
(124MB) are each past GitHub's 100MB per-file limit, and `chunks.json` holds the full extracted text
of 666 documents including commercial textbooks (Nise, Ogata) — fine to keep locally, wrong to
redistribute. They live in the **private** Hub dataset repo `atakankahya/controlai-rag-index`;
`controlai_rag/fetch_index.py` (`./run.sh --fetch-index`) pulls them with your HF token. It copies
out of the Hub cache rather than symlinking, because the index is mutated in place by uploads and by
`scripts/repair_*.py`. A public clone has no index and builds its own from `data/user_docs/`.

`build()` is checkpointed every 4,000 chunks to `embeddings.partial.npz` and resumes from it. An
80k-chunk build takes ~45 minutes, and writing only at the end meant one interruption discarded all
of it (observed at 72,008 of 80,370).
`retriever.py` fuses the two rankings with reciprocal rank fusion and gates on cosine similarity
(`MIN_COSINE = 0.62`). That number is a property of the model *and* the corpus, so re-measure it
with `./run.sh --calibrate` whenever the corpus changes size materially — it moved when the index
grew 8x. Current margins: in-domain worst best-match 0.678, off-domain best 0.571. Lexical candidates are scored densely too — without that, a chunk BM25 ranked first but
that fell outside the dense top-K was dropped for having no score rather than for being irrelevant.
`_is_low_value` discards back-of-book index pages, which match almost any control query because they
contain every term in the field, while explaining nothing.

The gate matters. BM25 scores are unbounded and corpus-relative, so the old threshold of 2.5 passed
essentially everything: a question about the Bode sensitivity integral retrieved Routh-Hurwitz
tables at score 19.7 and injected them as authoritative context. Returning nothing is a valid and
frequent outcome — the model then answers from its own knowledge.

`display_source_name()` maps raw indexed filenames (which carry owner initials, course codes, scan
artifacts) to clean citable labels. Never let a raw filename reach an answer.

Uploads through `/api/upload` are embedded immediately via `HybridRetriever.add_chunks`, or they
would have no vector and the cosine gate would make them permanently unreachable.

### `app.py` — serving
Local-only FastAPI. MLX keeps its compute stream in thread-local state, so every inference call
goes through one dedicated worker thread for the process lifetime; a lock serialises turns because
they all mutate the same KV cache. `_to_wire_events` translates the agent's event vocabulary
(`text`/`thinking`/…) into what `web/app.js` consumes (`token`/`thought`/…).

### Model artifacts
`adapters/` and `models/` hold earlier fine-tuning output. **They are gitignored — not backed up by
git.** The LoRA adapters are not loaded by default: `behavior_v1` emits a spurious empty
`<tool_call></tool_call>` as its first output on essentially every prompt, so no tool ever runs, and
it refuses fully-specified problems; `sft_v2` produces empty output when no tools are exposed and
string-typed numbers when they are. Both were measured against the base model, which routes and
formats correctly. `CONTROLAI_ADAPTER=<path>` loads one anyway for A/B work.

`configs/*.yaml` are MLX-LoRA training configs and `scripts/` holds the offline pipeline (corpus
discovery → extraction → dataset generation → training → evaluation). That pipeline is independent
of the serving path and uses `requirements-training.txt`/`requirements-corpus.txt`.

### Deployment (`app_space.py`, `engine_torch.py`, `requirements-space.txt`)
The demo Space (huggingface.co/spaces/atakankahya/ControlAI-Agent) runs Linux/NVIDIA on ZeroGPU,
where MLX does not exist. `CONTROLAI_BACKEND=torch` makes `app.py::_make_engine` build
`TorchEngine` instead of `LocalEngine`; `Embedder` switches on the same variable. That is the whole
switch — two branches, no orchestrator.

`engine_torch.py` mirrors `engine.py` rather than calling `model.generate`, because `generate`
cannot express either of the two things that matter: `DynamicCache.crop()` for prefix reuse across
tool steps, and injecting `</think>` to close an overrunning reasoning block. Loading is 4-bit NF4
(Qwen3-14B is ~28GB bf16, ~9GB quantised) with `device_map={"": 0}` — **never `"auto"`**, which
inspects free VRAM at load time, before ZeroGPU has attached hardware, and silently offloads to CPU.

**ZeroGPU platform gotchas, each learned by having the Space fail:**
- A `@spaces.GPU` function is only detected if wired to a real Gradio event handler. One called
  solely from a FastAPI route fails startup with "No @spaces.GPU function detected". Hence the
  hidden probe button.
- That function must be a module-level `def`; nested inside `with gr.Blocks():` the detection fails.
- Don't `mount_gradio_app()` the probe and then run uvicorn on the same port — Gradio's own server
  setup collides ("address already in use"). The probe launches on `port + 1`, non-blocking.
- **Never pass the model-holding object as an argument** to a `@spaces.GPU` function. ZeroGPU
  marshals arguments across a process boundary and tries to share the model's CUDA tensors, failing
  with `_share_cuda_: only available on CUDA` after emitting nothing — which reads exactly like
  "just slow". Reach the agent through the module global.

The Space needs `HF_TOKEN` as a secret: the retrieval index is in a private dataset repo and
`app_space.py::_fetch_index` pulls it at startup. Without it the Space still boots, and answers
from model knowledge alone.

**The Space's queries are embedded with the bf16 `Qwen/Qwen3-Embedding-0.6B` while the index was
built with the MLX 4-bit checkpoint.** Measured, the two produce vectors agreeing at cosine 0.96,
and against the real 80,370-chunk index the gate behaves the same: in-domain worst best-match 0.712
(MLX 0.708), off-domain best 0.528 (MLX 0.541). `MIN_COSINE = 0.62` therefore holds unchanged on
both. `CONTROLAI_MIN_COSINE` overrides it if that ever drifts.

### Benchmark (`benchmarks/`)
`controlbench_v1.jsonl` is the eval set; `SCOPE.md` defines the taxonomy and `README.md` the
data-hygiene rules (never let benchmark prompts leak into training data; split by `family`, not by
individual question). `scripts/eval_answer_quality.py` is the fast qualitative check across 18
domain cases.
