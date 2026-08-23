#!/usr/bin/env bash
# ControlAI launcher.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"
[ -d ".venv" ] && source .venv/bin/activate

case "${1:-}" in
  --cli|-c)
    shift
    exec python cli.py "$@"
    ;;
  --build-index)
    exec python -m controlai_rag.retriever --build
    ;;
  --fetch-index)
    exec python -m controlai_rag.fetch_index "${@:2}"
    ;;
  --ingest-corpus)
    exec python scripts/ingest_processed_corpus.py "${@:2}"
    ;;
  --calibrate)
    exec python scripts/calibrate_retrieval.py "${@:2}"
    ;;
  --help|-h)
    cat <<'USAGE'
ControlAI

  ./run.sh                     web console at http://127.0.0.1:8000
  ./run.sh --cli               interactive terminal chat
  ./run.sh --cli "question"    one-shot question
  ./run.sh --fetch-index       download the prebuilt index from the HF Hub
  ./run.sh --build-index       rebuild the dense retrieval index
  ./run.sh --ingest-corpus     merge data/processed/ chunks into the index
  ./run.sh --calibrate         re-measure MIN_COSINE against the current corpus

Environment:
  CONTROLAI_MODEL          MLX model id       (default mlx-community/Qwen3-14B-4bit)
  CONTROLAI_ADAPTER        LoRA adapter path  (default none)
  CONTROLAI_THINKING       off | auto | on    (default auto)
  CONTROLAI_THINK_BUDGET   max reasoning tokens (default 512)
USAGE
    ;;
  *)
    exec python app.py
    ;;
esac
