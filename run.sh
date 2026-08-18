#!/usr/bin/env bash
# ControlAI Universal Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check flags
if [ "$1" == "--cli" ] || [ "$1" == "-c" ]; then
    shift
    exec python cli.py "$@"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "ControlAI Launcher Usage:"
    echo "  ./run.sh            # Launch Web UI & automatically open browser (http://127.0.0.1:8000)"
    echo "  ./run.sh --cli      # Launch interactive terminal assistant"
    echo "  ./run.sh --cli \"<query>\" # Run single one-shot query in terminal"
    exit 0
else
    # Default: Launch Web UI & automatically open browser at http://127.0.0.1:8000
    exec python app.py
fi
