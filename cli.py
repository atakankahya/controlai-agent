#!/usr/bin/env python3
"""Interactive terminal client for ControlAI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.agent import ControlAgent
from controlai_agent.engine import DEFAULT_ADAPTER, DEFAULT_MODEL, LocalEngine

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


def converse(agent: ControlAgent, question: str, history: list[dict], verbose: bool) -> str:
    started = time.time()
    first_text: float | None = None
    answer = ""
    in_thought = False

    for event in agent.stream(question, history):
        kind = event["type"]
        if kind == "thinking":
            if not in_thought:
                print(f"{DIM}thinking… ", end="", flush=True)
                in_thought = True
            if verbose:
                print(f"{DIM}{event['text']}{RESET}", end="", flush=True)
        elif kind == "text":
            if in_thought:
                print(RESET, flush=True)
                in_thought = False
            if first_text is None:
                first_text = time.time() - started
            print(event["text"], end="", flush=True)
        elif kind == "tool_start":
            args = str(event["arguments"])
            print(f"\n{DIM}→ {event['tool']}({args[:100]}{'…' if len(args) > 100 else ''}){RESET}", flush=True)
        elif kind == "tool_end":
            print(f"{DIM}  {event['status']}{RESET}", flush=True)
        elif kind == "plot":
            print(f"\n{DIM}[plot saved: outputs/plots/{Path(event['url']).name}]{RESET}", flush=True)
        elif kind == "done":
            answer = event["answer"]
            if event["sources"]:
                print(f"\n\n{DIM}Sources: {'; '.join(dict.fromkeys(event['sources']))}{RESET}")
            stats = event["stats"]
            print(
                f"\n{DIM}{time.time() - started:.1f}s total · first token {first_text or 0:.2f}s · "
                f"{stats['cached_tokens']}/{stats['prompt_tokens']} prompt tokens cached · "
                f"{stats['decode_tps']} tok/s{RESET}"
            )
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="ControlAI -- offline control engineering agent")
    parser.add_argument("prompt", nargs="*", help="ask one question and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id or local path")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="optional LoRA adapter path")
    parser.add_argument(
        "--thinking",
        choices=("off", "auto", "on"),
        default=None,
        help="reasoning before answering (default: auto -- on for conceptual questions only)",
    )
    parser.add_argument("--think-budget", type=int, default=None, help="max tokens spent reasoning")
    parser.add_argument("--verbose", action="store_true", help="print reasoning and full tool output")
    args = parser.parse_args()

    print(f"Loading {args.model}…")
    engine = LocalEngine(model_id=args.model, adapter_path=args.adapter)
    kwargs = {}
    if args.thinking:
        kwargs["thinking"] = args.thinking
    if args.think_budget:
        kwargs["think_budget"] = args.think_budget
    agent = ControlAgent(engine=engine, **kwargs)
    print(f"Ready in {engine.load_seconds:.1f}s.\n")

    if args.prompt:
        converse(agent, " ".join(args.prompt), [], args.verbose)
        return 0

    print(f"{BOLD}ControlAI{RESET} — control engineering assistant. Ctrl-D or 'exit' to quit.")
    history: list[dict] = []
    while True:
        try:
            question = input(f"\n{BOLD}you ›{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            return 0
        if question.lower() in ("clear", "reset"):
            history.clear()
            agent.engine.reset_cache()
            print(f"{DIM}history cleared{RESET}")
            continue
        print()
        answer = converse(agent, question, history, args.verbose)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
