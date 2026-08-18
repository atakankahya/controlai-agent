#!/usr/bin/env python3
"""Interactive CLI for ControlAI Agent with deterministic tool calling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.orchestrator import ControlAIAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="ControlAI Offline Engineering Agent CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        help="Base model path or HuggingFace repo",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional fine-tuned LoRA adapter path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to execute in non-interactive mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw tool calls and intermediate steps",
    )
    args = parser.parse_args()

    print("Initializing ControlAI Agent...")
    agent = ControlAIAgent(model_path=args.model, adapter_path=args.adapter_path)
    print("ControlAI Agent initialized with deterministic control tools.")

    if args.prompt:
        result = agent.run(args.prompt, verbose=args.verbose)
        print("\n" + "=" * 50)
        print("AGENT RESPONSE:")
        print("=" * 50)
        print(result.final_response)
        if result.tool_traces:
            print("\n" + "-" * 50)
            print(f"EXECUTED TOOLS ({len(result.tool_traces)}):")
            for trace in result.tool_traces:
                print(f"  * {trace.tool_name}({trace.arguments}) -> {trace.result.get('status', 'done')}")
        return 0

    print("\n" + "=" * 60)
    print("🤖 Welcome to Control-LLM (Offline Control Engineering Agent)")
    print("=" * 60)
    print("Type your control engineering question or design problem.")
    print("Commands: 'exit' or 'quit' to end, 'verbose' to toggle tool details.\n")
    while True:
        try:
            user_input = input("\n[Engineer] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break
            result = agent.run(user_input, verbose=args.verbose)
            print(f"\n[ControlAI]\n{result.final_response}")
            if result.tool_traces and not args.verbose:
                tools_used = ", ".join(t.tool_name for t in result.tool_traces)
                print(f"\n(Verified using tools: {tools_used})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ControlAI.")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
