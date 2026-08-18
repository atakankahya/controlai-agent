#!/usr/bin/env python3
"""Control-LLM Interactive Terminal CLI.

Run standalone in terminal:
    python cli.py
Or single query:
    python cli.py "Design an LQR controller for A=[[0, 1], [-2, -3]], B=[[0], [1]]"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlai_agent.orchestrator import ControlAIAgent

console = Console()


def print_banner() -> None:
    banner_text = Text()
    banner_text.append("ControlAI Terminal Assistant\n", style="bold cyan")
    banner_text.append("Control Theory • Computational Python • Mathematical Simulation\n", style="dim white")
    banner_text.append("Commands: ", style="bold")
    banner_text.append("/clear ", style="yellow")
    banner_text.append("(new session) • ", style="dim")
    banner_text.append("/exit ", style="yellow")
    banner_text.append("(quit) • ", style="dim")
    banner_text.append("/help", style="yellow")

    console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))


def run_interactive(agent: ControlAIAgent) -> None:
    print_banner()
    history: list[dict[str, str]] = []

    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() in ("/clear", "clear"):
            history.clear()
            console.clear()
            print_banner()
            console.print("[yellow]Started a new chat session.[/yellow]")
            continue

        if user_input.lower() in ("/help", "help"):
            console.print(Panel(
                "• Type your question or control engineering prompt directly.\n"
                "• Ask to simulate ODEs, discretize state space matrices, or design LQR.\n"
                "• Plots are saved automatically to outputs/plots/.\n"
                "• /clear - Clear session history.\n"
                "• /exit  - Quit the terminal session.",
                title="Help & Guide",
                border_style="blue",
            ))
            continue

        # Run streaming agent response
        console.print()
        console.print("[bold cyan]ControlAI[/bold cyan]")

        accumulated_text = ""
        plots_generated = []

        try:
            for event in agent.run_stream(user_input, history=history):
                etype = event.get("type")

                if etype == "thought":
                    content = event.get("content", "")
                    console.print(f"  [dim cyan]› {content}[/dim cyan]")

                elif etype == "token":
                    content = event.get("content", "")
                    accumulated_text += content
                    sys.stdout.write(content)
                    sys.stdout.flush()

                elif etype == "plot":
                    plot_url = event.get("url", "")
                    plots_generated.append(plot_url)
                    console.print(f"\n  [bold green]📊 Plot saved:[/bold green] [underline white]{plot_url}[/underline white]")

                elif etype == "done":
                    final_resp = event.get("response", "")
                    if not accumulated_text:
                        accumulated_text = final_resp
                        console.print(Markdown(final_resp))

            sys.stdout.write("\n")
            sys.stdout.flush()

            # Record history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": accumulated_text})

        except Exception as err:
            console.print(f"[bold red]Error:[/bold red] {err}")


def run_oneshot(agent: ControlAIAgent, prompt: str) -> None:
    accumulated = ""
    for event in agent.run_stream(prompt):
        etype = event.get("type")
        if etype == "thought":
            console.print(f"[dim cyan]› {event.get('content')}[/dim cyan]")
        elif etype == "token":
            content = event.get("content", "")
            accumulated += content
            sys.stdout.write(content)
            sys.stdout.flush()
        elif etype == "plot":
            console.print(f"\n[bold green]📊 Plot saved:[/bold green] {event.get('url')}")
        elif etype == "done":
            if not accumulated:
                console.print(Markdown(event.get("response", "")))

    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="ControlAI Terminal Client")
    parser.add_argument("query", nargs="*", help="Optional one-shot query string")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-Instruct-2507-4bit", help="Model path")

    args = parser.parse_args()

    with console.status("[bold cyan]Loading ControlAI Engine on Apple Silicon (MLX)...[/bold cyan]"):
        agent = ControlAIAgent(model_path=args.model, adapter_path=args.adapter)

    if args.query:
        query_str = " ".join(args.query)
        run_oneshot(agent, query_str)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
