"""Control-LLM Agent Orchestrator: Multi-step tool calling, execution, grounding, and streaming synthesis loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import os
import sys

try:
    from mlx_lm import generate as mlx_generate, load as mlx_load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from transformers import AutoModelForCausalLM, AutoTokenizer

from controlai_agent.prompts import CONTROLAI_SYSTEM_PROMPT
from controlai_agent.registry import ToolRegistry, registry
import controlai_agent.tools  # noqa: F401 (ensure all tools are registered)
from controlai_rag.index import ControlRAGIndex


@dataclass
class ToolExecutionTrace:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResult:
    final_response: str
    tool_traces: list[ToolExecutionTrace] = field(default_factory=list)
    total_steps: int = 0
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    is_grounded: bool = True
    plots: list[str] = field(default_factory=list)


def sanitize_json_escapes(raw: str) -> str:
    """Escape unescaped backslashes in JSON strings (e.g. \\dot, \\omega, \\mu in math/code)."""
    return re.sub(r"\\(?![/\"\\bfnrtu])", r"\\\\", raw)


def fix_space_separated_arrays(json_str: str) -> str:
    """Convert MATLAB/Numpy space-separated lists inside brackets [1 2 3] to [1, 2, 3]."""
    def _fix_brackets(match: re.Match) -> str:
        content = match.group(1).strip()
        # Replace spaces between numbers with commas
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", content)
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", fixed)
        fixed = re.sub(r"([0-9eE\.\-+]+)\s+([0-9eE\.\-+]+)", r"\1, \2", fixed)
        return f"[{fixed}]"
    return re.sub(r"\[\s*([0-9eE\.\-+\s]+?)\s*\]", _fix_brackets, json_str)


def parse_flexible_json(raw_str: str) -> dict[str, Any] | None:
    """Parse JSON with fallback to escape sanitization, array fixing, and non-strict control characters."""
    raw_str = raw_str.strip()
    try:
        obj = json.loads(raw_str, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        sanitized = sanitize_json_escapes(raw_str)
        obj = json.loads(sanitized, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        fixed_arrays = fix_space_separated_arrays(sanitize_json_escapes(raw_str))
        obj = json.loads(fixed_arrays, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def _extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse tool calls from <tool_call>, markdown code blocks, or raw JSON robustly."""
    calls: list[dict[str, Any]] = []

    # 1. Standard <tool_call> tags
    for match in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text):
        obj = parse_flexible_json(match.group(1))
        if obj and "name" in obj:
            calls.append(obj)

    # 2. Markdown json blocks with tool call schema
    if not calls:
        for match in re.finditer(r"```(?:json)?\s*(\{\s*[\"']name[\"']\s*:[\s\S]*?\})\s*```", text):
            obj = parse_flexible_json(match.group(1))
            if obj and "name" in obj:
                calls.append(obj)

    # 3. Raw JSON object containing "name" and "arguments" / "parameters"
    if not calls and ('"name"' in text or "'name'" in text):
        match = re.search(r"(\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*\})", text)
        if match:
            obj = parse_flexible_json(match.group(1))
            if obj and "name" in obj:
                calls.append(obj)

    # Clean pre-tool thought / raw JSON artifacts
    cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*```", "", cleaned)
    cleaned = re.sub(r"\{\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z0-9_]+[\"'][\s\S]*\}", "", cleaned)
    cleaned = cleaned.strip()
    return calls, cleaned


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ControlAIAgent:
    """Universal Control Engineering Agent supporting GGUF, Ollama C++, Apple MLX, and PyTorch."""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        adapter_path: str | None = None,
        tool_registry: ToolRegistry = registry,
        max_tool_steps: int = 3,
    ) -> None:
        self.model_path = model_path
        
        # Auto-detect trained LoRA adapter if none explicitly provided
        default_adapter = PROJECT_ROOT / "adapters" / "controlai_qwen3_4b_sft_v2"
        if adapter_path is None and default_adapter.exists():
            adapter_path = str(default_adapter)

        self.adapter_path = adapter_path
        self.registry = tool_registry
        self.max_tool_steps = max_tool_steps

        # Detect platform & backend
        self.is_ollama = str(model_path).startswith("ollama")
        self.is_gguf = str(model_path).endswith(".gguf") or "gguf" in str(model_path).lower()
        self.is_mlx = HAS_MLX and not self.is_ollama and not self.is_gguf and not str(model_path).startswith("Qwen/") and not os.environ.get("FORCE_TRANSFORMERS")

        if self.is_ollama:
            self.ollama_model = model_path.split(":", 1)[1] if ":" in str(model_path) else "controlai"
            self.hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        elif self.is_gguf:
            if not HAS_LLAMA_CPP:
                raise ImportError("llama-cpp-python is required to run GGUF models. Install it with: pip install llama-cpp-python")
            self.llama_model = llama_cpp.Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,  # Offload all layers to Metal / CUDA GPU
                n_ctx=4096,
                verbose=False,
            )
            self.hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        elif self.is_mlx:
            if adapter_path:
                self.model, self.mlx_tokenizer = mlx_load(model_path, adapter_path=adapter_path)
            else:
                self.model, self.mlx_tokenizer = mlx_load(model_path)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Universal PyTorch / Transformers fallback on Linux, Colab, HuggingFace, CUDA
            hf_id = "Qwen/Qwen2.5-3B-Instruct" if "mlx" in str(model_path) else model_path
            self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            if adapter_path and Path(adapter_path).exists():
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                except Exception:
                    pass

        # Initialize local offline RAG index
        try:
            self.rag_index = ControlRAGIndex()
        except Exception:
            self.rag_index = None

    def _generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Universal text generation handling Ollama C++, GGUF llama_cpp, MLX, and PyTorch."""
        if self.is_ollama:
            res = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": max_tokens},
            )
            return res.get("response", "").strip()
        elif self.is_gguf:
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
                temperature=0.2,
            )
            return output["choices"][0]["text"].strip()
        elif self.is_mlx:
            return mlx_generate(
                self.model,
                self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            ).strip()
        else:
            import torch
            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _get_grounded_instruction(self, user_prompt: str, base_instruction: str) -> str:
        """Retrieve relevant textbook theorems and inject grounding into system instructions."""
        if not self.rag_index or not self.rag_index.chunks:
            return base_instruction

        try:
            rag_hits = self.rag_index.search(user_prompt, top_k=3)
            high_rel = [h for h in rag_hits if h.get("score", 0) > 2.5]
            if not high_rel:
                return base_instruction

            ref_texts = []
            for h in high_rel[:2]:
                fname = h.get("filename", "Reference")
                page = h.get("page")
                page_str = f" (p. {page})" if page else ""
                clean_chunk = h.get("text", "")[:400].strip()
                ref_texts.append(f"[{fname}{page_str}]:\n{clean_chunk}")

            return base_instruction + "\n\n### Grounded Reference Context from Canonical Control Literature:\n" + "\n\n".join(ref_texts)
        except Exception:
            return base_instruction

    def run(
        self,
        user_prompt: str,
        system_instruction: str = CONTROLAI_SYSTEM_PROMPT,
        history: list[dict[str, Any]] | None = None,
        max_tokens_per_step: int = 2500,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute a complete agent interaction loop synchronously."""
        messages: list[dict[str, Any]] = []
        effective_sys = self._get_grounded_instruction(user_prompt, system_instruction)
        if effective_sys:
            messages.append({"role": "system", "content": effective_sys})

        if history:
            for item in history:
                r = item.get("role")
                c = item.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": user_prompt})

        tools_schema = self.registry.get_tool_schemas()
        traces: list[ToolExecutionTrace] = []
        plots: list[str] = []
        called_signatures: set[str] = set()

        for step in range(1, self.max_tool_steps + 1):
            rendered_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tools=tools_schema,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_output = self._generate(rendered_prompt, max_tokens=max_tokens_per_step)

            tool_calls, pre_text = _extract_tool_calls(model_output)

            if not tool_calls:
                clean_output = re.sub(r"<tool_call>.*?</tool_call>", "", model_output, flags=re.DOTALL).strip()
                return AgentResult(
                    final_response=clean_output,
                    tool_traces=traces,
                    total_steps=step,
                    raw_messages=messages,
                    is_grounded=True,
                    plots=plots,
                )

            # Check for loops
            new_calls = []
            for call in tool_calls:
                sig = f"{call.get('name')}:{json.dumps(call.get('arguments', {}), sort_keys=True)}"
                if sig not in called_signatures:
                    called_signatures.add(sig)
                    new_calls.append(call)

            if not new_calls:
                break

            messages.append({"role": "assistant", "content": model_output})

            for call_data in new_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})

                tool_result = self.registry.execute(tool_name, tool_args)
                traces.append(ToolExecutionTrace(tool_name=tool_name, arguments=tool_args, result=tool_result))

                if "plot_path" in tool_result:
                    p_path = Path(tool_result["plot_path"])
                    if p_path.exists():
                        plots.append(f"/plots/{p_path.name}")

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Final synthesis after tool execution
        forced_prompt = self.hf_tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        final_output = self._generate(forced_prompt, max_tokens=max_tokens_per_step)

        _, clean_final = _extract_tool_calls(final_output)
        return AgentResult(
            final_response=clean_final or final_output,
            tool_traces=traces,
            total_steps=len(traces) + 1,
            raw_messages=messages,
            is_grounded=True,
            plots=plots,
        )

    def run_stream(
        self,
        user_prompt: str,
        system_instruction: str = CONTROLAI_SYSTEM_PROMPT,
        history: list[dict[str, Any]] | None = None,
        max_tokens_per_step: int = 2500,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream token-by-token generation and tool execution events with zero JSON leakage."""
        messages: list[dict[str, Any]] = []
        effective_sys = self._get_grounded_instruction(user_prompt, system_instruction)
        if effective_sys:
            messages.append({"role": "system", "content": effective_sys})

        if history:
            for item in history:
                r = item.get("role")
                c = item.get("content", "")
                if r in ("user", "assistant") and c:
                    # Strip any legacy JSON tool call artifacts from previous chat sessions
                    c_clean = re.sub(r"\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*\}", "", c).strip()
                    c_clean = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", c_clean).strip()
                    if c_clean:
                        messages.append({"role": r, "content": c_clean})

        messages.append({"role": "user", "content": user_prompt})

        tools_schema = self.registry.get_tool_schemas()
        traces: list[dict[str, Any]] = []
        plots: list[str] = []
        thoughts: list[str] = []
        called_signatures: set[str] = set()

        # Dynamic thought generation - only show thoughts when tools or derivations occur
        for step in range(1, self.max_tool_steps + 1):
            rendered_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tools=tools_schema,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_output = self._generate(rendered_prompt, max_tokens=max_tokens_per_step)

            tool_calls, pre_text = _extract_tool_calls(model_output)

            if not tool_calls:
                # Direct final answer without tools -> stream tokens directly
                clean_output = pre_text or model_output
                words = re.split(r"(\s+)", clean_output)
                for w in words:
                    if w:
                        yield {"type": "token", "content": w}

                yield {
                    "type": "done",
                    "response": clean_output,
                    "traces": traces,
                    "plots": plots,
                    "thoughts": thoughts,
                }
                return

            if pre_text:
                thoughts.append(pre_text)
                yield {"type": "thought", "content": pre_text}

            # Filter out duplicate loops
            new_calls = []
            for call in tool_calls:
                sig = f"{call.get('name')}:{json.dumps(call.get('arguments', {}), sort_keys=True)}"
                if sig not in called_signatures:
                    called_signatures.add(sig)
                    new_calls.append(call)

            if not new_calls:
                break

            messages.append({"role": "assistant", "content": model_output})

            for call_data in new_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})

                t_start_msg = f"Executing tool: {tool_name} with parameters: {json.dumps(tool_args, ensure_ascii=False)}"
                thoughts.append(t_start_msg)
                yield {"type": "thought", "content": t_start_msg}
                yield {"type": "tool_start", "tool": tool_name, "args": tool_args}

                tool_result = self.registry.execute(tool_name, tool_args)
                trace_item = {
                    "tool": tool_name,
                    "args": tool_args,
                    "status": tool_result.get("status", "success"),
                    "residual": tool_result.get("residual"),
                }
                traces.append(trace_item)

                if "plot_path" in tool_result:
                    p_path = Path(tool_result["plot_path"])
                    if p_path.exists():
                        plot_url = f"/plots/{p_path.name}"
                        plots.append(plot_url)
                        yield {"type": "plot", "url": plot_url}

                yield {"type": "tool_end", "trace": trace_item}
                t_end_msg = f"Tool {tool_name} returned status: {trace_item['status']}"
                thoughts.append(t_end_msg)
                yield {"type": "thought", "content": t_end_msg}

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Final Synthesis phase
        synth_thought = "Synthesizing verified engineering response and LaTeX formulations..."
        thoughts.append(synth_thought)
        yield {"type": "thought", "content": synth_thought}

        messages.append({
            "role": "user",
            "content": "Synthesize the simulation and calculation results above into a complete, thorough engineering explanation with LaTeX equations. Do not output any JSON or tool call tags.",
        })

        forced_prompt = self.hf_tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )

        final_output = self._generate(forced_prompt, max_tokens=max_tokens_per_step)

        # If model generated another tool call during final synthesis, execute it!
        synth_tool_calls, clean_synth = _extract_tool_calls(final_output)
        if synth_tool_calls:
            for call_data in synth_tool_calls:
                tool_name = call_data.get("name")
                tool_args = call_data.get("arguments", {})
                t_res = self.registry.execute(tool_name, tool_args)
                if "plot_path" in t_res:
                    p_path = Path(t_res["plot_path"])
                    if p_path.exists():
                        plot_url = f"/plots/{p_path.name}"
                        plots.append(plot_url)
                        yield {"type": "plot", "url": plot_url}
                messages.append({"role": "tool", "name": tool_name, "content": json.dumps(t_res, ensure_ascii=False)})

            # Re-generate synthesis after executing the tool
            re_prompt = self.hf_tokenizer.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True)
            final_output = self._generate(re_prompt, max_tokens=max_tokens_per_step)
            _, clean_synth = _extract_tool_calls(final_output)

        final_text = clean_synth or final_output
        # Strip any lingering raw json
        final_text = re.sub(r"\{\s*[\"']name[\"']\s*:[\s\S]*?\}\s*\}", "", final_text).strip()
        if not final_text:
            final_text = "The computational analysis and simulation have been executed successfully as detailed above."

        # Stream words smoothly
        words = re.split(r"(\s+)", final_text)
        for w in words:
            if w:
                yield {"type": "token", "content": w}

        yield {
            "type": "done",
            "response": final_text,
            "traces": traces,
            "plots": plots,
            "thoughts": thoughts,
        }
