# ControlAI: Open-Source Safety-Critical AI Agent for Control Systems Engineering

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/atakankahya/ControlAI-Agent)
[![GitHub stars](https://img.shields.io/github/stars/atakankahya/controlai-agent?style=social)](https://github.com/atakankahya/controlai-agent)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-ControlAI--Agent-blue)](https://huggingface.co/atakankahya/ControlAI-Agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**ControlAI** is an open-source, domain-specific AI assistant engineered specifically for **Control Systems Engineering, Dynamical Systems, Robotics, and Applied Mathematics**.

---

## Quickstart

ControlAI runs entirely on your own machine. Requires **Apple Silicon** (inference is MLX) with
at least 16 GB of unified memory; the default model needs about 8 GB on disk and is downloaded on
first run.

```bash
git clone https://github.com/atakankahya/controlai-agent.git
cd controlai-agent
pip install -r requirements.txt

./run.sh                  # web console, opens http://127.0.0.1:8000
./run.sh --cli            # interactive terminal chat
./run.sh --cli "design an LQR for A=[[0,1],[-2,-3]], B=[[0],[1]], Q=eye(2), R=1"
./run.sh --build-index    # build the dense retrieval index from data/user_docs/ (~45 min)
```

| Variable | Default | Purpose |
| :-- | :-- | :-- |
| `CONTROLAI_MODEL` | `mlx-community/Qwen3-14B-4bit` | any MLX model id or local path |
| `CONTROLAI_ADAPTER` | *(none)* | optional LoRA adapter |
| `CONTROLAI_THINKING` | `auto` | `off`, `auto` (conceptual questions only), or `on` |
| `CONTROLAI_THINK_BUDGET` | `512` | ceiling on tokens spent reasoning |

---

## Motivation: Safety-Critical Verification in Control Engineering

Standard large language models (LLMs) operate probabilistically without deterministic verification. When applied to physical systems—such as autonomous aerial vehicles, industrial manipulators, or power grids—general-purpose models present severe reliability challenges:
* **Numerical Hallucinations:** Estimating eigenvalues without characteristic polynomial evaluation, inverting singular matrices, or generating unstable feedback gains.
* **Lack of Formal Proof Structure:** Omitting boundary conditions, PBH rank tests, or domain-specific stability limits.
* **Physical Safety Violations:** A sign error in a state feedback gain leads directly to closed-loop instability in hardware.

**ControlAI addresses these limitations through a hybrid architecture:**
1. **Deterministic Scientific Sandbox:** Computes continuous/discrete algebraic Riccati equations (CARE/DARE), matrix exponentials, and Bode diagrams using LAPACK, SciPy, and CVXPY.
2. **4-Stage Mathematical Proof Standard:** Formulates system class, analytical theorems, closed-form derivations, and engineering breakdown limits.
3. **Dynamic Simulation & Plotting:** Solves nonlinear differential equations and renders verified trajectories directly in the interface.
4. **Offline Retrieval:** 80,000+ chunks of classical and modern control literature, indexed locally.
   The prebuilt index is not distributed -- it carries the full text of copyrighted textbooks --
   so `--build-index` builds one from whatever you put in `data/user_docs/`.
   Lexical and dense rankings are fused and gated on cosine similarity, so an unrelated passage is
   dropped rather than cited — the retriever returning nothing is a normal outcome.
5. **Bounded latency:** the fixed system-prompt and tool-schema prefix is prefilled once at startup
   and reused across turns, so answers begin streaming in well under a second.

---

## Architecture Overview

```mermaid
graph TD
    User([Engineering Query]) --> Frontend[Web Console / CLI]
    Frontend --> Agent[ControlAgent streaming tool loop]

    subgraph Local[Runs entirely on-device]
        Engine[LocalEngine - MLX, cached KV prefix]
        Model[Qwen3-14B-4bit]
        Retriever[Hybrid Retriever - BM25 + dense]
        Tools[Deterministic Tool Registry]
    end

    Agent --> Engine --> Model
    Agent --> Retriever
    Agent --> Tools

    Tools --> SciPy[SciPy / LAPACK / BLAS]
    Tools --> PythonControl[python-control]
    Tools --> PyExecutor[Sandboxed Python]
    Tools --> Verifier[Residual Verifier]

    Verifier --> Answer[Verified answer, plots, citations]
    Answer --> Frontend
```

---

## Key Capabilities

### 1. Modern Interactive Web Console (`web/`)
* **Real-Time Token Streaming (SSE):** Token-by-token fluid rendering with smart scroll retention.
* **LaTeX Formula Rendering:** KaTeX integration with math delimiter protection and syntax-highlighted code blocks.
* **Engineering Toolbar:** One-click insertion of state matrices, transfer functions, and control parameters ($\zeta, \omega_n$).

### 2. Deterministic Mathematical Tool Suite (`controlai_agent/tools/`)
* **State-Space Analysis:** Exact ZOH discretization, eigenvalue decomposition, controllability and observability Gramians, and Lyapunov equation solvers.
* **Frequency Domain:** Exact Bode magnitude and phase calculation, gain/phase margins, and Routh-Hurwitz stability criterion.
* **Controller Synthesis:** Continuous LQR (CARE), Discrete LQR (DARE), pole placement, PID tuning, and constrained Model Predictive Control (CVXPY QP).
* **State Estimation:** Kalman filter prediction and Joseph-stabilized measurement updates.
* **Nonlinear & Safety Filters:** Control Barrier Functions (CBF-QP safety filter), Nonlinear Dynamic Inversion (NDI), and Feedback Linearization.
* **Robust Control:** Kharitonov stability test, Small Gain theorem, and multiplicative uncertainty bounds.

### 3. Live Python Execution Sandbox (`python_executor.py`)
* Executes scientific Python routines using `numpy`, `scipy.signal`, `scipy.linalg`, `control`, and `matplotlib`.
* Automatically isolates and captures generated simulation plots to `outputs/plots/` and renders them in chat.

---

## Benchmark Results (ControlBench-v1)

> **These figures were measured against the previous architecture** — the fine-tuned Qwen3-4B LoRA
> served through the old orchestrator. That serving path has been replaced (base Qwen3-14B, no
> adapter, rewritten agent loop and retriever), and the suite has not yet been re-run against it, so
> treat the table as historical rather than as a description of the current build.

Evaluated across **50 multi-pillar benchmark problems**:

| Benchmark Pillar | Qwen3-4B Base (Text-Only) | ControlAI Agent (Ours) | Verification Mechanism |
| :--- | :---: | :---: | :--- |
| **Numerical Synthesis** *(CARE, DARE, ZOH, Kalman)* | 0.0% | **100.0%** | Deterministic SciPy/LAPACK solver with $10^{-10}$ residual verification. |
| **Theory & Proofs** *(PBH, Doyle 1978, Waterbed)* | 83.3% | **86.7% - 95.0%** | 4-Stage CoT Proof Standard with literature grounding. |
| **Code & Simulation** *(SciPy ODE, Matplotlib)* | 29.0% | **90.0%+** | Dynamic execution sandbox with automated plot capture. |
| **Safety & Traps** *(Uncontrollable / Ill-conditioned modes)* | 60.0% | **80.0%** | Pre-computation PBH rank verification. |
| **Real-World Case Studies** *(Aerospace, Drone, Automotive)* | 80.0% | **80.0%** | CBF-QP safety filtering and nonlinear inversion. |
| **Overall Score** | 47.3% | **87.3%** | Deterministic tool execution and mathematical grounding. |

---

## Open-Source and Community Support

ControlAI is an **open-source project** developed for control engineering researchers, robotics practitioners, and applied mathematicians.

### Contributing:
Contributions from the community are welcome across several areas:
* **Analytical Derivations:** Expanding formal proof templates for multi-input multi-output (MIMO) systems.
* **Nonlinear Control:** Implementing additional Control Lyapunov Functions (CLFs) and adaptive controllers.
* **Simulation Bridges:** Connecting agent tool calls with external simulation platforms (ROS 2, MATLAB/Simulink, Gazebo).
* **Benchmark Expansion:** Adding challenging test problems to `benchmarks/controlbench_v1.jsonl`.

---

## License

This project is licensed under the **[MIT License](LICENSE)**.
