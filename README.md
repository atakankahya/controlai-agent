# ControlAI: Open-Source Safety-Critical AI Agent for Control Systems Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atakankahya/controlai-agent/blob/main/ControlAI_Colab_Demo.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/atakankahya/controlai-agent?style=social)](https://github.com/atakankahya/controlai-agent)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-ControlAI--Agent-blue)](https://huggingface.co/atakankahya/ControlAI-Agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**ControlAI** is an open-source, domain-specific AI agent engineered specifically for **Control Systems Engineering, Dynamical Systems, Robotics, and Applied Mathematics**.

---

## Motivation: Safety-Critical Verification in Control Engineering

Standard large language models (LLMs) operate probabilistically without deterministic verification. When applied to physical systems—such as autonomous aerial vehicles, industrial manipulators, or power systems—general-purpose models present severe reliability challenges:
* **Numerical Hallucinations:** Estimating eigenvalues without characteristic polynomial evaluation, inverting singular matrices, or generating unstable feedback gains.
* **Lack of Formal Proof Structure:** Omitting boundary conditions, PBH rank tests, or domain-specific stability limits.
* **Physical Safety Violations:** A sign error in a state feedback gain leads directly to closed-loop instability in hardware.

**ControlAI addresses these limitations through a hybrid architecture:**
1. **Deterministic Scientific Sandbox:** Computes continuous/discrete algebraic Riccati equations (CARE/DARE), matrix exponentials, and Bode diagrams using LAPACK, SciPy, and CVXPY.
2. **4-Stage Mathematical Proof Standard:** Formulates system class, analytical theorems, closed-form derivations, and engineering breakdown limits.
3. **Dynamic Simulation & Plotting:** Solves nonlinear differential equations and renders verified trajectories.
4. **Offline RAG Knowledge Engine:** Grounded with 68,000+ chunks indexed across classical and modern control engineering literature.

---

## Live Demo and Access

* **Interactive Google Colab Demo:**  
  Run ControlAI in a cloud environment with zero local setup:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atakankahya/controlai-agent/blob/main/ControlAI_Colab_Demo.ipynb)

* **Hugging Face Model Repository:**  
  Access model weights and fine-tuning configurations: [atakankahya/ControlAI-Agent](https://huggingface.co/atakankahya/ControlAI-Agent)

---

## Architecture Overview

```mermaid
graph TD
    User([Engineering Query]) --> WebApp[Web Console & CLI]
    WebApp --> Orchestrator[ControlAI Agent Orchestrator]
    
    subgraph Core Engine [Hybrid Verification Engine]
        Brain[Foundation Model - Qwen3-4B]
        LoRA[Theory & Tool Adapter - SFT]
        RAG[Offline Knowledge Engine - 68,000+ Chunks]
        Tools[Deterministic Numerical Tool Suite]
    end
    
    Orchestrator --> Brain
    Orchestrator --> LoRA
    Orchestrator --> RAG
    Orchestrator --> Tools
    
    Tools --> SciPy[SciPy / LAPACK / BLAS Matrix Solvers]
    Tools --> PythonControl[Python Control ct.tf / step_response]
    Tools --> PyExecutor[Python Simulation Sandbox]
    Tools --> Verifier[Numerical Residual Verifier]
    
    Verifier --> FinalResponse[Verified Technical Synthesis & Plots]
    FinalResponse --> WebApp
```

---

## Key Capabilities

### 1. Web Console (`web/`)
* **Real-Time Token Streaming (SSE):** Word-level fluid rendering with smart scroll retention.
* **LaTeX Formula Rendering:** KaTeX integration with math delimiter protection and code block syntax highlighting.
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
* Automatically isolates and captures generated simulation plots to `outputs/plots/`.

---

## Benchmark Results (ControlBench-v1)

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

## Local Installation and Usage

### 1. Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/atakankahya/controlai-agent.git
cd controlai-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-corpus.txt
pip install -r requirements-training.txt
pip install rich
```

### 2. Launch Universal Assistant (Web & Browser Auto-Open)

```bash
./run.sh
```
* Starts the backend server and opens the browser interface at `http://127.0.0.1:8000`.

### 3. Interactive Terminal CLI

```bash
# Launch interactive terminal session
./run.sh --cli

# Or execute a single query directly
python cli.py "Design an LQR controller for A=[[0, 1], [-2, -3]], B=[[0], [1]], Q=diag([10, 1]), R=1"
```

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
