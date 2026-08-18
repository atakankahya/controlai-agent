"""Generate Dense Graduate-Level Control Theory SFT Dataset (Proofs, Theorems, Derivations)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("data/training/theory_sft_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THEORY_DOMAINS = [
    # 1. Frequency Domain & Limitations
    {
        "topic": "Phase Margin vs Damping Ratio Derivation",
        "templates": [
            "Derive the exact theoretical relationship between Phase Margin (PM) and damping ratio (zeta) for a standard second-order system. Show where the linear approximation holds and explain its mathematical breakdown.",
            "Explain the derivation of the phase margin formula PM = arctan(2*zeta / sqrt(sqrt(1+4*zeta^4) - 2*zeta^2)). Why does zeta approx PM / 100 fail for large phase margins?",
            "What is the analytical connection between open-loop frequency response phase margin and closed-loop time domain damping ratio? Provide the full derivation.",
        ],
        "response": (
            "### 1. Mathematical Context & Open-Loop Transfer Function\n"
            "Consider a canonical second-order open-loop system with unity negative feedback:\n"
            "$$\n"
            "L(s) = \\frac{\\omega_n^2}{s(s + 2\\zeta\\omega_n)}\n"
            "$$\n"
            "The gain crossover frequency $\\omega_c$ is defined by the condition $|L(j\\omega_c)| = 1$:\n"
            "$$\n"
            "|L(j\\omega_c)| = \\frac{\\omega_n^2}{\\omega_c \\sqrt{\\omega_c^2 + 4\\zeta^2\\omega_n^2}} = 1\n"
            "$$\n"
            "Squaring both sides and setting $u = (\\omega_c / \\omega_n)^2$ yields the quadratic equation $u(u + 4\\zeta^2) = 1$, which gives the unique positive root:\n"
            "$$\n"
            "\\omega_c = \\omega_n \\sqrt{\\sqrt{1 + 4\\zeta^4} - 2\\zeta^2}\n"
            "$$\n\n"
            "### 2. Exact Phase Margin Derivation\n"
            "The open-loop phase at crossover is $\\angle L(j\\omega_c) = -90^\\circ - \\arctan\\left(\\frac{\\omega_c}{2\\zeta\\omega_n}\\right)$. By definition, the Phase Margin (PM) is $\\text{PM} = 180^\\circ + \\angle L(j\\omega_c) = 90^\\circ - \\arctan\\left(\\frac{\\omega_c}{2\\zeta\\omega_n}\\right) = \\arctan\\left(\\frac{2\\zeta\\omega_n}{\\omega_c}\\right)$. Substituting $\\omega_c$ yields the exact closed-form expression:\n"
            "$$\n"
            "\\text{PM} = \\arctan\\left(\\frac{2\\zeta}{\\sqrt{\\sqrt{1 + 4\\zeta^4} - 2\\zeta^2}}\\right)\n"
            "$$\n\n"
            "### 3. Linearized Rule-of-Thumb Approximation\n"
            "Performing a Maclaurin series expansion of $\\text{PM}(\\zeta)$ around $\\zeta = 0$ in radians:\n"
            "$$\n"
            "\\text{PM}(\\zeta) \\approx 2\\zeta \\text{ rad} = 2\\zeta \\left(\\frac{180^\\circ}{\\pi}\\right) \\approx 114.6^\\circ \\cdot \\zeta \\approx 100^\\circ \\cdot \\zeta\n"
            "$$\n"
            "This gives the classical engineering rule of thumb:\n"
            "$$\n"
            "\\zeta \\approx \\frac{\\text{PM}^\\circ}{100}\n"
            "$$\n\n"
            "### 4. Engineering Breakdown Conditions\n"
            "- **Range of Validity:** The approximation $\\zeta \\approx \\text{PM}/100$ is accurate only for $0 \\le \\zeta \\le 0.6$ ($0^\\circ \\le \\text{PM} \\le 60^\\circ$).\n"
            "- **High Phase Margins ($\\text{PM} > 70^\\circ$):** The exact formula approaches $90^\\circ$ asymptotically as $\\zeta \\to \\infty$, whereas the linear rule predicts unphysical $\\text{PM} > 100^\\circ$.\n"
            "- **Non-Dominant Dynamics:** In high-order plants with unmodeled high-frequency poles/zeros or transport delay $e^{-sT_d}$, phase lag accelerates rapidly near $\\omega_c$, invalidating the second-order phase-damping correspondence."
        )
    },
    # 2. State-Space Controllability: PBH vs Kalman
    {
        "topic": "PBH Rank Test vs Kalman Controllability Matrix",
        "templates": [
            "Compare the Kalman Controllability Matrix rank condition with the Popov-Belevitch-Hautus (PBH) rank test. When is PBH preferred in engineering practice, and what modal insights does it provide?",
            "Explain why the Kalman rank condition C = [B, AB, ..., A^(n-1)B] is numerically ill-conditioned for high-order systems, and how the PBH test resolves this.",
            "What is the mathematical formulation of the PBH eigenvector and rank test for stabilizability and controllability?",
        ],
        "response": (
            "### 1. Mathematical Formulations\n"
            "Consider a continuous-time linear time-invariant (LTI) system $\\dot{x} = Ax + Bu$ with $x \\in \\mathbb{R}^n, u \\in \\mathbb{R}^m$.\n"
            "- **Kalman Controllability Matrix:**\n"
            "$$\n"
            "\\mathcal{C} = \\begin{bmatrix} B & AB & A^2 B & \\dots & A^{n-1}B \\end{bmatrix} \\in \\mathbb{R}^{n \\times nm}\n"
            "$$\n"
            "The system is controllable if and only if $\\operatorname{rank}(\\mathcal{C}) = n$.\n\n"
            "- **Popov-Belevitch-Hautus (PBH) Rank Test:**\n"
            "$$\n"
            "\\operatorname{rank}\\begin{bmatrix} sI - A & B \\end{bmatrix} = n, \\quad \\forall s \\in \\mathbb{C}\n"
            "$$\n"
            "In practice, this rank condition only needs to be verified at the eigenvalues $s = \\lambda_i \\in \\operatorname{spec}(A)$.\n\n"
            "### 2. Numerical Ill-Conditioning of the Kalman Matrix\n"
            "Computing $\\mathcal{C}$ involves matrix powers $A^k$. For systems with $n \\ge 5$ or widely separated eigenvalues, the columns of $A^k B$ align with the dominant eigenvector of $A$, causing $\\mathcal{C}$ to become severely ill-conditioned (condition number $\\kappa(\\mathcal{C}) \\to \\infty$). Numerical rank determination via SVD suffers catastrophic floating-point cancellation, falsely declaring controllable systems uncontrollable.\n\n"
            "### 3. Modal Insights Provided by PBH\n"
            "- **Uncontrollable Mode Identification:** If $\\operatorname{rank}[\\lambda_i I - A, B] < n$, the eigenvalue $\\lambda_i$ is directly identified as the uncontrollable mode. The left eigenvector $v_i^T A = \\lambda_i v_i^T$ satisfies $v_i^T B = 0$.\n"
            "- **Stabilizability Verification:** If all uncontrollable modes satisfy $\\operatorname{Re}(\\lambda_i) < 0$, the system is immediately proven to be stabilizable by state feedback, even if full controllability fails."
        )
    },
    # 3. Right-Half Plane Zeros & Waterbed Effect
    {
        "topic": "Right-Half Plane Zeros and Bode Sensitivity Integral",
        "templates": [
            "Why do Right-Half-Plane (RHP) zeros impose fundamental performance limitations in feedback control? Explain using the Bode sensitivity integral (waterbed effect).",
            "Derive why non-minimum phase zeros cause initial undershoot and upper-bound closed-loop bandwidth. Reference Poisson integral formulas.",
            "Explain the waterbed effect in feedback control systems and its relationship with Poisson's integral for open-loop unstable poles and non-minimum phase zeros.",
        ],
        "response": (
            "### 1. Definition of Non-Minimum Phase Zeros\n"
            "A zero $z \\in \\mathbb{C}$ with $\\operatorname{Re}(z) > 0$ is a Right-Half-Plane (RHP) zero. RHP zeros cannot be inverted by feedback without introducing unstable pole-zero cancellations in the open-loop gain.\n\n"
            "### 2. Time-Domain Effect: Initial Undershoot\n"
            "By the initial value theorem and Laplace integration, for a system with an RHP zero at $s = z > 0$, the step response $y(t)$ satisfies:\n"
            "$$\n"
            "\\int_0^\\infty e^{-zt} y(t) dt = \\frac{T(z)}{z} = 0\n"
            "$$\n"
            "Because $e^{-zt} > 0$ for all $t \\ge 0$, the output $y(t)$ must change sign at least once, causing an unavoidable **initial undershoot** (response starts in the wrong direction).\n\n"
            "### 3. Frequency-Domain Limitation: Bode Sensitivity Integral\n"
            "For an open-loop stable system with sensitivity function $S(s) = (I + L(s))^{-1}$ having relative degree $\\ge 2$, the Bode Sensitivity Integral states:\n"
            "$$\n"
            "\\int_0^\\infty \\ln |S(j\\omega)| d\\omega = \\pi \\sum_{i} \\operatorname{Re}(p_i)\n"
            "$$\n"
            "where $p_i$ are open-loop RHP poles. This is known as the **Waterbed Effect**: reducing sensitivity ($|S(j\\omega)| < 1$) in the low-frequency control bandwidth inevitably pushes $|S(j\\omega)| > 1$ into higher frequencies, amplifying noise and reducing stability margins.\n\n"
            "### 4. Poisson Integral Bandwidth Bounds\n"
            "By Poisson's integral formula over the right half-plane, an RHP zero at $z = \\sigma_z + j\\omega_z$ forces:\n"
            "$$\n"
            "\\omega_b \\le \\frac{z}{2}\n"
            "$$\n"
            "Closed-loop bandwidth is strictly bounded by approximately half the frequency of the slowest RHP zero."
        )
    },
    # 4. Separation Principle & Doyle 1978 Robustness Fragility
    {
        "topic": "Separation Principle and Doyle 1978 Counterexample",
        "templates": [
            "State the Separation Principle for Linear-Quadratic-Gaussian (LQG) control. Under what conditions does it hold, and why does LQG lack guaranteed robustness margins?",
            "Explain Doyle's 1978 paper 'Guaranteed Margins for LQG Regulators'. How does Loop Transfer Recovery (LTR) restore robustness?",
            "Why does deterministic LQR have guaranteed 60 deg phase margin while LQG can have arbitrarily small gain margin?",
        ],
        "response": (
            "### 1. Statement of the Separation Principle\n"
            "For linear systems subject to additive Gaussian white noise:\n"
            "$$\n"
            "\\dot{x} = Ax + Bu + w, \\quad y = Cx + v\n"
            "$$\n"
            "the optimal stochastic controller decomposes into two independent sub-problems:\n"
            "1. **Deterministic LQR State Feedback:** $u(t) = -K \\hat{x}(t)$, where $K = R^{-1} B^T P$ solves the Continuous Algebraic Riccati Equation (CARE).\n"
            "2. **Optimal Kalman State Observer:** $\\dot{\\hat{x}} = A\\hat{x} + Bu + L(y - C\\hat{x})$, where $L = Q_e C^T V^{-1}$ solves the Filter CARE.\n"
            "The closed-loop eigenvalues are precisely the union $\\operatorname{spec}(A - BK) \\cup \\operatorname{spec}(A - LC)$.\n\n"
            "### 2. The Robustness Fragility: Doyle's 1978 Counterexample\n"
            "While deterministic full-state LQR guarantees **gain margin $\\text{GM} \\in [1/2, \\infty)$** and **phase margin $\\text{PM} \\ge 60^\\circ$** via the Kalman return difference inequality:\n"
            "$$\n"
            "\\left\\| I + R^{1/2} K (j\\omega I - A)^{-1} B R^{-1/2} \\right\\| \\ge 1\n"
            "$$\n"
            "John Doyle proved in 1978 that **LQG regulators have no guaranteed stability margins**. Introducing the observer dynamic destroys the return difference inequality at the plant input. In the presence of infinitesimal plant parameter mismatches or high-frequency unmodeled dynamics, the closed-loop system can become unstable.\n\n"
            "### 3. Loop Transfer Recovery (LTR)\n"
            "To restore full LQR robustness at the plant input, Doyle and Stein developed **LQG/LTR**:\n"
            "By artificially injecting high fictitious process noise $W = W_0 + q^2 B B^T$ into the Kalman filter design as $q \\to \\infty$, the observer gain $L(q)$ asymptotically recovers the target LQR loop transfer function:\n"
            "$$\n"
            "\\lim_{q \\to \\infty} C(sI - A + BK + LC)^{-1} L = K(sI - A)^{-1} B\n"
            "$$\n"
            "provided the plant is minimum-phase (no RHP zeros)."
        )
    },
    # 5. Lyapunov Direct vs Indirect Methods
    {
        "topic": "Lyapunov Direct vs Indirect Stability Methods",
        "templates": [
            "Explain the fundamental difference between Lyapunov's Indirect Method (linearization) and Lyapunov's Direct Method. What are the limitations regarding domain of attraction and critical eigenvalues?",
            "When does Jacobian linearization fail to determine nonlinear stability? Contrast with LaSalle's Invariance Principle.",
            "Compare local asymptotic stability via eigenvalues with global asymptotic stability via positive definite Lyapunov functions.",
        ],
        "response": (
            "### 1. Lyapunov's Indirect Method (Linearization)\n"
            "For an autonomous nonlinear system $\\dot{x} = f(x)$ with equilibrium $f(0) = 0$, the system is expanded via Taylor series about the origin:\n"
            "$$\n"
            "\\dot{x} = A x + \\text{h.o.t.}, \\quad A = \\left. \\frac{\\partial f}{\\partial x} \\right|_{x=0}\n"
            "$$\n"
            "- **Asymptotic Stability:** If $\\operatorname{Re}(\\lambda_i(A)) < 0$ for all eigenvalues, the origin is locally asymptotically stable.\n"
            "- **Instability:** If any $\\operatorname{Re}(\\lambda_i(A)) > 0$, the origin is unstable.\n"
            "- **Critical Inconclusive Case:** If $\\operatorname{Re}(\\lambda_i(A)) \\le 0$ with at least one eigenvalue on the imaginary axis ($\\operatorname{Re}(\\lambda) = 0$), linearization is **strictly inconclusive**; stability is governed entirely by higher-order terms.\n\n"
            "### 2. Limitations of the Indirect Method\n"
            "1. **No Domain of Attraction:** Linearization proves stability only in an infinitesimal open neighborhood $B_\\epsilon(0)$. It provides zero information about large-signal stability.\n"
            "2. **Linear Systems Only:** It cannot handle non-differentiable non-smooth nonlinearities (e.g. saturation, deadzone, Coulomb friction).\n\n"
            "### 3. Lyapunov's Direct Method & LaSalle's Invariance\n"
            "Constructs a continuously differentiable positive definite energy function $V(x) > 0$ for $x \\neq 0$ and $V(0) = 0$:\n"
            "- **Asymptotic Stability:** $\\dot{V}(x) = \\nabla V(x) \\cdot f(x) < 0$ for all $x \\neq 0$.\n"
            "- **Global Asymptotic Stability (GAS):** Requires $V(x)$ to be radially unbounded ($V(x) \\to \\infty$ as $\|x\| \\to \\infty$).\n"
            "- **LaSalle's Invariance Principle:** If $\\dot{V}(x) \\le 0$ (negative semi-definite), trajectories converge to the largest invariant set $M$ contained in $\\{x \\mid \\dot{V}(x) = 0\\}$, enabling rigorous proofs of asymptotic stability even when $\\dot{V} = 0$ along certain manifolds."
        )
    },
    # 6. Small Gain Theorem vs Passivity
    {
        "topic": "Small Gain Theorem vs Passivity in Robust Control",
        "templates": [
            "Contrast the Small Gain Theorem with the Passivity Theorem in nonlinear and robust feedback control. When is passivity preferred over small gain?",
            "Explain why Passivity allows infinite gain margin while Small Gain is conservative at high loop gains.",
            "Compare L2-gain norm conditions with positive realness in closed-loop stability analysis.",
        ],
        "response": (
            "### 1. Small Gain Theorem ($\mathcal{L}_2$-Gain Based)\n"
            "Consider a feedback interconnection of two stable operators $H_1: \\mathcal{L}_2 \\to \\mathcal{L}_2$ and $H_2: \\mathcal{L}_2 \\to \\mathcal{L}_2$.\n"
            "The closed-loop system is $\\mathcal{L}_2$-gain stable if the product of their induced norms is strictly less than unity:\n"
            "$$\n"
            "\\gamma(H_1) \\cdot \\gamma(H_2) < 1, \\quad \\text{where } \\gamma(H) = \\sup_{u \\neq 0} \\frac{\\|Hu\\|_2}{\\|u\\|_2}\n"
            "$$\n"
            "- **Limitation:** Small gain is magnitude-based and ignores phase entirely. When high loop gain is required at low frequencies for tracking/disturbance rejection, small gain becomes excessively conservative.\n\n"
            "### 2. Passivity Theorem (Energy / Phase Based)\n"
            "An operator $H$ is passive if for all input signals $u$ and truncation times $T$:\n"
            "$$\n"
            "\\langle u, Hu \\rangle_T = \\int_0^T u^T(t) y(t) dt \\ge 0\n"
            "$$\n"
            "The **Passivity Theorem** states that the negative feedback interconnection of two passive systems (with at least one being strictly passive) is **always stable**, regardless of loop gain magnitude $\\gamma$.\n\n"
            "### 3. Engineering Preference in Robotics & Physics\n"
            "- Passivity is preferred in physical systems (robotics, electromechanical actuators, spacecraft) where actuators and sensors are **collocated** (e.g. torque motors and tachometers measuring conjugate power variables $P = \\tau \\cdot \\omega$).\n"
            "- Because physical dissipation inherently guarantees passivity, feedback controllers designed with passive architectures achieve **infinite gain margins** and remain stable under arbitrary unknown payload variations."
        )
    }
]


def generate_theory_dataset(count: int = 3600) -> None:
    train_file = OUTPUT_DIR / "train.jsonl"
    valid_file = OUTPUT_DIR / "valid.jsonl"

    all_examples = []
    for i in range(count):
        domain = random.choice(THEORY_DOMAINS)
        prompt_tmpl = random.choice(domain["templates"])
        
        prefixes = [
            "",
            "Please provide a rigorous mathematical derivation: ",
            "As an advanced control engineering specialist: ",
            "Perform a graduate-level analysis: ",
        ]
        q_text = random.choice(prefixes) + prompt_tmpl
        
        example = {
            "id": f"theory_sft_{i+1:05d}",
            "messages": [
                {"role": "system", "content": "You are Control-LLM, a premier AI research scientist and expert engineering agent specialized in control systems engineering, applied mathematics, robotics, and dynamical systems."},
                {"role": "user", "content": q_text},
                {"role": "assistant", "content": domain["response"]},
            ]
        }
        all_examples.append(example)

    random.shuffle(all_examples)
    split_idx = int(0.9 * len(all_examples))
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]

    with train_file.open("w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with valid_file.open("w", encoding="utf-8") as f:
        for ex in valid_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(train_data)} training and {len(valid_data)} validation theory SFT examples in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_theory_dataset(3600)
