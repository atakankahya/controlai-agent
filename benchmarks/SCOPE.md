# ControlAI Benchmark Scope

## Purpose

Measure whether a local model can reason, calculate, code, and communicate safely across control-systems engineering. The benchmark language is English. Vendor-specific PLC programming is outside the first version.

## Knowledge domains

1. Classical control and loop shaping
2. State-space systems and geometric properties
3. State estimation and filtering
4. Optimal control
5. Robust control and uncertainty
6. Model predictive control
7. Nonlinear control
8. Adaptive control
9. System identification
10. Sampled-data, numerical, and implementation issues

## Task types

Each domain should eventually contain more than one of these task types:

1. **Concept:** Explain a definition, limitation, or trade-off.
2. **Derivation:** Derive a condition, controller, estimator, or closed-loop map.
3. **Numerical:** Compute a result for a concrete system.
4. **Code:** Produce executable Python and verify the result.
5. **Critique:** Find and correct an error in a proposed solution.
6. **Design:** Make and justify an engineering design choice.
7. **Underspecified:** Detect missing information instead of inventing it.

## Difficulty levels

- **Foundation:** One standard idea with explicit assumptions.
- **Intermediate:** Multiple connected steps or a design trade-off.
- **Advanced:** Uncertainty, constraints, nonlinearities, or competing objectives.

## Evaluation dimensions

Answers are scored separately for:

- technical correctness;
- completeness;
- executable numerical verification when requested;
- explicit assumptions and conventions;
- compliance with requested format;
- calibrated uncertainty and safety.

## Leakage rule

Benchmark prompts and close numerical or textual variants must not enter training data. Splits are organized by problem family, not by randomly shuffling nearly identical questions.

