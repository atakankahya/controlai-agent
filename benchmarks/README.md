# ControlAI benchmark

`v0.jsonl` is the 12-item qualitative seed. `v1_dev.jsonl` is the 300-item,
machine-reference development benchmark. Its six families are absent from both
SFT train and validation.

## Rules

- Never copy benchmark prompts, references, close paraphrases, or numerical
  variants into SFT, continued-pretraining, preference, or synthetic datasets.
- Split by `family`, not by individual question. A parameter change does not
  create an independent test example.
- Keep benchmark answers in English.
- Score the saved model response against every rubric item. Do not let the model
  see `reference` or `rubric` while generating its answer.
- MATLAB and Simulink code may be statically reviewed now. Mark execution status
  separately until a licensed MATLAB runner is connected.

## Growth path

1. **v0 seed:** at least one checked item in every domain.
2. **v0.1:** 5 families per domain and a mix of concept, derivation, numerical,
   code, critique, design, and underspecified tasks.
3. **v1 development:** 300 family-separated items with machine-readable ground
   truth. The later release gate remains a private 1,000-item/200-family test.

Run the structural and numerical checks with:

```bash
python scripts/validate_benchmark.py benchmarks/v0.jsonl
python scripts/validate_benchmark_v1.py benchmarks/v1_dev.jsonl
```

Run a one-question MLX smoke test from an activated project environment:

```bash
python scripts/run_benchmark_mlx.py --limit 1
```

If that succeeds, resume and finish the remaining questions by omitting
`--limit`. The runner appends after every answer and skips completed IDs:

```bash
python scripts/run_benchmark_mlx.py
```

For v1, always name the output explicitly:

```bash
python scripts/run_benchmark_mlx.py \
  --benchmark benchmarks/v1_dev.jsonl \
  --output benchmarks/responses/qwen3_4b_base_v1_dev.jsonl
```

The benchmark runner sends the system prompt, question, language, and word limit
to the model. Rubrics and references remain hidden. Use a different `--output`
path for each model/configuration so runs cannot be mixed. The original
`qwen3_4b_instruct_v0.jsonl` run is retained as a prompt-wiring baseline; the
corrected default output is `qwen3_4b_instruct_v0_1.jsonl`.

For a fast, deterministic regression signal on v1, run:

```bash
python scripts/score_benchmark_v1.py \
  benchmarks/responses/qwen3_4b_base_v1_dev.jsonl
```

This checks derived numbers, methods/APIs, conclusions, and word limits. It is
explicitly provisional; final model selection still requires rubric review of
the saved answers, especially Kharitonov and generated MATLAB code.
