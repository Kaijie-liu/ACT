ConfigNet Driver Guide (CLI + Semantics + Tests)

Purpose
- This document defines the unified ConfigNet driver semantics and how to run it.
- The driver only orchestrates existing L1/L2 logic; it does not change algorithms.

Immutable semantics (must hold)
1) Level 1 policy
- If a concrete counterexample is found, CERTIFIED is forbidden.
- Enforced via policy gating (see `act/pipeline/confignet.py`).
- Note: Confignet L1 now delegates to `validate_verifier`'s counterexample
  validation (solver soundness check). Do not conflate it with L2 bounds
  validation logic.

2) Level 2 strict behavior
- Any alignment failure, missing bounds, or NaN/Inf in bounds is an ERROR.
- Strict mode must preserve this behavior (see `validate_bounds_per_neuron`).

3) Bounds compare output fields
- Per-neuron comparison outputs canonical fields:
  `violations_total`, `violations_topk`, `layerwise_stats`.
- Field names are contractually stable.

Driver rules
- Orchestration only: the driver calls existing L1/L2 functions.
- Do not re-implement or change L1/L2 semantics.
- JSONL v1/v2 coexist: v1 is untouched; v2 adds fields.

CLI entrypoints
There are two equivalent ways to run ConfigNet:
- Module CLI: `python -m act.pipeline.confignet ...`
- Pipeline CLI wrapper: `python -m act.pipeline.cli confignet ...`

Supported commands (from `act/pipeline/confignet.py`)
1) sample
- Samples instance specs only.
- Example:
  `python -m act.pipeline.confignet sample --num 3 --seed 0 --out_json out.json`

2) level1
- Runs concrete input sampling + Level 1 checks (JSONL v1).
- Example:
  `python -m act.pipeline.confignet level1 --num 2 --seed 0 --n-inputs 50 --device cpu --dtype float64 --out_jsonl l1.jsonl`

3) level2
- Runs Level 2 bounds checks directly (no L1), writes simple JSONL payloads.
- Example:
  `python -m act.pipeline.confignet level2 --num 2 --seed 0 --n-inputs 5 --tf-modes interval --device cpu --dtype float64 --out_jsonl l2.jsonl`

4) l1l2
- Unified driver: sampling -> L1 -> L2 -> policy -> JSONL v2.
- Example:
  `python -m act.pipeline.confignet l1l2 --num 2 --seed 0 --n-inputs 50 --tf-modes interval --device cpu --dtype float64 --out_jsonl l1l2.jsonl`

5) validate_verifier
- Build confignet instances and run `validate_verifier` on them.
- This path uses the verification validator (solver/back_end checks), not the
  sampling-based ConfigNet L1 check.
- Modes:
  - `counterexample`: calls `validate_counterexamples`
  - `bounds`: calls `validate_bounds`
  - `bounds_per_neuron`: calls `validate_bounds_per_neuron`
  - `comprehensive`: calls both L1+L2 in validator
- Examples:
  `python -m act.pipeline.confignet validate_verifier --mode counterexample --num 2 --seed 0 --device cpu --dtype float64`
  `python -m act.pipeline.confignet validate_verifier --mode bounds --num 2 --seed 0 --tf-modes interval --n-inputs 5`
  `python -m act.pipeline.confignet validate_verifier --mode bounds_per_neuron --num 2 --seed 0 --tf-modes interval --n-inputs 5 --atol 1e-6 --rtol 0.0 --topk 10`
  `python -m act.pipeline.confignet validate_verifier --mode counterexample --families mlp --num 2 --seed 0`

6) smoke
- Quick sanity check for sampling + builder + forward pass.
- Example:
  `python -m act.pipeline.confignet smoke --num 3 --seed 0 --device cpu --dtype float64`

Common flags
- `--num`: number of instances
- `--seed`: base seed
- `--n-inputs`: number of sampled inputs (L1/L2)
- `--tf-modes`: transfer function modes for L2 (default: interval)
- `--device`: cpu/cuda
- `--dtype`: float32/float64
- `--out_json` / `--out_jsonl`: output paths
- `--no-strict-input`: disable strict input sampling checks (L1/L1L2)
- `--no-strict`: disable strict mode for L2 bounds per-neuron
- `--run-solver`: enable solver path (L1L2 only)
- `--solver`: torchlp / gurobi_lp / gurobi_milp

JSONL outputs
- v1 (Level1-only driver): see `act/pipeline/confignet.py`
- v2 (Unified L1L2 driver): see `act/pipeline/confignet_io.py`

Tests
Confignet tests are consolidated into one file:
- `pytest -q act/pipeline/tests/test_confignet.py`

Verification tests are consolidated into one file:
- `pytest -q act/pipeline/verification/tests/test_verification.py`

For a full run:
- `pytest -q`
