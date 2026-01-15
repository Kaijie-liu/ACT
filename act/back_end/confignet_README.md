ConfigNet Driver Guide (examples_config workflow)

Purpose
- Confignet samples instances, builds models, and materializes them into
  `examples_config.yaml` plus ACT Net JSON files.
- Validation is handled elsewhere (use `act/pipeline/verification/validate_verifier.py`
  or pipeline CLI). Confignet does not run L1/L2.

File layout (3 files)
- `act/back_end/confignet.py`: CLI + orchestration + builders
- `act/back_end/confignet_spec.py`: schema + sampler + seeds
- `act/back_end/confignet_io.py`: examples_config helpers + nets JSON writer

Design contracts (must hold)
1) examples_config ownership
- The hand-authored base `examples_config.yaml` is never overwritten.
- Generated entries are replaced by prefix (`cfg_seed...`) on each run.
- Names are prefixed with `cfg_seed...` to avoid collisions.

CLI entrypoints
Two equivalent ways to run:
- Module CLI: `python -m act.back_end.confignet ...`
- Pipeline CLI wrapper: `python -m act.pipeline.cli confignet ...`

Supported commands (from `act/back_end/confignet.py`)
1) sample
- Samples instance specs only.
- Example:
  `python -m act.back_end.confignet sample --num 3 --seed 0 --out_json out.json`

2) generate
- Sample instances and materialize them into `examples_config.yaml` + nets JSON.
- Example:
  `python -m act.back_end.confignet generate --num 2 --seed 0 --device cpu --dtype float64`

Common flags
- `--num`: number of instances
- `--seed`: base seed
- `--device`: cpu/cuda
- `--dtype`: float32/float64
- `--out_json`: output path (sample only)
- `--examples-config`: path to examples_config.yaml (default: back_end/examples)
- `--nets-dir`: ACT Net JSON directory (default: back_end/examples/nets)

Generated entries
- Confignet writes generated entries under `networks:` by prefix.
- Re-running Confignet removes entries with the `cfg_seed` prefix and replaces them.

Tests
- `pytest -q act/pipeline/tests/test_confignet.py`
- `pytest -q act/pipeline/verification/tests/test_verification.py`
- `pytest -q`
