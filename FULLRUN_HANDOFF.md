# HyZor Strict Pure-HybridZ Frozen Handoff

This handoff records the current strict pure-HybridZ result. The 2026-06-24
frozen artifact is historical provenance; the 2026-06-25 soundfix below is the
current reporting source because it fixes an instance-level disjunct aggregation
bug in the package runner.

## Strict Rule

- Count only ACT-HybridZ results produced by the HybridZ engine itself.
- `CERT` means HybridZ proved the unsafe set empty.
- `ADV` means HybridZ produced an exact reachable unsafe witness.
- ORT/PyTorch replay is an audit guard only. It must not upgrade an engine
  `UNKNOWN` into a counted HybridZ verdict.
- Do not count input split, sampling, CROWN/backward rescue, Gurobi-only proof,
  LP-witness promotion, phase-fix rescue after HybridZ returned `UNKNOWN`, or
  ReLU triangle relaxation.

## Canonical Soundfix Artifact

Path:

```text
/data1/Kane/ICSE/act_hybridz_soundfix_20260625
```

Key files:

- `FINAL_HYBRIDZ_RESULTS_20260625_SOUNDFIX.csv`
- `FINAL_CROSS_TOOL_RANKING_20260625_SOUNDFIX.csv`
- `SOUNDFIX_ERRATUM_20260625.md`
- `metaroom_2023_soundfix_detail_20260625.csv`

Soundfix headline:

```text
1763 / 2213 = 977 CERT + 786 ADV, P0=0
```

Important correction:

- Old metaroom row `100 CERT / 0 ADV / 100 V+A` is superseded.
- Current sound row is `94 CERT / 1 ADV / 95 V+A / 5 TIMEOUT`.
- Root cause: the package runner promoted an instance to CERT when any split
  disjunct was CERT. The fixed rule is: any disjunct ADV makes the instance
  ADV; an instance is CERT only when every split disjunct is CERT.
- Current metaroom remains rank #1 by V+A (`95` vs abCROWN/NeuralSAT `94`).

## Historical Frozen Artifact

Path:

```text
/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25
```

Key files:

- `FINAL_HYBRIDZ_RESULTS_20260624.csv`
- `FINAL_CROSS_TOOL_RANKING_20260624.csv`
- `FINAL_HYBRIDZ_RESULTS_20260624.md`
- `FROZEN_SUMMARY.json`
- `UNRESOLVED_FRONTIER.csv`
- `UNRESOLVED_FRONTIER_SUMMARY.csv`
- `FUTURE_WORK_HYBRIDZ_20260624.md`
- `_DETAIL.csv`
- `_INDEX.csv`
- `_MANIFEST.sha256`

Historical frozen headline, superseded for reporting:

```text
1768 / 2213 = 983 CERT + 785 ADV, P0=0
```

## Soundfix Table And Rank

| Bench | N | CERT | ADV | V+A | Unsolved | Rank | Best | Gap |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| safenlp_2024 | 1080 | 432 | 647 | 1079 | 1 | 2 | abCROWN 1080 | 1 |
| metaroom_2023 | 100 | 94 | 1 | 95 | 5 | 1 | OURS 95 | 0 |
| sat_relu | 100 | 50 | 50 | 100 | 0 | 1 | OURS+NeuralSAT 100 | 0 |
| malbeware | 150 | 131 | 19 | 150 | 0 | 1 | OURS 150 | 0 |
| cersyve | 12 | 5 | 6 | 11 | 1 | 1 | OURS 11 | 0 |
| acasxu_2023 | 186 | 86 | 34 | 120 | 66 | 5 | nnenum 186 | 66 |
| linearizenn_2024 | 60 | 39 | 1 | 40 | 20 | 5 | nnenum+PyRAT 60 | 20 |
| dist_shift_2023 | 72 | 70 | 0 | 70 | 2 | 1 | OURS 70 | 0 |
| tllverifybench_2023 | 32 | 5 | 12 | 17 | 15 | 3 | PyRAT 30 | 13 |
| cora_2024 | 180 | 19 | 6 | 25 | 155 | 1 | OURS 25 | 0 |
| relusplitter | 220 | 41 | 2 | 43 | 177 | 3 | abCROWN 113 | 70 |
| cgan_2023 | 21 | 5 | 8 | 13 | 8 | 2 | PyRAT 19 | 6 |

Best-position summary:

- Rank 1: `metaroom_2023`, `sat_relu`, `malbeware`, `cersyve`,
  `dist_shift_2023`, `cora_2024`.
- Rank 2: `safenlp_2024`, `cgan_2023`.
- Rank 3: `tllverifybench_2023`, `relusplitter`.
- Lower rank: `acasxu_2023`, `linearizenn_2024`.

## Productization Status

Current mainline work has moved the strict HybridZ frontend, benchmark runner,
frozen-count metadata, reporting, and self-tests into package-owned modules:

- `act/back_end/hybridz_config.py`
- `act/back_end/hybridz_tf/sparse_ops.py`
- `act/back_end/solver/sparse_hz.py`
- `act/back_end/solver/solver_hz_verdict.py`
- `act/pipeline/hybridz_benchmark_runner.py`
- `act/pipeline/hybridz_results.py`
- `act/pipeline/hybridz_projected_relu_mip.py`
- `act/pipeline/hybridz_projected_utils.py`

`scripts/` remains local and untracked by user request. It should not be pushed
unless the user explicitly changes that rule.

Important remaining acceptance item:

- A complete frontend soundfix-suite run with
  `--hybridz-require-frozen-match` is still pending after package cleanup. The
  soundfix artifact above is the current reporting source until that full rerun
  passes.
- 2026-06-25 package rerun note: a first `safenlp_2024` frontend rerun missed
  frozen parity by one ADV because iid704 is a HiGHS B&B edge case under the
  19s engine budget.  The runner now includes a fixed benchmark-wide
  `normal_seed2` branch (`HZ_HIGHS_OPTIONS=random_seed=2`) for all safenlp
  instances.  Full safenlp package rerun evidence is saved at
  `/data1/Kane/ICSE/act_hybridz_safenlp_seed2_full_20260625`: it matches the
  frozen safenlp table exactly (`432 CERT / 647 ADV / 1079 V+A / 1 UNKNOWN`,
  zero per-instance result mismatches, `P0=0`).  The recovered iid704 result is
  a pure HybridZ MILP witness (`milp_objective_bound`), not a replay/audit
  promotion.

Acceptance command:

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark \
  --category frozen \
  --hybridz-require-frozen-match \
  --hybridz-results-dir /data1/Kane/ICSE/act_hybridz_frontend_full_FINAL
```

Short smoke examples already exercised through the normal ACT frontend:

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark \
  --category safenlp_2024 \
  --max-instances 1 \
  --solvers hybridz \
  --hybridz-results-dir /tmp/act_hybridz_safenlp_frontend_smoke_current
```

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark \
  --category frozen \
  --max-instances 1 \
  --hybridz-workers 2 \
  --hybridz-results-dir /tmp/act_hybridz_frozen_smoke_current
```

## Local Regression Commands

These lightweight checks are the current fast guardrail before any commit:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_config
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_results
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.solver.solver_hz_verdict
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf
git diff --check
```

## Future Work

Future directions are written in:

- `docs/FUTURE_WORK_HYBRIDZ_20260624.md`
- `/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/FUTURE_WORK_HYBRIDZ_20260624.md`

High-value directions:

- `dist_shift_2023`: sounder and tighter S-curve operators. This is the cleanest
  non-binary wall and must be validated benchmark-wide before becoming default.
- `cgan_2023` and `tllverifybench_2023`: sparse exact-HZ propagation, block or
  Schur presolve, and representation-drop avoidance.
- `acasxu_2023`, `linearizenn_2024`, `relusplitter`, `cora_2024`: exact forward
  bound tightening, projected/compressed ReLU formulations, and open-source
  HiGHS/SCIP portfolio improvements for binary-MIP scale walls.
- `safenlp_2024`: one remaining UNKNOWN. Any change must be benchmark-wide and
  pass the full frozen-suite gate before replacing the frozen headline.
- CIFAR-style models: lazy/matrix-free exact HybridZ remains research work. It
  is not part of the current frozen claim.
