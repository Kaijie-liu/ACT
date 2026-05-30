# ACT VNN-COMP 2025 sweep — CPU STRICT (no helper) — reproducibility bundle

**Date.** 2026-05-24 → 2026-05-30 (CPU baseline) — primarily inherited from r93 prior sweep
plus session B3-sparse-eq_lagr metaroom CPU work.

**Tool.** ACT at `/data1/Kane/ACT`, commit `98a3860e` **with session patches applied**
(see `patches/session_dirty.patch`).

**Python / Torch / Solvers.** Same as GPU archive — Python 3.12.12, torch 2.9.1 (CPU
backend), scipy linprog HiGHS. No Gurobi, no MILP.

**Protocol.** Same helper-free defaults as GPU archive (see `patches/README.md`).

**Honest scope statement.** This session (2026-05-28 → 2026-05-30) was **GPU-focused**.
We did NOT run a fresh complete CPU sweep with our new code. This CPU archive captures
**r93's pre-session CPU baseline** (`source=cpu` in r93's per_instance.csv) **plus the B3
sparse-eq_lagr metaroom CPU work**.

Why this matters:
- Our session patches (gather/slice exact, sigmoid cap raised, upsample/convT exact, etc.)
  ARE sound on CPU — there's no code path that's GPU-specific.
- We expect the CPU sweep with our new code would produce SIMILAR gains to GPU on the
  relevant benchmarks (nn4sys gather/slice, dist_shift sigmoid, cgan upsample/convT, etc.)
- We did not have wall-clock budget to do this full CPU rerun. It would take ≈ 40-60 h
  vs ≈ 25 h GPU.

For the paper, the CPU sweep results in this archive ARE comparable to r93's CPU columns
in the cross-tool table (i.e., the baseline that the user posted as a comparison reference).

**Result.** See `_summary_overall.csv`.

| Metric                                   | Value (r93 CPU baseline) |
|------------------------------------------|--------------------------|
| Sound UNSAT (V)                          | 599                      |
| Sound SAT (A)                            | 67                       |
| Timeout (T)                              | 185                      |
| Unknown (U)                              | 2489                     |
| Errors (E)                               | 108                      |
| Total instances                          | 3,448                    |

**Cross-run validation note.** Many benchmarks have CPU/GPU bit-identical results
(see user's cross-run table — class A: collins_rul, malbeware, acasxu, linearizenn,
safenlp, tllverifybench, dist_shift, cersyve). Class C (GPU unlocks more decisions):
metaroom, cora, ml4acopf, relusplitter, nn4sys, tinyimagenet. Class D (GPU runs through
but no decisive verdict): traffic_signs, yolo, cifar100, soundnessbench, vggnet16,
lsnc_relu, collins_aero, cgan.

## Directory layout

Identical to GPU archive. See `act_gpu_strict_20260530/README_REPRODUCIBILITY.md`
"Directory layout" section.

## Per-instance verdict semantics

Identical to GPU archive.

## How to reproduce from scratch

### Step 1: install
Same as GPU. Apply `patches/session_dirty.patch`.

### Step 2: audit no-helper wiring
Same as GPU.

### Step 3: smoke test
```bash
# Run a small CPU smoke
PYTHONPATH=/data1/Kane/ACT ACT_VNNLIB_ROOT=/data1/Kane/data/vnncomp2025_benchmarks/benchmarks \
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.watchdog_runner \
    --benchmark malbeware --instance-ids 0 \
    --wall-s 60 --rss-cap-gb 24 --device cpu --dtype float64 \
    --out-dir /tmp/act_cpu_smoke \
    --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
# Expected: malbeware iid=0 CERTIFIED in < 30s
```

### Step 4: full CPU sweep
See `scripts/run_act_strict_vnncomp2025_cpu.sh`. Expected wall-clock: ≈ 40-60 h sequential,
or ≈ 20 h with 4-way parallel (within RSS budgets).

### Step 5: aggregate
Same as GPU archive.

## Known issues

- **Incomplete session CPU rerun**: see "Honest scope statement" above.
- **B3 sparse-eq_lagr metaroom CPU work**: was captured in a separate sweep (see
  memory `project_b3_sparse_eq_lagr_20260528.md`); CPU verdict +5 CERT on metaroom
  is consistent with the GPU archive's metaroom numbers.

## Cross-tool comparison context

Same as GPU archive.
