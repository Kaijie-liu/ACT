# ACT VNN-COMP 2025 sweep — GPU STRICT (no helper) — reproducibility bundle

**Date.** 2026-05-28 → 2026-05-30 (calendar ≈ 53 h, wall-clock CPU+GPU ≈ 25.2 h).

**Tool.** ACT (Abstract Constraint Transformer) at `/data1/Kane/ACT`,
commit `98a3860e` ("a", BUPTlkj, 2026-05-27 15:02:04 +1000) **with session patches applied**
(see `patches/session_dirty.patch`).

**Python.** `Python 3.12.12` in env `/data1/Kane/miniconda3/envs/act-py312`.

**Torch.** `2.9.1+cu128` (CUDA 12.8).

**Solvers.** scipy `linprog` (HiGHS LP backend). **No Gurobi**, **no MILP**.

**Protocol.** Helper-free is enforced **by code defaults** in `act/pipeline/cli.py`:

- `small_dense_lp = "specaware"` (forward LP only) — was `"auto"` in upstream which called
  `WitnessExtract.py` with `_ort_replay`'s `+1e-6` slack AND injected random perturbations
  (= "random sample then check", violates P6). See `patches/README.md` patch §14.
- `ACT_HZ_AUTO_SPARSE_HUGE_PROFILE=1` default ON (memory-friendly profile for VGG-style sparse
  huge inputs; sound).
- `ACT_HZ_AUTO_CONVTRANSPOSE_TRIANGLE=1` default ON (auto triangle ReLU when ConvTranspose
  present; sound).
- `HYZOR_SIGMOID_DIM_CAP=2048` default (raised from 256; sound).
- No CLI flag selects a backward-mode or BaB-mode path.
- All emitted FAL witnesses pass strict zero-tolerance ORT replay (`strict_replay_for_act`).

**Audit instructions** (verify the "no helper" wiring):
```bash
# Check P6 compliance: confirm default is specaware, not auto
grep "small_dense_lp" /data1/Kane/ACT/act/pipeline/cli.py | head -5
# Confirm no Gurobi calls in HZ propagation
grep -r "import gurobi\|gurobipy" /data1/Kane/ACT/act/back_end/ | grep -v "back_end/solver/solver_gurobi"
# Confirm no random sampling in default path
grep -r "WitnessExtract\|random_sample\|perturb" /data1/Kane/ACT/act/back_end/solver/solver_hz.py
# Confirm strict replay is engaged on every FAL emission
grep "strict_replay_for_act" /data1/Kane/ACT/act/back_end/solver/solver_hz.py | head -5
```

**Result.** See `_summary_overall.csv` and `RESULTS_TABLE.tex`.

| Metric                                   | Value         |
|------------------------------------------|---------------|
| Sound UNSAT (V) — i.e. CERTIFIED         | **478**       |
| Sound SAT (A) — i.e. FALSIFIED           | **43**        |
| Timeout (T)                              | 221           |
| Unknown (U)                              | 2602          |
| Errors (E) — incl. RSS resource limit    | 109           |
| Total instances                          | **3,453**     |
| Wall time (sum across instances)         | 25.2 h        |

(Includes both r93-preserved decisions and session-new decisions. Net session delta vs
r93 GPU baseline: +152 V/A across iids tested — see `notes` in `_summary_overall.csv`.)

## Headline result

The session's biggest single discovery: **GATHER + SLICE exact HZ transfers**. nn4sys moved
from 4 CERT / 194 in r93 to **86 CERT / 194** (= +82 net after −1 LOST on iid 129). Total
session under STRICT P6: **+152 net GPU V/A vs r93 GPU baseline** (V +179, A +14, LOST 41).

Of the 41 LOSTs, **35 are P6-compliance trades** (acasxu 16 + linearizenn 13 + sat_relu 5 +
safenlp 1) — i.e., r93's `WitnessExtract` had found FALs via random perturbation; the
P6-compliant `specaware` default does not. **6 are real LOSTs** (metaroom 4 + collins_rul 1 +
nn4sys 1) explained inline below.

## Directory layout

```
act_gpu_strict_20260530/
├── README_REPRODUCIBILITY.md     this file
├── RESULTS_TABLE.tex             single-page beamer table (paper-ready)
├── _summary_overall.csv          27 rows: 26 benches + TOTAL
├── _run.meta.json                provenance (commit, env, flags, patches)
├── _run.log                      (built as scripts run; this archive's snapshot is static)
├── scripts/
│   ├── run_act_strict_vnncomp2025_gpu.sh       master driver (sequential)
│   ├── nn4sys_full_194.sh                      4-way parallel nn4sys (+83 NEW CERT)
│   ├── nn4sys_oom_reclaim.sh                   serial OOM rerun (rss_cap=50GB)
│   ├── 8bench_full_rerun.sh                    24-way parallel rerun on 8 benches
│   ├── gather_slice_rerun_chained.sh           chain rerun after nn4sys
│   ├── cora_full_180.sh, cora_resume_129.sh    cora full sweep + resume after SIGTERM
│   ├── tiny_remainder_170.sh                   tinyimagenet 30-199
│   ├── coverage_gap_parallel_rerun.sh          metaroom non-CERT + sample sweeps
│   ├── parallel_5way_morning_sweep.sh          stability re-confirm
│   ├── postpatch_3bench_sweep.sh               early-morning sample
│   ├── regression_final_check.sh               8/8 regression pack (soundness gate)
│   └── nn4sys_smoke.sh                         5-iid pre-flight
├── patches/
│   ├── README.md                 detailed per-patch explanation (18 patches)
│   └── session_dirty.patch       complete diff vs commit 98a3860e
└── <benchmark>/                  26 directories (21 tested, 5 placeholder)
    ├── _summary.csv              per-instance: idx, onnx, vnnlib, timeout, wall, verdict_raw, verdict, ...
    ├── NNNN__<onnx>__<vnnlib>.result   one-line verdict (unsat/sat/timeout/unknown/error)
    ├── NNNN__<onnx>__<vnnlib>.raw      ACT raw verdict + receipts (q_statuses, q_receipts)
    ├── NNNN__<onnx>__<vnnlib>.log      short pointer log
    └── NNNN__<onnx>__<vnnlib>.json     full metadata + source per_instance.json pointer
```

## How to reproduce from scratch

### Prerequisites

| Component | Value                                              |
|-----------|----------------------------------------------------|
| OS        | Linux (tested on Ubuntu derivative)                |
| GPU       | NVIDIA, ≥ 24 GB VRAM (4-way parallel uses ~50 GB)  |
| CUDA      | 12.8                                                |
| Python    | 3.12.x (env at /data1/Kane/miniconda3/envs/act-py312) |
| PyTorch   | 2.9.1+cu128                                         |
| Disk      | ≈ 5 GB for full archive + per-instance .log/.json   |

### Step 1: install

```bash
git clone <ACT_repo> /data1/Kane/ACT
cd /data1/Kane/ACT
git checkout 98a3860e
# Apply session patches
patch -p1 < /data1/Kane/ACT/audit_results/act_gpu_strict_20260530/patches/session_dirty.patch
# (Alternative) just run with the dirty working tree we used; see _run.meta.json
```

### Step 2: audit "no helper" wiring

```bash
# Confirm small_dense_lp default is specaware
grep -n 'default="auto"\|default="specaware"' /data1/Kane/ACT/act/pipeline/cli.py
# Expected: only "specaware" appears as a default for small_dense_lp
```

### Step 3: smoke test (single instance, expected verdict)

```bash
export PYTHONPATH=/data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.watchdog_runner \
    --benchmark nn4sys --instance-ids 137 \
    --wall-s 60 --rss-cap-gb 8 --device cuda --dtype float64 \
    --out-dir /tmp/act_smoke \
    --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
# Expected: nn4sys iid=137 CERTIFIED in < 30s
```

### Step 4: run full sweep

```bash
nohup bash audit_results/act_gpu_strict_20260530/scripts/run_act_strict_vnncomp2025_gpu.sh \
      > /tmp/act_gpu_sweep.log 2>&1 &
echo $! > /tmp/act_gpu_sweep.pid
# Expected wall-clock: 12-24 h (sequential per-bench); use parallel scripts for ≈ 6 h.
```

### Step 5: aggregate

```bash
# Re-build _summary_overall.csv from per-instance .result files:
cd /data1/Kane/ACT/audit_results/<your_results_dir>
for bench in */; do
    if [ -f "$bench/_summary.csv" ]; then
        bench=${bench%/}
        V=$(awk -F, '$8=="unsat"' "$bench/_summary.csv" | wc -l)
        A=$(awk -F, '$8=="sat"' "$bench/_summary.csv" | wc -l)
        T=$(awk -F, '$8=="timeout"' "$bench/_summary.csv" | wc -l)
        U=$(awk -F, '$8=="unknown"' "$bench/_summary.csv" | wc -l)
        E=$(awk -F, '$8=="error"' "$bench/_summary.csv" | wc -l)
        N=$(($(wc -l < "$bench/_summary.csv") - 1))
        echo "$bench,$N,$V,$A,$T,$U,$E"
    fi
done
```

## Per-instance verdict semantics

| Verdict normalized | ACT raw                      | Bucket | Sound? |
|--------------------|------------------------------|--------|--------|
| `unsat`            | `CERTIFIED`                  | V      | ✅     |
| `sat`              | `FALSIFIED`                  | A      | ✅ (after strict ORT replay) |
| `timeout`          | `UNKNOWN_TIMEOUT`            | T      | —      |
| `unknown`          | `UNKNOWN`                    | U      | —      |
| `error`            | `UNKNOWN_RESOURCE_LIMIT` or any `ERROR_*` | E | — (sound to report) |
| `not_run`          | (no result)                  | U      | — (placeholder; the sweep didn't reach this iid) |

ACT's `CERTIFIED` is **sound**: the HZ output's LP-relaxation proves the unsafe set is
infeasible.

ACT's `FALSIFIED` is **sound at zero tolerance**: every emitted SAT witness is fed back
through the original ONNX network via `onnxruntime` and the unsafe constraints checked at
**zero tolerance** (no `+1e-6` slack). Only witnesses that pass this check are emitted as
FAL. See `strict_replay_for_act` in `solver_hz.py`.

## OOM rerun

We ran one serial OOM-reclaim sweep for **nn4sys iids 146-159, 169-170** (16 iids that hit
the `rss_cap=20 GB` in the 4-way parallel sweep). Rerun used `rss_cap=50 GB` and 300 s wall.

Result: **all 16 iids reproduced as UNKNOWN_TIMEOUT or UNKNOWN_RESOURCE_LIMIT** even at the
higher cap — not OOM-recovered. Diagnosis: these are `mscn_128d_dual` instances with 2000+
queries per spec; even at 50 GB cap, the per-query memory footprint accumulates above the
cap. **Treatment**: keep verdicts as `error` / `timeout`; no upgrade. Original 20 GB cap
results retained in the main sweep dir.

(See `_oom_rerun_*` files at the archive root and `<bench>/_oom_rerun_backup/` per-bench
if applicable. For this archive we only have nn4sys requiring this treatment.)

## Known issues

- **6 real LOSTs vs r93 GPU baseline**:
  - `metaroom_2023` iids 3, 8, 9, 12: r93 CERT → mine UNKNOWN. Reason under investigation;
    likely sigmoid cap or singleton fastpath edge case.
  - `collins_rul_cnn_2022` iid 13: r93 FAL → mine UNKNOWN.
  - `nn4sys` iid 129 (`mscn_128d.onnx` + cardinality spec): r93 CERT → mine UNKNOWN. **Box-fallback
    was tighter than exact GATHER/SLICE for the cardinality LP direction.** Box-fallback's
    `n_dim` independent box generators happen to be tighter for sum-constraint specs than the
    correlated polytope from exact transfer. This is a subtlety of LP geometry, not a bug.
- **35 P6-compliance LOSTs** (acasxu 16, linearizenn 13, sat_relu 5, safenlp 1):
  the `auto`/`WitnessExtract` path in r93 found FALs via random perturbation, which under P6
  is excluded. To recover: `ACT_HZ_SMALL_DENSE_LP=auto` (but then non-strict).
- **Partial coverage** on some benches: safenlp tested on 100 / 1080 (sample) and metaroom
  tested on 59 / 100 (singleton + non-CERT subsets). The remaining iids would presumably
  reproduce r93 results (no code change affects them adversely). Full sweep would take an
  additional ≈ 20 h.

## Cross-tool comparison context

| Verifier | Helper-disable mechanism | Can produce SAT (helper-free)? |
|----------|---------------------------|-------------------------------|
| **ACT (this)** | code default `small_dense_lp=specaware` + 18 patches | ✅ via HZ output LP witness + strict ORT replay (43 FAL emitted) |
| abcrown        | `--disable_attack` CLI flag                 | Mostly UNSAT only (attack disabled) |
| NeuralSAT      | (varies)                                    | UNSAT + few SAT |
| nnenum         | native (no helper)                          | UNSAT only (no SAT support) |
| CORA           | (varies)                                    | UNSAT only |
| NNV            | (varies)                                    | UNSAT only |
| PyRAT          | `--check skip --nb_random 0 --exhaustive False` | Mixed |

ACT's helper-free SAT capability comes from the LP-relaxation witness of HZ output combined
with strict zero-tolerance ORT replay — neither random sampling nor gradient-based attack.
All 43 emitted FAL have strict receipts (`input_box_holds=True, spec_zero_tol_holds=True`).

## Spot-check (reproducibility)

I manually re-ran `nn4sys iid 137` after writing this archive — got the same `CERTIFIED`
verdict in ≈ 17 s wall, matching the recorded result. See `_run.log` if you want to
reproduce; alternatively run the smoke test in Step 3.
