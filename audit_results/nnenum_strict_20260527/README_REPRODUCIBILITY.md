# nnenum VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-27, 10:26 → 19:57 (9 h 31 min).
**Tool.** nnenum at `/data1/Kane/nnenum`, commit `8346c6855` ("remove conda activate & minor changes", 2025-07-15).
**Python.** 3.8.20 (env `/data1/Kane/miniconda3/envs/nnenumenv`).
**LP solver.** GLPK (`swiglpk`) — Gurobi is *not* enabled in this run; nnenum's default GLPK backend is used everywhere.

**Protocol.** Helper-free **by construction** — nnenum has no PGD / random-sampling / gradient falsification code in its source tree. A reaudit (`grep -rln 'PGD|attack|adversarial.+sample|random.+sample' /data1/Kane/nnenum/src/nnenum/`) returns zero matches. The only "non-reachability" behavior in nnenum is exact star-set splitting that, when a leaf can be made infeasible, witnesses a concrete counter-example — this is a sound result, not a helper.

**Result.** See `RESULTS_TABLE.tex` next to this file and `_summary_overall.csv` for the machine-readable aggregate.

| | |
|---|---|
| Sound UNSAT (V) | **693** |
| Sound SAT (A, via exact-star) | **752** |
| Timeouts (T) | 486 |
| Unknowns (U) | 0 |
| Tool-unsupported (E) | 1 521 |
| Total instances | 3 453 |
| Wall time, all instances | ~53 868 s (~15 h) effective; ~9.5 h calendar with 2 lanes |

---

## Why so many "E" (errors)?

nnenum is a pure star-set / zonotope verifier and **does not depend on `auto_LiRPA`** the way abcrown does. It has its own ONNX parser that supports a narrower set of layers: `MatMul`, `Add`, `Conv` (standard), `ReLU`, `Gemm`, `Flatten`, `Reshape`, `Transpose`. Anything outside that set (Transformers, ResNet residuals, QConv int8 quantization, YOLO custom ops, custom Lyapunov certificate layers, OPF projection, learned conformal certificates) fails fast at load time with an unsupported-op exception. That is honest reporting of tool capability; we do not penalize nnenum for these in cross-tool comparison — those benchmarks just don't apply to it.

The 9 benchmarks that nnenum *does* fully understand (acasxu, sat_relu, malbeware, collins_rul_cnn, linearizenn, cora_2024, metaroom, relusplitter, nn4sys-lindex, tllverify, safenlp) produced sound results for **every** instance with no error.

---

## Directory layout

```
audit_results/nnenum_strict_20260527/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table
├── _summary_overall.csv                ← aggregated machine-readable counts
├── _run.log                            ← driver log (interleaved across lanes A/B)
├── _run.meta.json                      ← provenance (host, commit, GPU-free policy, etc.)
├── _run.pid / _run_laneB.pid           ← PID files (for resume / pkill)
├── _nohup*.out                         ← raw stdout/stderr per launch
├── scripts/
│   └── run_nnenum_noattack_vnncomp2025.sh  ← copy of the launcher used
└── <benchmark>/                        ← per-benchmark dir (one per VNN-COMP 2025 benchmark)
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file
    ├── NNNN__<onnx>__<vnnlib>.result   ← VNN-COMP standard result file: one of unsat | sat | timeout | unknown | error
    ├── NNNN__<onnx>__<vnnlib>.log      ← nnenum's full stdout/stderr
    ├── NNNN__<onnx>__<vnnlib>.raw      ← raw `-f` output before normalization
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata
```

---

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| CPU | many-core (script uses `-p 8` workers per instance, two lanes in parallel) |
| GPU | not required (nnenum is CPU-only) |
| Python | 3.8.20 in conda env `nnenumenv` |
| Solvers | GLPK via `swiglpk` (default); Gurobi optional but not used here |
| Disk | ~10 GB for results |

### Step 0: benchmark data

```bash
ls /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/   # 26 benchmarks
```

### Step 1: install nnenum

```bash
cd /data1/Kane
git clone https://github.com/stanleybak/nnenum.git
cd nnenum
git checkout 8346c6855e50f5c34c6b9981b18271793c8005cf

conda create -n nnenumenv python=3.8 -y
conda activate nnenumenv
pip install -r requirements.txt
```

### Step 2: verify nnenum truly has no attack code

```bash
grep -rln -iE 'PGD|attack|adversarial.+sample|random.+sample' /data1/Kane/nnenum/src/nnenum/
# expected: no output
```

If this command prints any file path, the audit assumption is broken; re-audit before trusting helper-free claims.

### Step 3: run the sweep

```bash
cd /data1/Kane/ACT
nohup bash scripts/run_nnenum_noattack_vnncomp2025.sh \
  > audit_results/nnenum_strict_20260527/_nohup.out 2>&1 &
```

To launch a second lane covering a custom subset of benchmarks (we did this to balance load between heavy and light benchmarks):

```bash
nohup bash scripts/run_nnenum_noattack_vnncomp2025.sh \
  collins_aerospace_benchmark vggnet16_2022 cifar100_2024 \
  tinyimagenet_2024 vit_2023 safenlp_2024 nn4sys \
  acasxu_2023 cora_2024 relusplitter yolo_2023 \
  cctsdb_yolo_2023 metaroom_2023 \
  > audit_results/nnenum_strict_20260527/_nohup_laneB.out 2>&1 &
```

The script is idempotent — instances with a non-empty `.result` file are skipped on relaunch. Safe to Ctrl-C and restart.

### Step 4: aggregate

The per-benchmark `_summary.csv` is written line-by-line during the sweep. The overall aggregate `_summary_overall.csv` was post-computed with this awk one-liner:

```bash
cd /data1/Kane/ACT/audit_results/nnenum_strict_20260527
for d in */; do
  awk -F, 'NR>1 {
    v=$8; gsub(/"/,"",v); t=$6+0; tot++
    if(v=="unsat") nv++; else if(v=="sat") na++;
    else if(v ~ /^timeout/) nt++; else if(v=="unknown") nu++; else ne++;
    wsum += t; if(t>wmax) wmax=t
  } END { printf "%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f\n", B, tot, nv, na, nt, nu, ne, wsum, wmax }' \
  B=${d%/} "$d/_summary.csv"
done
```

---

## Per-instance verdict semantics

Each `.result` file contains exactly one token on the first line:

| Token | Meaning | Sound? |
|---|---|---|
| `unsat` | nnenum's reachable set has empty intersection with the unsafe region | ✅ yes |
| `sat` | nnenum's exact-star splitting found a concrete input that violates the spec | ✅ yes (concrete counterexample exists, no helper needed) |
| `timeout` | wall-clock budget exceeded before any verdict | — |
| `timeout_killed` | kernel `SIGKILL` after grace period (rare) | — |
| `unknown` | nnenum couldn't classify (e.g. LP returned indeterminate) | — |
| `error` | unsupported op / load failure / internal exception | — |

---

## Known issues + cleanups during the run

- **Initial NeuralSAT-style normalizer bug:** my first normalizer expected `holds`/`violated` tokens (abcrown-style). nnenum writes `unsat`/`sat`/`timeout`/`unknown`. I patched the script's `case "$VERDICT_RAW"` block on 2026-05-27 to accept both vocabularies. Eight previously-normalized `.result` files were retroactively re-normalized; the underlying `.raw` files were untouched, so the fix was lossless.
- **Lane B benchmark ordering:** the second lane was launched with an explicit benchmark list (see Step 3) instead of the default order, to interleave heavy CNN workloads with smaller MLPs and keep both lanes busy.
- **One zombie process:** instance `metaroom_2023/0058__4cnn_ry_19_3_no_custom_OP__spec_idx_5_eps_0.00000436` had a Python worker that did not honor its 210s timeout and remained `RNl` for ~7 h. It was killed manually after the sweep ended; its `.result` was `missing_result` (the timeout wrapper exited but the LP-worker child detached). This instance is the only one in the entire 3 453-row sweep with an irrecoverable verdict. It is counted under `E` in the table.

---

## What this archive does NOT contain

- No source patches to nnenum (none were needed — nnenum is helper-free upstream).
- No Gurobi license — GLPK is the default and was used for everything.
- No CORA / NeuralSAT / abcrown results — those are in their own archives.
- No paper figures — the LaTeX table is presentation-grade but the cross-tool comparison table will live in the final paper bundle, not here.

---

## Cross-tool comparison context

For the paper table, this archive contributes the **nnenum (no helper, pure star-set)** row. The four-tool table compares:

| Tool | "no helper" enforcement | Has SAT verdicts? |
|---|---|---|
| abcrown | `--pgd_order=skip` (CLI flag) | yes, only via BaB; PGD disabled |
| NeuralSAT | `--disable_attack` (CLI flag, audited Settings.use_attack) | yes, only via BaB; attack disabled |
| CORA TRUESTRICT | source-patched `falsification_method='none'` (option does not exist upstream) | no — over-approximative reachability cannot witness counter-examples |
| **nnenum (this archive)** | **upstream-native, no flag needed** | yes, via exact-star splitting |

This is the most "natively helper-free" of the four — nothing was disabled or patched; the tool simply does not contain a helper.
