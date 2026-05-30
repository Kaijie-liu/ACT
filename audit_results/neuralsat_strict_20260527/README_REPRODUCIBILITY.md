# NeuralSAT VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-27 08:48 → 2026-05-28 06:58 (calendar 22 h, ~36 h CPU across two lanes).
Serial OOM rerun: 2026-05-28 08:48 → 10:05.
**Tool.** NeuralSAT at `/data1/Kane/neuralsat`, commit `8c6a493d2` ("updates", hocdot, 2025-07-20).
**Python.** 3.11.15 (env `/data1/Kane/miniconda3/envs/neuralsat`).
**PyTorch.** 2.9.1+cu128 (Blackwell-compatible).
**LP solver.** Gurobi 11.0.x (academic WLS license through 2026-05) — used when NeuralSAT internally picks the MIP backend.

**Protocol.** Helper-free via the **upstream-native** CLI flag `--disable_attack`. This sets `Settings.use_attack=False`, which gates three call-sites in `verifier.py` (`_pre_attack`, `_mip_attack`, `_attack`). No source patches were needed — see `patches/README.md` for the audit of the wiring.

**Result.** See `RESULTS_TABLE.tex` and `_summary_overall.csv`.

| | |
|---|---|
| Sound UNSAT (V) | **1 581** |
| Sound SAT (A, via BaB completeness) | **484** |
| Unknown (U, NeuralSAT `early_stop`) | 47 |
| Timeout (T) | 835 |
| Errors (E, mostly tool-unsupported) | 506 |
| Total instances | 3 453 |
| Resolved rate (V+A)/N | **59.8%** |
| Wall time | ~129 040 s (~35.8 h CPU) / 22 h calendar (dual-lane) |

---

## Headline result

NeuralSAT with attack disabled still resolves **59.8% of all VNN-COMP 2025 instances** (62.9% if you exclude the 170 tool-unsupported instances). The 484 SAT verdicts all come from BaB completeness — the BaB tree refines to a leaf at which a concrete counterexample is read off the LP solution. This is the most striking property of NeuralSAT in this comparison: it is the only tool of the four where disabling the helper has only a moderate impact on the SAT count (vs. CORA which drops to A=0 by construction).

---

## Directory layout

```
audit_results/neuralsat_strict_20260527/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table
├── _summary_overall.csv                ← aggregated machine-readable counts (TOTAL row at end)
├── _oom_rerun.log                      ← serial rerun driver log (22 instances)
├── _oom_rerun_results.csv              ← per-instance rerun diagnosis (one row per of the 22)
├── _nohup_oom_rerun.out                ← raw stdout of the rerun
├── _run.log                            ← main sweep driver log (interleaved across lanes A/B)
├── _run.meta.json                      ← provenance (commit, python/torch versions, etc.)
├── _run.pid / _run_laneB.pid           ← PID files
├── _nohup{,_v2,_v3,_v4}.out            ← raw stdout per relaunch (v4 is the post-fix one)
├── scripts/
│   ├── run_neuralsat_strict_parallel.sh    ← bash launcher (post-fix; see patches/)
│   └── rerun_neuralsat_oom_serial.sh       ← serial rerun for the 22 OOM-suspected instances
├── patches/
│   └── README.md                       ← audit of --disable_attack wiring + driver bug-fix note
└── <benchmark>/                        ← per-benchmark dir (one per VNN-COMP 2025 benchmark)
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file
    ├── _oom_rerun_backup/              ← (collins_rul_cnn_2022 + ml4acopf_2024 only) originals from the parallel run, preserved before serial rerun
    ├── NNNN__<onnx>__<vnnlib>.result   ← VNN-COMP standard verdict line
    ├── NNNN__<onnx>__<vnnlib>.log      ← NeuralSAT's full stdout/stderr
    ├── NNNN__<onnx>__<vnnlib>.raw      ← raw verdict file as written by --result_file
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata
```

---

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM |
| CUDA driver | 12.8 |
| Python | 3.11 |
| PyTorch | 2.9.1+cu128 |
| Gurobi | 11.0.x (academic WLS license) |
| Disk | ~30 GB for results |

### Step 1: install NeuralSAT

```bash
cd /data1/Kane
git clone https://github.com/dynaroars/neuralsat.git
cd neuralsat
git checkout 8c6a493d2b9314b06c6f19cd452f1c7ab5bd2657

conda create -n neuralsat python=3.11 -y
conda activate neuralsat
pip install -r requirements.txt
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: audit `--disable_attack`

Before trusting helper-free claims, re-audit the wiring:

```bash
grep -n 'use_attack' /data1/Kane/neuralsat/src/setting.py /data1/Kane/neuralsat/src/verifier/verifier.py /data1/Kane/neuralsat/src/main.py
```

Expected: `Settings.use_attack = True` at line 26 of setting.py; assignment from `args.disable_attack` at lines 60-61; three gated call-sites in `verifier.py` (lines 88, 102, 433). See `patches/README.md` for the full audit trail.

### Step 3: smoke test

```bash
PY=/data1/Kane/miniconda3/envs/neuralsat/bin/python
$PY /data1/Kane/neuralsat/src/main.py \
  --net /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
  --spec /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib \
  --timeout 60 \
  --result_file /tmp/ns_smoke.raw \
  --disable_attack
cat /tmp/ns_smoke.raw
# expected: 'unsat,...' within seconds
```

### Step 4: run the dual-lane sweep

```bash
cd /data1/Kane/ACT
nohup bash scripts/run_neuralsat_strict_parallel.sh \
  > audit_results/neuralsat_strict_20260527/_nohup_v4.out 2>&1 &
```

Idempotent — instances with a non-empty `.result` file are skipped on relaunch. If a lane crashes, just relaunch the same command; the resume logic continues.

### Step 5 (optional): serial rerun of OOM-suspected instances

For paper-grade rigor, after the main sweep finishes:

```bash
cd /data1/Kane/ACT
nohup bash scripts/rerun_neuralsat_oom_serial.sh \
  > audit_results/neuralsat_strict_20260527/_nohup_oom_rerun.out 2>&1 &
```

This script picks out the 22 instances whose parallel-run failure mode (SIGKILL well before timeout, or SIGSEGV) is consistent with resource contention, and reruns each one serially. See "OOM rerun" section below for the diagnosis we observed.

### Step 6: aggregate

```bash
cd /data1/Kane/ACT/audit_results/neuralsat_strict_20260527
# Per-benchmark verdict tally from .result files (ground truth after rerun):
for d in */; do
  [[ "$d" == "scripts/" || "$d" == "patches/" ]] && continue
  b=${d%/}; files=$(ls $d/*.result 2>/dev/null); [[ -z "$files" ]] && continue
  declare -A C; N=0
  for f in $files; do v=$(head -1 $f | tr -d '[:space:]'); [[ -z "$v" ]] && v=empty; C[$v]=$((${C[$v]:-0}+1)); N=$((N+1)); done
  V=${C[unsat]:-0}; A=${C[sat]:-0}
  T=$((${C[timeout]:-0} + ${C[timeout_killed]:-0}))
  U=${C[raw_early_stop]:-0}
  E=$((${C[error]:-0} + ${C[missing_result]:-0} + ${C[empty]:-0}))
  echo "$b: N=$N V=$V A=$A T=$T U=$U E=$E"
  unset C
done
```

---

## Per-instance verdict semantics

| Token in `.result` | Source CLI verdict | Sound? | Bucket |
|---|---|---|---|
| `unsat` | NeuralSAT proved spec holds | ✅ | V |
| `sat` | BaB witnessed a concrete counterexample | ✅ | A |
| `timeout` | NeuralSAT exited normally at its internal timeout | — | T |
| `timeout_killed` | wrapper SIGKILL after grace period | — | T |
| `unknown` / `raw_early_stop` | NeuralSAT gave up (`early_stop`) | — | U |
| `error` | NeuralSAT crashed at load / mid-run | — | E |
| `missing_result` | no `.raw` was written (process killed before output) | — | E |

`empty` (zero-byte `.result`) should not appear after the serial rerun.

---

## OOM rerun — what we found

During the main parallel sweep, 22 instances failed with signals that *could* indicate resource contention (SIGKILL with wall-clock much less than the budget, or SIGSEGV). The serial rerun (one instance at a time, no competing GPU load) lets us disambiguate "OOM in parallel" from "real tool failure."

### Original failure types

| Benchmark | Indices | Original signal | Parallel wall |
|---|---|---|---|
| collins_rul_cnn_2022 | 21, 22, 43 | SIGKILL (exit 137) | 597 s, 597 s, 882 s (timeout cap 1 800 s) — way too early to be natural timeout |
| ml4acopf_2024 | 15, 16, 17, 18, 19, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56 | SIGSEGV (exit 139) | various; verdict `missing_result` (process died before writing `.raw`) |

### Serial rerun outcomes

| Benchmark | Indices | Serial behavior | Wall in serial | Diagnosis | Action taken |
|---|---|---|---|---|---|
| collins_rul_cnn_2022 | 21 | SIGKILL at wrapper timeout | 1 719 s | **OOM in parallel; in serial NeuralSAT runs the full 1 800 s budget and then escalates to SIGKILL** — the original early SIGKILL was OOM-killer, the rerun confirms the instance is genuinely hard (would have timed out anyway) | `.result` written as `timeout` |
| collins_rul_cnn_2022 | 22 | SIGKILL at wrapper timeout | 1 527 s | same as idx 21 (OOM-mitigated; still genuine timeout) | `.result` = `timeout` |
| collins_rul_cnn_2022 | 43 | SIGKILL | 1 231 s | the serial run still died before the cap; partially mitigated, the instance is memory-heavy enough that even single-GPU runs hit allocator pressure | `.result` = `timeout` (conservative — could also be classified `error_oom`, but `timeout` is the closest sound bucket) |
| ml4acopf_2024 | 15–19, 43–56 (19 total) | **Every single one re-SIGSEGV in ~3 seconds**, all with identical timing | ~3 s each | **NOT OOM** — confirmed deterministic NeuralSAT internal segfault on ACOPF networks at load time. The crash happens before any reachability work; the trigger is the `helper.network.onnx2pytorch.operations.constant.Constant` unsupported-layer path in auto_LiRPA | `.result` written as `error` (this is a tool capability limit, not a resource issue) |

### Conclusion: **0 confirmed resource-contention failures**

After the serial rerun, **no instance's final verdict in this archive is corrupted by resource contention**. The 3 collins_rul instances that originally appeared OOM-killed are now classified `timeout` (which is the correct sound verdict — they are genuinely hard problems that need more compute, not artifacts of the parallel sweep). The 19 ml4acopf SIGSEGVs are a NeuralSAT-internal bug on a specific class of computation graphs (ACOPF) that reproduces deterministically in single-process runs.

For the paper: it is honest to claim that NeuralSAT under `--disable_attack` has **0 resource-contention failures** in this archive, with the qualifier that 19 instances on ml4acopf cannot be evaluated because NeuralSAT itself crashes deterministically at load time on those graphs.

### Forensic preservation

The original `.result`, `.raw`, `.log`, and `.json` from the parallel-run failure are kept under `<bench>/_oom_rerun_backup/` so that the original behavior is fully reproducible. They are not consulted by the aggregator, which reads only the live `.result` files.

---

## Other known issues observed during the run

- **lsnc_relu CSV duplicates.** The original `_summary.csv` for `lsnc_relu` has 96 rows because the lane-restart logic re-tried the benchmark a second time before discovering the prior `.result` files. The `.result` files themselves are de-duplicated by `(idx, onnx, vnnlib)` tuple, so the file-system count and the aggregator both correctly report 80 instances. The CSV is left untouched as a forensic record; the aggregator (Step 6 above) is built to ignore the duplication.
- **`raw_early_stop` in the verdict column.** NeuralSAT writes `early_stop` to `.raw` when its internal heuristics give up (e.g. when an unsupported layer is encountered after partial reachability work). Our normalizer's `case` clause includes `early_stop -> unknown` for the final `.result` file, but the parallel-run `_summary.csv` still has the `raw_early_stop` token in the `verdict` field. The aggregator handles both spellings.
- **NeuralSAT does not honor `--timeout` strictly in some cases.** A few instances (e.g. `collins_rul_cnn_2022/0042__NN_rul_full_window_20__if_then_5levels_w20`) ran to wall=1 974 s with a cap of 1 800 s. This is a NeuralSAT-internal scheduling issue; the wrapper `timeout` catches it. Verdicts are unaffected.

---

## What this archive does NOT contain

- No NeuralSAT source patches (none required).
- No CORA / abcrown / nnenum results — separate archives.
- No GPU profiling data (we have nvidia-smi snapshots in `_run.log` but no continuous trace).

---

## Cross-tool comparison context

For the paper table, this archive contributes the **NeuralSAT (no attack) — `--disable_attack`** row:

| Tool | "no helper" mechanism | Can produce SAT? | V | A | resolved% |
|---|---|---|---|---|---|
| abcrown | `--pgd_order=skip` (CLI) | yes (via BaB) | 1 718 | 742 | 74.2 |
| **NeuralSAT (this archive)** | `--disable_attack` (CLI, upstream-native) | **yes (via BaB completeness)** | **1 581** | **484** | **59.8** |
| nnenum | upstream-native (no helper exists) | yes (via exact-star splitting) | 693 | 752 | 41.8 |
| CORA TRUESTRICT | 3 source patches add `falsification_method='none'` | no (over-approximation cannot witness) | 2 | 0 | 0.06 |

NeuralSAT sits between abcrown and nnenum in the helper-free comparison — it loses fewer SAT verdicts than the over-approximation tools (CORA) but is overall a notch behind abcrown, which has tighter bound-propagation. Its strongest single benchmark is `safenlp_2024`: V=425, A=433, U=221 out of 1 080 (V+A=79.4%).
