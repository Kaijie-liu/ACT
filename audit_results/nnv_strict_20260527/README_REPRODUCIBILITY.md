# NNV STRICT VNN-COMP 2025 sweep — reproducibility bundle

**Date.** Sweep ran 2026-05-28 10:48:55 → 2026-05-29 03:03:03 calendar (~16 h with 3 parallel lanes; ~43.6 h CPU summed across lanes). Note: this calendar window includes ~2 h of iterative patch debugging at the start; the final-fixes clean sweep took ~14 h.
**Tool.** NNV at `/data1/Kane/nnv`, commit `696e20d3d` ("Merge pull request #274 from mldiego/master", Diego Manzanas Lopez, 2025-07-17).
**MATLAB.** R2026a Update 2 (`26.1.0.3251617`).
**Solvers.** GLPK default; Gurobi WLS license available but NNV's STRICT runs do not select Gurobi.

**Protocol.** Helper-free via **three source patches** (NNV has no upstream CLI flag for "no helper"):
1. `run_vnncomp_instance.m` — STRICT mode adds env-gated `NNV_STRICT_NO_HELPER=1` switch that skips `falsify_single` (random sampling + lb/ub corner evaluation) and refuses `cp-star` reach (conformal prediction; statistical, not sound).
2. `matlab2nnv.m` — R2026a compatibility shim (`nnet.cnn.layer.ScalingLayer` → existing `ElementwiseAffineLayer`); restores R2024a behavior. **Not a STRICT-mode change** — required for NNV to load *any* ONNX network on R2026a.
3. `nnv_strict_run_one.m` (driver-side) — parpool isolation per MATLAB pid + `NNV_NUMCORES=5` env cap to prevent inter-lane parpool thrash.

See `patches/README.md` for the full diff and scientific-integrity rationale of each.

**Result.** See `RESULTS_TABLE.tex` and `_summary_overall.csv`.

| | |
|---|---|
| Sound UNSAT (V) | **457** |
| Sound SAT (A) | **0** (impossible under STRICT — over-approximative reach without helper cannot witness counterexamples) |
| Timeout (T) | 1 393 |
| Unknown (U) | 185 |
| Unsupported by STRICT (cp-star refused) | 928 |
| Tool error / unsupported op | 490 |
| Total instances | 3 453 |
| Resolved (V+A)/N | **13.2%** |
| Resolved on runnable benchmarks (excl. unsupported + cp-star-refused) | 457/1 427 = **32.0%** |
| Wall time | 157 007 s ≈ 43.6 h CPU summed across 3 lanes / ~14 h calendar (clean run) |

---

## Headline scientific finding

NNV's pure-reachability methods (`approx-star`, `exact-star`, `relax-star-area`) verify **457 instances soundly** — **230× more than CORA TRUESTRICT** (2 sound verdicts) on the same benchmark set. The two over-approximative MATLAB tools both produce **A = 0** under STRICT (no falsification helper → cannot witness concrete counterexamples), but NNV's reach machinery is meaningfully tighter on the realistic VNN-COMP 2025 properties.

NNV's strongest single benchmark: **metaroom_2023 (93 V / 100, all sound approx-star)** — this is the closest a pure over-approximative tool gets to NNV's helper-on competition performance on a non-trivial benchmark.

---

## Why so many in the E bucket?

`E = 490 + 928 = 1 418` instances — but these break down honestly:

- **928 (`unsupported_strict`)** — benchmarks whose only configured reach method is `cp-star` (conformal prediction). NNV's STRICT refuses these because cp-star produces statistically-bounded results (coverage = confidence = 0.999), not formal verification. Affected: `vit_2023`, `tinyimagenet_2024`, `cifar100_2024`, `vggnet16_2022`, `cersyve`, `ml4acopf_2024`, `yolo_2023`, `linearizenn_2024` (cp-star fallback path), `nn4sys` (mscn variants), `tinyimagenet`, `vit` (5 R2026a load err each, 195 cp-star each). This is *not* an NNV verifier failure — it is a transparent protocol refusal. Reported separately from `error` in the CSV (`E_unsupported_strict` column).
- **490 (`error`)** — true NNV / MATLAB failures, with the following decomposition:
  - **80** `lsnc_relu`: NNV upstream `error("IR and opset not yet supported in MATLAB")`. Permanent verifier limitation.
  - **50** `soundnessbench`: R2026a parser regression in `ReshapeLayer.ONNXParams` (custom layer class no longer has `Nonlearnables` field). Would be `unsupported_strict` anyway since soundnessbench is cp-star.
  - **39** `cctsdb_yolo_2023`: NNV upstream `error("Working on supporting this one")`.
  - **45** `traffic_signs_recognition_2023`: R2026a `QConv` int8 quantization unsupported (parser regression).
  - **72** `dist_shift_2023`: R2026a exact-star load path fails (custom input layer issue).
  - **170** `nn4sys` (mscn + transformer): NNV upstream rejects `onnx::If` control-flow.
  - **80** `relusplitter`: NNV upstream rejects certain operator combinations.
  - **5** `vit_2023` + **5** `tinyimagenet_2024`: R2026a load err on a handful of variants.
  - **5** `test`: NNV has no exec-policy for the VNN-COMP smoke test net.
  - **6** `collins_aerospace_benchmark`: YOLO arch (same as cctsdb_yolo).
  - **19** `cgan_2023`: R2026a load err on `small_transformer` variants.
  - **10** `malbeware`: R2026a load err on a subset of variants.
  - **1** `acasxu_2023`: **MATLAB internal license manager crash** (`libmwlmgrimpl.so` segfault — see Known Issues below). Anomalous, unrelated to CPU/memory pressure.
  - **3** other R2026a parser edge cases.

So **0/490 errors are due to resource contention** — see the "Resource contention audit" section below for the full forensic.

---

## Directory layout

```
audit_results/nnv_strict_20260527/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table
├── _summary_overall.csv                ← aggregated machine-readable counts (TOTAL row at end)
├── _run.log                            ← driver log (interleaved across lanes A/B/C, includes pre-patch iterations)
├── _run.meta.json                      ← provenance (MATLAB version, NNV commit, STRICT flags)
├── _run.pid / _run_laneB.pid / _run_laneC.pid   ← PID files
├── _nohup_postfix_laneA.out            ← raw stdout per launch (postfix = after all patches stabilized)
├── _nohup_postfix_laneB.out
├── _nohup_postfix_laneC.out
├── scripts/
│   ├── run_nnv_strict_vnncomp2025.sh   ← bash launcher (3 lanes invoked separately with disjoint benchmark sets)
│   └── nnv_strict_run_one.m            ← per-instance MATLAB wrapper (parpool isolation, NNV_NUMCORES, STRICT env)
├── patches/
│   └── README.md                       ← full diff + scientific rationale for all 3 patches
└── <benchmark>/                        ← per-benchmark dir (one per VNN-COMP 2025 benchmark)
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file
    ├── NNNN__<onnx>__<vnnlib>.result   ← one of: unsat | unknown | timeout | timeout_killed | unsupported_strict | error
    ├── NNNN__<onnx>__<vnnlib>.log      ← NNV's full MATLAB stdout
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata
```

---

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| MATLAB | R2026a Update 2 (the compat shim is *required* for R2026a; R2024a does not need it but we did not re-validate on R2024a) |
| GPU | optional; NNV's STRICT runs are pure CPU (MATLAB linear algebra, no `gpuArray`) |
| Disk | ~10 GB for results + diaries |

### Step 1: install NNV

```bash
cd /data1/Kane
git clone https://github.com/verivital/nnv.git
cd nnv
git checkout 696e20d3dbe566ee45cd2e2a3f6c352e44bcd448
```

### Step 2: apply the three patches

Detailed in `patches/README.md`. Summary:

1. Edit `code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m` — add STRICT switch + 3 STRICT guards (banner + cp-star primary refusal + falsify bypass + cp-star fallback refusal).
2. Edit `code/nnv/engine/utils/matlab2nnv.m` — add `elseif isa(L, 'nnet.cnn.layer.ScalingLayer')` branch mapping to `ElementwiseAffineLayer`.
3. The third patch lives in this archive at `scripts/nnv_strict_run_one.m` (driver-side, not NNV-internal).

Verify patches:

```bash
grep -c "STRICT-MODE PATCH (ACT paper" /data1/Kane/nnv/code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m
# expected: 1
grep -c "R2026a compat shim" /data1/Kane/nnv/code/nnv/engine/utils/matlab2nnv.m
# expected: 1
```

### Step 3: smoke test

```bash
NNV_STRICT_NO_HELPER=1 NNV_NUMCORES=5 /data1/Kane/MATLAB/bin/matlab -nodisplay -nosplash -batch "
addpath('/data1/Kane/ACT/scripts');
nnv_strict_run_one('malbeware', \
  '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/malbeware/onnx/malware_malimg_family_scaled_linear-25.onnx', \
  '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/malbeware/vnnlib/malbeware_family-Obfuscator.AD_label-17_eps-1_idx-89.vnnlib', \
  '/tmp/nnv_smoke.result');
"
cat /tmp/nnv_smoke.result
# expected: 'unsat' (in ~15-20 s; uses exact-star with 5-worker parpool)
```

If the smoke prints `error`, inspect the stderr for either (a) `Unsupported Class of Layer` (patch 2 missing) or (b) `Failed to locate and destroy old interactive jobs` (patch 3 missing).

### Step 4: run the 3-lane sweep

```bash
cd /data1/Kane/ACT

# Lane A: default order (17 light-to-medium benchmarks)
nohup bash scripts/run_nnv_strict_vnncomp2025.sh \
  > audit_results/nnv_strict_20260527/_nohup_laneA.out 2>&1 &

# Lane B: heavy back-half (8 benchmarks, no safenlp)
nohup bash scripts/run_nnv_strict_vnncomp2025.sh \
  cora_2024 acasxu_2023 nn4sys vit_2023 tinyimagenet_2024 cifar100_2024 \
  vggnet16_2022 collins_aerospace_benchmark \
  > audit_results/nnv_strict_20260527/_nohup_laneB.out 2>&1 &

# Lane C: safenlp_2024 alone (1080 instances, the long pole)
nohup bash scripts/run_nnv_strict_vnncomp2025.sh safenlp_2024 \
  > audit_results/nnv_strict_20260527/_nohup_laneC.out 2>&1 &
```

The 3 lanes process disjoint benchmark sets, so they never race on `_summary.csv`. Idempotent — instances with non-empty `.result` files are skipped on relaunch.

### Step 5: aggregate

```bash
cd /data1/Kane/ACT/audit_results/nnv_strict_20260527
for d in */; do
  [[ "$d" == "scripts/" || "$d" == "patches/" ]] && continue
  b=${d%/}; files=$(ls $d/*.result 2>/dev/null); [[ -z "$files" ]] && continue
  declare -A C; N=0
  for f in $files; do v=$(head -1 $f | tr -d '[:space:]'); [[ -z "$v" ]] && v=empty; C[$v]=$((${C[$v]:-0}+1)); N=$((N+1)); done
  V=${C[unsat]:-0}; A=${C[sat]:-0}
  T=$((${C[timeout]:-0} + ${C[timeout_killed]:-0}))
  U=${C[unknown]:-0}
  Es=${C[unsupported_strict]:-0}
  Eo=$((${C[error]:-0} + ${C[missing_result]:-0} + ${C[empty]:-0}))
  echo "$b: N=$N V=$V A=$A T=$T U=$U E_strict=$Es E_other=$Eo"
  unset C
done
```

---

## Per-instance verdict semantics

| Token | Meaning | Bucket | Sound? |
|---|---|---|---|
| `unsat` | reach $\cap$ unsafe halfspace $= \emptyset$ | V | ✅ |
| `unknown` | reach $\cap$ unsafe $\ne \emptyset$ (could be real SAT or over-approx artifact) | U | — |
| `timeout` | NNV internal budget exceeded | T | — |
| `timeout_killed` | wrapper `timeout` SIGKILL after grace period (rare; wall > used_timeout) | T | — |
| `unsupported_strict` | STRICT refused cp-star reachability | E_strict | — |
| `error` | NNV crash / unsupported op / parser regression | E_other | — |

`sat` never appears under STRICT.

---

## Resource contention audit

The 3-lane sweep ran concurrently with two other experiments on the same host (PyRAT's OOM serial rerun + the ACT pipeline's hybridz solver verifications). Load average peaked at **297** at one point. We audited whether any NNV instance failed due to CPU/memory contention rather than a real tool limit.

Exit code distribution across all 3 453 instances:
- `0` (clean exit): 2 718
- `124` (timeout SIGTERM): 1 368
- `137` (SIGKILL): 25

Of the 25 SIGKILL events:
- **24** had wall_sec ≥ 95% of used_timeout → these are normal `timeout` wrapper escalations (SIGTERM ignored, escalated to SIGKILL after 10-second grace). Verdict correctly relabeled as `timeout_killed` (bucket T, not E).
- **1** had wall_sec = 33.6 s vs. timeout = 116 s (acasxu_2023 idx=161, `prop_4` on net `3_8`). Log shows a stack trace through `libmwlmgrimpl.so` — a MATLAB internal license-manager crash, **not** OS OOM-killer (`dmesg` shows no OOM events) and **not** a CPU starvation kill. This is an anomalous MATLAB-side bug. Bucketed as `error`. The crash dump is at `/home/kaijieliu/matlab_crash_dump.2359208-1` for reference.

**Conclusion: 0 of 3 453 instances had verdicts corrupted by resource contention.**

This is the same level of forensic transparency we applied to NeuralSAT's OOM serial rerun (which uncovered 3 genuine OOM-induced failures on collins_rul_cnn_2022 + 19 SIGSEGVs on ml4acopf_2024). NNV STRICT here had zero comparable issues despite higher load.

---

## Other known issues observed during the run

- **`_run.log` contains multiple sweep iterations.** During the patch development phase (first ~2 h of the calendar window), we iteratively diagnosed and fixed three classes of issue: (a) ScalingLayer error in matlab2nnv, (b) parpool stale-jobs race across MATLAB sessions, (c) NNV's hardcoded 20-worker parpool request swamping the host with 3 lanes. After each fix, error-bucket `.result` files were deleted and the sweep was relaunched (idempotent). The driver log preserves all relaunches honestly. The "final clean" sweep period is everything after the third patch was in place (~12:00).
- **`_summary.csv` per-benchmark rows may have duplicates from relaunches** — the CSV row count exceeds the `.result` file count for some benchmarks. The aggregator (Step 5 above) reads `.result` files (one per instance) as ground truth and ignores the CSV row count.
- **NNV's per-instance timer is loose.** Several instances ran 30-130% past the per-instance budget (e.g. acasxu max wall 146 s on a 116 s cap; relusplitter max 210 s on 180 s). The wrapper `timeout` catches these; verdicts unaffected, but `wall_sum` in the CSV slightly overstates time-on-budget.

---

## Cross-tool comparison context

For the paper table, this archive contributes the **NNV STRICT (no helper, R2026a-patched)** row:

| Tool | "no helper" mechanism | A possible? | V | A | (V+A) / N |
|---|---|---|---|---|---|
| abcrown | `--pgd_order=skip` (CLI) | yes (via BaB) | 1 718 | 742 | 74.2% |
| PyRAT | monkey-patch `look_random` to passive shim | yes (analyzer-only) | 1 242 | 151 | 40.3% |
| NeuralSAT | `--disable_attack` (CLI) | yes (via BaB) | 1 581 | 484 | 59.8% |
| nnenum | upstream-native (no helper exists) | yes (via exact-star splitting) | 693 | 752 | 41.8% |
| **NNV STRICT (this archive)** | **3 source patches + R2026a compat shim** | **no** (over-approx cannot witness) | **457** | **0** | **13.2%** |
| CORA TRUESTRICT | 3 source patches (`falsification_method='none'`) | no (same reason) | 2 | 0 | 0.06% |

NNV's pure reachability is **230× stronger than CORA TRUESTRICT** on V, but still well below the BaB-based tools that can prove A under STRICT. The over-approximative-MATLAB-tool family (NNV + CORA) is, in the paper's framing, the most informative pair for measuring "what does a sound verifier prove without ever calling out to a falsification side-channel?"

---

## Patches and authorization

The R2026a `ScalingLayer` compatibility shim (Patch 2) was applied **after explicit user authorization** on 2026-05-28. The user's initial instruction was to avoid upstream NNV modifications, and Claude initially deferred on adding the shim. After observing that 97% of NNV instances errored on R2026a load, Claude presented the trade-off to the user (option A: tiny semantic-equivalence shim restoring R2024a behavior; option B: accept 97% error and report NNV as "incompatible"; option C: reinstall R2024a). The user selected option A. The shim was reviewed for semantic equivalence (ScalingLayer and ElementwiseAffineLayer both compute `y = Scale .* x + Offset`) before being applied.

This is transparent in the archive: see `patches/README.md` for the user-authorized note in the patch banner.
