# ACT CPU / GPU Run Index — r93_rerun branch

**Built**: 2026-05-26 (UTC)
**Owner location**: `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS/`
**Purpose**: single source of truth for every CPU and GPU run on the r93 code
snapshot. Future GPU re-validation work reads from here. Do NOT delete the
upstream source dirs — every per-benchmark folder here contains *symlinks*
(`_source_<label>`) back to the authoritative per-instance JSONs; the CSVs are
derived views.

## Per-benchmark layout

Each `<bench>/` subdir contains:

- `_source_cpu`            — symlink to authoritative CPU run dir
- `_source_gpu`            — symlink to authoritative GPU run dir (if any)
- `_source_cpu2`           — secondary CPU run (e.g. overnight round 2 subset)
- `_source_cpu_<variant>`  — alternative CPU configs (specaware / auto / witness / R9 / passes3)
- `_source_sidecar`        — SATSidecar-bridged path (cersyve only)
- `per_instance.csv`       — one final-outcome row per `(source, official iid)`, including watchdog-bounded outcomes and source receipt fields

## Status × benchmark table (paper-grade, derived from per_instance.csv)

Formal-run format: V / A / U / E / N — CERTIFIED / FALSIFIED / UNKNOWN / ERROR / total.
Bounded-smoke format: V / A / U / TO / RSS / E / N, where `TO` and `RSS`
are authoritative watchdog `UNKNOWN_TIMEOUT` and `UNKNOWN_RESOURCE_LIMIT`
records and are not semantic verdicts.

### A. CPU + GPU both run, **bit-identical, formal-qualified**

| Benchmark | CPU (V/A/U/E/N) | GPU (V/A/U/E/N) | Source CPU dir | Source GPU dir |
|---|---|---|---|---|
| `collins_rul_cnn_2022` | **39/11/12/0/62** | **39/11/12/0/62** | `collins_full62_v2_20260525T120407Z` | `collins_gpu_full62_cudafix_20260525` |
| `malbeware` | **123/13/14/0/150** | **123/13/14/0/150** | `malbeware_R93` | `malbeware_gpu_full150_cudafix_20260525` |
| `acasxu_2023` (`cpu_auto`) | **73/15/98/0/186** | **73/15/98/0/186** | `capability_rebaseline_…/acasxu_C_auto` | `acasxu_gpu186_t30_20260525T154939Z` |
| `linearizenn_2024` (`cpu_R9`) | **13/0/47/0/60** | **13/0/47/0/60** | `capability_rebaseline_…/linearizenn_R9` | `linearizenn_gpu60_20260525T144623Z` |

### B. CPU + GPU both run, **divergent — do NOT use GPU counts**

| Benchmark | CPU (V/A/U/E/N) | GPU (V/A/U/E/N) | Diff iids | Status |
|---|---|---|---|---|
| `sat_relu` | **1/18/81/0/100** | **1/21/78/0/100** | 5 stable (34, 50, 56, 86, 92) | Investigate before GPU promotion |

### C. CPU formal-qualified, **no GPU run yet**

| Benchmark | CPU (V/A/U/E/N) | Source CPU dir | GPU candidate |
|---|---|---|---|
| `tllverifybench_2023` (`cpu_witness`) | **1/2/29/0/32** | `capability_rebaseline_…/tllverify_witness` | low priority (LP-bound) |
| `dist_shift_2023` | **0/0/72/0/72** | `capability_rebaseline_…/dist_shift_2023` | not useful (no decisions yet) |
| `safenlp_2024` (`cpu_auto`) | **333/10/737/0/1080** | `capability_rebaseline_…/safenlp_B_auto` | timing study candidate |

### D. Overnight CPU serial Round 1 (2026-05-25 14:42 → 15:38, ~57 min)

| Benchmark | CPU (V/A/U/TO/RSS/E/N) | Source dir | GPU TODO |
|---|---|---|---|
| `lsnc_relu` | **0/0/78/2/0/0/80** | `overnight_cpu_20260525T144203Z/lsnc_relu` | low priority (0 decisions) |
| `ml4acopf_2024` | **5/0/54/10/0/0/69** | `overnight_cpu_20260525T144203Z/ml4acopf_2024` | optional after scaling work |
| `nn4sys` (iids 0–49 + 105,106) | **2/0/50/0/0/0/52** | `overnight_cpu_20260525T144203Z/nn4sys` | strict watchdog only |
| `relusplitter` (iids 0..29) | **3/0/21/6/0/0/30** | `overnight_cpu_20260525T144203Z/relusplitter` | bounded expansion completed in Round 2 |
| `yolo_2023` (iids 0..9) | **0/0/0/0/10/0/10** | `overnight_cpu_20260525T144203Z/yolo_2023` | needs memory/scaling work |

### E. Overnight CPU serial Round 2 (2026-05-25 22:52:41 → 2026-05-26 01:27:20 UTC, complete)

| Benchmark | Range | CPU (V/A/U/TO/RSS/E/N) | Source dir |
|---|---|---|---|
| `nn4sys` | iids 50..104 | **0/0/55/0/0/0/55** | `overnight_cpu_round2_…/nn4sys` |
| `relusplitter` | iids 30..219 | **4/0/21/150/15/0/190** | `overnight_cpu_round2_…/relusplitter` |
| `cersyve` (native) | iids 0..11 | **0/0/12/0/0/0/12** | `overnight_cpu_round2_…/cersyve` |
| `cora_2024` smoke | iids 0..9 | **1/0/1/8/0/0/10** | `overnight_cpu_round2_…/cora_2024` |
| `soundnessbench` smoke | iids 0..9 | **0/0/0/0/10/0/10** | `overnight_cpu_round2_…/soundnessbench` |
| `collins_aerospace_benchmark` | iids 0..5 | **0/0/3/3/0/0/6** | `overnight_cpu_round2_…/collins_aerospace_benchmark` |
| `traffic_signs_recognition_2023` smoke | iids 0..4 | **0/0/5/0/0/0/5** | `overnight_cpu_round2_…/traffic_signs_recognition_2023` |
| `vggnet16_2022` smoke | iids 0..4 | **0/0/0/5/0/0/5** | `overnight_cpu_round2_…/vggnet16_2022` |
| `metaroom_2023` smoke | iids 0..4 | **2/0/0/0/3/0/5** | `overnight_cpu_round2_…/metaroom_2023` |
| `cifar100_2024` smoke | iids 0..4 | **0/0/4/1/0/0/5** | `overnight_cpu_round2_…/cifar100_2024` |
| `tinyimagenet_2024` smoke | iids 0..4 | **0/0/0/0/5/0/5** | `overnight_cpu_round2_…/tinyimagenet_2024` |

Round 2 contains **308 attempted instances, 0 ERROR, 7 CERTIFIED**, with every
bounded termination retained in its per-benchmark CSV.  `cora_2024` iid 8 and
`metaroom_2023` iids 1 and 4 are CERT smokes only; they do not promote either
benchmark to formal-qualified status.

### F. Sidecar-only (not native ACT, but strict-clean receipts)

| Benchmark | Verdicts | Source dir |
|---|---|---|
| `cersyve` (3 strict FAL via SATSidecar bridge) | 3 FAL (iids 1, 5, 9) | `cersyve_strict_fal/` |

### G. Still blocked by a conversion/design issue

- `cgan_2023` — post-fix smoke >1 h, ~18.6 GiB RSS on first inst
- `cctsdb_yolo_2023` — OnnxSlice dynamic starts/ends; rank-1 input `[12296]`; R17 LUT_BOUNDS scaffold ready, converter detector not implemented
- `vit_2023` — flatten output numel 75 ≠ expected 5 (shape lineage gap)

### H. Bounded CPU smoke observed, not formal-qualified

- Resource/time ceiling only: `vggnet16_2022` (5/5 TO),
  `soundnessbench` (10/10 RSS), `tinyimagenet_2024` (5/5 RSS).
- Completed UNKNOWN-only or mixed bounded smoke: `traffic_signs_recognition_2023`
  (5 UNKNOWN), `cifar100_2024` (4 UNKNOWN + 1 TO),
  `collins_aerospace_benchmark` (3 UNKNOWN + 3 TO).
- New decisive smoke results requiring later full qualification:
  `cora_2024` (1 CERT + 1 UNKNOWN + 8 TO) and
  `metaroom_2023` (2 CERT + 3 RSS). Their CERT rows agree with the official
  `zero_tol` and `small_tol` UNSAT labels.

## Diagnostic notes baked into the data

These divergences inside the per-benchmark CSVs are intentional — they record
the migration path, not errors:

- **`collins_rul_cnn_2022/cpu_rebase`** (62 × `ERROR_RuntimeError`) — pre-R-fix
  buggy run from the May-24 rebaseline. Replaced by `_source_cpu` (the
  post-fix `collins_full62_v2`). Keep for paper-trail of the fix.
- **`linearizenn_2024/cpu_witness`** (60 × ERROR) — same: pre-fix. Authoritative
  CPU is `_source_cpu_R9`.
- **`malbeware/cpu_rebase`** (100 inst, partial) — earlier rebaseline subset
  (not full 150). Authoritative CPU is `_source_cpu` (malbeware_R93).
- **`cersyve/cpu_rebase`** (12 × ERROR) — pre-R16 ready-check; closed since.
- **`cersyve/cpu`** (3 rows, verdict='?') — sidecar bridge output; verdicts are
  in the receipts under `_source_sidecar`. The 3 strict-clean FAL are real.
- **`sat_relu/cpu_witness`** (1 V / 49 FAL / 50 UNK) — a *different config* from
  the authoritative `cpu` (1 V / 18 FAL / 81 UNK). The 49-FAL number is NOT
  the canonical CPU result; it's a witness-extraction sidecar variant.
- **`safenlp_2024/cpu_passes3`** (284 V) — earlier pass=3 variant. Authoritative
  is `cpu_auto` (333 V).
- **`acasxu_2023/cpu_base`** (61 V, no FAL) and **`cpu_specaware`** (73 V, no
  FAL) — config sweep on the same code. Authoritative is `cpu_auto`, which
  matches GPU exactly (73 V + 15 FAL).

## Per-benchmark CSV schema

```
source,iid,verdict,internal_status,reportable_status,count_bucket,wall_s,
watchdog_status,watchdog_synthetic,strict_bounded_failure,run_status,
peak_rss_mb,onnx_model,vnnlib_spec,q_statuses,q_reportables,q_receipts,
error,json_path
```

- `source`: which `_source_*` symlink the row came from (cpu / gpu / cpu_auto / ...)
- `iid`: instance index
- `verdict`: `CERTIFIED` / `FALSIFIED` / `UNKNOWN` /
  `UNKNOWN_TIMEOUT` / `UNKNOWN_RESOURCE_LIMIT` / `ERROR_*` /
  `RECEIPT_ONLY` (sidecar)
- `wall_s`: per-instance wall time
- `watchdog_synthetic`: true when the row is the authoritative bounded outcome
  substituted for a killed in-flight child
- `q_statuses`, `q_reportables`, `q_receipts`: serialized query-level details
- `json_path`: absolute path to the selected final record JSON

`build_csvs.py` treats watchdog synthetic rows as authoritative for their iid,
so completed-child rows written before a forced kill cannot incorrectly appear
as normal UNKNOWN/CERT/FAL results.

## Official Label Cross-check

- Executable audit: `soundness_check.py`
- Machine-readable summary: `OFFICIAL_CROSSCHECK_SUMMARY.json`
- Row-level disagreements: `OFFICIAL_CROSSCHECK_DISAGREEMENTS.csv`
- Interpretation and receipt notes: `SOUNDNESS_VS_VNNCOMP_OFFICIAL.md`

The current audit includes the Round 2 `cora_2024`, `metaroom_2023`,
`cifar100_2024`, `tinyimagenet_2024`, `soundnessbench`, and native `cersyve`
sources wherever an official SAT/UNSAT longtable label exists. All newly observed CERT
rows agree with official labels.  `collins_rul_cnn_2022` retains nominal
official-label disagreements that are documented separately with strict ACT
receipt evidence; they must be reported as discrepancies rather than silently
collapsed.

## How to use this index later

**Re-running GPU on a benchmark currently CPU-only**:
1. Cite the CPU source from the table above (column "Source CPU dir").
2. Use the same code snapshot (r93_rerun = current `act` HEAD; check
   `git log -1` if uncertain).
3. Write GPU output to a new dir under `r93_rerun_20260525T083118Z/`
   following pattern `<bench>_gpu<N>_cudafix_<date>/`.
4. After completion, add a `_source_gpu` symlink to the new dir under
   `CONSOLIDATED_RESULTS/<bench>/` and re-run the build script (below).

**Building per-instance CSVs after adding new sources**:
```bash
cd /data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS
python3 build_csvs.py    # re-aggregates from all _source_* symlinks
```

**Verifying CPU/GPU bit-identity for a benchmark**:
```python
import pandas as pd
df = pd.read_csv("<bench>/per_instance.csv")
cpu = df[df.source == "cpu"].set_index(["iid","query_idx"]).verdict
gpu = df[df.source == "gpu"].set_index(["iid","query_idx"]).verdict
diffs = cpu.compare(gpu)
print(f"divergent rows: {len(diffs)}")
print(diffs)
```

## Upstream pointers

- Code snapshot: r93 = current `act` HEAD on machine; `act.pipeline.watchdog_runner`,
  `act.back_end.analyze` (R16 ready-check), `act.back_end.layer_schema`
  (FLOOR/CEIL/ROUND/SIN/COS/LUT_BOUNDS LayerKinds)
- Canonical VNNLIB root: `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks`
- Run scripts: `/data1/Kane/ACT/scripts/overnight_cpu_serial.sh` (Round 1),
  `/data1/Kane/ACT/scripts/overnight_cpu_serial_round2.sh` (Round 2)
- Support matrix: `r93_rerun_…/SCORED_BENCHMARK_SUPPORT_MATRIX.md`
- Morning report: `r93_rerun_…/MORNING_REPORT.md`

---

## I. Round 4 / 5 / 6 — ACT-side fixes + GPU full sweep + OOM rerun (2026-05-26 → 27)

### I.1 Code fixes (saved as `act_fixes_diff/*.patch`)

| # | File | Bug | Fix | Patch |
|---|---|---|---|---|
| 5 | `onnx_converter.py:convert_onnx_to_pytorch` | Round-4 "raw-first" strategy regressed ml4acopf/lsnc_relu/yolo/collins_aero (skipping `onnxsim.simplify` left their shape annotations dirty) | Swap to **simplify-first, raw fallback** — preserves nn4sys mscn fix while restoring everything else | `01_onnx_converter.patch` |
| 6 | `interval_tf/tf_mlp.py:tf_lrelu` + `solver/solver_hz.py:lrelu dispatch` | Hard `params["alpha"]` lookup crashed on PyTorch-converted LeakyReLU (key is `negative_slope`) | Accept either key, default to 0.01 (ONNX/torch default) | `03_leaky_relu_alpha_keyerror.patch` |
| 7 | `interval_tf/tf_cnn.py:tf_upsample` | ONNX Resize/Upsample carries 4D `scale_factor=(1,1,2,2)` but `F.interpolate` expects only spatial scales | Strip leading non-spatial dims (verified all 1.0), pass only `(2,2)` to interpolate | `04_upsample_strip_nc_scale.patch` |

All three preserve soundness; verified end-to-end on representative iids (see
each patch's accompanying note in `nn4sys_lindex200_FIXES.md` and
`ROUND5_FIX_AND_RERUN.md`).

### I.2 GPU full-sweep (2026-05-26 13:24 → 20:30 UTC, 3 parallel streams)

Sources: `gpu_stream{1,2,3}_20260526T132419Z/*/`

| Benchmark | CPU baseline | GPU result | bit-identity vs CPU |
|---|---|---|---|
| safenlp_2024 | 333V/10A/737U | **333V/10A/737U** | ✅ 1080/1080 identical |
| tllverifybench_2023 | 1V/2A/29U | **1V/2A/29U** | ✅ 32/32 identical |
| dist_shift_2023 | 72 UNK | 72 UNK | ✅ identical |
| cersyve | 12 UNK | 12 UNK | ✅ identical |
| **metaroom_2023** | 37 CERT (CPU full) | **87 CERT** | +50 GPU-only CERTs (CPU was RSS-cap; GPU 96 GiB fits HZ); all match official UNSAT |
| cora_2024 | 15 CERT | 16 CERT + **4 FAL** | +1 CERT, +4 new FAL (all ORT-clean, match official SAT) |
| ml4acopf_2024 | 5 CERT | 4-6 CERT | bit-identical at CERT level; UNK distribution differs |
| relusplitter | 7 CERT | 7 CERT | match |
| nn4sys (post-Round-4 fix) | 4 CERT | 4 CERT | match |
| traffic_signs_recognition_2023 | 30 UNK + 15 RSS | 45 UNK | GPU unlocks (no RSS-cap) |
| vggnet16_2022 | 18 RSS | 18 TIMEOUT | bounded UNK both sides |
| yolo_2023 | 72 RSS | 72 UNK | GPU unlocks; runs but no decisions |
| **tinyimagenet_2024** | 200 RSS | **1 FAL** + 197 UNK + 2 ERR | GPU finds 1 new FAL; iid 6 is a **small_tol-residual disagreement** (official zero_tol=UNSAT, small_tol=SAT, ACT FAL strict-clean) |
| cifar100_2024 | 99 UNK + 100 RSS | 121 UNK + 79 OOM | OOM during contention window — all rerun later |
| cgan_2023 | 5 smoke TO | 21 TO | GPU got past `small_transformer`; needs more wall |

Pre-Fix-#5 4 benchmarks (lsnc_relu / ml4acopf / yolo / collins_aero) failed
GPU sweep with 100% ERROR — those were rerun in Round 5 (below) after the fix.

### I.3 Round 5 — re-run after Fix #5/#6 (2026-05-26 23:43 → 2026-05-27 02:48 UTC)

Sources: `round5_aftersimplify_20260526T234325Z/*` (CPU+CUDA per bench) +
`round5_cuda_20260526T234657Z/*` + `round5_collins_aero_postfix6_*/` +
`round5_yolo_cuda_catchup_20260527T025827Z/` (yolo CUDA finished 32/72 in
the original due to a watchdog 2s-timeout bug; catchup ran iids 32..71).

| Benchmark | CPU (V/A/U/E) | CUDA (V/A/U/E) | Notes |
|---|---|---|---|
| ml4acopf_2024 | 4V/0A/65U/0E (69) | 6V/0A/63U/0E (69)¹ | (+1 OOM during contention, recovered by OOM rerun) |
| lsnc_relu | 0V/0A/80U/0E (80) | 0V/0A/80U/0E (80) | bit-identical, all UNK |
| yolo_2023 | 0V/0A/0U/0E + 72 RSS-cap | 0V/0A/72U/0E (72) | GPU has the RAM CPU lacks |
| collins_aerospace | 0V/0A/2U/4E (6, 4 ValueError from Upsample bug — fixed in Fix #7) | 0V/0A/6U/0E (6, post-Fix-#6) | GPU now error-free |

¹ ml4acopf_2024 has no entry in the official `longtable.tex`, so individual
CERT iids cannot be cross-checked. The 6 CERT (iids 21, 37, 44, 46, 52, 58)
all carry strict ACT receipts.

### I.4 OOM rerun (2026-05-27 01:26 → 02:11 UTC)

Source: `oom_rerun_20260527T012601Z/*` — all 87 GPU OOM iids re-attempted
on an empty GPU. **Result: 0 OOM, 0 ERROR, +3 new CERT.**

| Benchmark | OOM iids re-attempted | New verdicts |
|---|---|---|
| ml4acopf_2024 | 58, 59, 60 | 1 CERT + 1 UNK + 1 TO |
| metaroom_2023 | 30, 33 | **2 CERT** (both match official UNSAT) |
| tinyimagenet_2024 | 66, 67 | 2 UNK |
| relusplitter | 14 | 1 UNK |
| cifar100_2024 | 79 iids (100..198 range) | 79 UNK |
| **TOTAL** | **87** | **0 OOM, 0 ERROR, +3 CERT** |

The OOM cases were exclusively GPU contention with another user's heavy
job. With the GPU empty, all 87 instances completed cleanly.

### I.5 Status post-Round-5

**Active ERROR status across CONSOLIDATED_RESULTS**:
- 0 ERROR on every benchmark except collins_aerospace CPU (4 ValueError from
  Upsample 4D bug — fixed in Fix #7, not yet re-run).
- Round 5 confirmed Fix #5+#6 work: lsnc_relu/ml4acopf/yolo/collins_aero all
  now run on GPU without ERROR (previously 100% ERROR).

**metaroom_2023 final**: 89 CERT (87 GPU + 2 OOM-rerun) — all match official
UNSAT under both tolerances. Major bit-identity win.

**Soundness audit summary post-Round-5**:
- 0 new official-label conflicts on CERT/FAL agreement rows.
- collins_rul disagreements: still 18 zero_tol / 6 small_tol (well-documented).
- tinyimagenet iid 6 FAL: NEW small_tol-residual disagreement
  (official zero_tol=UNSAT, small_tol=SAT, ACT FAL strict-clean) — disclose
  in paper alongside collins_rul.

**Source naming convention** for new symlinks:
- `_source_round5_cpu` / `_source_round5_cuda` — Round 5 main sweep
- `_source_round5_cuda_pre_fix6` — original CUDA sweep before LeakyReLU fix
- `_source_round5_postfix6` — collins_aero rerun after Fix #6
- `_source_round5_cuda_catchup` — yolo CUDA iids 32..71 after watchdog bug
- `_source_oom_rerun` — 87 OOM iids re-attempted on empty GPU
- `_source_gpu_full` — original GPU full sweep (May 26)

