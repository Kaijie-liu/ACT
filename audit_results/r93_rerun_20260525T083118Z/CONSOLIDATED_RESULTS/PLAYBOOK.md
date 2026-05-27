# ACT r93 CPU / GPU Run Playbook

**Last updated**: 2026-05-26 UTC
**Self-contained location**: `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS/`
**Scope**: everything needed to reproduce, extend, or re-audit the May-25 r93
ACT runs across VNN-COMP 2025's 25 scored benchmarks.

This file is the entry point. Read it top to bottom before touching anything.

---

## 1. What's in this directory

```
CONSOLIDATED_RESULTS/
├── PLAYBOOK.md                            # ← you are here
├── MASTER_INDEX.md                        # benchmark-by-benchmark CPU/GPU status
├── SOUNDNESS_VS_VNNCOMP_OFFICIAL.md       # cross-check vs zero_tol / small_tol
├── SAT_RELU_CPU_GPU_DIVERGENCE_AUDIT.md   # device-divergence audit on sat_relu
├── HZ_VERIFICATION_FLOW.md                # end-to-end ACT hybridz trace + no-sampling audit
├── FINAL_RESULTS_TABLE.md                 # ← LATEST: 25 benchmarks × CPU/GPU with TO/RSS breakdown
├── nn4sys_lindex200_FIXES.md              # ACT bug fixes #1-#4 from Round 4 (mscn family)
├── ROUND5_FIX_AND_RERUN.md                # Fix #5 (simplify-first) + Round 5 rerun
├── OFFICIAL_CROSSCHECK_SUMMARY.json       # machine-readable cross-check
├── OFFICIAL_CROSSCHECK_DISAGREEMENTS.csv  # per-row disagreement detail
├── build_csvs.py                          # rebuild per-benchmark CSV from _source_*
├── soundness_check.py                     # rerun official-label audit
├── scripts/                               # all run scripts (Round 1/2/3/4 + GPU)
└── <bench>/
    ├── per_instance.csv                   # final outcome per (source, iid)
    └── _source_<label>                    # symlinks to authoritative run dirs
```

Layout invariant: never delete an upstream source dir — every `_source_*`
symlink in here points back to one. The per_instance.csv files are derived
views, regenerable from `build_csvs.py`.

---

## 2. Run chronology — what happened, in order

| Round | When (UTC) | Wall | Mode | Instances | Scripts | Notable |
|---|---|---|---|---|---|---|
| Capability rebaseline | 2026-05-24 22:57 onwards | ~12 hr | CPU formal-qualified | ~1900 | (pre-existing) | First clean canonical CPU runs for acasxu/collins_rul/dist_shift/linearizenn/malbeware/safenlp/tllverify/cersyve/sat_relu |
| GPU bit-identity | 2026-05-25 | ~5 hr | GPU formal-qualified | ~458 | (`*_gpu*_cudafix_*` dirs) | 4 benchmarks bit-identical to CPU; sat_relu 5 stable iid diffs |
| Round 1 overnight CPU | 2026-05-25 14:42 → 15:38 | 56 min | strict-watchdog serial | 241 | `overnight_cpu_serial.sh` | First overnight CPU on 5 unseen benchmarks; 0 ERROR; +10 first-time CERTs |
| Round 2 overnight CPU | 2026-05-25 22:52 → 2026-05-26 01:27 | 2h35m | strict-watchdog serial | 308 | `overnight_cpu_serial_round2.sh` | 11 entries (smokes); 0 ERROR; +7 CERTs incl. first metaroom + cora |
| Round 3 sequential (aborted) | 2026-05-26 12:10 → 12:54 | 44 min | strict-watchdog serial | 50 (cora 10..59) | `overnight_cpu_full_coverage.sh` | Killed to switch to parallel; cora partial 50/170 preserved |
| Round 3 parallel | 2026-05-26 02:59 → 06:25 | 3h26m | 3 parallel streams | 917 | `overnight_cpu_full_stream{1,2,3}.sh` | All 9 smoke-only benchmarks fully covered; **+35 metaroom CERTs**; +14 cora CERTs; nn4sys lindex_200+ had 61 ERROR (later fixed in Round 4) |
| sat_relu CPU determinism recheck | 2026-05-26 02:03 | 35s | strict-watchdog serial | 5×2 reruns | (inline) | Confirmed CPU fully deterministic on the 5 CPU/GPU divergent iids |
| Round 4 fix-and-rerun | 2026-05-26 10:51 → 12:33 | 1h42m | 3 parallel | 71 | inline + restored data files | 4 ACT bug fixes; 3 benchmark data files restored from `large_models.zip`; nn4sys 61-ERROR family **→ 0 ERROR + 2 new CERTs** |
| GPU stream sweep | 2026-05-26 13:24 → 20:30 | ~7h | 3 parallel GPU streams + auto-ingest | 2741 | `gpu_stream{1,2,3}_*.sh` + `gpu_auto_ingest.sh` | metaroom +50 CERT (vs CPU); cora +1 CERT +4 FAL; tinyimagenet +1 FAL; bit-identical confirmed on safenlp/tllverify/dist_shift/cersyve |
| Round 5 (Fix #5/#6) | 2026-05-26 23:43 → 2026-05-27 02:48 | 3h05m | 2 parallel (CPU+CUDA dual + CUDA-only) | 666 | `round5_rerun_after_fix.sh` + `round5_cuda_only.sh` | Restored ml4acopf/lsnc/yolo/collins_aero after Round 4 raw-first ONNX regression; Fix #6 LeakyReLU `alpha` key |
| OOM rerun | 2026-05-27 01:26 → 02:11 | 45min | sequential GPU | 87 | `oom_rerun.sh` | All 87 GPU OOM iids retried on empty GPU; **0 OOM reproduced**, +3 CERT (metaroom×2 + ml4acopf×1) |
| yolo CUDA catchup | 2026-05-27 02:58 → 03:25 | 27min | single CUDA | 40 | inline | yolo iids 32..71 (R5 had a watchdog 2s-timeout bug at iid 31) |
| collins_aero CPU post-Fix-#7 | 2026-05-27 03:43 → 03:57 | 14min | single CPU | 6 | inline | Fix #7 Upsample 4D scale_factor; closed all canonical-source ERRORs |

---

## 3. The 7 ACT-side fixes (Round 4 + Round 5/6 + Round 7)

**Round 4 (nn4sys mscn family, 2026-05-26)**:

All applied to `/data1/Kane/ACT/` source (r93 snapshot, code change durable):

| # | File | Bug | Fix | Affected |
|---|---|---|---|---|
| 1 | `act/front_end/vnnlib_loader/onnx_converter.py:convert_onnx_to_pytorch` | Hardcoded opset 10→13 + onnxsim breaks node-name refs (KeyError 'Slice_9' from onnx2torch) | **Raw-first strategy**: try `convert(onnx.load(path))` first; only escalate to opset upgrade + simplify when raw fails | Any pre-opset-13 model where onnxsim renames |
| 2 | same file, both `convert_onnx_to_pytorch` + `get_onnx_input_shape` | `.onnx.gz` passed to ONNX parser as raw bytes | Detect `.gz` suffix → decompress to tempfile (or `io.BytesIO`) before parse. Also fall back to sibling `.onnx.gz` when `.onnx` symlink is broken | nn4sys `mscn_2048d*` family + any future .gz path |
| 3 | `act/pipeline/verification/utils.py:_convert_OnnxBinaryMathOperation` | Var-var mul/add/sub/div only handled scalar broadcast (one side `len==1`) | Use `torch.broadcast_shapes(xs, ys)` to compute target shape; insert EXPAND helper on either side that needs growing. `tf_expand` already supports arbitrary `(in_shape, out_shape)` via `broadcast_to` | nn4sys mscn `(3,128)×(3,1)` pattern + similar |
| 4 | `act/pipeline/verification/utils.py:_convert_OnnxSplit13` | Opset 10 carries split sizes as ONNX node attribute `split=[6,1]`, but handler only read input tensor → fell through to equal-axis path and crashed on `axis_size (7) % num_splits (2) != 0` | Read `mod.split` attribute (set by onnx2torch from ONNX attr) before falling through to input-tensor / equal-axis paths | nn4sys mscn `Split_10`/`Split_21`/`Split_32` |

Verified end-to-end: nn4sys iids 129 (mscn_128d) and 137 (mscn_128d_dual)
both CERTIFIED in ~24s and ~23s respectively after the fix; pre-fix both
were ERROR_RuntimeError.

---

## 4. Benchmark data files restored

The 2026-05-26 audit found 3 benchmark ONNX files missing or corrupt on disk
(unrelated to ACT code). These were restored from the `large_models.zip`
distribution at `/data1/Kane/data/_large_models_download/large_models.zip`
(2.6 GB, sourced from `vnncomp2024` directory inside the zip).

| Benchmark | File | Pre-restore state | Restored size | Notes |
|---|---|---|---|---|
| nn4sys | `onnx/mscn_2048d.onnx` | Broken symlink → nonexistent target | 100 MB (100,937,498 bytes) | Symlink was to `../../nn4sys_2023/...` which doesn't exist |
| nn4sys | `onnx/mscn_2048d_dual.onnx` | 134-byte Git LFS pointer | 151 MB (151,478,151 bytes) | Pointer's `size:` field matched 151478151 |
| nn4sys | `onnx/mscn_2048d_dual.onnx.gz` | 20 KB wget log (`b"--2025-07-15 19:24:46-- https://rwth-aachen.sciebo.de/..."`) | 140 MB | Quarantined original at `_backup_bad_files/mscn_2048d_dual.onnx.gz.wget_log` |
| cgan_2023 | `onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx` | Missing | 272 MB (272,784,587 bytes) | Was the blocker for cgan formal runs |

ONNX sanity (post-restore):
- `mscn_2048d.onnx`: 68 nodes, opset 10
- `mscn_2048d_dual.onnx`: 147 nodes, opset 10
- `cGAN_imgSz32_nCh_3_small_transformer.onnx`: 319 nodes, opset 9

---

## 5. The standard run discipline (paper-grade)

Every CPU/GPU run in this archive obeys:

1. **Process-level watchdog wrapper** (`act.pipeline.watchdog_runner`):
   - `--wall-s` per-instance wall budget (excluding startup grace)
   - `--rss-cap-gb` aggregate process-tree RSS cap (kill on overshoot)
   - `--startup-grace-s` allowance for cold start
   - `--grace-kill-s` SIGTERM → SIGKILL escalation
   - `--strict-bounded-failure` → bounded UNKNOWNs count as `run_status=FAILED`
2. **Thread caps** to prevent CPU oversubscription:
   - `OMP_NUM_THREADS=1`
   - `MKL_NUM_THREADS=1`
   - `OPENBLAS_NUM_THREADS=1`
3. **Per-instance JSON** is the unit of truth. Watchdog-killed instances get a
   *synthetic* `per_instance_<bench>_watchdog_iid<N>_<ts>.json` carrying
   `status=UNKNOWN_TIMEOUT` or `UNKNOWN_RESOURCE_LIMIT`. CLI-completed
   instances get the normal `per_instance_<bench>_<ts>.json`. Both forms
   are retained; `build_csvs.py` picks the watchdog synthetic when both
   exist for the same iid (the watchdog killed an in-flight child).
4. **No sample substitution** for soundness-critical evaluation. The
   `_evaluate_constant_subgraph(allow_sample_substitution=False)` gate is
   load-bearing and must not be re-enabled.
5. **Pre-flight before any new run**:
   - `ps -ef | grep -E 'watchdog_runner|act\.pipeline'` should be empty
   - `nvidia-smi` for GPU runs — check free memory
   - `free -h` and `uptime` — confirm RAM and load headroom
6. **Sequential by default; parallel only if no contention**. Per the
   `feedback_parallel_execution_20260515` memory: CPU/HZ runs that need
   precision-critical timing must be sequential. Verdict-coverage runs
   may be parallel with low risk of `UNKNOWN_TIMEOUT` inflation.

---

## 6. The 5 audit invariants to check before publishing any number

For every CERT / FAL count quoted in a paper:

1. **Source provenance**: cite the `_source_<label>` symlink path.
2. **Watchdog completeness**: total inst = V + A + UNK + TO + RSS + ERR; no rows dropped.
3. **Official cross-check passes**: `python3 soundness_check.py` shows
   `disagree=0` for the row (or the disagreement is the documented
   collins_rul iids 0/22/47 family — these are official-side label errors
   per `SOUNDNESS_VS_VNNCOMP_OFFICIAL.md`).
4. **Bit-identity (if GPU)**: compare CPU and GPU per-iid verdicts. 0 diff =
   formal-qualified; ≥1 stable diff = audit (run sat_relu-style ORT replay).
5. **No silent ERROR**: any `verdict.startswith('ERROR')` requires a known
   root cause (e.g. `nn4sys_lindex200_FIXES.md`) or must be re-investigated.

---

## 7. How to add a new run + ingest

```bash
# 1. Launch the run (write per-bench dir under r93_rerun_…/)
cd /data1/Kane/ACT
PY=/data1/Kane/miniconda3/envs/act-py312/bin/python
export PYTHONPATH=/data1/Kane/ACT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/<my_run_label>_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $OUT
$PY -m act.pipeline.watchdog_runner \
    --benchmark <bench> --instance-ids 0,1,2 \
    --wall-s 60 --startup-grace-s 8 --poll-interval-s 0.5 \
    --rss-cap-gb 24 --grace-kill-s 3 \
    --device cpu --dtype float64 --strict-bounded-failure \
    --out-dir $OUT --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks

# 2. Symlink into CONSOLIDATED_RESULTS
cd /data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS
mkdir -p <bench>          # if first time
ln -sfn $OUT <bench>/_source_<my_label>

# 3. Rebuild CSVs
python3 build_csvs.py

# 4. Re-run official cross-check
python3 soundness_check.py
```

For GPU: same recipe with `--device cuda` (and matching `--dtype float64`
to preserve CPU bit-identity).

---

## 8. Quick reference — paths

- **Canonical VNNLIB**: `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/`
- **Official ground truth**: `/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/{zero_tol,small_tol}/longtable.tex`
- **Python env**: `/data1/Kane/miniconda3/envs/act-py312/bin/python`
- **ACT source**: `/data1/Kane/ACT/`
- **Run scripts** (this dir): `CONSOLIDATED_RESULTS/scripts/`
- **Round nohup logs**: `r93_rerun_…/stream{1,2,3}.nohup.log`, `r93_rerun_…/gpu_stream{1,2,3}.nohup.log`
- **Quarantined corrupt data**: `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/_backup_bad_files/`
- **Large model archive**: `/data1/Kane/data/_large_models_download/large_models.zip` (2.6 GB)

---

## 9. Known open items

- **mscn_2048d** HZ solver matmul shape mismatch: `solver_hz` prints
  `mat1 and mat2 shapes cannot be multiplied (2048x6 and 18x1)` and falls
  back to a sound TIMEOUT verdict. Separate from the conversion fixes; not
  blocking soundness. Investigation deferred.
- **CCTSDB** dynamic Slice with rank-1 input `[12296]`: R17 `LUT_BOUNDS`
  scaffold is implemented in `act/back_end/`, but converter-side pattern
  detection has not been wired up. `vit_2023` shape-lineage gap is a
  separate blocker.
- **collins_rul iids 0/22/47**: official zero_tol AND small_tol mark these
  UNSAT; ACT FAL receipts + independent ORT replay both show the witnesses
  are valid counterexamples (1000/1000 box samples reach the unsafe
  region). Report as label discrepancy needing organizer reconciliation;
  do not silently downgrade ACT's 11 collins_rul FAL count.
- **sat_relu 5 stable iid CPU/GPU diffs**: both sides sound (all 5 ORT-
  clean); divergence is float-ordering LP sampler diversity. Either side
  may be reported; do not silently pick one without disclosing.

---

## 10. Cross-references

- ACT consolidated index (HyZor memory): `~/.claude/projects/-data1-Kane-HyZor/memory/reference_act_consolidated_results.md`
- Official-label audit (HyZor memory): `~/.claude/projects/-data1-Kane-HyZor/memory/project_act_vs_official_soundness_audit_20260526.md`
- sat_relu divergence audit (HyZor memory): `~/.claude/projects/-data1-Kane-HyZor/memory/project_sat_relu_cpu_gpu_divergence_20260526.md`
- Older support-matrix prose (NOT authoritative — superseded by this index): `r93_rerun_…/SCORED_BENCHMARK_SUPPORT_MATRIX.md`
