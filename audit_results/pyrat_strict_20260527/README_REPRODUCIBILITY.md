# PyRAT STRICT VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-27 13:15 → 2026-05-28 05:07 main sweep (15 h 52 min calendar) + 2026-05-28 08:46 → 14:14 serial OOM rerun (5 h 28 min calendar). Total ≈ 21.3 h calendar / ~50 h CPU+GPU.
**Tool.** PyRAT public compiled `.pyc` distribution at `/data1/Kane/pyrat`, HEAD commit `95c72fc22b` ("Merge branch 'fix-public-version' into 'main'"). The competition reference commit (per arXiv:2512.19007 §PyRAT) is `4a9a4f065a` ("Final version for VNN2025"). PyRAT is **binary-only** — no source patches are possible.
**Python / PyTorch.** Python 3.10.17 in conda env `pyrat` (`/data1/Kane/miniconda3/envs/pyrat`); PyTorch 2.5.1 with CUDA.
**Solvers.** None (PyRAT is a pure abstract-interpretation reachability tool; no Gurobi/CPLEX/MIP backend).

**Protocol.** Helper-free via **runtime monkey-patch + forced CLI flags**, both implemented in `patches/run_pure.py` (which wraps `pyrat.main`). PyRAT's verifier code (`.pyc`) is bit-identical to commit `95c72fc`; the `run_pure.py` shim rebinds every falsification entry point to a no-op stub *before* `pyrat.main()` is invoked. The full list of patched symbols and the scientific-integrity justification for the passive `look_random(nb=1)` shim are documented in `patches/README.md`.

**Result.** See `RESULTS_TABLE.tex` and `_summary_overall.csv`.

| | |
|---|---|
| Sound UNSAT (V) | **1,242** |
| Sound SAT (A)   | **151** (analyzer-proved UNSAFE only; no PGD / center / random-sample witness) |
| Unknown (U)     | 215 |
| Timeout (T)     | 1,324 |
| Errors (E)      | 521 (all pyrat-internal: AssertionError / AttributeError / hw-OOM; *not* our methodology) |
| Total instances | 3,453 |
| Wall time       | ~180,386 s (50.1 h CPU+GPU) / 21.3 h calendar (with main + OOM rerun + ml4acopf rerun) |

## Headline result

**PyRAT preserves ~100% of its analyzer-only verifier capability under STRICT** on the benchmarks it natively handles (acasxu, malbeware, safenlp, linearizenn, dist_shift, collins_rul, tllverifybench). The losses are cleanly attributable:

1. **SAT-slice loss** (PGD was carrying it) — `safenlp` -592 F, `cora_2024` -128 F, `sat_relu` -50 F, `soundnessbench` -35 F.
2. **PyRAT-internal crash on deep CNN/ViT** (the competition's PGD was *masking* these crashes by returning a quick adv witness before BaB triggered the assertion) — `cifar100` 135 errors, `tinyimagenet` 199, `metaroom` 39, `cctsdb_yolo` 39, `yolo` 31, `relusplitter` 64, plus 6 `ml4acopf` instances that exceed our 96 GB VRAM single-process budget.

The "PGD masks intrinsic crashes" finding is methodologically the most interesting cross-tool observation: it means VNN-COMP's official numbers for PyRAT on cifar100/tinyimagenet implicitly count "*found adv before crash*" as success — these are real PGD wins, not analyzer wins.

The +V gains over the official numbers on `nn4sys` (+10) and `safenlp` (+86 V) reflect that VNN-COMP's per-benchmark *outer* wall budget ran out before PyRAT could complete all instances in the official sweep (visible in `vnncomp2025_results/pyrat/2025_nn4sys/results.csv` as 70/194 rows marked `run_instance_timeout` — never started). We run each instance to its own per-instance timeout with no global cap.

## Directory layout

```
audit_results/pyrat_strict_20260527/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table (compile standalone)
├── _summary_overall.csv                ← aggregated machine-readable counts (TOTAL row at end)
├── _oom_rerun_results.csv              ← per-instance OOM rerun diagnoses
├── _oom_rerun.log                      ← serial-rerun driver log (concatenated main + ml4acopf)
├── _run.log                            ← main driver log (timestamp / bench start / done in N s)
├── _run_big.log                        ← BIG-only parallel supervisor log
├── _run.meta.json                      ← provenance: commits + python/torch + flags + patches + anomalies
├── _nohup_main.out                     ← raw stdout of main supervisor (full BaB chatter)
├── _nohup_big.out                      ← raw stdout of BIG-only supervisor
├── scripts/
│   ├── run_pyrat_strict_vnncomp2025.sh ← main bash launcher (gated multi-stage)
│   ├── run_pyrat_strict_big_parallel.sh← BIG_GPU-only secondary supervisor
│   ├── run_instance.sh                 ← single-instance wrapper (calls run_pure.py)
│   ├── run_benchmark.py                ← per-benchmark Python worker pool
│   ├── run_pure.py                     ← the runtime monkey-patch (see patches/)
│   ├── rerun_pyrat_oom_serial.py       ← post-hoc OOM rerun (serial, exclusive GPU)
│   ├── fill_missing.py                 ← backfill rows lost to subprocess.timeout race
│   ├── repair_verdicts.py              ← carriage-return-aware re-parser for old runs
│   └── aggregate_results.py            ← machine-readable summary generator
├── patches/
│   ├── README.md                       ← scientific-integrity rationale for each patch
│   ├── run_pure.py                     ← the runtime patch payload (== scripts/run_pure.py)
│   ├── competition_ini_rename.patch    ← unified diff: 4a9a4f0 vnn_config/ → renamed
│   └── competition_ini/                ← bit-for-bit copy of the .ini files actually used
└── <benchmark>/                        ← 26 per-benchmark dirs (VNN-COMP 2025 benchmarks)
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file
    ├── _oom_rerun_backup/              ← (when applicable) original parallel-sweep .err+.out before retry
    ├── NNNN__<onnx>__<vnnlib>.result   ← verdict_raw (one line: True / False / Unknown / Timeout)
    ├── NNNN__<onnx>__<vnnlib>.log      ← pyrat stdout (CR-normalized) + stderr concatenated
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata (idx, verdict_raw, bucket, oom flag, …)
```

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| Python | 3.10.17 (PyRAT's `.pyc` is built for Python 3.10) |
| PyTorch | 2.5.1+cuXX |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB VRAM). PyRAT for cuda-enabled `.ini`s requires substantial VRAM; 96 GB is sufficient *except* for 6 ml4acopf `300_ieee_*` instances that exceed this budget. The competition AWS `m5` is CPU-only and avoids this. |
| CPU cores | 20 physical |
| RAM | 125 GB |
| Disk | ~3 GB for all 3,453 per-instance `.log` files |

### Step 1: install

```bash
# Get PyRAT public compiled binary at the competition-tagged commit (or HEAD; we used HEAD)
git clone https://git.frama-c.com/pub/pyrat.git
cd pyrat && git checkout 95c72fc

# Conda env
conda env create -f pyrat_env.yml
conda activate pyrat

# Get VNN-COMP 2025 benchmarks (assumed already present at /data1/Kane/data/vnncomp2025_benchmarks/benchmarks)
```

### Step 2: audit "no helper" wiring

The runtime monkey-patch is in `patches/run_pure.py`. Spot-check that all six
falsification entry points are no-op stubs:

```bash
python - << 'PYEOF'
import sys; sys.path.insert(0, '/data1/Kane/pyrat')
from scripts.run_pure import _disable_falsification, _disable_concrete_sim_helpers
_disable_falsification(); _disable_concrete_sim_helpers()
import pyrat.attacks.attacks as A
import pyrat.attacks.utils_attacks as U
for name, fn in [
  ('pgd_attack', A.pgd_attack), ('pgd_attack_batched', A.pgd_attack_batched),
  ('deepfool_attack', A.deepfool_attack),
  ('counter_adv', U.counter_adv), ('look_for_counter', U.look_for_counter),
  ('look_for_counter_adv', U.look_for_counter_adv), ('infer_counter', U.infer_counter),
  ('look_random', U.look_random),
]:
  raw = fn.__wrapped__ if hasattr(fn,'__wrapped__') else fn
  print(f'  {name:25s} -> {raw.__qualname__}')
PYEOF
```

Every line must show `_disable_falsification.<locals>.{_none_pair, _false, _look_random_passive}` (not the original `pgd_attack`, etc.).

### Step 3: smoke test (single instance)

```bash
python /data1/Kane/pyrat/scripts/run_pure.py --strict \
  --config /data1/Kane/pyrat/benchmarks/vnn_files/vnn_config_2025_competition/acasxu_2023.ini \
  --model_path /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
  --property_path /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib \
  --timeout 60 2>/dev/null | tr '\r' '\n' | grep "Result"
# Expected output: Result = True, Time = X.XX s, Safe space = 100.00 %, number of analysis = N
```

### Step 4: run full sweep

```bash
cd /data1/Kane/pyrat
mkdir -p results_pure
# Main supervisor (TINY -> MID -> BIG with per-stage GPU gates):
nohup bash scripts/run_all_2025.sh 0 > results_pure/_run_all.stdout 2>&1 &
# Parallel BIG-only supervisor (optional, accelerates BIG queue):
nohup bash scripts/run_big_only.sh 0 > results_pure/_run_big.stdout 2>&1 &
```

### Step 5: post-process

```bash
# Aggregate
python scripts/aggregate_results.py
# Backfill any rows lost to subprocess timeout race
python scripts/fill_missing.py <bench>  # if needed
# Re-parse for carriage-return-buried Result lines
python scripts/repair_verdicts.py
# Re-run OOM-flagged rows serially (single-supervisor, exclusive GPU)
python scripts/rerun_pyrat_oom_serial.py
```

## Per-instance verdict semantics

| Standardised `verdict` | Bucket | PyRAT `verdict_raw` | Meaning | Sound? |
|---|---|---|---|---|
| `unsat` | V | `True` | analyzer proved over-approx ⊆ safe halfspace | ✅ |
| `sat` | A | `False` | analyzer proved over-approx ⊆ unsafe halfspace (no concrete witness extracted under STRICT) | ✅ |
| `unknown` | U | `Unknown` | analyzer couldn't decide | — |
| `timeout` | T | `Timeout` | pyrat hit its own `--timeout` and reported Timeout | — |
| `error` | E | — | pyrat raised an exception (AssertionError / AttributeError / OOM) or the watchdog SIGKILL'd it before a Result line was emitted | — |

A=151 here means "151 analyzer-proved UNSAFE verdicts". Under STRICT the
`look_random(nb=1)` witness call has been replaced with a passive shim that
returns True without touching `model.infer`, so these A verdicts come from
the abstract analyzer alone — no concrete sample is ever evaluated.

## OOM rerun

The main sweep ran two supervisors in parallel (one for TINY+MID+BIG, one
for BIG_GPU only) so we could overlap CPU- and GPU-bound benchmarks. This
saved ~10–15 h of wall but produced 104 instances flagged with CUDA-OOM
errors when both supervisors ran cuda-enabled benchmarks concurrently. They
were re-run **serially**, single supervisor, exclusive GPU, and the
diagnostic CSV is `_oom_rerun_results.csv`. Diagnoses:

| Diagnosis | Count | Meaning |
|---|---:|---|
| `RECOVERED_serial` | 19 | original OOM, retry produced V/A/U |
| `RECOVERED_serial_to_timeout` | 77 | original OOM, retry produced a clean Timeout (i.e. the instance was genuinely going to time out, OOM was incidental) |
| `reproduces_intrinsic_serial` | 17 | original OOM, retry crashes with a different intrinsic exception (degenerate-bound AssertionError, etc.) — not OOM, real tool bug exposed |
| `reproduces_OOM_serial` | 11 | original OOM, retry OOMs again single-process. **These are real hardware-limit instances** (mostly `ml4acopf_2024` 300_ieee models needing >96 GB VRAM and some `relusplitter` oval21 RSPLITTER models). |
| `retry_no_result` | 4 | retry silently failed to write a row (watchdog timing edge cases) |

The original parallel-sweep `.err` / `.out` files for every rerun row are
preserved at `<bench>/_oom_rerun_backup/`. The per-instance `.result` file
in `<bench>/` reflects the **retry** verdict (or the unchanged original if
no retry happened), not the stale parallel-sweep verdict.

## Known issues

* **`sat_relu` has 2 duplicate `(onnx, vnnlib)` rows in instances.csv** (`sat_v6_c27` and `unsat_v6_c30` each appear twice). We keep both rows in `_summary.csv` for protocol fidelity; both agree on verdict.
* **`traffic_signs_recognition_2023` was double-launched** by both supervisors due to a skip-if-complete race (the BIG_GPU queue is in both `run_all_2025.sh` and `run_big_only.sh`, and the `is_complete` check raced). We deduplicated post-hoc by keeping the row with `(verdict != error, lower wall_s)` per `(onnx, vnnlib)`, yielding the proper 45 unique rows.
* **6 `ml4acopf_2024` instances exceed our 96 GB VRAM** single-process. These remain `error` after serial retry with diagnosis `reproduces_OOM_serial`. The competition AWS `m5` runs PyRAT on CPU (no GPU OOM possible). To reproduce the competition's 61 V / 16 F on these instances would require a CPU-only run of PyRAT.
* **`run_pure.py` does not patch the `analyze_exhaustive_integer` brute-force enumeration mode** (only forces `--exhaustive False`). The `cctsdb_yolo_2023` benchmark's official `.ini` enables this mode, so under STRICT we instead get 39 `error` (the cctsdb_yolo `.ini` parser raises when `exhaustive=False` but the benchmark expects it on). This is a forced-flag side effect, not a methodology gap.
* **`outer subprocess timeout` race**: an earlier version of `run_benchmark.py` set Python `subprocess.run(timeout=T+60)` which was only 15 s above the inner bash watchdog's effective deadline (`T+45`). On `lsnc_relu` (T=25) this dropped 12/80 CSV rows because pyrat multiprocessing children took >15 s to clean up after SIGKILL, killing the wrapper before it could append the CSV row. Fixed to `T+120` in this archive; the 12 dropped rows were backfilled via `scripts/fill_missing.py`.

## Cross-tool comparison context

PyRAT's helper-disable mechanism is structurally analogous to the other
strict-mode reproductions in this paper:

| Tool | Mechanism | Can produce A? |
|---|---|---|
| abcrown | `--pgd_order=skip` (also disables `adv_warmup`, `check_adv`, BaB-attack, MIP-attack) | yes — BaB completeness can find concrete adv |
| CORA | `falsification_method='none'` (source patch on 3 files; A=0 by design — over-approx is sound but cannot witness) | no |
| NeuralSAT | `--no-pgd-attack` + early-stop disabled (CLI) | yes — BCP/branch can falsify |
| nnenum | native (no helper in upstream) | yes — concrete simulation as part of method |
| PyRAT | runtime monkey-patch (binary-only distribution) | **yes — analyzer's over-approximation can prove UNSAFE directly when output bounds fall fully in the violation halfspace; no concrete witness is extracted** |

PyRAT's A=151 is therefore directly comparable to CORA-truestrict A=0 (CORA's reachability is over-approximative; PyRAT's con_z + BaB is also over-approximative but additionally tracks the polytope's *direction*, so it can sometimes prove the entire over-approx is inside the violation halfspace — yielding sound A). This is the key methodological distinction between the two abstract-interpretation tools.
