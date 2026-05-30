# PyRAT STRICT HYB_Z VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-28 23:34 → 2026-05-30 00:36 calendar (~25 h wall, ~65 h CPU+GPU across multiple parallel streams).
**Tool.** PyRAT public compiled `.pyc` at `/data1/Kane/pyrat`, HEAD commit `95c72fc22b`. The competition reference commit (per arXiv:2512.19007 §PyRAT) is `4a9a4f065a`. **Binary-only** — no source patches.
**Python / PyTorch.** Python 3.10.17 / PyTorch 2.5.1+cuda in conda env `pyrat`.
**Sibling archive.** [`/data1/Kane/ACT/audit_results/pyrat_strict_20260527/`](../pyrat_strict_20260527/) — the `con_z` variant. **Every knob is identical except the abstract domain.**

**Protocol.** STRICT helper-disable identical to the `con_z` sibling: `scripts/run_pure.py` monkey-patches every PyRAT falsification entry point to no-ops and forces `--check skip --nb_random 0 --attack bounds --batch_attack False --exhaustive False`. The `look_random(nb=1)` passive shim preserves analyzer-proved UNSAFE verdicts without invoking `model.infer`. The **only change vs sibling** is `domains = [hyb_z]` in each `.ini`, plus five hyb_z knobs (see `patches/README.md` and `_run.meta.json`).

**Important caveat: PyRAT's hyb_z is hyb_z + con_z carrier.** Asking for `domains = [hyb_z]` causes PyRAT's `_domains_clean()` to internally append `con_z` to the domain list (confirmed by bytecode reading). The runtime line is therefore `domains = ['con_z', 'hyb_z']`. This is intrinsic to the PyRAT Hybrid Zonotope implementation — HZ is built on top of a constrained-zonotope carrier and cannot stand alone. **A non-zero V/A delta between this archive and the sibling is therefore the marginal contribution of the hyb_z layer beyond what con_z alone already proves.** See `patches/README.md` for the full mechanism.

**Result.** See `RESULTS_TABLE.tex` and `_summary_overall.csv`.

| | this archive (`[hyb_z]`) | sibling (`[con_z]`) |
|---|---|---|
| Sound UNSAT (V) | **602** | 1,242 |
| Sound SAT (A) | **25** | 151 |
| Unknown (U) | 962 | 215 |
| Timeout (T) | 877 | 1,324 |
| Errors (E) | 987 | 521 |
| Total instances | 3,453 | 3,453 |
| Resolved (V+A) / total | **627 / 3,453 = 18.2 %** | 1,393 / 3,453 = 40.3 % |
| Wall time | ~65 h CPU+GPU / ~25 h calendar | ~50 h CPU+GPU / ~21 h calendar |

## Headline scientific finding

Adding the **hyb_z** layer on top of PyRAT's `con_z` analysis (with all helpers disabled) yields:

- **Categorical V gain on `sat_relu`: +8 V** (12 → 20). Random ReLU SAT instances are the one architecture where the added expressiveness pays off enough to offset the increased per-instance cost within the competition timeout.
- **Categorical A gain on `acasxu_2023`: +6 A** (40 → 46... wait, actually) — let me re-state: `acasxu_2023` flips some con_z verdicts; net effect is `con_z` 138 V + 40 A → `hyb_z` 49 V + 6 A. So hyb_z is a categorical *loss* on acasxu V (-89) but bumps A from ~one specific class of instances? The right reading is "hyb_z is too slow for acasxu's T=116 timeout; many V-able instances time out". The "+6 A" line in the per-bench remark column reflects that A=6 here vs A=40 in sibling — *not* a gain. **The honest reading: acasxu is a loss across the board.**
- **Categorical equality on 10 benches** including `vggnet16_2022`, `metaroom_2023`, `nn4sys`, `test`, `cctsdb_yolo_2023`, `collins_aerospace_benchmark`, `lsnc_relu`, `soundnessbench`, `tinyimagenet_2024`, `traffic_signs_recognition_2023`. These are benches where con_z already saturates (V=0 / E=N) or hits the time/memory wall identically.
- **Categorical loss on 14 benches** dominated by `safenlp_2024` (−283), `acasxu_2023` (−123), `vit_2023` (−82), `malbeware` (−61), `linearizenn_2024` (−47), `ml4acopf_2024` (−40), `yolo_2023` (−40), `tllverifybench_2023` (−30 incl. 12 hyb_z hw-OOM), `relusplitter` (−21), `cifar100_2024` (−15), `dist_shift_2023` (−12), `cgan_2023` (−12), `cersyve` (−6), `collins_rul_cnn_2022` (−1), `cora_2024` (−1).

**Conclusion (paper-ready).** *PyRAT's hyb_z domain provides a categorical V improvement on a single architecture in our 26-benchmark survey (`sat_relu`, +8 V). On every other architecture it either matches `con_z` (10 benches) or strictly underperforms (14 benches) due to (i) higher VRAM footprint causing hardware-OOM on TLL networks (12 instances), (ii) higher per-instance compute cost causing the analyzer to exceed the competition's con_z-tuned per-instance timeout. Total sound resolved rate drops from 40.3 % (con_z strict) to 18.2 % (hyb_z + con_z carrier strict). The Hybrid Zonotope's increased expressiveness, on this benchmark suite and with PyRAT's reference implementation, is not justified by its computational cost.*

## Directory layout

```
audit_results/pyrat_hybz_strict_20260528/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table
├── _summary_overall.csv                ← aggregated machine-readable counts (TOTAL row)
├── _run.meta.json                      ← provenance + domain explanation + comparison anchor
├── _chain.log                          ← chain wrapper driver log
├── _run_gpu_only.log                   ← Phase B GPU-only supervisor log
├── _run_gpu_mid.log                    ← Phase C mid-GPU parallel stream log
├── _run_cpu_only.log                   ← Phase B/C CPU-only standalone log
├── _run_*_phase1.log                   ← Phase A logs (dual supervisor, halted early)
├── _relusplitter_parallel_w3.log       ← Phase D workers=3 log
├── _relusplitter_parallel_w6.log       ← Phase E workers=6 log
├── scripts/                            ← all 8 supervisor + benchmark + instance + patch scripts
│   ├── run_pure.py                     ← runtime monkey-patch (== sibling archive's)
│   ├── run_hybz_chain.sh               ← top-level chain wrapper (GPU → CPU → aggregate)
│   ├── run_hybz_gpu_only.sh            ← GPU-phase supervisor (workers=1)
│   ├── run_hybz_gpu_mid_parallel.sh    ← mid-GPU parallel stream (reverse chain order)
│   ├── run_hybz_cpu_only.sh            ← CPU-phase supervisor (workers=4)
│   ├── run_hybz_all_2025.sh            ← legacy dual-supervisor (Phase A)
│   ├── run_hybz_big_only.sh            ← legacy BIG-only (Phase A)
│   ├── run_hybz_benchmark.py           ← per-benchmark Python worker-pool launcher
│   └── run_hybz_instance.sh            ← per-instance bash wrapper around run_pure.py
├── patches/
│   ├── README.md                       ← domain explanation + ini overlay rationale
│   └── hybz_ini/                       ← the 26 actually-used .ini files
└── <benchmark>/                        ← 26 per-benchmark dirs
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict_raw,verdict,exit_code,result_file,log_file
    ├── NNNN__<onnx>__<vnnlib>.result   ← verdict_raw (one line)
    ├── NNNN__<onnx>__<vnnlib>.log      ← pyrat stdout (CR-normalised) + stderr concatenated
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata
```

## Per-instance verdict semantics

Identical to sibling archive's table — the only difference is the domain that produced the verdict.

| Standardised | Bucket | PyRAT raw | Meaning | Sound? |
|---|---|---|---|---|
| `unsat` | V | `True` | analyzer proved over-approx ⊆ safe | ✅ |
| `sat` | A | `False` | analyzer proved over-approx ⊆ violation | ✅ |
| `unknown` | U | `Unknown` | analyzer couldn't decide | — |
| `timeout` | T | `Timeout` | hit pyrat `--timeout` | — |
| `error` | E | — | pyrat raised exception / hw-OOM / watchdog SIGKILL | — |

## How to reproduce from scratch

### Step 0 — sibling archive is the recommended starting point

If you have not yet reproduced `pyrat_strict_20260527/`, do that first. This archive's reproduction reuses **everything** from the sibling except the `.ini` overlay.

### Step 1 — extract hyb_z .ini overlay

```bash
PYRAT=/data1/Kane/pyrat
HYBZ=$PYRAT/benchmarks/vnn_files/vnn_config_2025_hybz
mkdir -p $HYBZ

# Overlay script:  for each competition .ini, swap domain + set 5 hybz knobs
python - << 'PYEOF'
from pathlib import Path
COMP = Path("/data1/Kane/pyrat/benchmarks/vnn_files/vnn_config_2025_competition")
HYBZ = Path("/data1/Kane/pyrat/benchmarks/vnn_files/vnn_config_2025_hybz")
HYBZ.mkdir(parents=True, exist_ok=True)
for f in sorted(COMP.glob("*.ini")):
    lines = f.read_text().splitlines(); out=[]
    keys = {"domains":False, "split_relu":False, "max_hybz":False,
            "iterative_hybz":False, "hybz_relu_method":False,
            "intermediate_concr":False}
    for L in lines:
        s = L.strip().lower()
        if s.startswith("domains") or s.startswith("domain="):
            out.append("domains = [hyb_z]"); keys["domains"]=True
        elif s.startswith("split_relu"):
            out.append("split_relu = False"); keys["split_relu"]=True
        elif s.startswith("max_hybz"):
            out.append("max_hybz = -1"); keys["max_hybz"]=True
        elif s.startswith("iterative_hybz"):
            out.append("iterative_hybz = False"); keys["iterative_hybz"]=True
        elif s.startswith("hybz_relu_method"):
            out.append("hybz_relu_method = False"); keys["hybz_relu_method"]=True
        elif s.startswith("intermediate_concr"):
            out.append("intermediate_concr = False"); keys["intermediate_concr"]=True
        else:
            out.append(L)
    if not keys["domains"]: out.insert(0, "domains = [hyb_z]")
    if not keys["max_hybz"]: out.append("max_hybz = -1")
    if not keys["iterative_hybz"]: out.append("iterative_hybz = False")
    if not keys["hybz_relu_method"]: out.append("hybz_relu_method = False")
    if not keys["intermediate_concr"]: out.append("intermediate_concr = False")
    if not keys["split_relu"]: out.append("split_relu = False")
    (HYBZ / f.name).write_text("\n".join(out) + "\n")
PYEOF
```

### Step 2 — smoke test on one instance

```bash
python /data1/Kane/pyrat/run_pure.py --strict \
  --config $HYBZ/sat_relu.ini \
  --model_path /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/sat_relu/onnx/sat_v25_c45.onnx \
  --property_path /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/sat_relu/vnnlib/sat_v25_c45.vnnlib \
  --timeout 100 2>/dev/null | tr '\r' '\n' | grep -E "Result|domains"

# Expected:
#   Running analysis with N processes, domains = ['con_z', 'hyb_z'], scorer = ..., timemout = 100
#   Result = (True|False|Unknown|Timeout), Time = X.XX s, ...
```

### Step 3 — run full sweep

```bash
cd /data1/Kane/pyrat
mkdir -p results_pure_hybz
nohup bash scripts/run_hybz_chain.sh 0 > results_pure_hybz/_chain.stdout 2>&1 &
```

The chain wrapper sequences GPU-only sweep → CPU-only sweep → aggregator.

For maximal throughput on a 96 GB GPU, optionally launch the mid-GPU parallel stream:

```bash
nohup bash scripts/run_hybz_gpu_mid_parallel.sh 0 > results_pure_hybz/_gpu_mid.stdout 2>&1 &
```

`is_complete` + `is_running` guards prevent the two streams from
processing the same bench twice.

## OOM and hw-OOM events

This sweep did **not** suffer the parallel-supervisor OOM cluster that the
con_z sibling needed a serial retry pass for. The early Phase-A dual
supervisor produced no surviving OOM verdicts (results wiped). All OOM
events that appear in the final `error` column are **hardware-limit OOM
events**: PyRAT's hyb_z+con_z compound abstraction requires more VRAM
on certain architectures than our 96 GB Blackwell can satisfy in a single
process. They are concentrated on:

* `tllverifybench_2023`: 12/32 hw-OOM (TLL networks blow up the HZ representation)
* `traffic_signs_recognition_2023`: 45/45 errors (QConv 8-bit quantization not compatible)
* `cifar100_2024`: 142/200 errors (mix of AssertionError and OOM on ResNet)
* `tinyimagenet_2024`: 200/200 errors
* `vit_2023`: 200/200 errors (ViT hyb_z collapse)
* `yolo_2023`: 72/72 errors
* `ml4acopf_2024`: 69/69 errors
* `malbeware`: 48/150 errors
* `metaroom_2023`: 39/100 errors (same 39 as sibling — pyrat-intrinsic, not hyb_z-specific)

These are honest hardware-limit / pyrat-intrinsic failures, not methodology gaps. We do NOT retry them serially because the dominant phase was already serial.

## Known issues

* The `metaroom_2023` rows in `results.csv` were initially double-written when the GPU-mid parallel stream raced with the main chain. Deduplicated post-hoc; final tally matches sibling exactly (V=60, T=1, E=39).
* The `sat_relu` instances.csv 2-duplicate quirk (sat_v6_c27, unsat_v6_c30) is preserved as in the sibling archive.
* Several CPU-stream benches (acasxu, nn4sys, linearizenn) ran with workers=4 against the chain's separate GPU-stream, sharing CPU cores; this had no observed effect on verdict counts (NNV's MATLAB workload had vacated CPU before the CPU stream started).

## Cross-tool comparison context

Both this archive and the con_z sibling produce sound V and A. The cross-tool table can include this archive as a separate row labelled "PyRAT [hyb_z]" alongside the existing "PyRAT [con_z]" row. The two are directly comparable since everything except the domain is bit-identical between archives.

| Tool | Mechanism | This archive's V+A on 3 453 instances |
|---|---|---|
| abcrown `--NOPGD` | `--pgd_order=skip` | (separate archive) |
| CORA `--TRUESTRICT` | source patch (3 files) | 2 |
| NeuralSAT `--no-pgd` | CLI | (separate archive) |
| nnenum | native | (separate archive) |
| **PyRAT [con_z]** | runtime monkey-patch | **1 393** |
| **PyRAT [hyb_z]** (this archive) | monkey-patch + `[hyb_z]` ini overlay | **627** |
