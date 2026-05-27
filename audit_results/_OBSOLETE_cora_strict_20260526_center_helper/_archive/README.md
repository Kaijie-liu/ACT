# CORA STRICT sweep — VNN-COMP 2025 (archived 2026-05-27)

## What this is

Single full sweep of the **CORA** verifier (MATLAB,
`/data1/Kane/cora-vnncomp2025`) on every VNN-COMP 2025 benchmark under a
**strict, no-helper** configuration. The goal is to read out CORA's
*pure-reachability* capability with every falsification / random-sampling
helper switched off.

Run window: 2026-05-26 23:45 → 2026-05-27 03:53 (≈ 4 h 07 min).
Driver log: `_run.log`. Per-instance `.json`/`.log`/`.result` in
`<benchmark>/`. Aggregated truth: [`per_bench_summary.csv`](per_bench_summary.csv).

## Configuration ("STRICT")

Recorded in `_run.meta.json`:

```json
{
  "falsification_method": "center",
  "refinement_method": "naive",
  "note": "STRICT: no zonotack vertex sampling, no fgsm gradient, no nrSamp random samples. Pure reachability + center-of-box deterministic eval."
}
```

What each CORA knob does:

| CORA knob (default) | STRICT value | What it kills |
|---|---|---|
| `falsification_method = zonotack` | `center` | vertex-sampling falsifier (plus `nrSamp` random samples) |
| `falsification_method = fgsm`     | (off, via above) | gradient-based adv search |
| `refinement_method = zonotack` (or `zonotack-layerwise`) | `naive` | bound-refinement helpers |

Per-instance timeouts come from each benchmark's `instances.csv` (no cap).
GPU gate: each benchmark waits until ≥ 50 GB VRAM is free before starting
(parallel-safe alongside other GPU work).

## Benchmark classification

We ran **26 benchmarks** total. CORA officially submitted **16** of them to
VNN-COMP 2025 (per `/data1/Kane/data/vnncomp2025_results/cora/results.csv`);
the other **10** we ran for completeness. The participant paper
(arXiv:2512.19007) only lists 9 in CORA's section but the actual published
`results.csv` covers 16 — we use the `results.csv` truth.

| Class | Count | Benchmarks |
|---|---:|---|
| **official** | 16 | acasxu_2023, cersyve, cifar100_2024, collins_rul_cnn_2022, cora_2024, dist_shift_2023, linearizenn_2024, malbeware, metaroom_2023, relusplitter, safenlp_2024, sat_relu, soundnessbench, test, tinyimagenet_2024, tllverifybench_2023 |
| **supplemental** | 10 | cctsdb_yolo_2023, cgan_2023, collins_aerospace_benchmark, lsnc_relu, ml4acopf_2024, nn4sys, traffic_signs_recognition_2023, vggnet16_2022, vit_2023, yolo_2023 |

## Headline result

26 benchmarks, **3 453** instances, **14 821 s** total wall (≈ 4 h 07 min).

VNN-COMP convention: `unsat` = property holds (V), `sat` = counter-example
exists (A). Column tally:

| Slice | N | V | A | U | E | resolved % |
|---|---:|---:|---:|---:|---:|---:|
| 16 officially-submitted benches | 2 709 | **15** | **26** | 2 668 | 0 | **1.51 %** |
| 10 supplemental benches | 744 | 0 | 0 | 744 | 0 | 0.0 % |
| **Grand total (26 benches)** | **3 453** | **15** | **26** | **3 412** | **0** | **1.19 %** |

Compare to CORA's **official** numbers on the same 16 benchmarks (helpers
enabled, deduped from `results.csv`):

| Slice | N | V | A | U | resolved % |
|---|---:|---:|---:|---:|---:|
| Official CORA (16 benches, full helpers) | 2 722 | 951 | 947 | 824 | **69.7 %** |
| STRICT (this run, 16 same benches)       | 2 709 | 15  | 26  | 2 668 | **1.51 %** |
| **Delta**                                 | −13$^{\dagger}$ | **−936** | **−921** | +1 844 | **−68.2 pp** |

$^{\dagger}$ Official N differs from STRICT N because (a) `acasxu/cersyve/...`
all carry +1 spurious duplicates in `results.csv` beyond the warmup row we
already deduped, and (b) `sat_relu` is missing 1 unique instance in CORA's
submission. STRICT N matches each benchmark's `instances.csv` exactly.

**STRICT loses 68.2 percentage points of resolve rate.** Almost all of CORA's
official solving capability comes from its falsification/refinement helpers,
not from raw reachability.

Only **4 benchmarks** keep non-zero solves once helpers are off:

* `tllverifybench_2023`: **11 V + 17 A** (official 16 V + 17 A → A retained,
  V drops a bit)
* `collins_rul_cnn_2022`: **1 V + 5 A** (official 40 V + 23 A)
* `cersyve`: **1 V + 2 A** (official 8 V + 5 A)
* `test`: **2 V + 2 A** (smoke baseline, official 4 V + 1 A)

### Important caveat about the A column

All 26 A verdicts in STRICT mode were produced by CORA's
**deterministic center-of-box evaluation**, not by an adversarial search.
With `falsification_method=center` CORA evaluates the network at the
geometric centre of the input box; if that single concrete output violates
the property, CORA reports `sat`. So the STRICT A column measures
"how often the box centre alone happens to be a counter-example", not
"how often pure reachability proves SAT". On `tllverifybench_2023` the
17 official A verdicts already had a violating centre, which is why
STRICT matches official on A. Elsewhere the centre rarely violates.

## Repository layout

```
cora_strict_20260526/
├── _archive/
│   ├── README.md                  (this file)
│   ├── ppt_cora_strict.tex        (one-page beamer table)
│   ├── per_bench_summary.csv      (STRICT + official cross-tabulated, deduped)
│   └── scripts/
│       ├── cora_strict_sweep_runner.m       (MATLAB driver)
│       └── run_cora_strict_vnncomp2025.sh   (bash launcher)
├── _run.log               (driver log, 1 line per instance + 5-line progress)
├── _run.meta.json         (frozen STRICT config of this run)
├── _run.pid               (driver PID)
├── _nohup.out             (initial stdout, before normalizer/diary patch)
├── _nohup_v2.out          (stdout after restart with per-instance diary)
└── <benchmark>/
    ├── _summary.csv       (idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict,...)
    ├── NNNN__model__prop.json     (per-instance metadata: verdict, walltime, etc.)
    ├── NNNN__model__prop.log      (full MATLAB stdout for that instance)
    └── NNNN__model__prop.result   (VNN-COMP verdict file)
```

## How it was driven

Top-level command (cf. `scripts/run_cora_strict_vnncomp2025.sh`):

```bash
RESULTS_ROOT=/data1/Kane/ACT/audit_results/cora_strict_20260526 \
CORA_GPU_FREE_GB_MIN=50 \
CORA_GPU_CHECK_INTERVAL_S=45 \
nohup bash /data1/Kane/ACT/scripts/run_cora_strict_vnncomp2025.sh \
   > $RESULTS_ROOT/_nohup.out 2>&1 &
```

That invokes the MATLAB driver `cora_strict_sweep_runner.m` which:

1. Adds `/data1/Kane/cora-vnncomp2025` to the MATLAB path; requires
   `prepare_instance.m` to be patched to `center` + `naive` so benchmarks
   do not silently fall back to default falsifier.
2. Walks benchmarks in a **light → heavy** order so the heavy GPU jobs
   (`vit_2023`, `tinyimagenet_2024`, `cifar100_2024`, `vggnet16_2022`,
   `collins_aerospace_benchmark`) run *last*, after concurrent ACT stream-3
   heavy-CNN load has had time to drain.
3. Before each benchmark, polls `nvidia-smi` until ≥ 50 GB free; never
   pre-empts another tool.
4. Inside a benchmark, instances run **sequentially** (no MATLAB parallelism)
   to keep CORA deterministic. Each instance writes its own
   `.json` / `.log` / `.result` triple alongside the
   benchmark's `_summary.csv`.

## Reproducing later

```bash
RESULTS_ROOT=/tmp/cora_strict_repro \
bash /data1/Kane/ACT/audit_results/cora_strict_20260526/_archive/scripts/run_cora_strict_vnncomp2025.sh
# single benchmark only:
RESULTS_ROOT=/tmp/cora_strict_repro \
bash /data1/Kane/ACT/audit_results/cora_strict_20260526/_archive/scripts/run_cora_strict_vnncomp2025.sh tllverifybench_2023
```

To diff against this archive after a re-run:

```bash
diff <(cut -d, -f1,7 cora_strict_20260526/<bench>/_summary.csv | sort) \
     <(cut -d, -f1,7 /tmp/cora_strict_repro/<bench>/_summary.csv | sort)
```

## Caveats / runtime experience

* `nn4sys` ate 81 minutes alone (4860 s wall, 194 timeouts) — CORA's
  reachability hits per-instance timeout on most rows. Dominant cost item.
* `safenlp_2024` (1080 instances) took 25 minutes, average 1.4 s each: CORA
  exits fast with `unknown` here, no actual heavy compute.
* `tllverifybench_2023` is the lone "interesting" benchmark — STRICT CORA
  retains A fully (17 vs 17 official) and only loses some V (11 vs 16
  official). Worth a deeper look.
* `cifar100_2024`, `tinyimagenet_2024`: 100 % unknown but still cost 17 min
  and 40 min respectively due to per-instance reachability setup overhead
  before timing out.
* GPU gate fired cleanly: no contention with the concurrent ACT stream-3.
* `sat_relu` official `results.csv` has only 99 unique (onnx,vnnlib) pairs
  vs 100 in the benchmark's `instances.csv`; CORA appears to have missed
  one instance during their official submission. Recorded as
  `official_N_dedup=99` in `per_bench_summary.csv`.

## Cross-reference

Companion strict-mode experiments on other tools:

* **abcrown** `--NOPGD`: `/data1/Kane/ACT/audit_results/abcrown_nopgd_20260525/`
  (see `README_REPRODUCIBILITY.md` there).
* **NeuralSAT** `--disable_attack`:
  `/data1/Kane/ACT/audit_results/neuralsat_strict_20260527/` (sweep in
  progress 2026-05-27).
* **PyRAT** 2025 strict sweep, live under `/data1/Kane/pyrat/` driven by
  `/data1/Kane/pyrat/scripts/run_all_2025.sh` (in progress 2026-05-27).

Together these four strict runs let us measure *what survives in
helper-free mode* across four independent verifiers.

## Audit history of this archive

* 2026-05-27 earlier: first draft of `ppt_cora_strict.tex` had two bugs
  identified by the user — (a) `relusplitter` was misclassified as
  supplemental (CORA actually submitted it), (b) every "official X/Y"
  Remark was off by 1 because CORA's `results.csv` includes a duplicated
  warmup row per benchmark. This README and the LaTeX are the corrected
  revision.
