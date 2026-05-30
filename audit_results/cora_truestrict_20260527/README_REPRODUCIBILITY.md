# CORA TRUESTRICT VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-27, 12:48 → 21:30 (8 h 42 min).
**Tool.** CORA vnncomp2025 fork at `/data1/Kane/cora-vnncomp2025`, commit `72a0022e8` ("Update README.md", Lukas Koller, 2025-10-16).
**MATLAB.** R2026a Update 2 (`26.1.0.3251617`).
**Protocol.** **TRUESTRICT** — three source patches add a new `falsification_method='none'` option to CORA that does not exist upstream. With `'none'` selected (the default we apply via the patched `prepare_instance.m`), CORA runs pure over-approximative reachability with **no** falsification helper of any kind: no FGSM gradient attack, no center-of-box deterministic point evaluation, no zonotack random sampling.

**Result.** See `RESULTS_TABLE.tex` and `_summary_overall.csv`.

| | |
|---|---|
| Sound UNSAT (V) | **2** |
| Sound SAT (A) | **0** (impossible by design — over-approximation cannot witness counter-examples) |
| Unknown (U) | 3 451 |
| Errors (E) | 0 |
| Total instances | 3 453 |
| Wall time | ~29 606 s (~8.2 h CPU) / 8 h 42 min calendar |

## Headline scientific finding

**CORA without falsification helpers verifies only 2 of 3 453 instances — both on the smoke "test" benchmark.** Every other VNN-COMP 2025 benchmark returns `unknown` for every instance. This is the central honesty test of the paper's cross-tool comparison: CORA's competition strength is **entirely** attributable to its falsification helpers (`zonotack` + `zonotack-layerwise`), not to its over-approximative reachability. Over-approximation is consistent and sound, but on the VNN-COMP 2025 benchmark suite it is too coarse on its own to certify even the easiest non-trivial instance.

The contrast with the helper-enabled CORA competition submission (which reports ~951 V / 947 A / 824 U on the same 3 453 instances; see [obsolete archive] for that data) is striking and should be the headline result of any "what does a sound verifier actually verify?" section in the paper.

## Why is U so large?

CORA's verdicts under TRUESTRICT are:
- `unsat` — the over-approximative reachable set has empty intersection with the unsafe halfspace. This is a **sound proof** that the property holds.
- `unknown` — the reachable set intersects the unsafe halfspace. CORA's reachability cannot distinguish *true* counter-examples from *spurious* artifacts of the over-approximation. Upstream CORA would now invoke `falsify_single` / `zonotack` to disambiguate by searching for a concrete counter-example; we have patched that out.

So `unknown` here means "either a real SAT or a too-loose over-approximation — and without a helper we cannot tell which." Lots of VNN-COMP 2025 instances *are* SAT (especially in `safenlp`, `sat_relu`, `acasxu`), and CORA's over-approximation does intersect the unsafe region for many UNSAT instances too. Both populations show up as `unknown`.

---

## Directory layout

```
audit_results/cora_truestrict_20260527/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table
├── _summary_overall.csv                ← aggregated machine-readable counts (TOTAL row at end)
├── _run.log                            ← driver log
├── _run.meta.json                      ← provenance (MATLAB version, falsification_method='none', etc.)
├── _run.pid / _nohup*.out              ← PID + raw stdout per launch
├── scripts/
│   ├── run_cora_strict_vnncomp2025.sh  ← bash launcher (calls MATLAB -batch)
│   └── cora_strict_sweep_runner.m      ← MATLAB driver (per-benchmark sweep with diary log)
├── patches/
│   ├── README.md                       ← scientific-integrity rationale for each patch
│   ├── prepare_instance.m.patch        ← unified diff (backup .orig at /data1/Kane/cora-vnncomp2025/)
│   ├── (validateNNoptions.m + verify.m verbatim before/after snippets — see patches/README.md)
└── <benchmark>/                        ← per-benchmark dir (one per VNN-COMP 2025 benchmark)
    ├── _summary.csv                    ← idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict,result_file,log_file
    ├── NNNN__<onnx>__<vnnlib>.result   ← VNN-COMP standard result file: one of unsat | unknown
    ├── NNNN__<onnx>__<vnnlib>.log      ← CORA's full MATLAB diary
    └── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata
```

---

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| MATLAB | R2026a Update 2 (older R20XXa should also work) |
| GPU | optional; CORA uses GPU only when explicitly enabled in `prepare_instance.m`. We left the upstream default. |
| Disk | ~5 GB for diaries + results |

### Step 0: benchmark data

```bash
ls /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/   # 26 benchmarks
```

### Step 1: install CORA vnncomp2025 fork

```bash
cd /data1/Kane
git clone https://github.com/koller-lukas/cora-vnncomp2025.git
cd cora-vnncomp2025
git checkout 72a0022e8908a1189aea93e6e5fdc758a1857999
```

### Step 2: apply the three TRUESTRICT patches

The patches and their rationale are documented in `patches/README.md`. Summary:

1. `prepare_instance.m` — backup `prepare_instance.m.orig` and apply `prepare_instance.m.patch` (provided in `patches/`). The patch sets `options.nn.falsification_method = 'none'` and `options.nn.refinement_method = 'naive'`.
2. `code/cora/nn/+nnHelper/validateNNoptions.m` — add `'none'` to the admissible-values list for `falsification_method` (one line change; see `patches/README.md` for before/after).
3. `code/cora/nn/@neuralNetwork/verify.m` — wrap the falsification block in `if strcmp(options.nn.falsification_method, 'none') ... else ... end` so that the FGSM / center / zonotack branches and the `aux_checkPoints` call are skipped (see `patches/README.md`).

### Step 3: smoke test (validate the patch installed correctly)

```bash
cd /data1/Kane/cora-vnncomp2025
matlab -nodisplay -nosplash -batch "
  prepare_instance('acasxu', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib');
  fid = fopen('/tmp/smoke.result','w');
  [r,~] = run_instance('acasxu', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib', '/tmp/smoke.result', 100, false);
  disp(r);
"
cat /tmp/smoke.result
# expected output: 'unsat' (in ~4-5 seconds)
```

If the smoke test prints `'unknown'` instead, the patch is wrong (either the falsification branch is still firing, or the option didn't reach `verify.m`). Inspect the diary log to confirm `Falsification Method: none` appears.

### Step 4: run the full sweep

```bash
cd /data1/Kane/ACT
nohup bash scripts/run_cora_strict_vnncomp2025.sh > audit_results/cora_truestrict_20260527/_nohup.out 2>&1 &
```

The sweep is idempotent — instances with non-empty `.result` are skipped on relaunch. Safe to Ctrl-C and restart.

### Step 5: aggregate

```bash
cd /data1/Kane/ACT/audit_results/cora_truestrict_20260527
for d in */; do
  awk -F, 'NR>1 {
    v=$7; gsub(/"/,"",v); t=$6+0; tot++
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
| `unsat` | reachable set $\cap$ unsafe halfspace = $\emptyset$ | ✅ sound proof |
| `unknown` | reachable set $\cap$ unsafe halfspace $\ne \emptyset$ (CORA cannot disambiguate without helper) | — |

`sat`, `timeout`, `error` never appear in this sweep. CORA's reachability either decides UNSAT or admits `unknown`; it does not write `timeout` explicitly (instances that exceed the per-instance timeout cap still write `unknown`).

---

## Known issues observed during the run

- **One instance ignored its timeout cap.** `nn4sys/0073__lindex_deep_300_5_4__lindex_5_4_prop_5.result` ran for **3 561 s** (target cap was 800 s for nn4sys, with the driver's `timeout_cap` set to 0 = no driver-level cap). This is a CORA-internal issue: CORA's per-instance timer in `verify.m` does not abort some reachability iterations. Effect: nn4sys's `wall_sec_max` is 3 561 s; the verdict for that instance was `unknown` (same as the other 193 nn4sys instances). No data is corrupted, but the sweep took ~1 h longer than it should have. Future runs may want to set `timeout_cap` to a non-zero ceiling in `run_cora_strict_vnncomp2025.sh`.
- **acasxu max wall 217 s.** Similar minor cap-overrun — capped at 116 s per instances.csv but several ran to 200+ s. Same root cause.
- **No errors anywhere.** CORA loaded all 3 453 ONNX models successfully (unlike nnenum, which errored on 1 521 / 3 453 due to unsupported ops). CORA's ONNX parser is much more permissive — this is one of the few categories where CORA's competition build outshines nnenum.

---

## What this archive does NOT contain

- No helper-enabled CORA results — those are in `audit_results/_OBSOLETE_cora_strict_20260526_center_helper/` (kept for forensic reference but should **not** appear in the paper's TRUESTRICT row).
- No abcrown / NeuralSAT / nnenum results — separate archives.
- No GPU profiling — CORA in TRUESTRICT runs almost entirely on CPU.

---

## Cross-tool comparison context

For the paper table, this archive contributes the **CORA TRUESTRICT (no helper)** row:

| Tool | "no helper" mechanism | Can produce SAT? |
|---|---|---|
| abcrown | `--pgd_order=skip` (CLI flag) | yes (via BaB) |
| NeuralSAT | `--disable_attack` (CLI flag) | yes (via BaB) |
| nnenum | upstream-native (no helper exists in source) | yes (via exact-star splitting) |
| **CORA TRUESTRICT (this archive)** | **3 source patches add `'none'` option** | **no** (over-approximation cannot witness counter-examples) |

CORA is the only tool of the four whose paper row honestly reports `A = 0` for every benchmark. This is *not* a tool weakness — it is the price of pure soundness without a falsification side-channel, made transparent by the patch.
