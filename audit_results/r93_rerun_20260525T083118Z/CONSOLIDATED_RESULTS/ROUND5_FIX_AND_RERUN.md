# Round 5 — converter ordering swap + rerun

**Date**: 2026-05-26 UTC (after the 23:30 UTC audit)
**Trigger**: Round 4's "raw-first" ONNX converter strategy fixed nn4sys mscn
(opset 10 + onnxsim corruption → `Slice_9` KeyError) but **silently regressed**
ml4acopf_2024, lsnc_relu, yolo_2023, and collins_aerospace_benchmark. The GPU
re-run sweep surfaced these as 100% ERROR on those 4 benchmarks. CPU was
hitting the same error (verified manually) but the regression had been masked
because Round 1's CPU baselines were captured BEFORE the Round 4 fix landed.

## Root cause

Round 4's `_convert_raw()` skipped `onnxsim.simplify`. The mscn family needed
that to be skipped (simplify renamed `Slice_9` and broke onnx2torch's lookup),
but ml4acopf + lsnc + yolo **need** `onnxsim.simplify` to clean up shape
annotations and collapse trivial sub-graphs so ACT's downstream
`_convert_OnnxTranspose` etc. see the expected rank.

Concretely on ml4acopf iid 0 (`14_ieee_ml4acopf-linear-residual.onnx`, opset
14): every Transpose has perm `[1,0]` over an input of shape `(1, 11)`. Without
simplify, the ACT-side tracer sees `self.shape` as rank-1 (the leading batch
was elided somewhere) and `_convert_OnnxTranspose` raises
`ValueError: OnnxTranspose: perm rank 2 != input rank 1`.

## Fix (act/front_end/vnnlib_loader/onnx_converter.py)

**Swap the order of the two paths.** Try the historically-working pipeline
(opset upgrade + `_preprocess` + `onnxsim.simplify` + `shape_inference`)
FIRST; only fall back to the raw graph (no simplify) when the full pipeline
raises. nn4sys mscn still works because the full pipeline raises `KeyError`
on Slice_9 → the raw fallback kicks in and that's the model where raw works.

```python
# Previously (Round 4):
try: pytorch_model = _convert_raw()              # ml4acopf etc. regressed
except: pytorch_model = _convert_with_full_pipeline()

# Now (Round 5):
try: pytorch_model = _convert_with_full_pipeline()   # default
except: pytorch_model = _convert_raw()               # nn4sys mscn fallback
```

Saved patch: `act_fixes_diff/01_onnx_converter.patch` (193-line diff
replaces the Round-4 187-line diff).

## Verification (one-instance per benchmark)

After swap, each previously-failing combination now runs through:

| benchmark/device | iid | Round 4 result | Round 5 result |
|---|---|---|---|
| ml4acopf_2024 / cpu  | 0   | `ERROR_ValueError` 0.3s   | `UNKNOWN` 0.82s ✓ |
| ml4acopf_2024 / cuda | 0   | `ERROR_ValueError` 2.8s   | `UNKNOWN` 5.86s ✓ |
| lsnc_relu / cpu      | 0   | `ERROR_IndexError` 0.3s   | `UNKNOWN` 6.51s ✓ |
| lsnc_relu / cuda     | 0   | `ERROR_IndexError` 0.3s   | `UNKNOWN` 32s ✓ |
| yolo_2023 / cpu      | 0   | (RSS-cap pre-R4)          | `UNKNOWN_TIMEOUT` 48.6s ✓ |
| yolo_2023 / cuda     | 0   | `ERROR_NotImplementedError` 2.1s | `UNKNOWN` 20.82s ✓ |
| collins_aerospace / cpu  | 0 | `UNKNOWN_TIMEOUT` (3-of-6 finished) | `UNKNOWN_TIMEOUT` 46s ✓ |
| collins_aerospace / cuda | 0 | `ERROR_WATCHDOG_EXIT_NONZERO` | `UNKNOWN_TIMEOUT` 45.6s ✓ |
| **nn4sys / cpu (regression check)** | 137 | `CERTIFIED` 22.8s | `CERTIFIED` 33.07s ✓ (preserved) |
| **nn4sys / cuda (regression check)** | 137 | `UNKNOWN` ~37s | `CERTIFIED` 36.62s ✓ |

**The single swap fixes all 4 GPU-only ERROR benchmarks AND preserves the
nn4sys lindex_200+ fix from Round 4. No new code, just reordering.**

## Round 5 rerun

`scripts/round5_rerun_after_fix.sh` runs the 4 regressed benchmarks on BOTH
CPU and GPU under the new converter:

- ml4acopf_2024 0..68 (69 inst), wall=60s, RSS=8 GiB
- lsnc_relu 0..79 (80 inst), wall=60s, RSS=8 GiB
- yolo_2023 0..71 (72 inst), wall=90s, RSS=24 GiB
- collins_aerospace_benchmark 0..5 (6 inst), wall=120s, RSS=8 GiB

CPU sources for these benchmarks needed refresh too because the Round 4 fix
had also broken their CPU paths (the regression had been silent — Round 1
ran the pre-Round-4 code; subsequent CPU verdicts for these would have
inherited the ERROR).

## Output

- BASE: `r93_rerun_…/round5_aftersimplify_<ts>/<bench>_{cpu,cuda}/`
- README: per-benchmark counts after each completes
- After completion: symlink each result into CONSOLIDATED_RESULTS as
  `<bench>/_source_round5_cpu` and `<bench>/_source_round5_cuda`, then
  `python3 build_csvs.py` + `python3 soundness_check.py`.

## Why this isn't another regression risk

The simplify-first path WAS the pre-Round-4 default. Every benchmark that
ran correctly before Round 4 (collins_rul, malbeware, acasxu, linearizenn,
sat_relu, safenlp, tllverifybench, dist_shift, cersyve sidecar, cifar100,
metaroom, etc.) used exactly this path. Round 5 restores that default for
those benchmarks while keeping the raw fallback for nn4sys mscn. The
fallback is gated by `try / except` so it never executes when the default
succeeds.

If a future model fails in onnxsim.simplify (the way nn4sys mscn did) but
ALSO fails on raw (e.g. an op onnx2torch can't convert at all), the user
sees the raw failure — typically a clearer signal than the simplify error.
