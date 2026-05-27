# nn4sys lindex_200+ family — Round 3 ERROR fixes

**Date**: 2026-05-26 UTC
**Originally observed**: Round 3 stream 1 finished with 61 ERRORs for nn4sys
iids 107..193 (the lindex_200+ family in instances.csv). Roots traced to four
independent issues:

## Root-cause inventory

| Models | Original error | # iids | Root cause |
|---|---|---:|---|
| `mscn_128d.onnx` | `NotImplementedError: Var-var 'mul' size mismatch (384 vs 3) at mul_39: non-scalar broadcast not yet supported` | 4 | `_convert_OnnxBinaryMathOperation` only handled scalar broadcast |
| `mscn_128d_dual.onnx` | `RuntimeError: ONNX conversion failed: ... 'Slice_9'` (KeyError from onnx2torch) | 23 | Hardcoded opset 10→13 upgrade plus onnxsim simplify rename a node and break Slice_9 reference inside onnx2torch |
| `mscn_2048d.onnx` | `RuntimeError: ONNX conversion failed: ...mscn_2048d.onnx.gz: Error parsing message with type 'onnx.ModelProto'` | 11 | Broken `.onnx` symlink → runner falls back to `.onnx.gz`, but converter tries to parse gzip bytes as raw protobuf |
| `mscn_2048d_dual.onnx` | same parsing failure | 23 | **Benchmark data file corruption**: `.onnx.gz` is 20 KB and contains a wget log header `b"--2025-07-15 19:24:46-- https://rwth-aachen.sciebo.de/..."`, not real ONNX content. Not an ACT bug; cannot be fixed on the verifier side |

## Fixes applied

### 1. `act/front_end/vnnlib_loader/onnx_converter.py:convert_onnx_to_pytorch`

- Replaced the hardcoded "always upgrade opset 10→13 + simplify" pipeline with a
  **raw-first strategy**: try `convert(onnx.load(path))` with only the
  conservative `_preprocess_onnx_for_onnx2torch` patch + shape inference.
  Only fall back to opset upgrade + onnxsim when the raw path fails.
- Added `.onnx.gz` auto-decompression to a tempfile when the passed `onnx_path`
  ends in `.gz` (handles the broken-symlink fallback case for mscn_2048d).
- Added `.onnx.gz` sibling fallback when the `.onnx` path is missing.

### 2. `act/front_end/vnnlib_loader/onnx_converter.py:get_onnx_input_shape`

- Same `.onnx.gz` fallback as `convert_onnx_to_pytorch`; uses `gzip.open` +
  `io.BytesIO` so `onnx.load` reads the decompressed bytes directly without
  a tempfile.

### 3. `act/pipeline/verification/utils.py:_convert_OnnxBinaryMathOperation`

- Extended var-var broadcast handling from "scalar (len==1) only" to **general
  numpy/PyTorch broadcasting**. Uses `torch.broadcast_shapes(xs, ys)` to
  compute the target, then inserts an EXPAND helper on whichever side(s)
  don't already match the target. The existing `tf_expand` already supports
  arbitrary `(input_shape, output_shape)` via `broadcast_to(out_shape)`, so
  no transfer-function change was needed.

### 4. `act/pipeline/verification/utils.py:_convert_OnnxSplit13`

- Opset 10 carries split sizes as the ONNX node *attribute* `split`
  (e.g. `[6, 1]`); opset 13 moved it to an input tensor. The handler only
  read the input tensor (and fell through to the "equal-axis count from
  children" path when no input), which threw `ValueError: equal-axis split
  requires axis dim (7) divisible by num_splits (2)` for non-equal opset-10
  splits like mscn_128d Split_10 ((?,3,7) → (?,3,6) + (?,3,1)).
- Now reads `mod.split` (set by onnx2torch from the ONNX attribute) before
  falling through.

## Soundness note

None of these errors produced false CERTIFIED or FALSIFIED — they were honest
conversion failures that returned `ERROR_RuntimeError` / `ERROR_NotImplementedError`.
The fixes restore the ability to run the verifier on these instances; the
resulting verdict is then whatever the analyzer finds within wall budget
(`CERTIFIED` for mscn_128d_* family in ~22s; `UNKNOWN_TIMEOUT` for mscn_2048d
family due to a separate solver-side matmul shape mismatch — see below).

## Verified end-to-end (representative iids, wall=60s)

| iid | model | pre-fix | post-fix |
|---|---|---|---|
| 129 | mscn_128d.onnx | `ERROR_NotImplementedError` | **CERTIFIED** (24.4s) |
| 137 | mscn_128d_dual.onnx | `ERROR_RuntimeError` (Slice_9) | **CERTIFIED** (22.8s) |
| 160 | mscn_2048d.onnx | `ERROR_RuntimeError` (gz parse) | `UNKNOWN_TIMEOUT` 60s (solver shape mismatch, sound TO not error) |
| 171 | mscn_2048d_dual.onnx | `ERROR_RuntimeError` (gz parse) | `ERROR_RuntimeError` (corrupt .gz on disk; not ACT-fixable) |

## Outstanding work (separate)

`mscn_2048d.onnx` HZ analyzer prints solver fallbacks:

```
L38 FALLBACK (RuntimeError): mat1 and mat2 shapes cannot be multiplied (2048x6 and 18x1)
L41 FALLBACK (RuntimeError): mat1 and mat2 shapes cannot be multiplied (2048x2048 and 6144x1)
...
```

This is a separate solver-side issue (matmul shape alignment inside `solver_hz`),
not a conversion bug. The fallback is sound (returns `UNKNOWN_TIMEOUT` not a
wrong verdict). Investigation deferred.

## Rerun source

`r93_rerun_…/nn4sys_fix_rerun_20260526T105058Z/` — all 61 originally-ERROR iids,
sequential, OMP=MKL=OPENBLAS=1, wall=60s, RSS cap=24 GiB, strict-bounded-failure.
