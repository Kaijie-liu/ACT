# CCTSDB dynamic Slice — sound representation design (NOT YET IMPLEMENTED)

Run root: `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/`
Authored 2026-05-25 after the R12 fail-closed gate eliminated the
sample-substitution path. This is a design proposal; no code lands
under this file. The implementation work is queued behind
ViT/ml4acopf/cgan scalability work.

## Actual structure of `slice_23` in `cctsdb_yolo_2023/onnx/patch-1.onnx`

The FX inspection of `slice_23` shows:

```
call_module  slice_23     target=Slice
  args = [initializers_onnx_initializer_18,   # data: STATIC initializer (tensor T)
          concat,                              # starts: dynamic, from input
          concat_1,                            # ends:   dynamic, from input
          initializers_onnx_initializer_19,    # axes:   STATIC initializer
          initializers_onnx_initializer_20]    # steps:  STATIC initializer
```

Walking back the `starts` chain:

```
concat = Concat(unsqueeze_31, unsqueeze_35)
unsqueeze_31 = Unsqueeze(cast_9)
cast_9       = Cast(gather_8)
gather_8     = Gather(input_1, init_5)   # input_1[init_5]  (init_5 is a static index)
# Identical structure for the second axis via gather_11 with init_6.
```

And `ends`:

```
concat_1 = Concat(unsqueeze_32, unsqueeze_36)
unsqueeze_32 = Unsqueeze(add_28)
add_28       = Add(cast_9, init_16)        # cast_9 + offset_const
unsqueeze_36 = Unsqueeze(add_30)
add_30       = Add(cast_12, init_17)       # cast_12 + offset_const
```

So:

* The **data** being sliced is a **static initializer** `T = init_18`.
* The **start** position on each axis is a **single input coordinate**
  picked up by `Gather(input_1, fixed_index)`, then cast to int.
* The **end** position on each axis is `start + offset`, where the
  offset is a static initializer.
* Therefore **`end - start` is constant** → the output spatial shape
  is **statically known**.

## Why this matters

The model is a YOLO patch cropper: it takes a fixed feature map `T`
and crops out a sub-window whose **position depends on two input
coordinates** but whose **size is fixed**. From the VNNLIB:

* `X_12288` (start row) is bounded in `[0, 62]` (integer-valued in
  practice, though the spec declares Real).
* `X_12289` (start col) is bounded in `[0, 62]`.

Both bounds come straight from the VNNLIB input box, so the box is
the universe of possible crop origins.

## Why sample substitution is wrong here

`_evaluate_constant_subgraph(slice_starts, allow_sample_substitution=True)`
returns the sample's value of `X_12288, X_12289` (some single integers,
typically the center of the input box). The ACT IR built on that
substitution then bakes the crop position to those single integers,
and any verification claim is only correct for that sample — not for
the entire input box `[0, 62]²`.

The R12 fail-closed gate (`torch2act.py:331`) blocks this implicit
downgrade. CCTSDB therefore now reports `cannot resolve starts/ends`
rather than silently emitting an unsound IR. This is the correct
behaviour; the task is to add a sound representation, not to remove
the gate.

## Sound representations — three options ranked by precision

### Option A: LUT_BOUNDS layer (envelope)

Most general; least precise. Emit a single layer that produces the
**element-wise envelope** of the crop over the entire `(X_12288,
X_12289) ∈ [0, 62]²` box:

```
For each output spatial position (i, j) in the FIXED output window:
    candidate_source_positions = {
        (start_h + i, start_w + j)
        for start_h in [0, 62]
        for start_w in [0, 62]
        # taking integer values in the box (62*62 = 3844 candidates,
        # or 63*63 = 3969 if we include both endpoints).
    }
    out_lb[i, j] = min over candidate (h, w) of T[h, w]
    out_ub[i, j] = max over candidate (h, w) of T[h, w]
```

* **Sound**: every concrete crop is contained in `[out_lb, out_ub]`.
* **Implementation cost**: one new `LayerKind.SLICE_LUT_ENVELOPE`,
  REGISTRY entry, transfer function (interval only — HZ would lose
  inter-position correlations anyway), torch2act handler, schema test,
  containment test.
* **Conversion cost**: per output position, a fixed `O(K²)` min/max
  over a static tensor (cheap; T is a small static feature map).
* **Verification cost**: identical to a CONSTANT layer once emitted
  (zero-indegree, seeded by `analyze()`).
* **Precision cost**: very loose; the envelope merges 3969 different
  crops into one. Likely yields UNKNOWN on most CCTSDB instances
  because the union is wide enough to satisfy the unsafe region too.

### Option B: Disjunctive verification (case split)

Convert one VNNLIB instance into K² × Q sub-instances, where K² is the
number of valid integer (start_h, start_w) pairs (~3969) and Q is the
existing per-instance VNNLIB query count. Each sub-instance has a
STATIC slice (no dynamic op needed at all). The overall verdict is

  * CERTIFIED iff every sub-instance verifies CERT.
  * FALSIFIED iff any sub-instance falsifies (witness valid for at
    least one start position).
  * UNKNOWN otherwise.

* **Sound**: case-splitting on the integer-valued input vars is
  exhaustive over the integer lattice in the box.
* **Implementation cost**: orchestrator-level; the cli emits N
  sub-runs and joins. No new LayerKind.
* **Verification cost**: ~3969× the per-instance cost on a 25 s
  budget per sub-instance = ~28 hours per instance. Prohibitive
  without parallelism + heavy pruning.
* **Precision**: tight — each sub-instance is exact.
* **Soundness concern**: the VNNLIB declares the start vars as Real,
  not Int. Case-splitting on integers is correct iff the model's
  `Cast(...Int)` truncates / rounds; we must inspect that cast. If
  it's `floor`, integers `0..62` are sufficient; if it's `round`,
  half-integer ties may need additional coverage. The handler must
  reject any cast policy it does not yet model.

### Option C: Hybrid LUT_BOUNDS + lazy refinement

Start with Option A's envelope; if the resulting verdict is UNKNOWN
and the unsafe region is "close" to the envelope's upper bound, split
the start range into K halves and retry the envelope on each. This is
the standard tree-of-boxes refinement that BaB-style verifiers use,
adapted to crop ranges.

* **Sound**: each level remains a sound envelope; finer envelopes
  refine, never widen.
* **Implementation cost**: highest; reuses Option A's LayerKind but
  needs a verifier-side refinement loop.

## Rank-1 native input convention (orthogonal but related)

CCTSDB's ONNX declares `input` as shape `[12296]` (rank-1, no batch).
The current ACT data loader (`onnx_converter.py:205`) treats the
first dim as batch and warns "ONNX model has batch size 12296".
`onnxruntime`'s session refuses `(1, 12296)` for this model (forward
fails on internal reshape mismatch). The model **only** accepts
`(12296,)`.

This is independent of the dynamic slice. The fix is:

* Treat a rank-1 ONNX input whose VNNLIB has the same numel as a
  rank-1 tensor, not a batched one.
* Add an explicit `original_rank` field to the ACT Net to remember
  the model's native input rank.
* Adjust ORT-replay helpers and `_eval_unsafe_strict` to pass the
  flat array straight through when `original_rank == 1`.

This needs to land before Option A/B/C, because the LUT envelope is
indexed by the input shape of the **original** model, not the
batched view ACT currently assumes.

## Decision

* **Today**: do **not** implement any of A/B/C. Status of CCTSDB
  remains *Blocked — data-dependent OnnxSlice*. The fail-closed gate
  is the only correct interim behaviour.
* **Next implementation slot for CCTSDB**: do the **rank-1 input
  convention fix** first (one-day work), then **Option A** as the
  smallest sound representation (one-day work for the layer + tests,
  plus a containment audit against ORT). Option A's likely-UNKNOWN
  verdict is acceptable as a starting point; Option C is only worth
  building if Option A's UNK rate is unacceptably high.
* **Never**: re-enable `allow_sample_substitution` in any handler
  that touches dynamic shapes / indices / branches. The R12 gate is
  load-bearing.

## Test plan when the implementation lands

1. Unit test for the LUT_BOUNDS transfer: build a tiny `(T, start_box)`
   case where the expected envelope is hand-computed; assert
   tf returns exactly that.
2. Sampling-based containment: for the CCTSDB patch-1 model, draw N
   inputs uniformly from the VNNLIB input box, forward via ORT to get
   `y_ort`, and verify `y_ort` lies inside the ACT envelope at every
   output position. Reuse the
   `scripts/audit_nn4sys_ort_containment.py` shape for nn4sys.
3. Rank-1 native input regression: assert ORT and ACT both accept
   `(12296,)` flat input and produce matching shape.
4. End-to-end 5-instance gate under the strict watchdog
   (`--strict-bounded-failure --rss-cap-gb 4 --wall-s 60`). Expected
   counts: `5 UNK + 0 ERR` is the success criterion for Option A.

Until all four exist, CCTSDB stays at `Blocked` in the support matrix.
