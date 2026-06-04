# ACT Frontend / Parser Cleanup Plan — Yolo Slice (and friends)

**Status**: optional engineering work, deliberately separated from the
verification main line. **This document does NOT describe a
verification-capability improvement** and any results from it
**MUST NOT modify the 253 V/A headline** without a separate gate.

---

## 1. Why this is a separate doc

The 2026-06-04 results memo (`research/results_20260604.md`) freezes the
project's verification-capability at **253 V/A** under the principle set
in `paper_skeleton_20260604.md` Section 2. Three closure analyses
(CIFAR-ImageHZ, VGG/Tiny-ImageHZ, CIFAR final-tail per-neuron hull)
delineate the precision ceiling.

The 39 cctsdb_yolo_2023 ERROR rows in the canonical sweep are
**not a verifier output** — they are an ONNX frontend parser gap.
Fixing the parser is engineering work, comparable to adding support
for a new ONNX opset. It is therefore:

- tracked here, not in the verification roadmap §7;
- gated on its own correctness criterion (parser-level), not on a
  V/A gate;
- explicitly NOT counted toward the 253-V/A baseline.

This separation matters for the paper's audit-receipt contract:
the paper distinguishes engineering changes (parser support) from
research changes (relaxation tightening), and they will not be merged
under the same header.

---

## 2. The specific gap

### 2.1 Symptom

In the §9 clean canonical sweep run on cctsdb_yolo_2023, all 39 iids
return:

```text
UNSUPPORTED_AS_UNKNOWN:
  ValueError: OnnxSlice at slice_23: cannot resolve starts/ends
```

The failure happens in ACT's ONNX-graph translation, BEFORE any
verification work runs. The verifier never gets a chance to see the
model in HZ form.

### 2.2 Root cause (advisor 2026-06-04, corrected)

The ONNX `Slice` operator's `starts` and `ends` parameters can be
provided in two ways:

1. as static attributes on the node, or
2. as additional inputs (the post-opset-10 form).

`slice_23` in the cctsdb-yolo model uses form (2), and the `starts` /
`ends` inputs are **NOT constants that can be folded**. They are
**data-dependent expressions of the input box**, of the shape

```text
Gather(input, const_idx) -> Cast -> Unsqueeze -> Add(+1)
```

i.e. the slice indices come from reading a value out of the network
input, casting it to int, and arithmetic. A constant-folding pass
would (a) fail to evaluate them, or (b) silently substitute the input
box's CENTER and produce an unsound result — the resulting Slice
would be the crop at one sample point, not the envelope over the
input box.

**Constant folding is therefore the WRONG fix.**

### 2.3 Correct fix: bounded dynamic Slice envelope

The fix is an interval-resolved Slice that takes a sound over-
approximation across all possible runtime slice windows derivable
from the input box.

```text
For each Slice op whose starts/ends are data-dependent:
  1. Walk the index expression subgraph.
  2. Recognize the supported patterns:
       Gather(input, const_idx)
       Cast(int<->float, on resolved interval)
       Unsqueeze(on resolved interval)
       Add / Sub by constant
       Clip by [min, max] constants
     For each, derive an integer interval [s_lb, s_ub] from the
     VNNLIB input box bounds. (Walk fails-closed if any subgraph
     node is outside this allow-list.)
  3. If the SLICED tensor is a static initializer or a sealed-
     bounds tensor (i.e. one whose per-element bounds are already
     known at parse time via LUT_BOUNDS infrastructure), build the
     element-wise envelope across every window
       w_k = sliced[s_lb + k : s_lb + k + window_size]
     for k in [0, s_ub - s_lb], take min/max per output position.
     The resulting tensor is a sound over-approximation that
     contains every realizable crop.
  4. If the SLICED tensor is a general runtime activation (i.e. NOT
     a sealed-bounds tensor), the fix fails closed: raise
     UNSUPPORTED_AS_UNKNOWN. No silent fallback.
```

This uses the existing LUT_BOUNDS infrastructure (referenced for
discoverability):

- `test_lut_bounds_envelope.py` — the envelope test pattern
- `tf_mlp.py:213` — LUT bound table consumer
- `hybridz_tf.py:80` — LUT bound table producer

### 2.4 Why the failed-closed branch is acceptable

If `slice_23` (or any other Slice) slices into an activation whose
bounds are not pre-computed as LUT_BOUNDS, the fix returns
`UNSUPPORTED_AS_UNKNOWN` — the iid stays UNKNOWN, not silently
"covered". This preserves the soundness contract: an iid is never
moved into V/A by a parser change that did not actually compute
sound bounds.

---

## 3. Hard rules for this cleanup

| Rule | Why |
|---|---|
| R1: this work does NOT modify the 253 V/A headline | the paper's central result is the principle-respecting verification number; engineering changes to the parser are separate |
| R2: if a yolo iid moves from ERROR to V or A after the cleanup, it is reported as "covered after parser fix", NOT as a V/A delta against 253 | otherwise we re-introduce the "+N V/A from a non-verification change" pattern the paper explicitly avoids |
| R3: no benchmark-name-gated patches | the parser change must apply to ALL Slice ops, not just `slice_23`; otherwise the fix is a benchmark patch in disguise |
| R4: the parser fix must not change behavior on benchmarks that already pass (CIFAR, Tiny, nn4sys, malbeware) | parity smoke required: rerun §9 cifar + tiny + nn4sys + malbeware after the parser change and confirm 253 V/A unchanged |
| R5: NO retroactive reframing of the 39 ERROR rows | they were 39 ERROR in 2026-06-04; a parser fix in 2026-06-XX does not retroactively make them V/A in the 2026-06-04 result |

---

## 4. Scope

| In scope | Out of scope |
|---|---|
| Index interval resolver for the allow-listed subgraph patterns (Gather(input,const_idx) / Cast / Unsqueeze / Add±const / Sub±const / Clip(min,max)) | general symbolic-shape analysis, full ONNX graph evaluation |
| LUT_BOUNDS envelope construction for Slice over static or sealed-bounds tensors | Slice over arbitrary runtime activations (those stay UNSUPPORTED_AS_UNKNOWN) |
| Generic OnnxSlice path for bounded dynamic starts/ends, with cctsdb `slice_23` as the first regression case | benchmark-name-gated or node-name-gated handling that only recognizes `slice_23` |
| A parity-smoke harness that reruns CIFAR + Tiny + nn4sys + malbeware and asserts unchanged V/A | any change in HZ propagation or LP code |
| A standalone receipt format note: each "covered after parser fix" iid is recorded with `parser_fix_stamp` so it is distinguishable from the 2026-06-04 baseline | folding the new rows into the 2026-06-04 combined summary |

---

## 5. Acceptance criteria (engineering gate)

The cleanup is acceptable iff ALL of:

1. **parity_smoke_pass**: rerunning the §9 canonical sweep on
   cifar100_2024 + tinyimagenet_2024 + nn4sys + malbeware produces
   the identical V/A iid set as `audit_results/clean_canonical_combined_summary_20260604.json`.
2. **no_new_error**: the parser fix does not introduce any new
   ERROR row on any of the four parity-smoke benchmarks.
3. **slice_23_resolves_soundly**: the cctsdb-yolo iids that previously raised
   `OnnxSlice at slice_23` now either return a real verifier
   verdict (V / A / UNK) through a bounded dynamic-Slice envelope, or a
   different, more specific UNSUPPORTED error (which then becomes the
   next gap to investigate).
4. **soundness_unchanged**: the parser fix does not change any FAL
   receipt's `xi_star` for any cifar / tiny / nn4sys / malbeware
   iid that already produced a FAL receipt in the 2026-06-04 sweep.

A FAIL on any criterion → revert the parser change. Multiple revert/
re-attempt cycles are acceptable — this is engineering, not research.

---

## 6. Reporting

When the cleanup lands, it is reported as **one of**:

- "ACT Slice parser support added (engineering); N previously-ERROR yolo
  iids now produce verifier verdicts" — followed by a NEW table of
  yolo-only counts, separate from Table 2 in the paper skeleton.

- "ACT Slice parser support attempted but FAILED parity smoke; reverted"
  — with a short note on what broke.

- "ACT Slice parser support deferred" — if the engineering load is too
  high to justify against the gain.

NONE of these reportings modify the headline 253 V/A or are presented
as a verification-capability improvement.

---

## 7. Priority

Optional. Can run in parallel with the §10 paper handoff but does NOT
block it. If a deadline forces a trade-off, the paper handoff wins.

---

## 8. 2026-06-04 implementation update — fixed-shape subset landed, cctsdb still blocked

The bounded dynamic-Slice plan in §2.3 has now been partially
implemented as a **generic, fail-closed parser subset**:

- `act/pipeline/verification/torch2act.py` now plumbs sound BOX
  input bounds from `InputSpecLayer` into `_LayerGraphBuilder`.
- `act/pipeline/verification/utils.py::_convert_OnnxSlice` now tries
  a bounded dynamic-Slice `LUT_BOUNDS` fallback when `starts` / `ends`
  are not constants.
- The supported index grammar is deliberately small:

```text
constant
Gather(input, const_idx)
Cast(expr)              # float -> int uses trunc-toward-zero interval semantics
Unsqueeze(expr)
Concat(exprs)
Add/Sub(expr, constant)
```

- The sliced tensor must be a static initializer or a materialized
  constant layer. Slice over a general runtime activation still fails
  closed.
- Every dynamic window must have a fixed output shape over the whole
  VNNLIB input box. If the start interval can produce an out-of-bounds
  or empty slice, the parser refuses the `LUT_BOUNDS` approximation
  because a fixed-shape envelope would miscompile ONNX shape semantics.

This subset is covered by:

- `tests/test_dyn_slice_envelope_parser.py`
- `tests/test_lut_bounds_envelope.py`
- `tests/test_constant_eval_failclosed.py`

### What changed in the cctsdb diagnosis

The first optimistic interpretation was: cctsdb `slice_23` is a
dynamic start into a static tensor, so `LUT_BOUNDS` should turn the
39 ERROR rows into normal verifier rows.

That interpretation is incomplete. Instrumenting the actual failing
FX node showed:

```text
slice_23 data tensor shape: [1, 3, 64, 64]
axes: [1, 2]
start intervals: [0, 62], [0, 62]
end intervals:   [1, 63], [1, 63]
window:          1 x 1
```

Axis 1 has dimension 3, but the start range includes values 3..62.
Therefore the Slice can produce out-of-bounds / empty windows for
some inputs and non-empty windows for others. A single fixed-shape
`LUT_BOUNDS` layer cannot represent that ONNX behavior soundly.

So the correct current status is:

```text
fixed-shape bounded dynamic Slice: supported
cctsdb_yolo_2023 slice_23: still unsupported, now for a sharper reason
```

This is not a regression and not a verifier weakness; it is a
front-end symbolic-shape limitation. In the current engineering branch,
this sharper fail-closed condition is classified by the CLI as UNKNOWN
with the diagnostic preserved in the `error` field; the frozen
2026-06-04 paper table still records the original 39 ERROR rows.

### What remains for a real cctsdb cleanup

A proper cctsdb cleanup now requires one of the following larger
engineering routes:

1. **Symbolic / union-shape Slice abstraction.** Represent the fact
   that a dynamic crop may be empty for some input values and non-empty
   for others, without splitting the input box. This is the principled
   route, but it touches tensor-shape semantics across the parser and
   downstream shape bookkeeping.

2. **Prove the invalid-start region is unreachable.** This would need
   a forward abstract proof over the index-producing subgraph. If the
   proof succeeds, the fixed-shape `LUT_BOUNDS` subset can apply. If
   the proof fails, the iid stays UNKNOWN/ERROR. No sample substitution
   is allowed.

3. **Keep cctsdb parser-blocked for the paper.** This is the honest
   short-term position. The 253 V/A headline remains unchanged, and
   Table 2 keeps `cctsdb_yolo_2023: 39 ERR (frontend parser gap)`.

Input splitting over valid vs invalid start ranges would solve the
shape problem operationally, but it is not allowed under the project's
No-BaB / no input-splitting principle.

### Reporting

The implementation above is an engineering guardrail, not a
verification result. It does not change the 253 V/A headline. If the
new CLI classification is used in a post-freeze engineering sweep,
report cctsdb as `UNKNOWN (variable-shape Slice unsupported)` rather
than as 39 parser crashes. If a future cctsdb route succeeds, report it
as "covered after parser symbolic-shape cleanup", separately from the
frozen 2026-06-04 verification-capability table.

### Pickup checklist

1. Keep `tests/test_dyn_slice_envelope_parser.py` passing; it protects
   the fixed-shape dynamic Slice subset.
2. Add a cctsdb-specific negative parser test that asserts
   `slice_23` fails closed with the variable-shape reason.
3. If pursuing cctsdb further, design symbolic / union-shape Slice
   before touching verifier relaxation code.
4. Run parity smoke on cifar / tiny / nn4sys / malbeware before
   counting any parser cleanup output.
