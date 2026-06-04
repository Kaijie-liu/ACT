# Dense-Conv ImageHZ — Follow-up Research Plan (2026-05-31)

This document is the deliverable from "Step 4" of the user's audit plan
(2026-05-31). It builds on `dense_conv_forward_hz_plan_20260531.md`.

## Context: Steps 1-3 results (autonomous execution)

| Step | Target | Result |
|------|--------|--------|
| 1. SmallDenseDAG hardening | 2 bugs + 3-layer test suite | ✓ 12/12 tests PASS, S2 bug fixed, input-name bug fixed, 0 regression |
| 2. lsnc_relu McCormickDAG | +V/A on 81 lsnc instances | ✗ NOT IMPLEMENTED — pyrat 1/81, NNV/CORA/NeuralSAT skipped this benchmark entirely; structural ceiling for sound forward methods. Documented as negative. |
| 3. ACASXu pairwise k=2 hull | +5–20 V on UNK iids | ✗ 0 V LIFT — implemented `PairwiseHullLP.py`, ran on 5 UNK iids across prop_1/2/3/4/5: all stay UNK. Plus iid 91 timing regression (V→U at 15s wall when hull cuts slow bound LPs). Consistent with historical `project_pairwise_hull_negative_20260516` on ACASXu. |

The pattern across Steps 2-3 is structural: under strict P1-P6 (forward
triangle LP), the precision wall is at LP-relaxation tightness, not at
witness search. CORA closes the cersyve gap via polynomial zonotopes, not
via tighter LP cuts.

## Dense-conv current state

From `project_full_vnncomp_sweep_post_rollback_20260531.md`:

- cifar100_2024: V=0, A=0 (200 instances; 30 OOM, 4 ERR otherwise UNK)
- tinyimagenet_2024: V=0, A=1 (200 instances)
- yolo_2023: V=0, A=0 (72 instances)
- traffic_signs_recognition_2023: V=0, A=0 (45 instances)
- cctsdb_yolo_2023: V=0, A=0 (39 instances)

OOM rerun on cifar100/yolo at 32GB RSS produced 0 V/A — confirming the
0-decision floor is structural, not resource. The previous agent already
ran sequential low-concurrency rerun on 62 ERR/OOM iids; all converted to
sound UNK with 0 V/A recovered.

## Why current HZ is bottlenecked

Two compounding issues in the forward HZ propagation through dense conv:

1. **Generator-to-pixel ratio explodes**. Each perturbed input dim seeds
   one root generator. After dense matrix flattening of a Conv kernel, the
   per-channel generators multiply with kernel-weight matrices, and each
   intermediate feature map has its own (typically larger) generator set.
   At 32×32×3 input × 16 channel × 3×3 conv, we already have generators
   on the order of 10⁵.

2. **Independent per-neuron triangle hulls discard joint correlation**.
   Even with HZ (which has binary generators ξ_b ∈ {-1,+1}), the local
   ReLU relaxation per scalar neuron composes into output points that are
   feasible in the abstract LP but not produced by any real input.
   Result: phantom unsafe corners in the output halfspace; LP says
   "unknown"; ORT replay rejects the LP-derived witnesses; verdict UNK.

## What ImageHZ/ImageStar-HZ would change

The KEY shift: **generators stay in image tensor shape `(n_gen, C, H, W)`
through Conv, AvgPool, Pad, and Skip-Add. They only flatten at the final
Dense/Output layer.** This is exactly what NNV's ImageStar does in the
forward set-based domain.

Concretely, for HZ = (Gc, Gb, c, Ac, Ab, b):
- `Gc` represented as tensor of shape `(n_gen, C, H, W)` not flat `(n, n_gen)`
- Conv layer: `Gc'[g, c', h', w'] = sum_{c, dh, dw} kernel[c', c, dh, dw] * Gc[g, c, h+dh, w+dw]`
  This is just standard conv on each generator — exact, no relaxation.
- AvgPool: tensor mean, exact.
- ReLU: per-neuron triangle still applies, but the generator-to-pixel
  binding is preserved at the tensor level, so the JOINT structure is
  available to a block-level cut (instead of per-neuron).

The forward-only block-level ReLU cuts that ImageHZ enables:
- Whole-channel monotone bound: if a generator `Gc[g, c, :, :]` is
  uniformly non-negative for some channel c, then post-ReLU on that
  channel preserves the generator orientation.
- Patch-level convex hull: pick spatial 2×2 (or kxk) patches and add
  joint hull cuts over the (k×k) ReLU outputs in that channel.
- Channel-aware reduction: when a generator is sparse across channels
  (concentrated in few), keep all channels for that generator and
  reduce others.

This is what Bak's ImageStar paper does, and what CORA-Conv2024 reportedly
extends.

## Estimated effort

- Tensor-shape generator data structure + Conv/Pool/Add/Pad operators
  (exact): ~1000 LOC, 3-5 days.
- Block-level ReLU convex cuts (forward LP only): ~500 LOC + theory work.
- Soundness tests (point-eval ORT match on each operator): ~500 LOC.
- Integration with HZ wrapper for VNN-COMP benchmarks: ~300 LOC.
- Tuning + sweep across 5 CIFAR/Yolo UNK iids: ~1 day.

Total: ~2-3 weeks of focused engineering.

## Concrete first milestone (NOT included in this session)

Per user audit:
> 第一个 milestone 不设全 CIFAR 涨分，而是选 5 个 CIFAR/Yolo UNKNOWN，证明
> output unsafe LP phantom margin 明显下降。

Suggested 5 iids:
- `cifar100_2024 iid=0,2,4` (`CIFAR100_resnet_medium.onnx`)
- `tinyimagenet_2024 iid=0` (`TinyImageNet_resnet_medium.onnx`)
- `yolo_2023 iid=0` (`TinyYOLO.onnx`)

Diagnostic metric: for each iid, measure
1. Pre-ImageHZ: output unsafe LP minimum spec residual = `min { d - c·y :
   y ∈ HZ_output_box, c·y ≤ d }`. If > 0, LP feasible in unsafe halfspace
   (= phantom).
2. Post-ImageHZ: same measure with tensor-shape generators + block ReLU cuts.

Success criterion: ≥50% drop in phantom margin on at least 3 of 5 iids.

If the margin drop is real (3+ of 5), proceed to full benchmark sweep.
If the margin barely moves, the bottleneck is elsewhere (e.g., the conv
operator isn't the dominant phantom source) and the next research target
shifts to block-level ReLU coupling on the existing dense HZ.

## What NOT to do (per user audit)

- Continue rerunning CIFAR/Yolo with bigger RSS budgets — already proven
  the floor is structural, not memory.
- Add more output cuts on the current dense HZ — pairwise hull negative
  is direct evidence that adding cuts after the fact doesn't tighten
  enough.
- Use random/corner/PGD-style witness search — P6 violation.
- Default-enable any new abstraction across all benchmarks without
  per-bench soundness validation.

## Files referenced

- `dense_conv_forward_hz_plan_20260531.md` — earlier 3-direction sketch
- `project_full_vnncomp_sweep_post_rollback_20260531.md` — sweep raw
- `project_smalldense_dag_cersyve_20260531.md` — SmallDenseDAG hardening
- `project_pairwise_hull_negative_20260516.md` — historical pairwise k=2 negative
