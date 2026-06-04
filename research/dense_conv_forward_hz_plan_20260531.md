# Dense-Conv Forward-HZ Bottleneck Plan (2026-05-31)

## Constraints

This plan stays under the project principles:

- no CROWN-style backward bound propagation
- no autograd / gradients
- no Gurobi / MILP
- no fallback verifier
- no input or activation branch-and-bound
- no PGD / random-sample-then-check falsification

Allowed tools remain forward HZ abstractions, structured LPs via HiGHS/Scipy,
exact linear operator transfer, and strict ORT replay for LP-derived witnesses.

## What The Cross-Tool Data Says

The largest ACT gaps are not evenly distributed. They concentrate in:

- `cifar100_2024`: ACT 0 decisions vs abcrown 101
- `tinyimagenet_2024`: ACT 1 decision vs abcrown 140
- `yolo_2023`: ACT 0 decisions vs abcrown 62
- `cctsdb_yolo_2023`: ACT 0 decisions vs abcrown 39
- `traffic_signs_recognition_2023`: ACT 0 decisions, other strict tools also mostly fail

The tools that solve these are not doing a better version of plain forward HZ.
Their wins come from at least one forbidden mechanism:

- optimized backward LiRPA / CROWN-style bound propagation
- branch-and-bound over activation or input domains
- exact-star splitting or MIP-style complete search

The VNN-COMP report explicitly says the strongest tools converge to
GPU-enabled linear bound propagation plus branch-and-bound. The CIFAR/Tiny
benchmarks were also filtered to remove cases verified by vanilla CROWN, so
single-pass forward relaxations are intentionally disadvantaged.

## Current Engineering Findings

Recent fixes removed real implementation blockers:

- zero-width root generator pruning
- sparse/preconv path not blocked by dense dispatch guard
- exact sparse residual `Add` without densifying `SparseGcZ`
- constraint-prefix dedup for sparse shared-generator add
- exact row ops such as Gather/Slice/Pad/ReduceSum/Transpose where applicable

After these fixes, targeted runs show:

- `yolo_2023` no longer fails at the earlier sparse residual-add bottleneck,
  but high-budget runs still end UNKNOWN.
- `cifar100_2024` OOM representatives rerun sequentially become UNKNOWN, not V/A.
- `cctsdb_yolo_2023` ERROR representatives become fail-closed UNKNOWN due to
  unsupported data-dependent Slice.
- `tinyimagenet_2024` OOM representatives rerun sequentially become UNKNOWN.

So the current dense-conv failure is no longer primarily "ACT cannot run the
model"; it is "the forward relaxation contains phantom unsafe LP corners."

## Why Forward HZ Loses Here

Dense image boxes create one generator per perturbed input dimension. In early
conv layers, each ReLU sees a wide linear combination of many correlated input
generators. The local ReLU relaxation adds a convex hull independently per
activation. Across many conv/ReLU/residual blocks, these independent local hulls
compose into output points that are feasible in the abstract LP but not produced
by any real input.

This explains the repeated pattern:

- output LP or interval says unsafe is feasible
- extracted witness is rejected by strict ORT, or no input witness exists
- adding local cuts (`selective_chull`, multi-corner LP, D-filter, K=2) does not
  remove enough global phantom correlation

The hard missing information is cross-layer, spec-conditioned correlation.
ABCROWN gets it through backward optimized slopes and BaB split constraints.
Both are forbidden, so ACT needs a forward-only replacement.

## Principle-Compliant Research Direction

### A. Forward Spec-Conditioned Template Domain

Maintain a small bank of forward template directions alongside HZ:

- selected output-spec rows projected forward as auxiliary linear functionals
- per-block summary templates, not per-neuron backward bounds
- all propagation remains forward through the computation graph
- final certification uses HiGHS infeasibility over the forward template/HZ
  relaxation

This is not CROWN if we do not compute per-neuron backward lower/upper linear
bounds. The design goal is to preserve a few global correlations that the local
ReLU hull discards.

Risk: selecting useful templates without gradients/backward is the hard part.
Allowed heuristics include structural rows from VNNLIB specs, final affine
class-difference rows, and forward sensitivity estimates from interval/HZ widths
computed without autograd.

### B. Block-Level ReLU Coupling

Instead of adding cuts for individual ReLUs, add small group constraints for
conv blocks:

- choose a small set of unstable activations in the same channel/spatial patch
- introduce shared slack variables or aggregate inequalities over the group
- keep the number of extra constraints capped per block
- no branching; one abstract state for the full input box

This borrows the intuition of multi-neuron relaxations but must be implemented
as a single forward over-approximation. It is closer to PRIMA/k-ReLU in spirit,
but without MILP, backward bound propagation, or splitting.

Risk: previous joint K=2 experiments gave 0 lift on YOLO/VGG samples. A useful
version likely needs block/channel structure, not arbitrary pairs.

### C. Exact Linear-Operator Coverage First

For dense-conv benchmarks, every exact linear op matters because one box
fallback destroys all factor correlation. Keep scanning `_dispatch` and
converter paths for unsupported-but-linear ops:

- data-independent Slice/Gather/Pad/Transpose/Reshape/ReduceSum
- ConvTranspose/Upsample/Resize
- QConv dequantization and integer affine forms in traffic-signs
- YOLO decode linear pieces before data-dependent branches

This path already produced the largest ACT gains on non-dense-conv benchmarks.
For dense-conv it may mainly reduce ERROR/OOM, but it is still prerequisite work.

## Near-Term Experiments

1. Low-concurrency rerun all dense-conv ERROR/OOM iids.
   Goal: convert false OOM/EXIT_NONZERO into auditable UNKNOWN.

2. Build an offline diagnostic for each dense-conv UNKNOWN:
   count first layer where abstract unsafe feasibility becomes irreversible,
   number of unstable ReLUs, generator/constraint counts, and strict replay
   rejection mode.

3. Prototype a block-template diagnostic on one CIFAR medium and one YOLO iid:
   no new verifier claim; just measure whether selected block aggregate rows
   tighten the final unsafe LP margin.

Stop condition: if block templates do not reduce unsafe LP feasibility margin on
5 targeted iids, close it as negative and avoid another broad sweep.
