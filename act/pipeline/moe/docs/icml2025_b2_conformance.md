# ICML 2025 RT-ER B2 semantic conformance

Status: r1/r2 retained as failed preregistered runs; r3 completed and
independently audited with zero issues.

B2 closes the semantic boundary between the released epoch-130 checkpoint and
the interfaces used by B3. It is not an accuracy or robustness experiment. The
cohort is the first 1,000 inputs in the official Torchvision CIFAR-10 test order,
without filtering by correctness, route, or margin. Each input is checked twice:
as the clean uint8 image and as the independently audited B1 PGD-50 endpoint.

The ACT-side reference checks the following identities:

1. the released normalized-input affine router against the router folded into
   unit-pixel coordinates;
2. every released expert against `PixelNormalizedExpert` specialization;
3. explicit expert selection against the grouped hard-dispatch output;
4. predictions and routes against the independently replayed B1 endpoint;
5. eval-mode BatchNorm state and checkpoint/module hashes.

The pinned alpha-beta-CROWN environment then reconstructs all four specialized
experts from the same checkpoint and compares both direct PyTorch and
`auto_LiRPA.BoundedModule` concrete forward outputs with the frozen ACT-side
reference. Saving the complete 1,000 x 2 x 4 x 10 output arrays lets the
independent auditor recompute every maximum error and prediction agreement
without trusting worker summaries.

The r1 configuration evaluated all experts in batches of 40 while the released
hard-dispatch program calls one expert per input. Although routes agreed exactly
and the raw-pixel wrapper had zero error, CUDA batch-shape rounding produced a
maximum selected-logit difference of `3.34e-4`, exceeding the frozen `1e-4`
threshold. That run is permanently excluded and recorded in
`results/baseline/icml2025_rt_er_b2_seed0_r1_failure.json`. The r2 protocol
restores released batch-1 dispatch semantics without relaxing any tolerance.

The r2 run then exposed a separate identity boundary: the real-float32 B3 graph
and the released B1 literal-fp16-plus-autocast graph differ on four clean
predictions and one clean route over all 10,000 inputs. They are different
floating-point programs, so r3 reports the drift and forbids describing B3 as
certifying the literal mixed-precision execution. The independently frozen B3
20-sample cohort has exact clean route and prediction agreement across both
identities.

The current frozen config is
`act/pipeline/moe/configs/icml2025_b2_seed0_r3.json`. Passing requires exact
non-tie route and prediction agreement, finite errors below the registered
tolerances, all BatchNorm layers in deployment eval mode, unchanged official
source/checkpoint/endpoint identities, and an independent zero-issue audit.
Positive conformance does not imply a robustness certificate or validate any
CROWN lower bound. It establishes conformance only for the explicitly identified
real-float32 B3 program.

## Audited r3 result

Both 1,000-sample families pass. The maximum folded-router error is
`4.85e-7`; fixed-expert wrapping and selected gathering are exact under the
released batch-1 execution shape. Across PyTorch 2.9.1/CUDA 12.8 and PyTorch
2.11.0/CUDA 13.0, the maximum direct expert-logit difference is `1.26e-4`.
The concrete auto_LiRPA conversion differs from its direct PyTorch module by at
most `5.09e-5`, and predictions agree on all 2,000 inputs. All 80 BatchNorm2d
layers are in eval mode. The independent audit recomputed every error from the
saved logits and found zero issues.

The result preserves the literal-runtime differences instead of hiding them:
among the first 1,000 inputs, one clean and one adversarial prediction differ,
while two adversarial routes differ. Across all 10,000 clean inputs, four
predictions and one route differ. None intersects the frozen B3 cohort. The
tracked audit manifest is
`results/baseline/icml2025_rt_er_b2_seed0_r3_audit.json`.
