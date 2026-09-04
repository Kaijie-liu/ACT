# ICML 2025 RT-ER B2 semantic conformance

Status: protocol and executable stage frozen; result pending.

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

The frozen config is
`act/pipeline/moe/configs/icml2025_b2_seed0_r1.json`. Passing requires exact
non-tie route and prediction agreement, finite errors below the registered
tolerances, all BatchNorm layers in deployment eval mode, unchanged official
source/checkpoint/endpoint identities, and an independent zero-issue audit.
Positive conformance does not imply a robustness certificate or validate any
CROWN lower bound.
