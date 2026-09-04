# What exactly is the certified artifact?

An input transform is part of the program being certified, not merely a data
loader convenience.  This distinction became observable in the official
TinyImageNet RT-ER pipeline.  The released program converts decoded pixels to
`float16`, applies antialiased bilinear resize to 224 by 224, normalizes in
`float16`, and then evaluates an affine hard router.  Replacing that path by the
natural real-arithmetic abstraction changed 111 of 200,000 clean routing
decisions in the first excluded execution.  That run is retained for audit but
contributes no scientific endpoint.

A second fail-closed execution showed that naming the transform is still not a
complete semantics.  For the same nominal bilinear resize (`align_corners=False`,
`antialias=True`), the PyTorch 2.9.1 CUDA float16 kernel used by ACT produced
pixels as large as 255.125, whereas the released Blackwell-compatible PyTorch
2.11.0 runtime kept every materialized validation pixel in `[0,255]`.  The run
stopped before its first census row.  No clamp or silent runtime substitution
was introduced.

The accepted execution therefore materializes and hashes all literal resized
centres inside the released compatibility runtime before ACT performs the
real-affine support calculation.  It separately replays the released float16
normalization.  Literal normalization changes 42 of 200,000 clean routes
relative to the continuous real-affine abstraction, but an independent audit
establishes that the 42 disagreements have empty intersection with the formally
stable endpoints at every registered radius.  This is the hard condition that
prevents finite-precision route drift from inflating the reported applicability.

A third failure mode appears before network propagation. A verifier API may be
asked to certify a real-valued radius `epsilon>0` while its float32 frontend
materializes `x-epsilon` and `x+epsilon` as the same tensor as `x`. In the
AdvMoE numerical-reach probe, rank 3 has a positive requested radius below
approximately `1.49e-8` but an effective box width of exactly zero. The
resulting computation is a point check, not a certificate for the requested
real ball. All five observed CROWN sign transitions coincide with expansion to
a new representable float32 box. This is not an AdvMoE-specific property: a
one-dimensional identity network reproduces the same positive-request,
zero-width represented set in the executable regression suite.

The request and the represented set are therefore separate certificate
identity fields. ACT records the requested radius together with the dtype,
shape, hashes of the represented lower and upper tensors, hashes of per-side
deltas and total coordinate widths, effective one-sided radii, and zero-width
coordinate counts. A positive backend bound over a represented set that is
strictly smaller than the requested real ball cannot be promoted to a
real-domain certificate. This finding is scoped to the microscopic-radius
regime: at registered radii of at least `0.5/255`, the audited boxes are not
ULP-degenerate. It does not imply a material ULP error at ordinary robustness
radii.

A fourth ambiguity is stateful-layer mode. A fresh AdvMoE router with 19
BatchNorm layers denotes one function in eval mode with default running
statistics and another in train mode with current-batch statistics. Across 20
official-construction-order seeds on the ordered CIFAR-10 test stream, the
former is exactly single-expert collapsed for 13 seeds and the latter for 8;
median maximum expert share is 100% and 99.305%, respectively. The train-mode
row is an ordered-test co-batch diagnostic rather than the literal augmented
training stream. It shows that BatchNorm semantics materially changes the
degree of the observed collapse without explaining it away. Trained telemetry
therefore binds its mode and statistics identity and reports both registered
semantics from fresh checkpoint copies.

The distinction is stronger than a mode flag. In train mode, the route of one
image can depend on the other examples in its batch, so the router is not a
single-input function unless the co-batch rule is specified. Formal per-input
verification targets eval mode with checkpoint running statistics; the
train-mode row is only a training-dynamics diagnostic. We relate the
near-constant initialization to established signal-propagation/rank-collapse
and MoE load-balancing literature, and claim the routed manifestation and its
verification consequences rather than rediscovering those phenomena.

These observations motivate an explicit certificate identity.  Every B3 result
must bind all of the following fields:

- Python, PyTorch import version, TorchVision import and package-metadata
  versions, NumPy, CUDA runtime, and NVIDIA driver;
- official source commit, checkpoint hash, router hash, dataset archive hash,
  and ordered-input identity;
- the complete preprocessing graph, operation order, constants, dtypes, and
  input domain;
- both `requested_radius` and the fully identified `represented_set`, including
  the represented lower/upper tensors and per-coordinate effective widths;
- the mode of every stateful layer and the identity or derivation rule for its
  running or current-batch statistics;
- solver versions and the numerical, integrality, feasibility, positive-margin,
  and outward-rounding policies used to derive a verdict.

Missing or changed identity fields fail closed.  A label such as "bilinear
resize" is not enough to identify a certifiable program.  This is a scoped
artifact-semantics result: it does not claim that either runtime is generally
incorrect, nor that floating-point inference has a unique real-arithmetic
interpretation.

The reusable representation record and its point-collapse regression are in
`act/pipeline/moe/certified_artifact_identity.py` and
`act/pipeline/moe/test_certified_artifact_identity.py`. They diagnose the set
passed to a numerical backend; they do not retrofit outward rounding into
auto_LiRPA. ACT's HZ support path separately applies outward slack and
`nextafter` toward the unsafe direction before using a solver bound.

The excluded and accepted evidence is anchored by
`act/pipeline/moe/results/icml2025_rt_er/tinyimagenet_router_census_k20_20260830_r2.json`.
The B3 manifest schema is frozen in
`act/pipeline/moe/configs/icml2025_b3_seed0.json` and is materialized by the
prepare and CROWN workers rather than reconstructed after verification.

The B2 conformance gate exposes one further identity boundary on the final
checkpoint. The released B1 evaluation uses literal float16 preprocessing and
autocast, whereas B3 explicitly uses a real-affine normalization represented
in float32. They differ on four clean predictions and one clean route over the
ordered 10,000-image test set. Neither result is silently relabeled: B3 claims
refer only to the real-float32 program, and the independently frozen 20-sample
B3 cohort must agree under both identities. On that cohort, routes and
predictions agree exactly. The complete cross-runtime expert conversion then
passes on 2,000 clean/adversarial inputs with independent audit and zero issues.
