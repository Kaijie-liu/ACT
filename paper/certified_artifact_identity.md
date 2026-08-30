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

These observations motivate an explicit certificate identity.  Every B3 result
must bind all of the following fields:

- Python, PyTorch import version, TorchVision import and package-metadata
  versions, NumPy, CUDA runtime, and NVIDIA driver;
- official source commit, checkpoint hash, router hash, dataset archive hash,
  and ordered-input identity;
- the complete preprocessing graph, operation order, constants, dtypes, and
  input domain;
- solver versions and the numerical, integrality, feasibility, positive-margin,
  and outward-rounding policies used to derive a verdict.

Missing or changed identity fields fail closed.  A label such as "bilinear
resize" is not enough to identify a certifiable program.  This is a scoped
artifact-semantics result: it does not claim that either runtime is generally
incorrect, nor that floating-point inference has a unique real-arithmetic
interpretation.

The excluded and accepted evidence is anchored by
`act/pipeline/moe/results/icml2025_rt_er/tinyimagenet_router_census_k20_20260830_r2.json`.
The B3 manifest schema is frozen in
`act/pipeline/moe/configs/icml2025_b3_seed0.json` and is materialized by the
prepare and CROWN workers rather than reconstructed after verification.
