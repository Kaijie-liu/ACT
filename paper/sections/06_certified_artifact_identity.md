# What exactly is the certified artifact?

A robustness claim binds a mathematical property to an executable artifact.
That identity is often described by a model checkpoint and a perturbation
radius. Our audits show that this is insufficient for routed models: the input
transform, the runtime implementation of that transform, and the numerical set
actually passed to a verifier can each change the program being certified.

## Finding 1: preprocessing is part of the routed program

The released TinyImageNet RT-ER pipeline decodes pixels, converts them to
float16, applies antialiased bilinear resize to 224 by 224, normalizes in
float16, and evaluates an affine hard router. A natural real-arithmetic
abstraction of that transform changes 111 of 200,000 clean routing decisions
relative to the literal pipeline. Because a route determines which expert is
executed, this is not a harmless difference in a data loader: it changes the
program path.

We therefore excluded that run instead of treating the continuous transform as
the official artifact. The accepted census materializes and hashes the literal
resized centres before applying the real-affine support calculation, then
replays the released normalization. Literal normalization still changes 42 of
200,000 clean routes relative to the continuous abstraction. The formal stable
endpoint set has empty intersection with those 42 disagreements at every
registered radius, which is the fail-closed condition needed to use the
continuous support calculation without inflating applicability.

## Finding 2: an operation name does not determine its floating semantics

Even the full transform parameters do not identify one floating computation.
For nominally identical bilinear resize settings—`align_corners=False` and
`antialias=True`—the PyTorch 2.9.1 CUDA float16 kernel used by one environment
produced values up to 255.125, while the released Blackwell-compatible PyTorch
2.11.0 runtime kept the materialized validation pixels within the registered
range. The former run stopped before its first census row. We neither inserted
a clamp nor silently substituted the other runtime.

The finding is not that one runtime is generally wrong. It is that a transform
name is not a complete certificate identity. Runtime version, device kernel,
operation order, dtype, constants, and the materialized input identity belong
to the claim. This observation is especially visible in routed systems because
small numerical differences can change discrete execution paths.

## Finding 3: the requested and represented perturbation sets can differ

A third distinction arises at the verifier frontend. Suppose an API is asked to
certify a real-valued (L_\infty) ball of radius (\epsilon>0). A float32
frontend commonly constructs lower and upper tensors by evaluating
`x-epsilon` and `x+epsilon`. If (\epsilon) is below the local ULP scale, both
expressions can round back to `x`. The backend has then verified a singleton,
not the requested real ball.

This situation occurs concretely in the AdvMoE numerical-reach probe. For rank
3, a positive requested radius below approximately (1.49\times10^{-8})
produces an effective box width of exactly zero. Across five inputs, every
observed CROWN sign transition coincides with expansion to a new representable
float32 box. A one-dimensional identity-network regression reproduces the
positive-request, zero-width set without any MoE component, showing that the
phenomenon belongs to the frontend representation rather than the model.

Accordingly, `requested_radius` and `represented_set` are distinct identity
fields. Our runner records the dtype and shape, hashes of the centre and
represented lower and upper tensors, hashes of both one-sided deltas and every
coordinate width, effective one-sided radii, and the count of zero-width
coordinates. The tensors recorded by this identity function are the tensors
passed to the backend. A positive lower bound over a represented set that is
strictly smaller than the requested real ball cannot be reported as a
real-domain certificate.

This finding has a narrow numerical scope. At the registered robustness radii
of at least 0.5/255, the audited input boxes are not ULP-degenerate. We do not
claim a material ULP error at ordinary radii. The microscopic experiment also
does not measure a sound CROWN reach because the installed backend does not
provide a validated outward-rounding argument. ACT's HZ solver follows a
different policy: it applies an absolute-plus-relative outward slack and then
`nextafter` toward the unsafe direction before consuming a support bound.

## Certificate identity and fail-closed use

Every final result binds five groups of fields: source and checkpoint hashes;
ordered dataset and preprocessing identities; exact runtime and device
versions; requested and represented perturbation sets; and solver tolerances,
integrality, positive-margin, and outward-rounding policies. A missing or
changed field invalidates reuse of the certificate. This policy turns the
three observations into a constructive requirement rather than a post-hoc
warning.

The same discipline changes how negative results are handled. The continuous
TinyImageNet run, the out-of-range resize run, and the microscopic CROWN
positive-bound probe are preserved as auditable evidence but excluded from
scientific endpoints they cannot support. The compliant AdvMoE gap figure uses
ordinary radii and a dimensionless relaxation-inflation statistic instead of a
ULP-sensitive radius axis. Its approximately (10^{11}) values compare the
drop permitted by a relaxation with the drop observed by the strongest attack;
they are not certified approximation ratios or bounds on the unknown true
reachable margin.
