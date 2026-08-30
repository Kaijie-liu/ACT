# What exactly is the certified artifact?

A robustness claim binds a mathematical property to an executable artifact.
That identity is often described by a model checkpoint and a perturbation
radius. Our audits show that this is insufficient for routed models: the input
transform, the runtime implementation of that transform, the numerical set
actually passed to a verifier, and stateful-layer evaluation semantics can
each change the program being certified.

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

## Finding 4: initialization does not identify BatchNorm semantics

A BatchNorm network does not define one unambiguous function at
initialization. At least three objects can be meant by an “initial model”:
initial weights evaluated with default running means and variances, the same
weights evaluated in training mode with current-batch statistics, or initial
weights evaluated after a data-dependent running-statistics warm-up. The
released AdvMoE training process uses the second kind of function while a
fresh checkpoint loaded for evaluation normally uses the first.

We isolate the first two semantics on 20 official-construction-order router
initializations and the same ordered 10,000-image CIFAR-10 test stream. Under
eval mode with default zero means and unit variances, 13/20 seeds route every
image to one expert; the median maximum expert share is 100%. Current-batch
statistics reduce, but do not remove, the effect: 8/20 seeds remain exactly
collapsed, and the median maximum share is 99.305%. Collapse targets vary
across seeds. The absolute signed-score mean divided by its standard deviation
has median 9.159 in eval mode and 2.474 with ordered-test batch statistics.
Thus the seed-1234 10,000:0 result is neither a single-seed accident nor purely
an eval-default-statistics artifact, but BatchNorm semantics materially changes
its degree.

The train-mode row is a controlled co-batch diagnostic, not a replay of the
literal shuffled and augmented training stream. It is intentionally labelled
as such. More fundamentally, a training-mode BatchNorm router maps a batch to
a batch of routes: the route assigned to one image may change when its
co-batch changes. It is not a well-defined single-input function unless the
co-batch construction is part of the specification. Per-input formal
verification therefore uses eval mode and the checkpoint's recorded running
statistics. Train-mode BatchNorm is used only to study training dynamics, not
as a certification target.

Trained-checkpoint telemetry therefore reports route load, signed
offset, and margin distributions under two identities: eval mode with the
checkpoint's current running statistics, and a fresh checkpoint copy in train
mode over registered ordered test batches. Neither identity is allowed to
select a checkpoint. This distinction lets us ask when training escapes
initial load collapse without silently changing the function under study.

We do not position the observed near-constant initialization as a new general
theory of deep networks. Signal-propagation and rank-collapse phenomena at
random initialization, and MoE load imbalance during training, are established
topics. The new object here is their routed-program manifestation and its
certification consequence: an initial router may have no meaningful partition
to certify, while the function seen during training can differ from the
single-input function later presented to a verifier.

## Certificate identity and fail-closed use

Every final result binds six groups of fields: source and checkpoint hashes;
ordered dataset and preprocessing identities; exact runtime and device
versions; requested and represented perturbation sets; solver tolerances,
integrality, positive-margin, and outward-rounding policies; and stateful-layer
mode plus stored or batch-derived statistics. A missing or changed field
invalidates reuse of the certificate. This policy turns the four observations
into a constructive requirement rather than a post-hoc warning.

The same discipline changes how negative results are handled. The continuous
TinyImageNet run, the out-of-range resize run, and the microscopic CROWN
positive-bound probe are preserved as auditable evidence but excluded from
scientific endpoints they cannot support. The compliant AdvMoE gap figure uses
ordinary radii and a dimensionless relaxation-inflation statistic instead of a
ULP-sensitive radius axis. Its approximately (10^{11}) values compare the
drop permitted by a relaxation with the drop observed by the strongest attack;
they are not certified approximation ratios or bounds on the unknown true
reachable margin.
