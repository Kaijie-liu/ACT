# ICML 2025 Certificate Applicability Audit

## Asset identity

The authors publicly provide training, attack, and model-definition source at
`TIML-Group/Robust-MoE-Dual-Model`. They do **not** provide a trained checkpoint
or an implementation of the analytic certified radius in Theorem 5.4. These are
three different provenance claims:

| Asset | Availability | Permitted label |
|---|---|---|
| RT-ER training/attack/model code | public, pinned at `30ef94d...` | official author source |
| trained CIFAR-10 RT-ER parameters | not published | unavailable; reproduction or author-supplied artifact required |
| Theorem 5.4 certificate implementation | not published | author-paper formula reimplementation only |

No experiment may describe an ACT implementation as the author's official
verifier. No RT-ER route-radius distribution may be reported until trained
router weights are obtained through a disclosed reproduction or directly from
the authors.

## Paper theorem and explicit assumptions

The paper defines a robust MoE output

```text
F_R(x) = sum_i a_Ri(x) f_Ri(x),
```

where the routing weights are nonnegative and sum to one. Assumption 5.3 states
that every relevant expert output and every router weight is Lipschitz. Theorem
5.4 additionally uses an upper bound `M_Ri <= 1` on the relevant expert output
and derives a margin-over-Lipschitz certified radius.

Applying that theorem requires all of the following to be made operational:

1. the concrete model output must match the weighted-sum semantics used by the
   theorem;
2. the functions called `a_Ri` must be identified and must be nonnegative,
   normalized, and Lipschitz on the entire certified ball;
3. the expert score/probability represented by `f_Ri` must be identified;
4. sound expert Lipschitz constants must be supplied for every required class;
5. sound router-weight Lipschitz constants must be supplied;
6. the `M_Ri <= 1` output bound must hold for the represented quantity;
7. the clean prediction margin in the numerator must use the same output
   semantics and checkpoint;
8. all numerical bounds must be outward-safe.

The paper motivates Assumption 5.3 for sparse MoEs by saying robust training
tends to separate the selected router scores and make the top-n set locally
stable. That motivation is not a per-input proof of the radius over which a hard
dispatch remains stable.

## Released-code semantic audit

The released CIFAR-10 model executes (with `E=4` in the author script)

```text
Flatten -> Linear(3072,E) -> torch.argmax -> selected ResNet18 logits.
```

It does not compute a continuous mixture of all expert outputs. A hard one-hot
gate is locally constant, and hence Lipschitz with constant zero, only inside a
region that does not reach a routing tie. At a reachable argmax boundary it is
discontinuous. Therefore a hard-router application of Theorem 5.4 must either:

- prove that the requested radius lies strictly inside the route-stable region;
  or
- provide a different theorem that covers route changes.

The released experts return raw logits to `CrossEntropyLoss`; they do not apply a
softmax in `forward`. The paper's `M_Ri <= 1` condition is immediate for class
probabilities but not for unrestricted logits. A formula reimplementation must
state whether it certifies a softmax-wrapped classifier or raw-logit margins and
derive the corresponding constants. It may not silently interchange them.

These observations establish an **artifact--theorem applicability question**.
They do not, by themselves, establish that a published certificate is unsound.
That stronger claim would require the exact author checkpoint and certificate
procedure plus a sound counterexample or a proof that the reported conclusion
does not follow.

## Exact affine hard-route boundary

For a clean hard route `i` and affine router scores `r(x)=Wx+b`, with no input
clipping, the minimum tie-inclusive `L_inf` distance to competitor `j` is

```text
(r_i(x)-r_j(x)) / ||W_i-W_j||_1.
```

The route boundary is the minimum over competitors. Under a pixel box, the
support is clipped coordinate-wise. ACT implements the exact piecewise-linear
inversion in `act.back_end.moe.route_boundary.affine_top1_route_boundary` rather
than assuming the unclipped formula.

If the author router consumes normalized pixels `z = scale*x + shift`, callers
must first fold normalization into the affine map. For the released FFCV path,
the per-channel scale is `255/std` and shift is `-mean/std` when `x` is in
`[0,1]` pixel units.

The oracle applies the frozen `1e-9 + 1e-9*scale` outward slack and directed
rounding around the computed radius. A formal comparison uses:

- `epsilon < route_radius_lower`: route stability established;
- `epsilon >= route_radius_upper`: an alternative tie-inclusive route is
  geometrically reachable;
- overlap with the numerical bracket: undecided under the frozen tolerance.

## Certificate decomposition metric

For a fixed model, input distribution, norm, and radius, define

```text
A(epsilon) = {x : epsilon is strictly below the exact route boundary}.
```

Every reported certificate can then be partitioned into:

1. certified and route-stable;
2. certified and route-unstable;
3. uncertified.

This decomposition is a useful evaluation metric, but it does not eliminate the
need to execute a route-invariance verifier on its applicable subset and report
runtime. For the current bal010 model, the exact boundary is obtained through
the nonlinear-router HZ feasibility protocol. For the released RT-ER
architecture, the affine oracle applies once trained weights are available.

The denominator must be stated. A natural test-set decomposition and a
route-boundary-targeted cohort answer different questions and remain separately
labeled.

## Audit outcomes

Let `R_formula(x)` be a faithfully reimplemented analytic radius and
`R_route(x)` the exact hard-route boundary.

| Observation | Sound interpretation |
|---|---|
| `R_formula < R_route` | the hard route is stable throughout the claimed ball; the routing applicability condition is satisfied |
| `R_formula >= R_route` | the hard-gate Lipschitz premise fails somewhere inside the claimed ball; the theorem's applicability to that artifact is not established |
| Route A proves safe beyond `R_route` | a sound route-changing certificate unavailable to a route-invariance-only application |
| Concrete full model violates inside `R_formula` | potential unsoundness evidence, requiring independent replay and exact formula/checkpoint provenance |

An assumption failure is reported as `NOT_APPLICABLE` or
`ASSUMPTION_NOT_ESTABLISHED`, never automatically as `UNSAFE` or
`UNSOUND_CERTIFICATE`.

## Current blockers and next evidence

The official repository contains random initialization code, not trained router
weights. Consequently the following are not yet scientific results:

- RT-ER test-set route-boundary distributions;
- overlap between analytic radii and hard-route boundaries;
- certificate decomposition on the official model.

The next admissible routes are:

1. obtain the original checkpoint from the authors and record its hash and
   redistribution terms; or
2. perform the already specified official-code, paper-config reproduction after
   dependency authorization.

Large public MoE router weights are not a zero-cost substitute. Per-token route
radius distributions also require representative hidden activations, and router
tensors are commonly embedded in large checkpoint shards. Such a study must be
separately scoped as hidden-state local sensitivity, not input robustness or an
end-to-end certificate.
