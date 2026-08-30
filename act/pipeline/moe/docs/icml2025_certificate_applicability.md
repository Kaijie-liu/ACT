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

## Paper-only questions Q1 and Q2

The PMLR paper and its appendix were audited in full on 2026-08-29.

**Q1: Is Theorem 5.4 numerically evaluated?** No. The paper provides the
formula and proof, but no certified-radius table, curve, certified accuracy, or
per-input theorem evaluation. Tables 1--10 and Figures 2--7 report attack
accuracy, standard accuracy, training curves, or method ablations. They are not
numerical instances of Theorem 5.4 or 5.5. A reproduction would therefore be
the first disclosed numerical instantiation on an official-code model family,
not a reproduction of an author-reported certificate result.

**Q2: How are the Lipschitz constants obtained?** The paper does not specify a
procedure. It defines `L_Ri` and `r_Ri`, says that these constants have been
"optimized during RT-ER", and does not provide spectral-product bounds,
bound-propagation code, sampled-gradient estimates, or numerical constant
values. The nearby suggestion that a practical robust margin can be estimated
on a strong adversarial input is not a sound lower-bound procedure for a formal
certificate. Until a method is supplied, the formula is
`NOT_FORMALLY_INSTANTIATED`.

Any future implementation must classify constants before computing a radius:

- sound global or local upper bounds may support a formal result;
- empirical gradient/attack estimates are diagnostics only and must never be
  labeled certified;
- an unspecified constant source leaves the formula uninstantiated.

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
must first fold normalization into the affine map. The released CIFAR script
stores `mean_255` and `std_255` in raw uint8 units. Therefore, when `x` is in
`[0,1]`, the scale is `255/std_255` and the shift is
`-mean_255/std_255`. This equals `1/std_unit` and `-mean_unit/std_unit`
when statistics are first divided by 255. A regression test evaluates both
domains and requires matching router scores.

`affine_top1_route_boundary_batch` computes all test-set boundaries without a
Python loop over samples. Its general finite-box path uses grouped
`sort+cumsum`. For exact uint8-derived CIFAR inputs, the explicit
`compute_device="cuda", capacity_grid_steps=255` path validates every capacity
against the declared grid, then uses a 256-bin weighted histogram. A synthetic
10,000 by 3,072, four-expert run took 0.712 seconds on the recorded Blackwell
GPU. This is an engineering benchmark, not an RT-ER scientific result.

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

## Pre-registered five-leaf decision tree

Let `R_formula(x)` be a faithfully reimplemented analytic radius and
`R_route(x)` the exact hard-route boundary.

| Leaf | Observation | Sound interpretation |
|---|---|---|
| L1 | no disclosed sound method for `L_Ri`/`r_Ri`, or empirical estimates only | `NOT_FORMALLY_INSTANTIATED`; report estimates only as diagnostics |
| L2 | sound constants exist, but no sample reaches the smallest registered radius `0.25/255` | `VACUOUS_AT_REGISTERED_RADII`; theorem applicability is no longer the numerical bottleneck |
| L3 | a non-vacuous sound `R_formula < R_route_lower` | hard-route applicability is established for that input and radius |
| L4 | a non-vacuous `R_formula >= R_route_upper` | a route boundary lies inside the claimed ball; report `ASSUMPTION_NOT_ESTABLISHED` for the hard-gate application |
| L5 | Route A proves output safety at a radius beyond `R_route_upper` | a sound route-changing certificate outside a route-invariance-only application |

Numerical overlap between the outward brackets is `UNDECIDED_NUMERICAL`, not
assigned to L3 or L4. A concrete full-model violation inside `R_formula` is
potential unsoundness evidence, but that escalation requires independent replay
and exact formula/checkpoint provenance; it is not inferred from L1, L2, or L4.

An assumption failure is reported as `NOT_APPLICABLE` or
`ASSUMPTION_NOT_ESTABLISHED`, never automatically as `UNSAFE` or
`UNSOUND_CERTIFICATE`.

## Released training-gradient audit

The official-code, paper-config reproduction now supplies checkpointed model
weights. At epochs 10 and 20, all 248 expert parameter tensors changed, while
both router tensors were bitwise identical. The Adam checkpoint has state for
all 248 expert parameters and no state for either router parameter.

This is explained by the released source. The shared `Router.forward` returns
integer `argmax` indices and `get_second_expert` returns integer `topk` indices.
All four released training entry points (`cifar10_RT_ER.py`,
`tinyimagenet_RT_ER.py`, `cifar10_JTDMoE.py`, and
`tinyimagenet_JTDMoE.py`) form their losses from expert outputs selected through
those indices. None exposes a differentiable router-score path, an explicit
router loss, or a straight-through gate estimator. This source-level conclusion
covers the four scripts at author commit `30ef94d...`; the cross-epoch tensor and
optimizer-state evidence covers the running CIFAR-10 RT-ER seed-0 reproduction.

The paper treats the router as part of the robust-MoE parameters, says robust
training encourages selected-router-score separation and local route stability,
and says the experts' and router's Lipschitz constants have been optimized
during RT-ER. It does not specify how gradients pass through the released hard
`argmax` dispatch. We therefore record two related but distinct findings:

1. the hard-router differentiation mechanism is underspecified in the paper;
2. the released training artifact does not realize the stated router-optimization
   effect.

This is an artifact-centered result, not an objection to static routing itself.
Static or stochastic expert assignment is a legitimate published design, as in
[Hash Layers](https://arxiv.org/abs/2106.04426) and
[THOR](https://arxiv.org/abs/2110.04260). It also does not establish that
Theorem 5.4 is unsound. Neutral author contact is `CONTACT_MANAGED_BY_PI`; the
agent retains no dates or countdown.

## Current evidence and remaining scope

Because the epoch-20 router is identical to epoch 10, the seed-0 reproduction
already supplies an exact 10,000-input route-boundary census for the released
training artifact, subject to a bitwise drift guard at every later checkpoint.
It does not supply the unpublished author checkpoint. The analytic certificate
still requires a disclosed sound constants provider; the official checkpoint
trajectory alone cannot fill that gap.

The single-seed finding is not an isolated initialization. Twenty
preregistered seeds were reconstructed by instantiating the complete official
four-ResNet18 model before reading its router. Seed 0 matched the immutable
checkpoint router bit for bit. Across seeds, the mean exact route-boundary
reachability fractions at `0.5`, `1`, `2`, `4`, and `8/255` are 47.64%,
75.20%, 94.96%, 99.77%, and 99.9985%. At `8/255`, the formally route-stable
fraction has mean 0.0015% and range 0--0.02%; seed 0 is exactly 0%. Thus
route-invariance applicability is empty for the observed official-code seed-0
artifact and nearly empty under the released pipeline's initialization
distribution. This is not an output-safety conclusion.

Large public MoE router weights are not a zero-cost substitute. Per-token route
radius distributions also require representative hidden activations, and router
tensors are commonly embedded in large checkpoint shards. Such a study must be
separately scoped as hidden-state local sensitivity, not input robustness or an
end-to-end certificate.

## Implemented constants providers

`act.pipeline.moe.certificate_constants` now makes the previously implicit
constant choice executable and provenance-carrying:

- `SOUND_GLOBAL_SPECTRAL` composes induced `L_inf` upper bounds for supported
  affine, convolutional, normalization, activation, pooling, sequential, and
  audited official-CIFAR-ResNet structures. Unknown graphs are rejected rather
  than silently treated as sequential.
- `EMPIRICAL_GRADIENT_SAMPLED` records sampled gradient `L1` norms with the
  permanent status `DIAGNOSTIC_ONLY`.
- `AUTHOR_UNSPECIFIED` returns `NOT_FORMALLY_INSTANTIATED` without inventing a
  value.

For a probability reading, the provider composes the global logit bound with
the sound `1/2` `L_inf`-to-scalar softmax-coordinate bound and records
`M_Ri=1`. A raw-logit reading receives no global `M_Ri<=1` bound unless a
separate sound provider supplies one. A continuous softmax router is handled by
the same `1/2` composition; a hard-argmax router returns `NOT_APPLICABLE` at a
reachable tie and is never assigned the misleading global constant zero.

The Equation (8) evaluator preserves these statuses: empirical or missing
constants cannot produce a formally labelled radius, and unnormalized routing
weights are rejected. This is an ACT paper-formula reimplementation, not author
certificate code. Unit tests cover exact linear norms, convolution and
BatchNorm bounds, residual composition, unknown-graph rejection, provider
labels, normalized-gate requirements, and formula arithmetic.
