# AdvMoE router bracket and staged-verifier protocol

## Methodological regime

AdvMoE supplies the nonlinear-CNN-router point in a three-level route-analysis
ladder:

1. affine RT-ER router: exact closed-form pixel-space boundary;
2. verification-scale one-hidden-layer router: exact unrelaxed HZ feasibility;
3. AdvMoE CNN router: attack/lower-bound bracket with an explicit undecided
   band.

For `E=2`, clean route `e`, and competitor `1-e`, the property is the scalar
margin `r_e(x) - r_(1-e)(x)`. A concrete PGD input that changes literal router
argmax is a route-instability witness. A backend lower bound above the frozen
positive tolerance is evidence for stability only at that backend's numerical
soundness level.

## Numerical discipline

The installed auto_LiRPA path is not outward-rounded. Therefore this project
does **not** promote its positive IBP/CROWN lower bounds to formal `SAFE` or
formally route-stable. Pilot states are:

- `ATTACK_CONFIRMED_ROUTE_UNSTABLE`: concrete adversarial input replays to the
  other route within the registered box;
- `POSITIVE_NUMERICAL_BOUND_FILTER`: finite lower bound at least `1e-7`, with no
  conflicting witness;
- `UNDECIDED`: neither condition;
- any positive-filter/witness overlap is an audit failure.

Formal route-stability counts remain zero until an outward-rounded or otherwise
validated backend is supplied. Negative relaxation bounds are always UNKNOWN.

## Source-to-CROWN adapter

The literal router is first tested and its current auto_LiRPA rejection is
recorded. A fixed-shape adapter then performs two exact 32x32 specializations:

- channel-padding strided slices become fixed 1x1 stride-2 identity
  convolutions;
- dynamic full-spatial average pooling becomes `AvgPool2d(8)`.

Random, zero, and one inputs must produce bit-identical scores and routes before
any bound is accepted. The adapter rejects use outside `[B,3,32,32]` at its
entry gate.

## Deep-path specialization

For route 0 or 1, every MoE convolution is replaced by a static convolution
containing the selected contiguous weight slice. All 16 replacements must
succeed. Dynamic full-model output and the corresponding specialized path must
agree for concrete inputs within tolerance. The route-specialized model has no
dispatch operator and is the eventual expert/property backend input.

The staged verifier is:

1. determine route stability/uncertainty with the router bracket;
2. for a stable route, verify its one static deep path;
3. otherwise verify both static deep paths on the full input box;
4. if a property backend is inconclusive, attacks may establish only
   full-model replayed UNSAFE witnesses;
5. guarded-cell boxing and eta compilation are retained as one bounded
   ablation, with no expected advantage preclaimed.

## Frozen engineering pilot

While B1 runs, only the first 20 ordered CIFAR-10 test inputs are used, over
`{0.5,1,2,4,8}/255`. PGD uses 20 steps, two restarts, and step size epsilon/4.
The bound worker uses CPU, one thread, and IBP solely to validate the harness;
the paper target remains CROWN after the B1 resource gate. This pilot is not a
full-test census and is not used for prevalence or certification claims.

The full AdvMoE line remains bounded to seed-0 official-code training, init and
final full-test bracketed census, deterministic intermediate telemetry subset,
five-radius staged-verifier table, and one guard ablation. No ratio or expert
count sweep is added here.
