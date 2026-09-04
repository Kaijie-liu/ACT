# RT-ER: from a theorem statement to an executable applicability audit

We use the official RT-ER release as an artifact case study because it combines
an explicit certified-robustness theorem, public training code, and routed
CIFAR-10 and TinyImageNet models. The goal is not to adjudicate the paper's
intent. We ask which claims can be instantiated from the released artifact and
which assumptions can be checked on the models that its pipeline produces.

## Four breaks in the released evidence chain

First, the certification theorems are not numerically instantiated in the
paper. The release contains no per-input certified radii, certified accuracy,
or executable certificate code, and it does not specify how the Lipschitz and
range constants appearing in the formula should be computed. Our
reimplementation therefore treats the constant provider and the interpretation
of the gate output as explicit inputs. Sound global spectral constants,
empirical gradient diagnostics, and an unspecified-author branch receive
different labels; empirical estimates are never called certificates.

Second, the published model uses hard `argmax` routing, while the theorem is
stated using Lipschitz gate quantities. We do not label the theorem unsound.
Instead, the applicability checker first asks whether the required route
condition is established at the input and radius under evaluation. The formula
is evaluated only inside that domain.

Third, the released CIFAR-10 training script does not update its router. Source
gradients do not connect the training loss to router parameters, the optimizer
contains no router state, and checkpoint hashes remain unchanged across
epochs. This finding is scoped to the released script. Static routing is a
legitimate model design; the artifact question is whether the mechanism invoked
to motivate route separation exists in the released training pipeline.

Fourth, current general-purpose verifier frontends reject the dynamic-dispatch
graph at `GatherElements`. Route specialization removes the dispatch operator:
the same expert backend can then consume the static branch. This
reject-specialize-accept sequence motivates route conditioning as a composable
analysis layer rather than a replacement for neural-network verification.

## Exact applicability of the official affine router

The official CIFAR-10 router is affine and hard top-1. For a clean winner (i)
and competitor (j), the smallest box-constrained perturbation that reaches a
tie solves a piecewise-linear support equation. We implement it in closed form,
including input clipping, tie-inclusive semantics, a concrete route-boundary
witness, and a vectorized uint8-grid path. The all-test-set implementation runs
in under one second and agrees with an independent optimizer to numerical
tolerance.

This exact oracle makes theorem applicability measurable rather than inferred
from attacks. We reproduce the official full-model initialization order—four
experts are constructed before the router—because PyTorch's random-number
consumption changes the router if this order is reversed. Seed 0 is checked
against the epoch-10 checkpoint bit for bit. We then construct 20 official
router initializations and evaluate all 10,000 ordered CIFAR-10 test inputs.

At (8/255), the route-invariance applicability set is empty for 18 of 20
seeds. Each of the remaining two seeds retains at most two inputs. Thus the
official evaluation radius is a near-empty applicability regime across 200,000
input-seed pairs, not a peculiarity of one checkpoint. At (2/255), the mean
applicability is 5.038%, but it ranges from 1.25% to 12.98% across seeds. Since
the released code does not set a seed, the size of the certifiable precondition
varies by more than an order of magnitude under otherwise identical execution.
We call this an initialization lottery, not a distribution-free property of
all affine routers.

The radius grid prevents the comparison from degenerating into one dramatic
endpoint. At 0.5/255 and 1/255, route invariance retains a sizeable domain and
absolute expert certification can be measured. At 2/255 it is restrictive but
nonempty, making it the primary comparative point. At 8/255, near-emptiness is
reported as an applicability result rather than used to manufacture an easy
coverage win.

## Resolution and the geometry of an untrained affine router

The official TinyImageNet script provides a second dataset and architecture
family. Its router is again affine over the resized input, so the same oracle
applies once the literal preprocessing semantics are fixed. Across 20
initializations, applicability collapses more rapidly than on CIFAR-10. The
observed median-radius ratio agrees within about one percent with the scale
predicted from the input dimension and empirical second moment.

For default affine initialization, a pairwise clean margin remains order one,
whereas the (L_1) norm of the weight difference grows on the order of
(\sqrt d). The typical route radius therefore scales as

\[
  r^*_{\mathrm{route}}
  =\Theta\!\left((d\,\mathbb{E}[x^2])^{-1/2}\right)
\]

up to initialization- and data-dependent constants. CIFAR-10 and the official
224-by-224 TinyImageNet pipeline provide two empirical points consistent with
this order law. We do not treat the raw-resolution fold as an independent
third point, because resize changes the induced weight statistics, and we do
not extrapolate this affine result to deep nonlinear routers.

A synthetic grid was frozen before execution to test whether this account is
merely a fit to those two points. Across dimensions from 1,000 to 500,000,
20 default-initialized four-way routers, and two fixed input second moments,
the fitted slopes are `-0.5286` and `-0.4854`. Both pass the registered point
tolerance around `-1/2`. The result is deliberately not called a complete
preregistered confirmation: the Tiny-moment cluster-bootstrap interval
contains `-0.5`, while the CIFAR-moment interval ends at `-0.500086` and misses
it by `8.6e-5`. The composite gate therefore fails one of two interval checks.
This mixed result supports the order-of-growth mechanism while preserving its
sampling uncertainty and claim boundary.

The AdvMoE audit supplies a useful counterexample to an architecture-only
reading. Its initialized convolutional router sends all 10,000 official test
images to expert 0; the signed score difference has
`abs(mean)/standard_deviation=9.1406`. Strong attacks find no flip on the frozen
20-input diagnostic even when epsilon 1 spans the clipped pixel cube. These are
neither exact boundary nor formal stability results. They show that route-share
and offset diagnostics are necessary before interpreting a large local
margin-to-gradient scale as an architectural property.

## Comparison enabled by path conditioning

On the independent verification-scale route-boundary cohort, the explicit
route-invariance baseline and Route A use the same downstream backend and
budget. Route invariance solves 12 of 100 inputs; staged route conditioning
solves 68, a difference of 56 inputs. All 36 route-changing certificates are
unique to Route A. The baseline is cheaper because it abandons 76 unstable
inputs, so this is a coverage comparison rather than a speedup claim.

The seed-0 130-epoch official-code compatibility reproduction has now landed.
Ordered full-test SA is 34.22% and independently replayed PGD-50 RA is 32.70%,
versus paper values 77.81% and 69.09%; all 10,000 adversarial endpoints replay
and the audit reports zero issues. The complete trajectory peaks at 37.40% SA
at epoch 30 and remains within 32.96%--37.40% from epoch 20 onward. Its endpoint
RA/SA ratio is 0.95558. These are diagnostics of a run that learned little, not
a causal attribution. The paper/source comparison additionally finds that
optimizer family, weight decay, mixed precision, and exact augmentation are
underspecified, while the cyclic-LR wording and released initial rate are
semantically ambiguous. Per the frozen asymmetric rule, seed 1 must land before
any pipeline-level insufficiency wording.

The official-scale B3 r5 execution now completes all 318 feasible expert
branches with zero backend errors, zero incomplete bounds, and a zero-issue
independent audit. Its fixed-radius positive-filter counts favor Route A over
the same applicability-limited expert backend at every nonzero-coverage radius:
`17/12`, `14/8`, `7/3`, `2/0`, and `0/0` from 0.5 through 8/255. On 20 exact
route-boundary inputs, Route A filters nine while route invariance is
inapplicable on all 20. This closes the official-scale *numerical conformance*
comparison, but not a formal certificate comparison: the installed CROWN path
is not outward rounded, so formal SAFE remains zero by construction. We claim
an audited applicability and numerical-coverage gap, not formal superiority on
the official model.

## Responsible scope

The audit reports reproducible properties of released scripts, dependencies,
and generated models. It does not infer author intent or treat static routing
as an error. Contact is managed by the principal investigator under a frozen
one-reminder and one-neutral-public-issue policy; the repository records the
policy rather than personal correspondence dates. Closed or unavailable prior
artifacts remain survey evidence and are not presented as executed baselines.
