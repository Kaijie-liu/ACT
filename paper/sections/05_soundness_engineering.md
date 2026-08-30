# Soundness engineering

The main algorithm reduces a routed program to static verification obligations,
but that reduction is useful only if the implementation preserves its logical
quantifiers. This section describes the engineering rules that make a reported
status traceable from an abstract bound to the literal model. They are part of
the method, not post-processing conventions.

## Four statuses, not a Boolean answer

Every property query terminates as `SAFE`, `UNSAFE`, `UNKNOWN`, or `TIMEOUT`.
`SAFE` requires a sound lower bound for every feasible route and every property
row. `UNSAFE` requires a concrete input inside the registered input set whose
literal full-model forward pass violates the property. A negative objective in
a relaxation is not such an input. An unfinished branch, a solver limit, an
unlifted expert witness, and a failed sufficient condition remain distinct
unknown reasons rather than being collapsed into failure or unsafety.

This distinction is essential for selected-softmax top-2. Gate elimination is
a sufficient proof rule: if every feasible selected expert satisfies a linear
property, their non-negative normalized mixture satisfies it. A violation in
one expert alone does not imply that the weighted model violates it. When this
tier is inconclusive, the F0 fallback bounds the selected gate range and a
property-directed expert difference. A negative McCormick relaxation objective
still yields only `UNKNOWN_WEIGHTED_RELAXATION`. It becomes `UNSAFE` only after
the candidate variables recover an input and the selected-softmax model replays
the violation.

## Tie-inclusive route semantics

Top-k membership is an unordered set. At a tie, every set consistent with a
legal top-k choice is feasible and must be covered. We use weak route guards
and enumerate all feasible memberships under `ANY_LEGAL_TOPK` semantics. This
choice prevents an implementation's stable-sort order from silently becoming a
mathematical assumption.

The rule also exposes a common property-compilation error. For guard violation
score `g` and safety score `s`, the tempting reduction `max(g,s) >= 0` is
unsound at `g=0`: the compiled property passes even if `s<0`, although the tied
route is legal. Our eta reduction uses a positive guard margin and is sound but
conservatively rejects the strip `0 <= g < eta`. We retain both the proof and a
mutation test in which the zero-margin construction must fail. Backend
conformance is therefore checked against route semantics, not just tensor
shapes.

## Support bounds, big-M, and outward use

Route guards and ReLU phases introduce mixed-integer constraints. Every big-M
constant must be derived from a support bound over the same identified HZ
frame. A fast unconditioned bound can be used only where the configuration
explicitly permits the looser encoding; the exact-support path retains shared
generator constraints and rejects a bound from a different frame, route pair,
or width. The implementation records which support tier produced every
constant.

Solver objectives are floating-point results. Before a support lower bound can
be consumed by a `SAFE` decision, ACT applies the frozen absolute-plus-relative
slack and moves the result with `nextafter` toward the unsafe direction. SAFE
also requires optimal solver status, the registered feasibility and
integrality tolerances, and a corrected lower bound above the frozen positive
margin. A primal incumbent is never substituted for a proof bound. Numerical
failure is reported as `UNKNOWN_NUMERICAL`, not retried with an undisclosed
tolerance.

The guard-accounting audit enforces a separate conservation law:

```
binaries_before - binaries_after
  = lp_support_eliminated
  + milp_support_eliminated
  + structural_or_propagation_eliminated.
```

This prevents a total binary reduction from being attributed wholesale to LP
or MILP support. It also explains why a fast-unstable-neuron count and a binary
count may have different universes. The confirmatory experiment closes this
identity exactly while retaining each component and its time.

## Shared variables in weighted fallback

For a feasible pair `{a,b}`, F0 rewrites a linear property of the mixture as
`u + lambda*d`, where `u` is the property value of expert `b`, `d` is the
property-directed difference between experts, and `lambda` lies in the sigmoid
image of a guarded router-margin interval. It encodes neither exponentiation
nor division nor a segmented sigmoid graph.

The two experts are propagated from one guarded input frame. Input generators
and router constraints remain shared; expert ReLU binaries remain independent.
The product uses the standard McCormick hull over recorded bounds for `lambda`
and `d`. Tests cover zero margin, identical expert outputs, fixed-sign and
sign-crossing differences, outside-expert ties, multiple legal pairs, shared
generator identity, and random concrete points. A mutation control reverses a
McCormick inequality and requires the consistency test to fail. These tests
guard against a subtle but severe error: independently cloning both experts
would erase input correlation, while aliasing their ReLU variables would add a
false correlation.

## Concrete replay is the unsafe boundary

All attack and solver candidates are transformed back to the registered input
domain and checked against the represented perturbation set. The literal routed
model then recomputes route membership, selected weights, output logits, and
the property. Hard top-1 candidates additionally must execute the branch whose
constraints produced them. Weighted top-k candidates must use a legal tied set
and the full selected-softmax mixture. Only this replay can cross the boundary
from an abstract possible violation to `UNSAFE`.

Replay artifacts store the input, clean and adversarial routes and predictions,
property values, perturbation norm, model and checkpoint hashes, and runtime
identity. Independent audit reloads the checkpoint and replays every stored
endpoint rather than trusting the runner's labels. The B1 endpoint applies the
same policy to all 10,000 PGD-50 endpoints.

## Differential and metamorphic checks

Critical closed forms and solver adapters have independent references. The
affine route-boundary oracle is compared against SciPy LP, its vectorized uint8
path against the generic breakpoint implementation, and its generated witness
against the literal router. The incremental guarded-box backend is compared
coordinate by coordinate with fresh SciPy models on hundreds of thousands of
objectives. Lazy top-k enumeration is compared set-for-set with exhaustive
enumeration at `E=8`. F0 concrete points must satisfy every generated
McCormick constraint.

These are differential tests rather than repeated calls through a shared
helper. Audits recompute summaries from row-level artifacts, check expected row
counts and unique sample ranks, separate fixed-radius from boundary-adaptive
cohorts, and use clean sample rather than sample-radius rows as the statistical
cluster. A result is publishable only when its independent audit reports zero
issues.

## Negative runs are evidence, not debris

Failed or superseded result directories are preserved and permanently excluded
by identity. They document memory failures, bootstrap errors, runtime-kernel
differences, and changed preprocessing semantics. A repaired run receives a
new directory and an explanation of whether mathematics, orchestration, or
only the execution bootstrap changed. No successful summary overwrites a
failed directory.

This rule makes negative engineering results interpretable. Incremental model
reuse accelerates build-dominated guarded-box hull queries by 15.03 times, yet
does not accelerate search-dominated property MILPs and is about 10% slower in
the frozen residual cohort. We retain both results and use a hybrid backend
policy rather than reporting the favorable number as a universal speedup.

## Certificate identity

Finally, a certificate identifies more than a checkpoint and requested radius.
It binds ordered data and preprocessing, runtime and device versions,
stateful-layer mode and statistics, the represented lower and upper tensors,
solver tolerances, and the outward-bound policy. This catches four observed
failure modes: real-arithmetic preprocessing that changes routes, version-
dependent float16 resize kernels, float32 boxes that collapse below an input's
ULP, and BatchNorm initializations that define different functions in eval and
train mode. The companion artifact-identity section reports those findings;
the engineering consequence is simple: if any identity field is absent or
changed, the certificate fails closed.
