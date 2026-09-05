# Tie-safe Lagrangian hard-top1 guard compiler

## Purpose

This compiler is a narrow bridge from a hard-top1 routed program to a static
network backend that cannot directly retain input-side route halfspaces. It is
intended to test whether preserving router--expert dependence inside one graph
recovers information lost by full-box path verification.

For branch `i`, the tie-inclusive guard is

```text
m_ij = r_i - r_j >= 0, for every j != i.
```

For property row `s_l >= 0` and fixed nonnegative multipliers, the compiler
emits

```text
phi_l = s_l - sum_j mu_lj * m_ij.
```

On the legal branch `phi_l <= s_l`; proving `phi_l >= 0` over the original box
therefore proves the original row on that branch. Ties are retained because a
tied margin is zero. The full proof is in
`act/back_end/moe/proofs/lagrangian_top1_guard_compilation.md`.

## Implementation and fail-closed rules

- `LagrangianTop1GuardedProperty` embeds the router and expert in a single
  PyTorch graph with a shared input.
- Multipliers must be finite, fixed, and nonnegative. They may differ by
  property row.
- AdvMoE's optional ablation evaluates a frozen scalar multiplier grid and
  takes the best complete lower bound independently for each property row.
- A backend error, an incomplete call, a malformed bound matrix, or a
  non-finite value makes the aggregate `UNKNOWN`/`ERROR`; it cannot filter a
  row.
- A negative relaxation result is `UNKNOWN`. Only a concrete full-model replay
  can establish `UNSAFE`.
- Positive CROWN results remain
  `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`, because the installed backend has
  no outward-rounding contract.

## Evidence available now

The analytic toy conformance result is
`act/pipeline/moe/results/crown/lagrangian_guard_toy_conformance_20260906.json`.
On `x in [-1,1]`, branch 0 is legal for `x >= 0`. With safety `s=x+0.1` and
`mu=1`, the unguarded lower bound is approximately `-0.9`, whereas the
compiled row is the constant `0.1`. Replacing the safety offset by `-0.1`
keeps the tie unsafe and the compiled lower bound negative. Concrete grid
evaluation confirms `phi <= s` on every legal point.

An exact retained-HZ differential is recorded in
`act/pipeline/moe/results/crown/lagrangian_guard_exact_hz_differential_r2_20260906.json`.
For both the safe `s=x+0.1` and unsafe-tie `s=x-0.1` controls, exact guarded HZ
and the compiled `mu=1` property agree on the analytic lower bound within the
explicit `5e-9` comparison tolerance. The first execution used `1e-9` and
failed by about `1.1e-9` because the complete HZ support result includes
conservative numerical padding. That failure is preserved as
`lagrangian_guard_exact_hz_differential_r1_failed_20260906.json`; r2 changes
only the disclosed comparison tolerance, which remains below the project's
`1e-7` positive-margin threshold.

Together these controls establish the real-arithmetic reduction, executable
CROWN conformance, and a verification-scale exact-HZ differential. They do not
establish official-scale benefit, a formal CROWN certificate, or generic
novelty.

The exact negative control in
`act/pipeline/moe/results/crown/lagrangian_guard_incompleteness_control_20260906.json`
separates compiler incompleteness from backend relaxation. On `X=[-1,1]`, let
the legal branch be `x>=0` and let `s(x)=0.1-2*ReLU(-x)`. Retained-guard exact
HZ support proves the property positive on the legal half interval. For every
fixed `mu>=0`, however, `min_X(s-mu*x)<0`; the best exact value is `-0.9` at
`mu=1`. An official-scale non-improvement therefore cannot be attributed to
CROWN without a diagnostic that separates backend error, finite multiplier
search, and this intrinsic sufficient-reduction gap.

## Development controls and cost semantics

New schema-v2 executions include two diagnostic controls:

- `lagrangian_mu0_graph_matched` uses the same compiled property graph with
  exactly one frozen `mu=0` call. It distinguishes nonzero guard information
  from graph/property lowering effects.
- `lagrangian_separate_interval` combines independently computed safety and
  router-margin intervals as `lower(s)-mu*upper(m)`. Comparing it with the
  shared-input graph measures whether the backend exploited shared dependence;
  merely placing both computations in one graph does not guarantee that it did.

Mechanism and budget claims are separate. The mechanism result retains every
complete frozen-grid execution. The budget result accepts a method only when
the sum of its required graph construction and bound-call wall times fits the
same per-sample/radius cutoff. A completed overshoot remains in the raw
artifact but is labelled `UNKNOWN_BUDGET_EXHAUSTED` for the cost-matched
comparison. Attack time is excluded from positive acceptance.

Multiplier scale is also part of experiment identity. A normalized grid is
resolved once from the development cohort's median absolute clean router
margin, stored with its source hash, and then frozen. The runner and auditor
reject any mismatch between normalized coefficients, scale, and raw
multipliers. This prevents both silent routing-logit rescaling and post-holdout
grid expansion.

## Remaining experiment gates

The development execution is frozen in
`act/pipeline/moe/configs/advmoe_lagrangian_development_r1.json`. It reuses the
historical first 20 clean-correct inputs strictly as a development cohort,
uses normalized coefficients `{0,0.25,0.5,1,2,4}` divided by the frozen median
clean router margin `2.580275058746338`, and applies a 60-second total-wall
evidence cutoff independently to each method and sample-radius pair. The
selection and scale are separate hash-bound artifacts.

1. Execute the frozen, explicitly manifested 20-input development cohort and
   choose no settings after inspecting a later endpoint-excluded cohort.
2. Freeze a disjoint holdout manifest only after development and its budget
   behavior have been audited.
3. Run a paired official-scale comparison with identical expert backend,
   budget, samples, radii, and preprocessing.
4. Keep formal and numerical-filter endpoints separate in every table.

## Development r1 result

The frozen development run completed all 100 sample-radius rows in 6,000.45
seconds and independently audited `PASS` with zero issues. All grid executions
fit the common 60-second cutoff; the grid's median accounted time was 47.22
seconds versus 4.99 seconds for unguarded two-path CROWN.

Relative to the graph-matched `mu=0` call, a nonzero multiplier strictly
improved 241/1,800 property-row lower bounds (13.4%), tied 1,559, and worsened
none because the exact `mu=0` call is a member of the fail-closed grid. This
property-level effect did not cross the complete obligation: Lagrangian,
graph-matched `mu=0`, separate intervals, and unguarded two-path each filtered
the same 2/100 sample-radius rows, both at `0.5/255`. The paired input-cluster
difference is exactly zero in this development cohort.

The shared graph also did not dominate the separately intervalized control:
558 rows improved, 444 worsened, and 798 tied within `1e-7`. This is evidence
that graph/relaxation form materially changes the computed lower bound, not
evidence that CROWN consistently preserves the intended relation. One route
flip witness occurred, but no route-changing row had a positive Lagrangian
filter. Formal SAFE remains zero by construction, while all 17 prediction-flip
UNSAFE outcomes were full-model replays.

The frozen interpretation is therefore:

> Under the registered multiplier protocol and common budget, Lagrangian
> compilation changed property-level bounds but added no complete numerical-
> filter coverage.

It is not permissible to infer from this run alone whether the endpoint gap is
caused by backend relaxation, finite multiplier search, or intrinsic fixed-
multiplier reduction incompleteness. The confirmatory holdout remains locked;
running it without a development-level endpoint signal would spend a fresh
cohort only to test an unchanged negative configuration.
