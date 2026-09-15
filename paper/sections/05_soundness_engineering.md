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

The real-arithmetic reduction and the numerical backend are separate soundness
layers. The first proves that complete branch coverage, guarded propagation,
and the McCormick outer relaxation imply the original routed property. The
second asks whether a concrete solver execution supplies a trustworthy lower
bound for that reduction. Passing the first layer does not validate the second.

For the registered HZ/HiGHS path, ACT consumes a support lower bound only after
optimal status, the frozen feasibility and integrality tolerances, an explicit
absolute-plus-relative correction, and `nextafter` toward the unsafe direction.
A primal incumbent is never substituted for a proof bound. These checks form a
fail-closed policy for the pinned backend; they are not a universal proof about
arbitrary native floating-point solver implementations. A result that does not
satisfy the registered policy is `UNKNOWN_NUMERICAL`.

The installed CROWN/auto_LiRPA path is treated more conservatively. It has no
outward-rounding contract in this artifact, so even a finite positive lower
margin is only `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`. Changing a method
string to alpha-CROWN is insufficient: optimized modes must pass a configuration
gate requiring autograd and explicit optimization iterations. Negative CROWN
bounds remain UNKNOWN, and no CROWN filter is merged into formal SAFE counts.

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

The independent auditor also rebuilds the evidence portfolio rather than
trusting a runner-supplied aggregate. Concrete prediction flips are checked
against route-invariance, route-conditioned, eta, and portfolio output filters;
concrete route flips are checked against positive router filters. Result schema
v2 separates route, prediction, and joint witness counts, while the auditor
retains an explicit compatibility path for frozen schema-v1 artifacts.

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

The production evidence package materializes this identity rather than storing
only a summary row. It binds the model state and optional checkpoint, literal
center and represented lower/upper tensors, property, configuration, numerical
policy, exact route family, every invoked branch/property result, and the
state-transition trace. Packages are immutable directories and a separate
auditor recomputes their hashes, route/pair coverage, accepted F0 minima, and
unsafe witness obligations. The accepted outward-corrected minimum is recorded
separately from a solver's raw bound contribution, since the latter excludes
the represented output center in the HZ lowering. A structural audit checks
the evidence chain; it is not presented as an independent re-execution of a
SAFE solve.

## Scoped reuse and a first independently checked bound

An opt-in post-comparison implementation exports positive per-property facts
from already computed guarded Tier-1 output intervals. A pair guard implies
membership of each selected expert, so two facts for the same property extend
to that pair domain and imply a mixture lower bound equal to their minimum.
Facts bind the concrete request/model/domain/property, router frame, policy
and expert identity. The auditor reconstructs them from source intervals.
Partial MILP results are not reused. This version trusts the underlying HZ
propagation and verifies the reuse implication; it does not independently
establish the propagated interval endpoints.

Separately, a finite-box LP checker evaluates signed dual multipliers and their
stationarity residual with exact rational arithmetic. Residuals are minimized
on the finite input box instead of being discarded as numerical zero. Float
coefficients denote exact binary rationals. The checker does not call a solver
and does not trust a primal objective. On the explicit guarded control
x0>=1/2, x1=1/4, 0<=x<=1, it checks the x0+x1 lower bound as exactly 3/4.
It is not a MILP proof-tree checker or a validation of network-to-LP lowering.
The two-expert, three-class reuse control remains SAFE while reducing F0
property solves from two to one; both packages pass structural re-audit.
These are analytic implementation controls, not real-model speedup evidence.

The next export check uses an actual trained bal010 router at index 3000 and
2/255, retaining its clean top-2 guard {4,5}. The exported LP relaxes one
binary among 3,075 factors. An independent scalar checker verifies every
source-HZ constraint, factor box and exact objective combination and checks a
score4-score0 lower bound of approximately 9.4850718858 as an exact rational.
This extends the checker beyond hand-written LP controls. Its guarantee still
starts at the stored HZ: it does not independently validate network propagation
or certify the complete MoE output. No positive result is required by the
registered export query; the margin was fixed before its evaluation.

The next separately frozen control covers every required output obligation
of that same seed0/index3000 request. All28 route queries finish, yielding
pairs{2,4} and{4,5}; the checker inventories18 pair/classification obligations.
It rechecks36 sparse LP exports/duals, including27 expert properties,six
disagreement endpoints and three weighted output properties. Fifteen obligations
have positive checked membership facts on both experts and are discharged by
scoped reuse. Three residual F0 LP lower bounds remain negative, so the complete
request is UNKNOWN. No missing obligation is treated as a positive result.

This infrastructure control uses the universal normalized-weight range[0,1]
and rationally checked disagreement endpoints, not the production sigmoid-range
configuration. It establishes downstream checking for an entire supplied
request inventory but NOT a positive complete-network certificate. Network-to-HZ,
guard lowering,route exclusions and the F0 outer-HZ construction/floating
coefficients remain trusted. Exact rational LP checking cannot retroactively
prove those upstream transformations. See `request_lp_results.md` and its
hash-bound36-proof recheck for the explicit remaining trusted base.

A separately frozen order-only follow-up closes those three residuals without
an unchecked numerical sigmoid endpoint. For ordered pair(a,b),let m=r_a-r_b.
A checked lower bound on m implies lambda_a>=1/2 when nonnegative; a checked
lower bound on -m implies lambda_a<=1/2 when nonnegative. Both statements
follow from sigmoid monotonicity and sigmoid(0)=1/2,including ties. Unknown
signs retain the corresponding universal endpoint. Only dyadic endpoints are
used; no floating transcendental evaluation justifies the range.

In the same old request,the checked upper bound for r2-r4 is about-3.11463,
so lambda2 lies in[0,1/2]. Five new LP queries(two order bounds,three residual
outputs) give residual lower bounds2.60544,1.55273,4.10960. Rechecking all41
stored exports/duals establishes positive downstream evidence for all18
obligations,with minimum aggregated bound about.183047. This is explicitly
conditional on the same trusted upstream and F0 lowering,not a proof of the
native network implementation or an additional performance benchmark. The
R1 universal-weight UNKNOWN remains intact. See `request_lp_order_results.md`.

### Checking construction before the floating F0 boundary

The next, separately frozen R3 control removes one assumption rather than
refining the same margin again. It starts with the stored shared expert HZ
before floating F0 projection. Interpreting its coefficients as exact binary
rationals, it forms `u=q E_b+c` and `d=q(E_a-E_b)` by rational arithmetic on
the same factor vector. It appends lambda and w directly, uses the unchanged
checked R2 ranges, and constructs four rational McCormick inequalities for
`w=lambda*d`. Finite w bounds come from the four rational corner products;
binary factors are explicitly relaxed to their continuous boxes. There is no
floating center/radius recoding between these objects and the LP.

A separate checker, importing neither the builder nor a solver, reconstructs
the factor constraints, projections, objective, variable bounds and product
planes, and then checks the proposed LP dual. The request aggregator checks
the range sources, identical shared-HZ identities, scopes and full obligation
inventory. On the frozen request all18 obligations remain positive:15 reused
facts and3 residual LPs, with residual bounds about2.60544,1.55273,4.10960.
The aggregate minimum remains about.183047. The trusted base now excludes
`F0_outer_HZ_construction_and_floating_coefficients`; network/input-to-HZ,
guard lowering and route infeasibility exclusions remain assumptions.
This is an exactly checked outer-relaxation construction and conditional
request proof, not exact linearization of MoE or a native floating-point proof.
The R1 UNKNOWN and R2 result are retained. Source:
`act/pipeline/moe/results/request_lp_rational_review_20260914_r3.json`.

The convolutional all-obligation control uses a deliberately weaker, separately
identified boundary: supplied floating F0 HZ→continuous LP→exact dual/residual
check. It independently reconstructs scoped interval projections, checks each
of nine output obligations and accounts for all six candidate pairs, while
trusting the upstream exclusions. Input98 closes through eight LP proofs and
one interval fact (minimum0.1772745); input16 leaves two nonpositive LP bounds
and remains UNKNOWN. It does not inherit the pre-F0 rational construction
guarantee described above. A complete conditional request is therefore
distinguished from a positive single-property control, a production SAFE,
and an independently proved network execution. Source:
`act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json`.

A subsequent, separately frozen study makes the pre-F0 boundary explicit for
the convolutional input98 as well. Fresh ordered shared expert outputs are
projected in exact rational arithmetic; checked router order gives lambda1 in
[1/2,1], and two checked support bounds per residual establish its disagreement
rectangle. An independent construction checker verifies the same-factor
projections, all four McCormick planes, finite variable bounds and LP dual
evidence. All nine properties remain positive (eight residuals and one scoped
interval fact; minimum0.1772745). Floating F0 construction is removed from the
trusted base for this request; network/source binding, guard lowering and route
exclusions remain assumptions. This does not upgrade the old production
TIMEOUT or certify deployed floating-point execution. Source:
`act/pipeline/moe/results/conv_pre_f0_review_20260915_r2.json`.

The same completed proof was subsequently packaged independently of server
paths. Content-addressed array storage preserves the original logical-file
hashes while reducing428.19MB of proof dependencies to a7.18MB bundle including
the checker. After copying outside the checkout, Python `-I -S` reproduced all
nine checked obligations without model, dataset, historical directory or solver
reads. Removed obligations, substituted sources and changed properties were
rejected even after transport hashes were recomputed; damaged content was also
rejected. This makes a real conditional proof portable, not its upstream
network construction independently verified. Evidence:
`docs/portable_conv_proof_v1_review.json`.

### A request-parametric conditional evidence contract

The subsequent optional evidence interface derives E, C, all legal pairs and
explicit linear properties from the request, rather than naming a particular
input or clean route. For every feasible pair/property it requires either two
scope-bound positive membership facts or a checked pre-F0 rational LP bound.
An exhaustive pair partition is checked, but the truth of excluded-route
infeasibility remains an upstream assumption; unresolved routes never imply
complete safety. The checker also reconstructs property projections, HZ-to-LP
export, checked range bindings, McCormick constraints and exact rational dual
bounds. Missing evidence and checked nonpositive bounds are distinct UNKNOWN
states, neither a counterexample.

Under sound network/input-to-HZ and ordered-source binding, correct guards and
route exclusions, complete positive obligations imply the requested properties
for all tie-legal selected-softmax top-2 outputs on the represented box. This
is the uniform meaning of CHECKED_CONDITIONAL. It does not independently prove
the upstream network transformation or deployed floating-point execution.
The independently pinned portable checker and statement identities are also
part of the checking contract. Production HZ-policy acceptance and CROWN
numerical filters remain separate evidence grades. Controls cover changing
dimensions, all-tied multi-pair models, partial reuse, semantic mutations and
deadline failures, but are not evidence of new real-model coverage. Sources:
`docs/general_evidence_v1.md`, `docs/general_evidence_v1_controls.json`.

A subsequent optional engineering revision distinguishes exhaustion of a
proposal's checking reserve from exhaustion of the whole request deadline.
Only the former returns committed partial evidence to the unchanged checker;
it cannot promote an uncommitted or missing obligation. True deadline expiry,
invalid evidence and incomplete isolated checking remain failures. Analytic
multi-pair and portable partial-proof controls cover this handoff, but it has
not been used to replace any frozen real-request result or establish a speedup.
The stored-proof cost profile and controls are recorded separately in
`docs/evidence_handoff_v1.md`.
