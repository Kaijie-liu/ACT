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

There is a separate, upstream obligation even before solving. In the frozen
main-table entry, materializing a clipped epsilon box and then computing an
HZ midpoint/radius use ordinary binary64 arithmetic. A saved-only exact audit
finds that neither step provides a universal outward enclosure on the
100-input cohort: all requested-box comparisons fail and 98 reconstructed
input HZs have inward endpoints relative to their materialized boxes. The
`exact=True` representation flag is not a containment proof; corrections to
later solver bounds do not certify source conversion. The audit checks the
stored endpoints and frozen formula, not unrecorded historical layer states.
Even a proof about that input HZ would still require downstream propagation,
guard and weighted-output containment. Historical HZ-policy acceptance is
therefore not a source-complete real-network certificate. See the
[scope and gain ledger](../../docs/main_table_source_applicability_20260921.md).

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

## Independently checkable complete-request evidence

The production solver and the independent evidence path have different trust
contracts. Production scoped facts are extracted from guarded interval bounds,
bind model/input/property/expert/frame/domain identities, and transfer only
along membership-to-pair containment. Partial MILP searches are not positive
facts. The separate rational path reconstructs the justification for each
required fact or weighted bound from its supplied source.

For a finite-box LP
\(\min c^\top z+d\) subject to \(Az\le b, Ez=h, l\le z\le u\),
a proposal supplies multipliers \(y\le0\) and free \(v\). Defining
\(r=c-A^\top y-E^\top v\), the standard-library checker evaluates

\[
 L=d+b^\top y+h^\top v+\sum_j\min(r_jl_j,r_ju_j)
\]

in exact rational arithmetic. This is a valid lower bound for every feasible
point: multiply inequalities by nonpositive multipliers, use equalities, and
minimize the remaining residual coordinatewise over the finite box. No
approximately zero residual is discarded. Solver status and a reported
objective are proposals, not proof authority; optimality is unnecessary for a
valid sign-sufficient bound. This checker is not a MILP search-tree verifier.

The proof producer starts **before floating F0 construction**, with the stored
shared expert HZs. Stored float coefficients denote exact binary rationals.
Property projection recomputes \(u=q^\top E_b+c\) and
\(d=q^\top(E_a-E_b)\) in rational arithmetic on the same factor frame.
Checked gate and disagreement ranges bind the same request, pair and property.
An additional variable \(w\) and the four rational McCormick inequalities
enclose \(w=\lambda d\); corner products provide finite bounds. The checker
reconstructs these coefficients from sources instead of calling the floating
producer. Binary factors explicitly relaxed to continuous boxes yield an outer
enclosure, not exact nonlinear or integer optimization.

An optional bounded gate checker replaces coarse score-order ranges with
rational sigmoid enclosures, conditional on checked router-margin supports.
It verifies a positive exponential Taylor sum and remainder bound, outward
dyadic rounding, and range-reduction squaring; no floating exponential is
trusted. A replacement composition checker revalidates the original complete
obligation inventory and the specifically bound residual before substituting
new evidence. Neither a narrower gate nor a successful numerical proposal is
itself a request proof. The real one-residual follow-up remains unclosed
(Section8); unsupported ranges fail closed rather than triggering precision
search. The upstream trusted components below are unchanged.

The request inventory is derived from the declared expert/class counts,
properties and all tie-legal pairs. Each obligation needs either two checked
positive scoped facts or a checked positive residual bound. Coverage and
source bindings are checked even when facts are reused. The
[complete-request composition proposition](03_path_conditioned_method.md#complete-request-composition-theorem)
then applies, subject to the trusted components below. Missing evidence, a
nonpositive checked bound and an expired check are different incomplete states;
none is promoted to a complete positive result.

| Component | Independent rational path checks | Still assumed |
|---|---|---|
| Request and source identity | Pinned statement, property, pair and source bindings | Supplied source corresponds to the intended network/input |
| Reachable representation | Stored constraints and shared-factor consistency | Network-to-HZ soundness, guard lowering |
| Route inventory | Complete declared partition and required obligations | Truth of upstream route-infeasibility exclusions |
| Weighted lowering | Exact property projections, ranges, McCormick rows and objective | Real selected-softmax semantics of the model |
| Lower bounds and reuse | Exact dual/residual arithmetic and domain-scoped aggregation | Correct checker/interpreter execution |
| Deployment | No such proof | Preprocessing, floating kernels and dispatch equivalence |

## Source checking and non-transferable certificates

A supplied-HZ proof and a declared-source proof have different starting points.
The former may yield a positive conditional result while assuming expert
propagation; the latter must independently justify containment at every source
step. The current evidence must not be read as though both contracts were
satisfied by the same positive proof.

For the separately implemented declared-source path, the checker first
validates an outward enclosure of the pinned **represented** input box.
Affine construction supplies exact residual compensation: for each old factor
assignment, a fresh bounded error factor admits the exact affine output.
Subsequent sparse affine lifts retain defining equalities rather than reset
outputs to independent intervals. ReLU graphs and their range justifications,
pooling/flatten conventions, parameter identities and append-only factor
allocations are reconstructed independently. The join preserves common input
factors while keeping expert-private error and activation factors distinct.
Composing these pointwise extensions justifies the new expert enclosure for
the declared real graph.

Optional tighter ranges require checked bounds for both an expression and its
negation on the same source/request/domain/layer/row. Recentring keeps the
changed equality RHS; zero-width ranges do not remove defining relations.
Missing range evidence uses the registered outer-box fallback; invalid
evidence rejects. The output checker then independently constructs every
necessary weighted LP on the resulting factor frame and checks its candidate
bound. A source change invalidates old downstream evidence, even if dimensions,
the requested input or the checkpoint have not changed.

For the registered nonoverlapping average-pool/flatten/affine router, exact
coefficient extrema on original pixels also check route exclusions.
An outsider must strictly dominate a selected expert to exclude a pair;
zero margin does not remove tie-legal obligations. This discharges exclusions
for that declared graph, not for arbitrary routers or native dispatch.

| Evidence path | Independently checked starting point | Residual trust / observed limit |
|---|---|---|
| Stored pre-F0 HZ proof | Source identities, rational projections/ranges, McCormick construction, bounds and complete obligation aggregation | Expert propagation remains assumed; the historical input98 proof is positive under this contract |
| New declared-source path | Represented-box containment, complete expert enclosure, registered guards/routes, fresh weighted LPs and available bounds | Declared-graph/program correspondence and checker execution remain assumed; no complete positive output proof was obtained |
| Deployed floating program | No end-to-end equivalence proof | Preprocessing, requested versus represented perturbation set, kernels and dispatch are not discharged by either path |

The supplied-source contract table above remains the default for earlier
conditional proofs; it is not silently replaced by the extension. The final
source-range comparison produces only nonpositive or missing output evidence.
Complete source checking is therefore not complete positivity. Conversely,
the old positive bound cannot repair a source-containment defect without a
proof tied to that old source. A negative new bound does not refute the old
conditional statement or demonstrate model unsafety.

These distinctions are enforced by request, parameter, source, factor, LP and
property bindings, not by matching tensor shapes or an `exact` flag. The
[archived source-proof development](../appendices/source_proof_history.md)
retains the original local discrepancies, intermediate controls and final
no-solve stop decision; none changes production acceptance or the main tables.

Independent checker and statement identities must be pinned outside the bundle.
A hash confirms identity, not the mathematical correctness of a trusted source.
The real input 98 proof is portable: isolated checking needs no checkpoint,
dataset, historical directory or solver. Its bundle rejects missing properties,
source substitution and changed properties even when transport hashes are
recomputed but the external statement is unchanged. A fresh source-defined
three-expert all-tie control additionally exercises the standard verifier and
independent proof generator; it is not trained-model coverage evidence.
See [the reviewer workflow](../artifact_quickstart.md).

The generic optional mode uses a single request budget for source capture,
proposal, construction, serialization, checking and aggregation. Immutable
partial evidence may survive an exhausted proposal reserve, but no unchecked
or missing obligation is discharged. Caches change parsing cost only if source
identity, exact checks and scope validation are retained. These engineering
controls do not relax production optimal-status acceptance, nor establish
additional positive coverage: the 20-input convolutional evidence experiment
returns no complete positive requests.

The [historical checking sequence](../appendices/independent_checking_history.md)
retains supplied-F0 controls and their stronger trust assumptions separately
from pre-F0 rational reconstruction. The
[closed feasible-point diagnostics](../appendices/arithmetic_diagnostic_limits.md)
supply no checked upper witness and do not turn nonpositive lower bounds into
proved relaxation obstructions. Main guarantees are stated per complete
request, not per checked JSON or isolated positive LP.
