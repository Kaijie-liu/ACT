# Same source HybridZ endpoint and McCormick comparison

This separate finite G1 control compares weighted-property representations after
actual checked HybridZ propagation. It uses the four unchanged source declarations
in the [frozen protocol](../configs/hz_source_representation_20261001.json), not new
samples or trained models. No real/full-size request, native optimization, GPU,
new range tightening, source tuning or shared-versus-independent ablation is admitted.

Each arm creates the original declaration, encloses the input, executes actual
sparse affine/ReLU propagation with checked compensation and factor provenance,
and prepares the same shared-input joint HZ for every tie-legal unordered pair.
Both arms pay for their own creation and propagation. Their source, lowering,
joint HZ, gate and properties must match exactly. No historic positive certificate
or negative relaxation point is imported into a new matrix. All 33 pair/property
duties remain; empty guards are not removed, and the historical partial-reuse
fixture name does not authorize free reuse.

## One representation factor

Endpoint support retains the current exact objectives and unchanged batch
candidate kernel. It needs 42 distinct endpoint objectives across four cases.
MC uses the same factor vector and original constraints, adds a gate and product
variable, and uses the existing rational McCormick constructor. For each property:

    u = q B + offset
    d = q A - q B = d0 + sum(d_i xi_i)
    difference range = [d0 - sum(abs(d_i)), d0 + sum(abs(d_i))].

All original factors, including relaxed binary factors, retain their registered
[-1,1] bounds. The property constant occurs once in u and cancels from d. The
gate is the unchanged sign-derived router range. MC has 33 LP objectives, without
new range queries. The same projected CPU dual algorithm takes 128 updates on
each applicable matrix; both sides retain its exact zero-candidate fallback.
Different objective counts and MC matrices are part of the representation cost,
not an assertion of equal floating operations or globally optimal proposals.

The MC candidate kernel receives only finite floating copies of rational RHS
and box vectors. Source/LP identities and final checking always use original
exact coefficients. A new independent bridge rechecks source-to-HZ lowering,
recomputes the difference rectangle from the source, calls the independent MC
construction checker, and then checks residual-compensated bounds. It never
invokes the MC builder, propagation or candidate optimizer. Missing certificates
remain UNKNOWN and cannot remove obligations.

## Evidence and interpretation

A fixed diagnostic constructs the proposed weighted_sign pair {0,1}, class1 MC
point using **new factor identities**, not old coordinate positions. Original
input factors are zero, ReLU negative/positive/sign factors are -1/1/1, affine
error factors are zero, gate is 1/4 and product is -1/8. The entire newly checked
LP must accept the point exactly. If not, retain a failed diagnostic and do not
search for another point. A feasible objective at or below the acceptance gate
can show that this MC relaxation cannot certify the corresponding property.
It is not a counterexample to the original network or an LP-optimality claim.

Only an endpoint bound above the threshold and a checked feasible MC objective
below it establish certificate-level representation separation on the same duty.
Comparing two proposed lower bounds alone does not isolate representation from
optimization convergence. Fixed-gate cases are mathematically equivalent but
their different numerical parameterizations need not produce equal candidate
bounds. A [0,1] gate recovers expert-wise sufficient conditions, not additional
weighted precision. Complete request outcomes aggregate every duty.

## Execution and stopping

Freeze before implementation and execution. Keep the old mathematical modules
unchanged and use new files/results. Thirteen groups cover source/domain and
property identity, all MC planes and ranges, private/binary factors, missing and
stale evidence, exact point validation, deadlines, checker independence and full
cost/archive inventory. Eight normal executions retain their outcomes irrespective
of which arm has more positives. Alternate arm order across cases.

Each finite arm has one cooperative 300-second deadline including source creation,
propagation, preparation, proposals, serialization, checking and observation.
Imports and test/archive orchestration are separately visible. This does not
establish a new production hard-budget supervisor or an end-to-end speedup.
An independent saved-evidence audit repeats mathematical checks and comparison
bindings without solving. Preserve failures. Do not tune any frozen component
to turn a negative result positive; all G1–G6 remain open after this finite study.
