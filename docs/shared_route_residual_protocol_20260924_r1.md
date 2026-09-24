# Shared router residual evidence — bounded development protocol R1

## Goal and non-goals

The last real frontier comparison is sealed. Its online run spent 104.3811 s
preparing router evidence and 185.0334 s checking it, then timed out before an
accepted route receipt or any output obligation. The separately charged audit
could exclude 24/28 pairs. This motivates reducing repeated route-evidence work,
not diagnosing expert relaxation and not claiming a recovered real certificate.
See `frontier_proof_execution_result_20260924_r1.md`.

This opt-in implementation changes the representation and checking of the
**same fixed final-affine equality candidates**. It does not change the source
enclosure, output relaxation, margin, strict-sign acceptance, route semantics,
25% policy, training, native solver, or production verifier. No persistent cache,
finite-field elimination, extra query search or data-dependent tuning is added.
Inputs 98, 4088, 4096 and 4098 remain sealed. Historical main-table source gaps
and all external comparison results remain unchanged.

The scientific objective remains complete, source-accounted MoE output
certificates at a fixed total budget and fair external comparison. A faster
router-proof segment alone meets neither that objective nor an ISSTA acceptance
criterion. No venue outcome or external superiority is promised.

## Equality-residual composition lemma

Let a checked source HZ use one ordered factor vector xi in [-1,1]^n, with
binary factors explicitly relaxed to that box. Write router score i as

    r_i(xi) = c_i + G_i xi,     E xi = h.

Other equalities/inequalities and input constraints remain inherited. For any
equality potential z_i, define a_i = c_i + z_i h and R_i = G_i - z_i E.
On every feasible assignment, r_i = a_i + R_i xi exactly. Thus

    r_j - r_i >= a_j - a_i - sum_t |R_j[t] - R_i[t]| = L_ji.

Proof: substitute E xi = h in both affine expressions, subtract them in the
same factor coordinates, and minimize each remaining linear coordinate over
[-1,1]. This is an outer bound even if xi has further constraints. All arithmetic
is rational, including accumulation and residual correction. Equalities may
have arbitrary signed potentials. **Inequality potentials cannot be substituted
into this score identity**: subtracting two merely lower-bounding score formulas
is invalid. The schema forbids them and does not accept claimed residuals/bounds.

The bound equals the ordinary residual-corrected LP dual bound with inequality
dual zero and equality dual z_j - z_i. For the fixed final-affine provider, each
z_i eliminates its own final affine output factor, exactly matching the previous
two-factor pullback. This is an algebraic reorganization, not a new general LP
duality theorem or a stronger relaxation.

Importantly, R_j - R_i is computed BEFORE taking absolute values. Replacing it
by independently bounded scores loses common-input cancellation; e.g. scores
100x+1 and 100x have difference 1, not the separate-box lower bound -199.

For each unordered legal top-2 pair S, a checked L_ji > 0 with i in S and j not
in S proves S infeasible on the entire domain. Zero/ties never exclude S. All
other pairs remain obligations. Multiple remaining pairs do not prove multiple
reachable routes. The route receipt always declares no complete output proof.

## Independence and identity contract

`shared_route_residual/check.py` repeats the existing independent box, affine,
ReLU and factor-frame checks, then parses the final HZ once for residual algebra.
This is NOT a claim that the whole source trace is parsed only once. It validates
all stored constraints, even inequalities unused by the candidate family.
It never calls the candidate provider or a solver. Candidates bind source,
request (including output property), router trace, factor ordering and invocation.
Arbitrary equality potentials remain untrusted; bad potentials can weaken the
bound but cannot fabricate acceptance. No prior-run residual is cached.

Trust retained: declared graph/program correspondence, stored-center
preprocessing and standard-library checker/runtime. This checks the declared
real graph enclosure, not deployed floating-point execution. Experts and output
properties are not evaluated by this router-only study. New residual evidence
hashes are not mislabeled as hashes of old materialized LP matrices.

## Controls before the freeze

Exact differential against unchanged pairwise checking: analytic pruning,
ties, crossing routes, random networks, varying E/C/depth, zero radius, constant
scores, general signed equality potentials, binary factors and residuals.
Negative controls: missing rows, malformed CSR, nonfinite values, illegal
inequality potentials, source/property/factor/run transplant, inward input or
changed affine enclosure, mutation after a previous check. Fresh `python -S`
checks import no producer/model/native solver. Deadline, exception, partial
evidence, terminal overrun, relocation, full-cost and no-overwrite controls apply.

## Frozen synthetic timing design (no real experiment)

Three deterministic fixtures, E=8, C=10, width=32, three hidden ReLU layers,
seed=724, radius=1/8. `prunable` and `tied` use identical final router weights
1/32, with respectively descending offsets 3(8-i) and zero offsets. `random`
uses the unchanged seeded final layer. Experts are present only for source
identity validation; no expert propagation or output solve is timed/performed.

For each fixture: three repetitions of pairwise vs shared checking. Reverse arm
order on odd repetitions; 18 fixed calls total, sequential, CPU, two-thread env,
sampled 8 GiB limit, **30 s for each whole router segment**, reserving up to 2 s
for publication. No repeat after failure, size increase or deadline extension.
These are synthetic repeated measurements, not independent trained models.

Each arm independently generates/validates its source, builds the same router
trace, prepares and serializes its candidates, starts a fresh independent
`python -S` checker, validates source and all 56 ordered bounds, and serializes
the terminal/cost records. The measured total includes imports, parsing,
construction, candidate generation, serialization, checking, owned cleanup and
terminal publication. Final cost-ledger write and later audit are separately
identified; late publication invalidates success. No free prefix/candidate
sharing between arms. Study measures the ROUTER SEGMENT, not full MoE latency.

Primary control: exact equality of all 56 bounds, residual terms, nonzero
residual counts and all 28 pair decisions. Hashes use distinct schemas.
Report every status, full segment/build/check time, candidate bytes, sampled RSS,
and any adverse/tied-case overhead. A speed signal permits planning separate
full-budget integration; it cannot upgrade any historical timeout or certificate.
If equivalence fails, stop; if no cost signal, report it without tuning fixtures.

The executable config binds all existing 538 source/protocol identities plus
new implementation, tests, this protocol and the passing controls. Commit/push
the freeze before timing. Saved-only recheck and cost audit precede archive.
Do not automatically launch a real request after this study.
