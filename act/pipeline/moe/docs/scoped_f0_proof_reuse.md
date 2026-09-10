# Scoped F0 proof reuse and independent LP checking, v1

This opt-in implementation follows the completed four-arm R1; it does not
modify that run, its schedules or the default production configuration.
`configs/staged_verifier_proof_reuse_v1.json` differs from the default only by
`scoped_proof_reuse=true`. No optimal-status or positive-margin gate changes.

## What is actually reusable

Tier 1 currently does not emit per-property MILP proof objects. This version
therefore collects its already-computed guarded **output interval endpoints**
per expert, and constructs positive classification-margin facts. It neither
infers all properties from an UNKNOWN expert nor pretends a primal incumbent
is a proof. Whole-expert solver results and partial MILP infeasibility proofs
are not reused in v1; adding them requires a separate evidence contract.

Let X_i be the same request box intersected with legal top-2 membership of
expert i. For a legal pair S={i,j}, X_S is contained in both X_i and X_j,
including all ties. If a shared classification property has lower bounds
L_i>tol and L_j>tol on those membership domains, convexity gives

    q F(x) >= lambda L_i + (1-lambda) L_j >= min(L_i,L_j) > tol.

Only this property is skipped. Otherwise the original F0 encoding/solve runs.
If all properties for a pair are reusable, pair expert propagation and gate
support are also unnecessary and skipped. No independent variable IDs or
joint abstract arrays are reused: these are logical facts about the same
concrete input. There is no arbitrary guard-inclusion oracle or cross-request
cache.

Each fact binds request, model-state hash, represented lower/upper box,
classification property and competitor, router frame, numerical policy,
expert identity, membership guard kind, source interval and exact arithmetic
certificate. Auditor reconstructs a reused row from the Tier-1 source records
instead of trusting a positive number or a source-reference string. This is
still conditional on the correctness of HZ propagation and the recorded
membership construction; structural auditing does not independently prove
those endpoints. Missing or mismatched facts cannot justify skipping F0.

The normal absolute-plus-relative correction and nextafter are retained when
turning source margins into float acceptance values; an exact rational check
additionally confirms the claimed value does not exceed the supplied interval
LP bound. This does not prove that fixed slack covers all solver/propagation
errors. Preparation and pair work remain in measured end-to-end time; reused
rows report zero **solver** time, not zero total processing time.

## Independent finite-box LP checker

`act/back_end/solver/lp_certificate.py` checks the explicitly supplied problem

    min c*x+d, A*x<=b, E*x=h, l<=x<=u,

with finite l,u. For supplied y<=0 and unrestricted z, define

    r = c - A^T y - E^T z,
    L = d + b^T y + h^T z + sum_j min(r_j*l_j, r_j*u_j).

On every feasible input, c*x+d >= L. All products, residuals and comparisons
are computed with Python Fraction; approximate stationarity is never assumed.
The LP identity and dual dimensions/signs must match, and the claimed bound
must be at most L. JSON float coefficients mean their exact binary rationals;
explicit rational strings are also accepted. Infeasibility is not established
by this bound checker. A separate optional SciPy proposer supplies untrusted
duals; `check()` itself never invokes a solver or trusts its objective/status.

This is **not** a MILP search-tree checker, a proof of network-to-LP lowering,
or a deployed floating-point execution proof. No non-optimal MILP result is
newly accepted by the production verifier. The tiny guarded support control
has x0>=1/2, x1=1/4 in [0,1]^2 and objective x0+x1, yielding exactly 3/4.

## Validation and next evidence

The three-class two-expert control has two safety obligations. Both versions
are SAFE, but scoped reuse reduces F0 encoding/solve calls from two to one.
Tests mutate model, domain, frame, property, policy, expert, claimed bound and
dual signs; missing second-expert facts cannot skip a mixture obligation.
The opt-in configuration and base numerical policy are tested together with
the original staged/monolithic/comparison regressions.

Run `python -m act.pipeline.moe.proof_reuse_controls --output NEW_DIRECTORY`
on a clean feature branch for retained control packages and audit records.
This is a code/analytic-control stage, not a claimed speedup or new official-
scale certificate. A separately frozen observed-cohort engineering comparison
is needed before claiming saved calls or net coverage on bal010. The exact LP
checker needs a hash-bound actual HZ/LP export before it can check that larger
class of bounds. Neither gap is filled by more PASS labels on JSON records.

Retained controls have run at `cceadd326` and were independently rechecked:
`results/scoped_proof_reuse_controls_20260911_r1.json` records SAFE/SAFE,
2/1 F0 solves, 0/1 reused properties, two zero-issue package audits and the
exact 3/4 guarded-LP bound. The full focused regression suite passes 34 tests.
