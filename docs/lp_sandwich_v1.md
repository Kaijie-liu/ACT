# Exact LP primal/dual diagnostic component — V1

Implemented and controlled on 2026-09-19. **Not integrated into production;
no real-request re-solving, no new performance experiment, no changed SAFE gate.**
This addresses the missing *evidence interface*, not yet the 30 unresolved real
obligations. Prior source-cache and nonpositive-analysis results remain sealed.

## What the checker establishes

For the supplied finite-box continuous LP

`min cᵀz+d, A z ≤ b, E z = h, l ≤ z ≤ u`,

the standalone checker performs exact rational arithmetic on stored coefficients
(JSON floats mean their exact binary rationals). Given a candidate point z, it
checks every variable bound, every inequality and every equality, with **zero
feasibility tolerance**. Only an exactly feasible point yields

`U = cᵀz+d`, an upper bound on the LP minimum.

It independently checks a supplied dual candidate with y≤0 and unrestricted v:

`r=c−Aᵀy−Eᵀv`,

`L=d+bᵀy+hᵀv+Σ min(r_j l_j,r_j u_j)`.

The residual box term is retained. If both witnesses pass, `L≤LP optimum≤U`.
Exact equality L=U establishes optimality **of this supplied LP only**. Neither
native solver success nor its objective is used as proof evidence.

| Checked evidence | Diagnostic meaning |
| --- | --- |
| L > frozen nonnegative acceptance threshold | LP positive lower bound; not by itself a network certificate |
| Exactly feasible point with U ≤ 0 | This LP cannot prove a strictly positive minimum; not network UNSAFE |
| 0 < U ≤ threshold | This LP cannot meet the strict acceptance threshold; not a claim of nonpositive true margin |
| L ≤ threshold < U | Candidate-versus-relaxation gap remains unresolved |
| Candidate point not exactly feasible | No LP upper bound, irrespective of small residuals or solver success |
| Missing witnesses / only native infeasibility status | No new proof; unresolved |

A positive U alone does not prove LP positivity. An infeasible relaxed point
does not prove the LP is infeasible. All output records explicitly set
`network_SAFE=false` and `network_UNSAFE=false`.

## Files and identities

- `lp_sandwich/check.py`: standalone standard-library checker, no optimizer,
  model, ACT or array-library dependency.
- `lp_sandwich/propose.py`: opt-in one-shot SciPy candidate capture component.
  Records raw x, native status/message/objective, equality/inequality residuals
  and marginals, and lower/upper-bound residuals/marginals **before** independent
  checking. Native floating records are untrusted diagnostics, not certificates.
- `lp_sandwich/tests.py`, `controls.py`: analytic, mutation, relocation and
  deadline controls. Old frozen files are verified unchanged before and after.

The bundle contains one LP, its obligation statement, and optional primal/dual
candidates. The statement binds request ID, source/export hashes, ordered pair,
property index and coefficients, LP identity and acceptance threshold. Consumers
must supply the expected statement digest from their frozen request contract;
they must not derive authority solely from the candidate's own label. Hash
binding does not independently verify that a network or property was correctly
lowered into this LP.

The checker does **not** verify network→HZ, guard lowering, route exclusions,
McCormick construction or complete pair/property coverage. Those remain outside
this component. A feasible point in a continuous relaxation may use fractional
binary factors or nonphysical product assignments; it is not a full-model
counterexample.

## Portable checking and deadlines

Copy `check.py` (renamed `verify.py` if desired) and `bundle.json` anywhere:

```sh
python -I -S verify.py bundle.json \
  --bundle-sha256 EXPECTED_FILE_SHA256 \
  --statement-sha256 EXPECTED_STATEMENT_SHA256 \
  --timeout-seconds 30
```

The consumer must require **exit code 0 and exactly one valid JSON result**.
Nonzero exit, partial output or multiple JSON records cannot be accepted, even
if an earlier output fragment looks positive. The CLI counts parsing/loading
and final publication against its deadline, supports only (0,300] seconds,
caps input bytes at 128MiB and uses a POSIX alarm plus internal deadline checks.
Timeout exits 3, malformed/invalid evidence exits 2. Files, objective, variable
bounds, sparse layout, dual signs and claimed values are all validated.

`propose_to` requires an absolute deadline, makes at most one native call with
at most 60 seconds (also capped by remaining time), and never retries or repairs
a point. It preserves native output on subsequent checker failure. It is a
**component**, not a full request supervisor: the native call and filesystem
publication still require an outer process watchdog for any real experiment.
Its elapsed time is not asserted to be an end-to-end MoE execution time.

## Controlled results

[Attempt 001](lp_sandwich_controls_attempt001.json): 25/25 PASS.
[Attempt 002](lp_sandwich_controls_attempt002.json): **26/26 PASS**, adding a
three-variable sparse rational optimum control. No failed attempt was discarded.
Sources are bound separately in each numbered receipt.

Controls cover positive/nonpositive exact bounds; weak lower bounds for LPs
with opposite optimum signs; missing primal/dual; zero-width coordinates;
threshold equality; sign, identity, property and objective mutation; malformed
CSR/nonfinite/duplicate JSON; arbitrary equality dual signs; reference-checker
differential; relocation under `python -I -S`; actual CLI timeout; preservation
of native records after checking timeout; and no retry/overwrite.

Two particularly relevant actual native-solver controls:

1. `min x` on [-1,1]: the exact endpoint gives L=U=−1. This is independently
   checked LP optimality, explicitly NOT a network counterexample.
2. A zero-objective LP with exact equality `x=1/3`: SciPy reports success,
   but its stored binary float is not exactly 1/3. The checker rejects primal
   feasibility and emits no upper bound. The native record remains available.

These are analytic controls, not results on the convolutional requests. They
demonstrate why native optimality labels cannot fill the previous evidence gap.

## Next gate — not yet executed or frozen

Before using this on a real saved obligation, freeze a separate small diagnostic
with explicit source/export/LP/request/property hashes, a single native attempt,
outer watchdog, complete cost/terminal accounting and isolated checking. Use the
unchanged LP, not tighter ranges or a changed objective, and keep it separate
from cache/scheduling comparisons. No fallback to tolerance-based feasibility,
automatic rational projection, basis repair or extra solve is allowed on failure.

If the point fails exact feasibility, preserve **unresolved**, not “LP too loose”.
If a nonpositive feasible relaxed point is checked, only then is the corresponding
LP limitation established. Isolating a particular range or envelope would still
require its own bounded control. No real diagnostic is launched by this release.
