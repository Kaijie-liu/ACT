# Multimodular arithmetic controls — completed 2026-09-20

This is a separate candidate-generation component, **not a rerun of the four
real LPs**. Starting branch `feat/moe-route-verification`, clean HEAD
`85d01c7d32ecd48ce924f61ed6d743f0d10d19da`. No existing frozen source or result
was changed. No dependency was installed. CPU controls ran single-threaded at
nice 10; other workloads were inspected and left alone.

## Evidence

- Protocol: `docs/modular_basis_v1.md`.
- Controls: `docs/modular_basis_controls_attempt002.json`, **64/64 PASS**:
  18 new tests (including 24 random rational differential systems) and 46
  existing basis/primal/LP-checker regressions. No skipped tests.
- Fresh review: `docs/modular_basis_v1_review_r2.json`, **PASS, 0 issues**:
  127 hash-bound control artifacts, 30 successful systems rechecked against
  original rational equations, eight unresolved systems preserved. Five original
  LP bundles rechecked after relocation with `python -I -S`; four feasible
  points and one expected equality-violation rejection. No elimination or native
  solver was invoked by this fresh review. Solver/model/site imports absent in
  each isolated checker process, not necessarily in the control harness.
- Prior frozen primitive protocol and its real-result artifact hashes checked
  before/after controls and review. Real LP reconstructions/new real native
  calls: **0**. Synthetic native-capture regression tests are not real queries.

Attempt001 and its review also passed and remain preserved. Attempt002 re-runs
the same controls after removing surplus end-of-file blank lines only; it does
not alter arithmetic, cases, limits or acceptance. R2 binds the formatted source.

## Mechanism and adverse results

| Control | Frozen primitive / expected obstacle | New bounded modular outcome |
| --- | --- | --- |
| p=2^2100+1, equations p*x+y=p+1 and x+p*y=p+1 | raw `row_product` reaches 4201 bits at 4096 cap | one prime, exact (1,1); field products at most 59 bits, CRT 30 bits; original LP checker gives feasible U=-1 for objective -x |
| coefficients 1/2^2500 and 1/3^1600, exact zero solution | row denominator LCM exceeds cap | coefficientwise field mapping, one prime, exact (0,0) |
| first prime divides coefficient denominator | field not defined | skip that prime, next prime succeeds; no rational singularity claim |
| first prime singularizes a rationally nonsingular basis | field rank loss | skip that prime, next prime succeeds |
| x=first_prime+1 | first modular answer aliases x=1 | exact residual rejects alias; third prime reconstructs correct answer |
| corrupt field candidate (injected fault) | modular output wrong | two exact-residual rejections, bounded unresolved, no candidate |
| singular / inconsistent rational systems | no invertible selected field basis | fixed two-prime control exhausts; UNKNOWN, not infeasible LP |
| intrinsic answer (2^4200,2^2100) | answer exceeds existing rational cap | LIMIT at round 69, 2070-bit modulus; a trial candidate's exact residual produces a 4167-bit numerator |
| small fraction 1/2^2100, two-prime control | insufficient reconstruction information | unresolved, even though true answer's numerator/denominator individually fit 4096 bits |
| synthetic 4096-row lower bidiagonal system | sparse interface scale | exact all-1/3 vector in one prime, 4096 pivots, zero fill |

The first row establishes **an analytic case where avoiding full integer cross
products matters**, not a claim that all intermediate growth disappears. Its
checked U is an upper bound on this supplied LP's minimum, not an independently
checked lower bound, optimality claim, neural-network counterexample or SAFE.
The 4167-bit event belongs to a *trial residual*, not to a successfully recovered
large solution; no candidate or original-LP check was emitted for that case.

All original mapping and full-LP acceptance checks remain. Nonzero E residuals
are allowed in a candidate basis solution but rejected by the original LP
checker. Missing/changed row, coordinate and identity controls reject. Deadline
(including expiry after field solution), operation, input/live/fill/heap, CRT
bit and rational bit controls retain LIMIT/TIMEOUT without a candidate. The
fixed-box empty-basis and redundant-row mappings remain supported.

## Why candidate output does not rely on modular luck

For any prime not dividing source denominators, reduction is a homomorphism
from those rational coefficients into the field. Field solutions are merged
coordinatewise by CRT. Euclidean reconstruction proposes a fraction using the
symmetric bound floor(sqrt((M-1)/2)); this can fail or produce a premature alias.
Neither a small reconstructed fraction nor a field rank result is a proof about
the original LP. Every returned vector must first have zero residual in **all
original assembled rational equations**. It is then only `CANDIDATE_ONLY` until
the separate unchanged checker verifies original LP equalities, inequalities,
bounds, objective, and property/source identity. The reviewer does not call the
modular constructor to establish these exact feasibility facts.

## Limits and cost interpretation

The method replaces one large-integer elimination with repeated sparse field
factorizations. Fill, repeated work, CRT storage, symmetric reconstruction and
exact residual checking can become the next bottleneck. The 128-prime schedule
provides less than 3840 modulus bits; its symmetric reconstruction range is
roughly 1920 bits per numerator/denominator, **not completeness up to the input
4096-bit cap**. A wrong early candidate can itself exhaust exact-residual limits.
No increased cap, extra basis, implicit fallback, or unbounded reconstruction is
used. Explicit value guards are not a bound on hidden Python integer/Fraction
temporaries, interpreter allocations, or OS RSS.

The first 64-test run took 1.339 seconds; receipt wall time including its post-control
frozen-artifact verification was 2.803 seconds. The initial verification before
the receipt timer is excluded. Its fresh review took 3.057 seconds; the final
source-identical-mathematics rerun took 1.346 test seconds. Exact final receipt
and review wall costs are retained in their JSON records. These are
descriptive shared-machine **control costs**, not total request budgets or a
real-LP speedup. Module deadlines are cooperative, not an outer process watchdog.

## Next gate

The arithmetic controls justify a **separate supervision-integration stage**:
one original deadline, process cutoff, partial evidence, exhaustive cost
accounting, and unchanged exact original-LP checker. Only after that stage and
its controls should any real diagnostic be separately frozen/authorized. The
four primitive real failures remain 4/4 LIMIT. No real feasibility or LP-
relaxation explanation is upgraded by these analytic controls.
