# Original four LPs: rational sparse input fidelity and finite protocol freeze

Date: 2026-09-20. Started on clean pushed `feat/moe-route-verification`, HEAD
`b3d91358e94313f4d84288fd6edbd9dc45405fd7`.

## Outcome

**All four original LPs pass exact native import/readback.** The new reader
performs no optimization, basis construction or candidate generation. These
are four real **imports**, not four real solver runs. Historical four LIMIT
results and zero checked feasible U remain unchanged.

| Original job | Variables | Nonzero entries checked | Entries with abs(value)<=1e-9 | Whole preparation seconds | Native import seconds | Native peak RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input220_p0 | 7,397 | 246,558 | 16 | 6.810 | 1.170 | 110,864 |
| input222_p1 | 7,682 | 236,502 | 16 | 6.520 | 1.122 | 109,736 |
| input230_p2 | 9,482 | 348,915 | 16 | 9.455 | 1.573 | 148,700 |
| input232_p0 | 9,095 | 329,664 | 19 | 9.051 | 1.473 | 145,304 |

Every original objective coefficient, finite variable bound, equality/inequality
side and matrix entry agrees. Total **1,161,639 nonzero entries**, including all
67 small coefficients. The four LPs contain 33,656 variables and 21,392 rows
in total. This establishes fidelity for these serialized LPs and this reader,
not general correctness of SoPlex or of network→HZ/F0 construction.

Native objective omits only the exact constant; the recorded original LP and
eventual checker retain that constant. No free offset rounding, matrix pruning,
scaling, row deletion or source/statement substitution occurred. Names support
native reordering while preserving exact coordinate identities. Sparse text
export wraps long lines; no dense matrix or custom arithmetic solver is added.

## Accounting and independent review

Whole per-LP preparation sums to **31.836 s**. Total preparation including
receipt work is **31.936 s**. Each whole includes original file validation,
exact export, native import and rational comparison; native times are nested,
not additive. Original historical network/HZ/F0 generation was not rerun.
GNU time records native-child RSS; this is **not** Python/composite peak RSS
or a native solver memory result. Per-phase clocks and source/readback byte
sizes are retained. Shared-machine descriptive preparation timings are not a
paired speedup or a forecast of exact LP solving.

- [Import receipt](soplex_large_import_attempt001.json): four PASS, full raw
  identities and 24 artifacts, zero optimization calls and zero candidates.
- [Fresh independent review](soplex_large_import_review_attempt001.json):
  PASS, zero issues. A separate parser compares the saved native readback to
  the hash-bound original rational source without calling the exporter, reader
  or optimizer. Additional review cost **7.298 s**, outside preparation.
- [Controls attempt002](soplex_fidelity_controls_attempt002.json): **27/27 PASS**,
  six new tests plus21 unchanged checker regressions. Includes exact sparse
  import controls,512-variable line wrapping, missing/duplicate/changed entries,
  source binding, deadline/bit/shape limits and independent parser mutations.
- The latest suite performs11 analytic read-only imports and4 legacy analytic
  SciPy solver calls from the unchanged checker regressions. **No SoPlex
  optimization and no real LP optimization** occur in these controls.

Attempt001's27-test PASS remains saved. Its broad `optimization_calls=0`
metadata failed to distinguish the four legacy analytic regression solves.
Attempt002 instruments these calls and reports separate real/SoPlex/legacy
counts; no mathematical test result, threshold, input or real-import outcome
was changed. The earlier preliminary five-test run also used read-only imports.

The fixed reader binary is
`/data1/Kane/MOE/envs/soplex-8.0.3/bin/read_only_rational_v1`; compiled with the
same pinned source, static dependencies and compiler settings as the prior
compatibility probe, using `soplex_fidelity/reader.cpp`. Old installed binaries,
old compatibility sources, ACT environment and production checker are unchanged.
Reader identity is bound in both receipts and the new freeze.

## What is frozen, and what is not

The [finite comparison protocol](soplex_finite_comparison_v1.md),
[machine freeze](soplex_finite_comparison_v1_freeze.json) and
[separate freeze review](soplex_finite_comparison_v1_freeze_review.json) bind:

- the same four original LPs/properties, ordered once, no new observations;
- fixed SoPlex8.0.3 exact configuration, native free choice of basis;
- one300-second supplied-LP clock (proposal218 / check298 / publication300),
 8GiB address-space limits and explicit output/bit limits;
- full cost accounting and unchanged final rational LP feasibility checking;
- no retries, point repairs, new samples, outcome-based tuning or arithmetic
  expansion; old Python20M counts remain historical, not a C++ work metric.

**Status: PROTOCOL_FROZEN_NOT_EXECUTED; execution_ready=false.** The protocol
and input-fidelity work requested here are complete. The new external-path
supervisor, exact rational CLI-output admission, resource/deadline controls and
terminal/cost audit have not yet been implemented/validated. They require a
hash-bound execution addendum implementing these already-frozen rules before
explicit launch authorization. A protocol freeze is not an execution-readiness
or performance claim. The reserved real output directory has not been created.

Next work should close that narrow execution gate, not develop another exact
elimination method. The eventual study must close after its four outcomes;
unresolved candidate generation is a reportable limitation, not a reason to
defer the main MoE paper or repeat the arithmetic loop.
