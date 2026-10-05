# C8 V1: exact dyadic balancing of the same shared-factor HZ

Previous turn was a status-only report (no research progress); the preceding
C7 experiment supplied real positive set/storage evidence and negative native
ingestion evidence. Its immutable result hashes are
f63d45dd21d7bfa6d0c415cd24b10ac99345d053fa72ca82ad7a2bc9087928ef and
281e31ec287d55b4b14355043b2310fb184eabc1557910ded15c938de6de5467.
C7 V1 is closed for advancement; do not edit or rerun it.

## Fixed structural hypothesis, registered before implementation/evaluation

Keep C7's ORIGINAL-identity common-suffix DAG, all sources/term multiplicities,
all continuous/binary coordinate prefixes and every old predicate unchanged.
Only change the scales of fresh continuously defined coordinates and their
new equality rows. This is not a solver-state or iid-dependent rule. Source
models, baseline, production defaults and historical HyZor remain untouched.

For a required defining sum with nonzero stored float64 coefficients a_i and
parent exponents p_i, let e_i = frexp(abs(a_i)).exponent + p_i. Then each
summand magnitude is strictly less than 2^e_i. Set f = max(e_i) - 26 and
T = sum_i 2^max(e_i-f, 0). Compute T entirely in int64, rejecting more than
64,000,000 summands; T <= 64,000,000 * 2^26 < 2^52, so addition cannot overflow.
The new auxiliary exponent is E = max(0, f + bit_length(T-1)). Consequently
the real L1 sum is <= T*2^f <= 2^E, even for exponents far below f. Clipping
only rounds the BOUND upward; it never rounds or deletes a model coefficient.
This replaces number-times-maximum by a tighter integer sum envelope and
uses a common minimum coordinate scale 1. All emitted coefficients still
require exact finite reverse ldexp equality.

After normalizing a NEW defining row (including its own unit pivot), choose
the smallest nonnegative integer k that makes every nonzero coefficient
at least 2^-20 in magnitude. Multiply the entire equality and RHS by 2^k,
requiring every nonzero coefficient <=2^40 and every coefficient/RHS scaling
exactly reversible. Otherwise reject. Pivot 2^k stays positive; dividing it
out recovers the same triangular unique definition. No extra row-scale array
is needed: the pivot contains k, and all pivot bytes are already sealed and
charged. Old equality and inequality rows remain byte-identical, not scaled.
Output generator coefficients must lie strictly inside the unchanged native
matrix thresholds (1e-9, 1e15); this check does not replace actual ingestion.

Full independent audit must recover every original coefficient in ONE inverse
power-of-two step including k, recheck positive power-of-two pivots, redundant
auxiliary boxes using an independent arbitrary-integer exponent sum, all old
predicates and complete original term multiset. Focused Fraction elimination
must divide by the actual pivot and recover the real original affine program.
No rounded-materialized-matrix byte identity, numeric soundness or terminal
verdict follows from that theorem alone.

## Unchanged target, accounting and advancement gates

Use ONLY the same complete ADD75 snapshot
d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed plus its native
Dense77 append, all 200 outputs, 14 terms and 9 sources. Do not move to another
instance when this version fails. Retain all C7 source/shape/frame checks.

Charge encoding at 16 abstract scalar arithmetic operations per retained
coefficient/center plus 16 per new row, versus C7's 8. The extra allowance
covers the capped-exponent subtraction/clipping, integer shifts and reduction,
row range reduction, and whole-row forward/inverse scaling. This is the same
abstract construction-work accounting, NOT a CPU-instruction count or timing
claim. Support visits are added without discount. Fixed ceilings: whole work
256M, branch 200M, HZ entries 64M, measured construction 1 GiB, CPU threads 1,
worker memory 16 GiB, worker wall 240s, test wall 60s. No larger-cap retry.

Preserve the complete offline numeric-root ledger and the same two-largest,
distinct-content native expanded Conv reference lower-bound selection from
C7. Strict decrease in BOTH complete bytes and entries is required. Include
all retained checkpoint, graph, factor, source, predicate and reconstruction
payloads; measure construction before independent oracles. Oracle streaming
cap 256M original entries remains. No full-reference total or live-root proof
is inferred from this offline comparison.

Preregister exclusive outputs results/c8_dyadic_balance_20260905_v1/ and, ONLY
if its offline gates pass, results/c8_checkpoint_ingestion_20260905_v1/.
Supervisors bind all inherited/new sources, tests, artifacts and provenance,
and automatically preserve logs/results/exit records on success or failure.
Do not mutate a source version after its complete target run starts.

The fresh ingestion worker verifies the checkpoint/source hashes before
unpickling, rebinds process-local identities and independently re-audits all
rows. Use the SAME installed SciPy-bundled HiGHS and ordinary ACT lowering,
same options and no changed tolerances. Only passModel/getLp, never solve,
presolve or optimize. Compare the entire matrix, all row/column bounds and
integrality. All original predicates are included even though C8 does not
rescale them. Any dropped/changed coefficient rejects advancement. Freeze
both native wrapper and core binary. Record output-map range separately;
property/ReLU constraints have not yet been ingested.

Required tests: inherited 231; C8 equivalent rational/geometry/frame/mutation/
budget/default-off tests; independent exact sum bound over mixed magnitudes,
more precise bound than the C7 maximum envelope, positive row scaling and
RHS identity, coefficient-window rejection, resealed pivot corruption and
same-native-backend complete retention on a small scaled HZ fixture.

Even passing both runs is NOT live ReLU78 or a terminal solve. C5 live prefix,
live-slot/whole-state proof, ordinary terminal solve, concrete witness checks,
same-structure shadows, four-concurrent nonregression, per-family and full
2413 replay remain outstanding. Formal 1870/2413, independent E0 61/400,
default-off status, all nonconvex HZ and verification restrictions remain.
