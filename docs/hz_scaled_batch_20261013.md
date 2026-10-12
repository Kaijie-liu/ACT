# V3 scaled source and batched exact endpoint consumption

Implementation/control stage, 2026-10-13. The
[protocol](hz_scaled_batch_design_20261013.md) was committed before implementation
(`bf9fa3b52`, acceptance/order clarification `29b61b91b`). This is an optional
HybridZ representation/support change; production defaults and sealed inputs are
unchanged. No dense source, real model or CUDA execution is admitted here.

## Algorithm and guarantee

The new source represents coefficients as C=(I-R)^(-1)LD. It retains separate
input factors with independently checked clipped centers and outward scales D;
the first affine bank stores captured W rather than rounded W*D. Centers and
their compensation are still checked. ReLU and gate ranges use weighted norms;
later affine/reference and guard checks remain source-bound. Auxiliary scales
are one. This is a new enclosure, not an identity-preserving V2 cache.

The support consumer computes each objective adjoint once and propagates dual
columns together, charging one coefficient visit and every column's exact
arithmetic. Objective and dual terms combine at the same factor **before** abs;
equal-scale grouping shares multiplication, not input assignments. For each
endpoint it independently checks both raw proposed and explicit-zero candidates,
then selects their maximum. Every property takes the minimum of all required
endpoints, and all tie-legal unordered pairs are obligations.

Checker-owned receipts bind source, factors, property, raw candidates and the
same cumulative Work session. A fabricated or foreign receipt cannot replace
private checked values. Portable checking reconstructs the source and plan and
rechecks both candidates, not a serialized receipt. Raw immutable candidate
bytes are published **before** acceptance, so an exact-check timeout can retain
them without claiming success.

The implemented CPU path remains float64, one thread, 128 projected updates and
129 evaluations. It does not materialize the full constraint matrix. The native
local transpose can allocate another value bank; no single-allocation claim.
GPU acceleration remains a distinct open experiment, not inferred from CPU work.

## Implementation review and capacity reconciliation

Two read-only reviews checked source enclosure and endpoint acceptance. They
are AI code reviews, not independent human review. The first-affine checker was
changed to compare input weights per row and retain only re-compressed at-most-
two-term compensation tails across rows. This avoids retaining two dense
dictionaries for every row of a layer. All error factors still precede lifts.

The complete static upper bound remains 7,963,310 of 8,000,000, for the previously
fixed zero-center dense recipe only. Source: 4,072,655. The actual consumer call
sites refine the earlier conservative ledger:

| Term changed from design allocation | Allocation | Implemented upper bound |
|---|---:|---:|
| Gate / private plan | 28,337 | 28,335 |
| Native roster / checked roots | 613 | 600 |
| Dual remainder | 11,460 | 11,452 |
| Merge / weighted supports | 42,882 | **42,888** |
| Receipt / terminal metadata | 75 | 74 |

The six added exact calls outside weighted norms are explicitly included.
Other terms are unchanged, giving consumer 3,890,637 and combined 7,963,292.
The frozen bound has 18 counts of additional conservatism; this is not a new
cap or permission to skip checks. JSON/hash/I/O/native work remains fully timed
even where the existing Work policy only ticks the deadline. Capacity is still
conditional on the fixed inventory, not arbitrary clipped inputs or runtime.

## Archived controls

Pre-archive tests: 84 tests passed across V3, parent/audit controls and V1/V2
regressions. This includes complete archived-code inventory, all eight worker
and seven nested native phase fields, and raw publication before exact failure.
The initial test-import syntax error is recorded; no source,
threshold or numerical recipe was changed to obtain these results.

Execution used clean committed code
`a5fb2cfb434f50bfdd2a0cf349695a26292f4a2d`. The
[independent audit / compact results](hz_scaled_batch_20261013.json) bind the
unchanged archive `/data1/Kane/MOE/runs/hz_scaled_batch_20261013_r1`, protocol,
source and evidence hashes. All six source-worker calls completed in the fixed order and
width 8. The 84 tests also passed inside the supervised archive.

| Fixed captured source | Checked property duties | Endpoint candidates checked | Production Work | Parent seconds |
|---|---:|---:|---:|---:|
| mixed | 3/3 positive | 12 | 6,119 | 1.840 |
| deep | 0/3 positive; UNKNOWN_NONPOSITIVE | 12 | 9,587 | 1.902 |
| clipped | 3/3 positive | 12 | 7,153 | 1.864 |
| zero | 3/3 positive | 12 | 7,075 | 1.885 |
| half | 3/3 positive | 10 | 3,074 | 1.925 |
| guarded | 12/12 positive | 48 | 4,835 | 1.841 |

All **106** proposed/zero candidate bounds match independently expanded exact
V3 matrices; all 53 required endpoints are checked. The six relocated bundles
recompute source and endpoint evidence with `python -I -S`, without torch,
NumPy, SciPy, a checkpoint, solver or historical run access. The separate
stdlib archive audit passes. It checks sources and exact bounds as well as
the full execution-code inventory and terminal accounting.

Each of the six source-worker calls retains one 300s parent deadline and
cumulative 8M Work. The unit-suite and six-source relocation subprocesses each
also have a 300s parent deadline, but do not share one Work across their
separate requests; there are eight outer calls in total. Production
Work is recorded before the test-only expanded oracle; parent time above
includes that oracle, startup, capture, checking, serialization and reception.
Thus these times are not production speedup estimates. The entire control
archive took 15.948s before terminal publication (including unit and relocation
calls), occupies 1,240,059 logical bytes, and independent audit took 0.100s.
All eight worker phases and seven nested native phases are retained.

Half contains an empty guard, zero has radius zero, and clipped is a distinct
pre-frozen input domain: none upgrades deep's UNKNOWN or establishes a new
nonempty route-changing model result. No recipe, threshold, denominator, time
cap or failed result was changed after observing these outcomes.

All G1–G6 remain OPEN. These controls can validate the representation and exact
support path, not trained-checkpoint correspondence, physical GPU speedup,
external competitiveness or admission to a dense query. Small-source controls
are now closed. The next bounded stage is a separate full-query capacity freeze
for the unchanged fixed dense source, retaining 300s/8M and peak-memory/partial
evidence accounting. Do not repeat or enlarge these small controls, run only a
large source prefix, or infer GPU/real-model admission from the static ledger.
