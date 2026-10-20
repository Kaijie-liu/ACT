# A2 strong support comparison

This once-only mechanism screen asks whether global margin-tilted supports
provide a useful checked-bound/cost tradeoff against strong priced slicing.
It uses the unchanged synthetic ten-class, 3,079-factor HZ fixture and first
property from `margin_native_cases.py`, not a trained model or new test input.
The actual source, full property and scope are hash-bound. G1–G6 remain OPEN.

The [prior decision](margin_a2_decision_20261021.md) and
[dual-transfer derivation](margin_slice_transfer_20261021.md) exclude an
inherently tighter-domain claim for the current scalar rule. This execution
does not retry the old product control or repair its invalid subduration.

## Fixed arms and all originating queries

All arms use the same continuous P, original shared/private factors, property
constant, two gate anchors and existing sign/quarter-Lipschitz sigmoid enclosure.
Original binary factors are explicitly relaxed. New candidate generation uses
the same SciPy/HiGHS raw-marginal provider throughout; independent exact source,
slice, product and scalar checkers retain their acceptance rules and bit limits.

| Fixed order | Maximum native calls | Charged work |
| --- | ---: | --- |
| zero | 2 | Global zero-tilt supports and four-cell scalar check |
| finite | 4 | Zero plus fixed weighted-median tilts, matched-zero check |
| transfer | 0 additional, 4 originating | Finite arm plus exact slab transfer and dual blending |
| endpoint_2 | 4 | Two endpoints on each of two closed cells |
| N1_2 | 10 | Two signed-difference ranges, two endpoints, one product per cell |
| M3_2 | 10 | Same paid inputs plus existing sigmoid relation rows |
| endpoint_4 | 8 | Four closed cells |
| N1_4 | 20 | Four closed cells, same range/product/fallback accounting |
| M3_4 | 20 | Four closed cells, same range/product/fallback accounting |

Maximum actual calls78; separately charging transfer's origin gives82 across
standalone arm prices, not82 actual calls. A checked sign-stable N1 difference
range permits endpoint reuse and subtracts one product call, never its four
paid range/endpoint queries. M3 is not skipped on that basis. Each product arm
takes the better checked product and its own paid endpoint bound per cell;
all cells are then required. No answers cross independent arms. The transfer
reference alone explicitly reuses and pays for its originating finite supports.

Each call has a fixed1s HiGHS cooperative solve allowance, not a one-second
complete-query wall cap. Parsing, construction, native overrun, exact checks,
serialization and reception are inside the common hard300s batch. No extra
search, iterations, query retry, training, dependency change or GPU occurs.

## Closed grid and feasibility boundary

The factor-box margin range is [−5/16,21/16]. Uniform two-cell cut is1/2;
four-cell cuts are3/32,1/2,29/32. Both cover the entire box, including ties.
The unchanged P's continuous margin range is[−1/16,31/32]: its entry equalities
fix the first two factors, while z₂∈[−1,3/4] and β∈[−1,1] give
m=1/2+3z₂/8+3β/16. Private factors can be zero.
For μ in that interval, θ=(32μ+2)/33, z₂=−1+7θ/4 and β=−1+2θ give a feasible
assignment. Thus every chosen cell intersects P. These are analytic fixture
facts, not route-flip witnesses for a trained network.

The old uniform eight-cell grid has empty outer cells. Rather than discard
solver-infeasible cells without checked infeasibility evidence, this new freeze
uses two/four cells, chosen before any A2 numerical outcome. Nothing is omitted
from P and no source parameter is changed.

## Cost and decision contract

Primary mechanism effect is the exact finite lower-bound gain over both the
independent zero arm and matched zero certificates. Minimum material gain is
1/1024 in this fixture's property units. References are transfer and all fresh
slice arms; these are not equal-query methods.

Measured arm-variable time includes construction, all native calls, exact
checks, arm serialization, and both independent receptions. Every standalone
price adds the full shared batch cost, not one ninth. Transfer additionally
pays finite's originating variable cost. Shared source capture/import, wrapper
validation, copy, cleanup and accounting are therefore not free. Fixed-order,
shared-setup execution is not independently measured cold-start performance.

The controller reports actual batch elapsed and the measured-through-cut
allocation. It also adds the entire remaining hard-budget interval equally to
every arm as a conservative terminal-tail sensitivity bound. This upper bound
is not time actually spent. A cost signal must survive it; a signal present
only before that bound is explicitly gray/unresolved by tail accounting, not
evidence of no speedup. This avoids a new observation chain.

All required arms and both receptions must complete first:

1. Stop this fixed-tilt mechanism on this source if the gain is below1/1024,
   or a strong reference attains at least its bound at no greater priced cost.
2. A limited mechanism signal requires the gain and a finite cost at most0.8
   of the cheapest reference reaching finite's bound minus1/1024, including
   the conservative tail. This does not itself authorize a trained-model claim.
3. Otherwise report gray, with no automatic source/grid/tilt/time expansion.
4. Any missing required arm, unexpected execution error or failed acceptance
   makes the comparison inconclusive. Preserve the prefix and stop; never count
   an early arm as the winner over an unfinished late arm.

No population inference, exact LP optimality, network unsafety or full-request
SAFE follows from this one synthetic property. Raw dual status0 is only a
proposal. Negative checked bounds do not separate optimization and relaxation
loss without checked feasible upper bounds. Anchor, tilt, sigmoid enclosure and
scalar-cell losses remain separately identified but not automatically measured.

## Execution and evidence

Freeze binds code, trees, source/scope/property, order, grids, query budget,
effect/cost thresholds and new accounting faults. Existing owned-session
supervision is unchanged: worker phase200s, parent run/cleanup210/220s;
portable phase260s, cleanup280s, common terminal300s. One fresh admission≤60s
is included, no wait/retry loop. A new exclusive result directory prevents reuse.
`DEV_MARGIN_20261021_V1` retains8GiB AS/process, RSS/wait4 telemetry only,
cumulative account-only Work and all4096/8192-bit checks. Reception continues
the same deadline and counters. The run does not consume real-development slots.

Durable intent/return records bind native LP hashes to independently reconstructed
original-source objective/cell rosters, not just matching counts. Producer
selection of checked zero duals is retained. Four new accounting controls reject
negative/nonfinite durations, wrong native LP binding and missing proposal groups;
old source/slice/product/RSS fault rosters are not repeated. Partial prefixes
after a hard kill give only a lower bound on Work; missing tail accounting is
reported, not silently filled.

Files are bounded: common≤1MiB; each of nine arms≤2MiB; at most156 event files
each≤4KiB; nine progress manifests≤64KiB; worker, each of two session receipts,
portable receipt and terminal each≤256KiB; continuation≤4KiB. Portable code≤1MiB
is checked before execution. Data copies are limited to common/worker/nine arms.
These limits give less than43MiB retained run data/code and less than88MiB with
one lossless archive copy and bounded metadata, within128MiB. Publication checks
actual cumulative bytes plus partial-write/terminal/archive reserves; oversized
objects stop rather than raise limits. This is bounded rejection, not a promise
all possible solver outputs fit. Relocated `python -I -S` reads no ACT/native
libraries, checkpoint, training data or historical result directory.

The transfer CLI now changes `sys.path` and imports its historical case driver
only when run as a CLI; reusable transfer arithmetic is unchanged. This prevents
relocated reception from searching its run parent for unbound dependencies.
Older archives retain their execution revisions and are not replayed.

Implementation preparation includes AST/diff checks and read-only mathematical
and integration reviews. The subsequent once-only run is the runtime test;
static review is not an execution or human technical review. Real-stage ledger
remains0 stages/0 calls/0seconds; old failures and sealed boundaries stand.
