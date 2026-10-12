# Complete endpoint consumption on checked HybridZ views

The [finite protocol](hz_store_endpoints_design_20261013.md) was pushed at
`e7626fba820774ea488a9afb7d200433acecd353` before execution. One generation
attempt completed all five calls. The final independent standard-library audit
passes; 15 new unit controls and 68 existing regressions pass (83 total).
This is a component result, not a new real-model or GPU experiment.

## What changed in the algorithm path

[The native consumer](../act/back_end/moe/store_endpoints.py) reads the immutable
request-owned common block, per-expert private suffixes and pair-only guard rows.
Common/expert sparse matrices are constructed once per request, not once per
pair/property. Global factor IDs are retained; no repeated source dictionaries,
joint HZ or full LP export is used. Eight-expert calls each construct 74 sparse
matrices, including 28 pair-guard blocks, at both tile widths.

Targets are tiled in at most eight columns. Each uses the frozen 128-update CPU
float64 projected subgradient rule; the exact selected candidate is compared with
the exact zero candidate. [The independent checker](../scoped_source/check_store_endpoints.py)
reads each original constraint row once per tile, combines all contributions to
each shared residual, and only then applies the box support correction. Zero
multipliers avoid arithmetic, not source/row identity checking. Absent residual
coordinates are exactly zero. Private expert factors remain disjoint.

This changes actual HZ support consumption. It does not introduce a tighter
mathematical domain than the same underlying endpoint problems. Global factor
order and common/guard summation grouping differ from the old compact kernel;
bit-identical floating trajectories are not claimed. Small same-candidate exact
checks agree with the prior original-row evaluator, including residual coordinates.

An external contract fixes source/store, label, class count, margin, weighted
top-2 and `ANY_LEGAL_TOPK`. The checker derives all classification properties and
every pair/distinct gate endpoint itself. Source mode independently checks the
input enclosure, affine/ReLU/compensation and guard chain first. Supplied-HZ mode
has a separate guarantee and uses the full gate range [0,1]. The positive gate
remains strictly greater than 1/10000000; a missing candidate is UNKNOWN, while
a missing/corrupt inventory is rejected. No negative bound establishes UNSAFE.

## Fixed results

All times below are observed child launch through parent receipt, including
imports, source creation, native preparation, proposals, exact selection,
serialization and independent reception. Each call had one 300-second deadline.

| Frozen case | Positive duties / required | Checked endpoints | Terminal | Parent seconds |
|---|---:|---:|---|---:|
| Source8, tile8 | 252 / 252 | 504 | checked declared-source positive | 9.538 |
| Same Source8, tile1 | 252 / 252 | 504 | checked declared-source positive | 23.457 |
| Negative3, tied routers | 3 / 6 | 6 | UNKNOWN_NONPOSITIVE | 2.999 |
| Existing mixed affine/ReLU source | 0 / 3 | 6 | UNKNOWN_NONPOSITIVE | 3.037 |
| Large given-HZ, tile8 | 0 / 252 | 504 | UNKNOWN_NONPOSITIVE | 114.405 |

Source8 has **32 factors**, not 25,648. Both widths use identical source, store,
property and query identities; every checked obligation bound is exactly **2**.
Its experts are individually safe by construction. This is complete synthetic
source-to-output plumbing, not a relationship-dependent gain, a trained model,
or independent route-change witness evidence. Enumerating all pairs is not a
reachability proof.

The separate large given-HZ object has **25,648 distinct factors**, 9,400 active
factors and 2,700 constraints per pair. All 504 endpoint bounds are negative;
all 252 duties remain unproved. It establishes combined layout/target consumption,
not source provenance, source-complete positive capacity or model unsafety. Its
entire coefficient recipe is independently checked, not just its dimensions.
Do not splice this width result with the 32-factor source positive.

The source8 tile8/tile1 observations are not an interleaved timing benchmark or
real/GPU acceleration claim. Large-layout proposal time was 80.193 s, exact
selection 19.936 s and online independent reception 10.596 s. Those substeps
sit inside, rather than add to, the 114.405 s parent cost. Peak child RSS was
656,688 KiB including imported libraries. Sparse conversion counters exclude
RHS/objective conversions. Internal serialization time excludes final manifest
writing; the parent interval includes it. The parent driver's own initial import
and later offline audit are outside these call intervals.

The new composition explicitly permits 96M cumulative work visits, with no
per-tile reset. The large call used 37,032,544 visits. This is a separately frozen
composition policy, not a silent increase of the old 8M primitive budget or a
change to numerical acceptance. The harness is not production/GPU supervision.

## Independent checks, retained failure and limitations

[Unit controls](../scripts/test_store_endpoints_units.py) cover full inventory,
tile tails, duplicate/swapped/foreign slots, wrong property/source, bad dual signs,
overstated/noncanonical bounds, offset accounting, alias capture, deadline/work
refusal and exact residual differentials. The 83-regression receipt is explicitly
a transcribed tool observation, not a fabricated raw subprocess log.

[Relocation controls](../scripts/test_store_endpoint_portable.py) copy the source8
package and four standard-library files to a new location and invoke `python
-B -I -S`. Complete checking returns the identical result without checkpoint,
dataset, old run directory, NumPy, Torch or solver. A null last candidate leaves
503 checked endpoints and UNKNOWN; missing the final tile, an overstated bound,
foreign property or altered source is rejected for the intended reason.

The first archive auditor rejected ACT's three fixed import messages before the
final JSON receipt. Its code and failure are preserved. V2 accepts only that exact
prefix or a receipt-only stream. V3 additionally asserts each frozen label, margin,
mode, class count and tile width, and rechecks all large-recipe coefficients.
No source, candidate, original result, threshold or parent time was rewritten;
no numerical regeneration occurred. The audit binds the candidate configuration
to execution records/code, not by proving the optimizer trajectory.

These five normal calls cover full [0,1] gates and exact [1/2,1/2] ties. They do
not execute the sign-restricted half-interval branches; no blanket all-gate-branch
test claim is made. Network capture/checkpoint correspondence, trained-density
source capacity, convolution/pooling, native floating execution and actual CUDA
remain untested. The given-HZ/source guarantees must stay separate.

## Archive and decision

Raw archive: `/data1/Kane/MOE/baseline_runs/hz_store_endpoints_20261013_r1`.
The [compact record](hz_store_endpoints_20261013_r1.json) binds final evidence.
At the post-audit/relocation observation there were 2,175 regular files totaling
13,837,862 bytes, before the small final storage ledger: well below 128 MiB. This
is a final archive-admission cap, not a continuously enforced filesystem quota.
No historical data/checkpoint/failed evidence was removed.

Stop this finite source-width/endpoint-count/timing study. Complete endpoint
consumption is connected; do not rerun these fixtures to seek better bounds.
Next bind this schema to the existing bounded reception/supervision contract,
and treat the same block GPU candidate kernel as a separate device factor with
CPU-equivalence controls and fresh resource admission. Do not automatically retry
the previously refused GPU slots. Captured-source streaming remains a separate
real-intake blocker; these results do not authorize sealed inputs or a real freeze.
G1–G6 stay OPEN and external scientific comparisons are unchanged.
