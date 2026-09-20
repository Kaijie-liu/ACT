# Property-directed row selection: frozen finite result

Executed [the frozen protocol](property_ranges_v1.md) once at
`606cf2947f69b629a034d9805c2ff07c185fbef0`, in
`data/moe/results/property_ranges_conv98_20260920_v1`.
Both arms finished within their original300s budgets. Each then passed a
separate moved `python -I -S` source/selection/output recheck and terminal
audit, with zero issues. Read the [review](property_ranges_v1_review.json)
and [saved-only analysis](property_ranges_v1_analysis.json).

**Property-directed selection substantially improves the eight common
checked lower bounds, but produces no complete positive request.** The ninth
property has no usable multiplier candidate in that arm. No row replacement,
retry, extra query, changed cap or gate adjustment followed these outcomes.

## Request-level result

| Same conv98, sole legal pair[1,2] | Prefix | Property-directed |
|---|---:|---:|
| Required output properties | 9 | 9 |
| Independently checked positive bounds | 0 | 0 |
| Independently checked nonpositive bounds | 9 | 8 |
| Missing bound evidence | 0 | 1 (competitor7) |
| Complete positive requests | 0 | 0 |
| Native range calls / checked two-sided facts | 8 / 4 | 8 / 4 |
| Native output calls | 9 | 9 |
| Complete stored-source execution | 158.4005s | 185.7265s |
| Portable proof bytes | 155,513,468 | 154,943,765 |
| Separate moved recheck plus audit | 70.0641s | 68.5693s |

Prefix returns `UNKNOWN_NONPOSITIVE_BOUNDS`; property-directed returns
`UNKNOWN_MISSING_BOUND_EVIDENCE`. Both checks cover all seven prefix
transitions,16 remaining expert transitions, shared/private join, five exact
route exclusions, selection replay and all nine new output-LP constructions.
A missing output dual is kept in the denominator, not dropped or treated as
a zero bound. Complete source/obligation checking is not complete positivity.

For competitor7 the property arm's native call returned status1,
“Time limit reached”, no usable multiplier candidate, after about16.096s
against its unchanged16s solver cap. The owning300s budget remained intact.
This is bounded candidate-generation failure, not a model counterexample,
a checked feasible point or a proof that its LP cannot certify.

## What selection actually changed

The two arms' available full pre-query score/source inventories agree for
both experts; no snapshot is missing. Prefix selected rows0,1 per expert.
Property-directed selected expert1 rows10,30 and expert2 rows17,31, keeping
two rows per expert and ascending query order. No old output point was used
by either execution. All four newly selected ranges passed both signed-dual
checks and tightened strictly:

| Expert / row | Generator range | Checked range | ReLU consequence |
|---|---|---|---|
| 1 / 10 | [-14.989215, 16.082634] | [-1.733783, 3.546047] | unstable → unstable |
| 1 / 30 | [-14.504665, 12.050121] | [-3.487049, 1.124352] | unstable → unstable |
| 2 / 17 | [-2.967610, 6.141332] | [0.318939, 2.840346] | unstable → active |
| 2 / 31 | [-4.319121, 5.324047] | [-0.456463, 1.479280] | unstable → unstable |

All four property-selected rows were unstable; one becomes active while
three remain unstable. The prefix arm instead queried three already-inactive
rows plus one unstable row that became inactive. Both therefore finish with
the **same counts**:24,462 continuous factors,1,618 binary factors,11,580
equalities and3,240 inequalities. They retain3,072 shared input factors and
20 combined expert-output coordinates.

Equal counts do not mean equal constraints, source identity or bound quality.
The prefix joint is `cdc671b5ba8bf1bb8b10b288169247bb86ed0c9cc99b9221a66577682c16496a`;
the property-selected joint is `01bd25ce6b9a3d8cbdba79443be30c33cdc00f55972e879cd13d28f611857c38`.
Every output LP and candidate was freshly generated against its own source.
The prefix source reproduces the prior active-range source, but its runtime
and candidates here are a fresh trial, not copied from that result.

## All output bounds

Rounded displays below are backed by exact rationals in the archive.

| Competitor | Prefix checked lower bound | Property-directed checked lower bound |
|---|---:|---:|
| 1 | -73.10270276 | -65.11512852 |
| 2 | -66.22245016 | -58.76701509 |
| 3 | -59.34977209 | -49.94490895 |
| 4 | -65.15967917 | -56.52503499 |
| 5 | -64.66546093 | -53.93390670 |
| 6 | -76.33747910 | -65.01063963 |
| 7 | -74.75273589 | Missing |
| 8 | -67.34927758 | -57.02234171 |
| 9 | -70.78956668 | -61.52105969 |

Eight common recorded bounds improve by7.455435 to11.326839. No bound turns
positive. Competitor7 is excluded from this paired numeric difference because
its second bound is missing; the loss of available evidence remains explicit.

This is direct finite evidence that the new row choice yields more useful
**recorded checked lower bounds** at the same quota, even though binary counts
match. It does not establish the same difference between exact LP optima,
unique root-cause attribution or a complete-request benefit. There is no
checked primal/optimality witness and no new model-unsafety conclusion.
Earlier-source and weighted-product contributions remain unseparated.

## Complete cost, not just the range solver

| Phase, subprocess-inclusive | Prefix | Property-directed |
|---|---:|---:|
| build | 60.6578s | 60.9611s |
| propose | 26.7500s | 55.9716s |
| seal | 0.1810s | 0.1609s |
| check | 70.6980s | 68.5298s |
| Total through publication | 158.4005s | 185.7265s |

The execution difference is+27.3260s
(about17.25%). This single
fixed-order trial is not an unbiased speedup/slowdown estimate. In particular,
better recorded bounds did not make the output proposals cheaper here:
26.7500s versus55.9716s.

Selection and its snapshots cost2.0447s /2.0720s,
nested within build. Range generation totals32.5548s /
32.8679s, including native solving1.3366s /
1.5774s and exact prechecks23.5160s /
23.5849s. Assembly, conversion, layer propagation,
serialization and joint/output construction remain separately recorded in the
analysis. Do not add those subcosts again to the phase totals.

The final in-budget checker repeats the required mathematics independently:
source/selection54.7663s /54.4076s, output-bound checking15.6944s /13.9019s.
The fresh post-terminal audit is separately reported, not deducted from the
execution. Frozen historical capture/model loading is excluded, as registered;
all subsequent source propagation, scoring, range checks and packing is charged.

## Archive and reproducibility

Prefix portable manifest: `3786eb72fcce4e83588d2f119680613035fec79e25e05bce0efba93ae7cb9f59`.
Property portable manifest: `cc76faf839f1e15436b5dc92671b18349efeb05685321d686abe9d15e0c714c3`.
The [archive receipt](property_ranges_v1_archive.json) binds freeze, execution,
raw review, archived review, derived analysis, terminals and test status.
The committed review has identical JSON values to the raw review, with a
trailing newline added; their byte hashes are separately recorded. Raw input,
parameter/source matrices, candidate vectors and moved proof bundles remain
local and are not committed.

Rebuild only the descriptive analysis, without new solves:

```sh
python -S scripts/analyze_property_ranges.py docs/property_ranges_v1_review.json
python scripts/test_analyze_property_ranges.py
```

The four saved-analysis tests cover exact reproduction, missing/duplicate/
false-positive counts, selected-range/call inconsistency and incomplete-check
accounting.74 source/selection/diagnostic tests and six earlier analysis/main
table tests also pass:84total. All81 frozen execution-related files and13
captured artifacts remain unchanged. Frozen execution and failure records
are preserved; the new script is post-result reporting only.

## Disposition

The finite comparison is closed. It establishes a stronger range-selection
effect on available bounds, **not** a new complete certificate or a production
speed improvement. Do not increase queries, change ranges/gate or resume this
protocol automatically. Any future intervention requires a separate bounded
decision grounded in the still-unclosed complete output obligations, not just
a wish for more positive-looking intermediate numbers.

Input98 remains a single-route67.06%-accuracy model case. High-accuracy and
cross-family route-changing strict certificates remain open, as does native
floating execution equivalence. Main experimental tables and historical
negative results are unchanged.
