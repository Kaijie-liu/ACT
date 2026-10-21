# B2 external paths on the observed unresolved cohort

The [frozen catalog](b2_observed_freeze_20261021.json) follows the completed
[B1 diagnosis](b1_observed_results_20261021.md). It is a failure-selected
development study, not an independent population comparison or a G4 win claim.
The first controller stopped before creating its run directory or invoking a
real source. A tuple/list difference in serialized compatibility metadata caused
a false identity mismatch; actual source hashes were unchanged. The repaired
controller is running from commit `2caeae501636bdc6237be89b9a4202f3451df1d2`
in an independent code copy, PID916650. All ten local diagnoses are complete:
eight positive numerical filters and two UNKNOWN. The 41-request all-pair
aggregate queue has started; its results are not yet complete.
[Records](b2_observed_records_20261021.json) retain that
failure and will preserve every attempted and unexecuted entry and its cost.

## Complete local diagnosis results

These are ten selected guarded-pair diagnoses, not ten complete MoE requests.
All 61 selected properties have matching source, clause, reception and deadline
records. Eight positive groups cover 51 properties; the two UNKNOWN groups
contain ten properties, without a per-property result inferred from their joint
outcome. All ten calls were normally received, with no ERROR or outer TIMEOUT.
Their complete call costs sum to **798.385 seconds**. The tenth ledger charges
**1,377.052 seconds**, including the 576-second failed-start upper bound and
stage overhead, and records 5,014,252 artifact bytes at that point.

| Model and input | Pair | Selected properties | Numerical result | Complete call seconds |
| --- | --- | ---: | --- | ---: |
| Seed1 / 4113 | {2,5} | 1 | POSITIVE | 171.978 |
| Seed2 / 4113 | {2,7} | 4 | POSITIVE | 5.251 |
| Seed2 / 4113 | {3,7} | 3 | POSITIVE | 5.184 |
| Seed2 / 4113 | {4,7} | 9 | POSITIVE | 5.261 |
| Seed0 / 4129 | {1,4} | 9 | POSITIVE | 8.654 |
| Seed0 / 4129 | {4,5} | 8 | POSITIVE | 5.256 |
| Seed0 / 4140 | {3,6} | 1 | UNKNOWN | 293.338 |
| Seed2 / 4141 | {3,7} | 9 | UNKNOWN | 293.032 |
| Seed1 / 4142 | {2,6} | 9 | POSITIVE | 5.266 |
| Seed0 / 4150 | {4,5} | 8 | POSITIVE | 5.166 |

The first result resolves the same seed1/4113 property that ACT's original F0
left unresolved and its offline closed-slice endpoints made positive. GenBaB
visited 364 nonlinear domains. This case therefore no longer supports an
ACT-exclusive capability claim. The other source that was expensive to slice,
seed2/4113 pair{2,7}, is positive for all four selected properties in the initial
bound phase, without BaB.

ACT's saved-HZ support and the external scalar-max graph are not the same
objective or source lowering: their reported bound magnitudes cannot be compared
as safety margins. Both results remain numerical-policy evidence. The 169.112s
offline ACT slice receipts are not a complete online ACT timing baseline; the
171.978s external call includes loading, graph construction and reception. No
population win-rate, full-request victory or strict source-complete certificate
follows from the local cohort. Continue the fixed aggregate queue before selecting B3.

The seed0/4140 UNKNOWN ends after 1,048 visited domains and an internal search
timeout, with scalar-max lower bound -0.092588444. Of 524 logged splits, 486
select the same product-input coordinate; the final global bound is unchanged
for 510 rounds. The printed range eventually reads `[4.9320,4.9320]`, which is
not evidence of exact zero width. This supports a search-stagnation observation,
not a unique relaxation cause or model unsafety. B1's pointwise replacement of
lambda by sigmoid(m) is likewise not a checked feasible or global certificate.
The second UNKNOWN reaches 464 domains and bound -0.105093709 before its
internal timeout. Its 232 splits instead select expert ReLU inputs, with seven
domains remaining; the last rounds still eliminate some domains. The two cases
therefore do not support one common gate-relaxation or stagnation explanation.
Neither unresolved result changes this frozen configuration. Independent
read-only review checks all ten identities, guards, coverage and receptions;
it is not a bound reproof.

## Fixed order and semantics

| Order | Path | Entries | Meaning |
| --- | --- | ---: | --- |
| 1 | Compatibility-adapted GenBaB guarded pair | 10 | Fixed-order ACT unresolved properties, local diagnosis only |
| 2 | Compatibility-adapted GenBaB all-pair aggregate | 41 | Same materialized requests; all 28 pairs and 9 properties registered |
| 3 | ACT route frontend plus plain CROWN whole-box weighted pair | 19 | Fixed prefix, remaining stage budget only |

The last two groups consume 60 full-request slots; the first consumes 10 local
diagnostic calls. Each call has one 300-second budget, and all 70 entries share
the six-hour stage with a 128MiB artifact cap. An unexecuted suffix remains in
the denominator. All pair graphs within a request share its remaining deadline.
There are no free ACT route exclusions for the all-pair GenBaB aggregate.

GenBaB uses the pinned CPU/float64 alpha-CROWN and nonlinear BaB configuration,
with the three disclosed compatibility edits preserved. For each property,
`max(qF+c, outsider-minus-selected guards)>0` is the sufficient condition;
unsafe alternatives across properties form an OR. Equality remains tie-legal.
An unresolved pair stops its complete-request attempt, with later obligations
still missing; it does not imply all 252 obligations were solved. Static-pair
falsification is UNKNOWN unless replay validates the original dynamic model.

Positive results are numerical filters, not independently checked source-complete
certificates. GenBaB's frozen configuration disables PGD and BaB attack; B1's
additional PGD discoveries are not a fair solved-rate baseline. Sigmoid/softmax
real equivalence also does not establish bitwise native floating equivalence.

## Readiness and execution identity

The one vector-OR API control at execution commit
`47ce052face0c58701bad471e7d6413ab6864009` completed in 11.420994 seconds,
separately from the real stage. Its numerical positive was not a required gate.
The [control archive](genbab_cpu_ready_20261021.md) preserves the actual OR
specification, nonlinear split metadata, output and reception. Read-only review
found no guard direction, tie, missing-property or shared-deadline blocker.

The new immutable execution copy uses `b2_observed_20261021_r2`, preserving
the original R1 log. Canonical JSON comparison now treats tuple/list metadata
equally while checking all source hashes and configuration values. The passed
native control is reused by binding its original receipt, unchanged invocation
body and twelve worker, graph, backend, configuration and supervision files;
the repaired controller itself has a separate bound hash. Ten ordinary tests
pass in 0.008 seconds without further native calls.

The failed startup has no saved precise timer. Its log was created at epoch
1792572548, and the process and run directory were confirmed absent at
1792573123.2984083. We charge a conservative **576 seconds** against the same
six-hour stage, not zero and not a claimed measured runtime. The retained
931-byte log and new controller log count towards the unchanged artifact cap.
The frozen query catalog, configuration and per-request budget are unchanged;
ordinary UNKNOWN and TIMEOUT continue without a positive-result gate.
