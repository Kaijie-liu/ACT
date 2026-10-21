# B2 external paths on the observed unresolved cohort

The [frozen catalog](b2_observed_freeze_20261021.json) follows the completed
[B1 diagnosis](b1_observed_results_20261021.md). It is a failure-selected
development study, not an independent population comparison or a G4 win claim.
The current retained prefix has **37 complete-request attempts: 17 numerical
POSITIVE, 18 UNKNOWN and two ERROR**, plus ten local diagnoses (eight positive,
two UNKNOWN). Four GenBaB and 19 CROWN requests remain unexecuted. The second
ERROR is retained; only that untouched suffix will continue after the bounded
compatibility repair below. Earlier prefix sections are historical snapshots.

The first controller stopped before creating its run directory or invoking a
real source. A tuple/list difference in serialized compatibility metadata caused
a false identity mismatch; actual source hashes were unchanged. The repaired
controller ran from commit `2caeae501636bdc6237be89b9a4202f3451df1d2`
in an independent code copy, PID916650. It stopped after a backend heuristic
error: ten local diagnoses give eight positive numerical filters and two
UNKNOWN; sixteen complete-request attempts give eight numerical POSITIVE,
seven UNKNOWN and one ERROR. The remaining 25 GenBaB and 19 CROWN requests
have not executed. The failed request and all earlier results are retained.
[Records](b2_observed_records_20261021.json) retain that
failure and preserve every attempted and unexecuted entry and its cost.

## Second stopped prefix and initial identity read repair

Controller 290944 has exited. Request `b2_046`, seed2/4374, stopped after
290.540 seconds because its initial process observation rejected an invalid
state, session ID or start-time field. The original fields were not retained;
the offending PID has disappeared. Its membership and unique cause are unknown.
Two pair results were saved before the interruption, but there is no accepted
full aggregate. This remains ERROR, not a bound failure or model counterexample.

The compatibility change permits one immediate strict reread only for those
parsed-but-invalid fields during the initial development observation. A complete
valid reread is required within the same deadline. Persistent invalidity,
malformed records, PID mismatches and permission errors still reject. Subsequent
identity confirmation, cleanup and signalling remain strict; the proof-policy
path is unchanged. Twenty-seven ordinary tests pass in 3.892 seconds, with no
new model or solver call; independent read-only review found no material issue.
The earlier graph/API control is reused only for unchanged files, with this
observer version and its tests explicitly bound separately.

The 47 retained calls cost 7,597.772 seconds. R3 records 8,473.849 cumulative
seconds before its final write; all retained directories and logs, including
that write, occupy 65,558,760 bytes. The next immutable suffix carries **8,812
seconds**, a conservative confirmed-dead upper bound including the repair and
ordinary-test interval, plus four seconds for preflight, not measured solver time. All prior logs, failures and
new suffix artifacts remain inside the original six-hour/128MiB stage. Continue
only `b2_047` through `b2_069`; do not rerun either ERROR or any completed query.

## What produced the seventeen complete positives

For the fixed positive requests within `b2_010` through `b2_045`, log segments
and pair receipts distinguish three actual execution paths:

| Pair completion path | Pair calls |
| --- | ---: |
| Initial CROWN without alpha optimization or BaB | 389 |
| Alpha optimization without BaB | 78 |
| Actual nonlinear BaB, also using alpha optimization | 9 |
| Total, 17 requests times 28 pairs | 476 |

Nine requests close without BaB and eight use it. Only seed2/4179 closes every
pair directly with unoptimized initial CROWN. The `incomplete_only` receipt and
the phrase "verified with init bound" include alpha-optimized completions; they
must not be relabelled plain CROWN. The records bind each request and native log.

This shows that external successes cannot all be attributed to nonlinear
branching. It does not prove which steps are necessary under another
configuration, or isolate graph representation, guard encoding and bound
generation. Many of these pairs lie outside ACT's saved route roster, so the
pair counts are neither causal contributions nor cost percentages. All results
remain numerical filters, not source-complete proofs. Complete the remaining
strong-path queue before selecting B3.

## Fixed first twenty complete requests

The fixed prefix `b2_010` through `b2_029` contains ten numerical POSITIVE,
nine UNKNOWN and one retained ERROR. The unchanged queue continues; this is a
descriptive cross-analysis of that prefix, not its final result or a population
comparison. The records bind the B1 inventories and observations to the B2
outcomes. Independent read-only recomputation agrees, with no new solver calls.

The ten complete positives cross several distinct ACT obstacles:

| Model / input | B1 terminal | Reused | Accepted status0 positive | Unaccepted status1 raw positive | Other returned unresolved | Missing | B2 completion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Seed1 / 4113 | UNKNOWN | 13 | 4 | 0 | 1 | 0 | Includes BaB |
| Seed2 / 4113 | TIMEOUT | 3 | 8 | 15 | 0 | 1 | All initial bounds |
| Seed0 / 4129 | TIMEOUT | 0 | 1 | 14 | 2 | 1 | Includes BaB |
| Seed1 / 4142 | TIMEOUT | 0 | 0 | 8 | 0 | 1 | All initial bounds |
| Seed0 / 4150 | TIMEOUT | 0 | 1 | 7 | 0 | 1 | All initial bounds |
| Seed2 / 4179 | TIMEOUT | 5 | 4 | 8 | 0 | 1 | All initial bounds |
| Seed2 / 4223 | UNKNOWN | 10 | 11 | 6 | 0 | 0 | Includes BaB |
| Seed1 / 4232 | TIMEOUT | 1 | 0 | 7 | 0 | 1 | Includes BaB |
| Seed0 / 4275 | ERROR | 1 | 5 | 12 | 0 | 0 | Includes BaB |
| Seed2 / 4275 | TIMEOUT | 0 | 9 | 7 | 1 | 1 | All initial bounds |
| Total | | 33 | 43 | 84 | 4 | 7 | Five initial / five with BaB |

These 171 obligations belong to the saved B1 route rosters. B2 instead covers
all 252 pair/property obligations in each positive request; those denominators
are not interchangeable. The seven missing B1 obligations were attempted but
had no saved return. The four other unresolved returns comprise the status0
nonpositive bound at Seed1/4113, two negative raw status1 duals, and one status1
return without a raw dual. None establishes an exact negative optimum.

Seven of these ten requests have positive raw duals for all returned remaining
unresolved obligations, but five of the seven still lack an output return and
another has the rejected ERROR terminal. The raw dual multipliers were not
independently checked. Thus external closure supports investigating proof
acquisition and incomplete coverage as well as bound strength; it does not
justify removing ACT's acceptance gate. Five external positives require no
actual BaB, so the external gains cannot all be attributed to nonlinear
branching. Differences in representation and bound generation remain mixed.
No B1 terminal or source guarantee is upgraded, and B3 selection remains pending
the complete strong-path table.

## Stopped prefix and compatibility repair

The sixteen complete-request attempts cost 2,722.531 seconds. The eight
POSITIVE results each cover all 252 registered obligations. Across the seven
UNKNOWN requests, 972 properties have positive receipts, 54 belong to unresolved
pair groups, and 738 have no receipt. Missing properties are not filled using
local diagnoses or ACT route exclusions.

| Request | Numerical result | Complete seconds | Positive / unresolved / missing |
| --- | --- | ---: | ---: |
| Seed1 / 4113 | POSITIVE | 187.611 | 252 / 0 / 0 |
| Seed2 / 4113 | POSITIVE | 21.852 | 252 / 0 / 0 |
| Seed0 / 4129 | POSITIVE | 132.992 | 252 / 0 / 0 |
| Seed0 / 4140 | UNKNOWN | 293.256 | 180 / 9 / 63 |
| Seed2 / 4141 | UNKNOWN | 294.727 | 189 / 9 / 54 |
| Seed1 / 4142 | POSITIVE | 33.526 | 252 / 0 / 0 |
| Seed0 / 4150 | POSITIVE | 31.133 | 252 / 0 / 0 |
| Seed1 / 4150 | UNKNOWN | 293.626 | 243 / 9 / 0 |
| Seed2 / 4150 | UNKNOWN | 294.857 | 99 / 9 / 144 |
| Seed2 / 4179 | POSITIVE | 26.997 | 252 / 0 / 0 |
| Seed0 / 4195 | UNKNOWN | 294.038 | 54 / 9 / 189 |
| Seed1 / 4195 | UNKNOWN | 293.847 | 90 / 9 / 153 |
| Seed1 / 4212 | UNKNOWN | 292.263 | 117 / 0 / 135 |
| Seed2 / 4223 | POSITIVE | 155.405 | 252 / 0 / 0 |
| Seed1 / 4232 | POSITIVE | 24.572 | 252 / 0 / 0 |
| Seed0 / 4237 | ERROR | 51.830 | No accepted aggregate |

Seed2/4223 is informative for proof acquisition: B1 returned UNKNOWN despite
complete output-candidate coverage and six positive raw dual values that its
policy did not accept. The external path closes all obligations numerically;
27 pairs finish in initial bounds and {0,6} uses BaB, visiting 228 domains.
This supports investigating evidence acquisition, without independently checking
those raw duals or promoting B1's result. Seed1/4232 also closes numerically;
27 pairs finish initially and {4,6} uses two BaB domains.

Seed1/4212 instead shows incomplete coverage: all 13 returned pairs are positive,
including ACT's only registered feasible pair {0,6}, but 15 external pairs are
missing. Its full result stays UNKNOWN. This is distinct from an unresolved
returned pair and does not establish that the extra routes are redundant under
an independently checked source contract.

Seed0/4237 stopped in `bbps._fast_backward_propagation` with
`AttributeError: 'NoneType' object has no attribute 'ndim'`. The accumulated
heuristic contribution was absent. The failure occurred at pair {5,6}, after
25 pair results; its 225 partial positives are not an accepted aggregate.
The saved B1 counterexample concerns {6,7}/property5, which was not attempted,
so there is no observed same-obligation numerical conflict.

All 26 calls total 3,520.917 seconds. The stage record reports 4,099.986 seconds
including the retained 576-second startup charge and stage overhead, before
its own final write. Final bound artifacts occupy 30,563,368 bytes; the earlier
29,151,959-byte measurement excludes the 1,411,409-byte final record. The original
backend snapshot and stopped directory remain unchanged.

The repaired, separate backend snapshot adds only an empty-lA BBPS fallback:
when none of a split node's consumer coefficients is present, return zero
expected-improvement scores and the existing split points. These are not output
bounds. Any active coefficient keeps the old path; missing required dictionaries
still raise an error. The downstream masks and actual bound/acceptance computations
are unchanged. Zero scores can affect search order and cost, so this is a disclosed
compatibility version, not a performance-equivalent or algorithmic improvement.

Twenty-four ordinary controls pass, including execution of the patched method's
AST for empty, partial and active coefficient maps, shape preservation, missing
metadata rejection, and retention of ERROR rows during suffix continuation.
No checkpoint or solver call was used. Two outdated patch-scope test assertions
were corrected; an unavailable pytest invocation was replaced by existing
unittest without installing dependencies.

The suffix is running as PID290944 from immutable `f8eb02e6a`, in new directory
`b2_observed_20261021_r3`. Its start receipt confirms `b2_026` is the first
launch and all 26 earlier attempts are retained. The existing API control applies
to unchanged graph, worker and supervision files; the new heuristic has its own
ordinary tests. Carry **4,393 seconds** against the same six-hour stage: 4,373
is a conservative upper bound observed after PID916650 was confirmed absent,
and 20 seconds covers ordinary test execution. This is not measured solver time.
All old artifacts, controller logs and the new suffix remain charged to the same
128MiB cap. The original 300-second requests, query order and numerical gates
are unchanged. No failed or completed request is rerun.

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

## First complete request results

Each request below registers all 28 unordered pairs and nine output properties.
The three POSITIVE requests have 252 unique positive receipts each, with no
missing obligations. The two UNKNOWN requests retain their unresolved and
unexecuted obligations. Each
request independently loads its frozen model and materialized input and
constructs each attempted pair graph under its shared deadline; it borrows neither ACT route exclusions nor earlier
local diagnosis conclusions. Read-only review checks identities, complete
coverage, deadlines, cleanup and reception, not the numerical bounds themselves.

| Request | B1 online result | B1 complete seconds | GenBaB complete result | GenBaB complete seconds |
| --- | --- | ---: | --- | ---: |
| Seed1 / 4113 | UNKNOWN | 55.467 | Numerical POSITIVE | 187.611 |
| Seed2 / 4113 | TIMEOUT | 296.096 | Numerical POSITIVE | 21.852 |
| Seed0 / 4129 | TIMEOUT | 296.117 | Numerical POSITIVE | 132.992 |
| Seed0 / 4140 | UNKNOWN | 38.444 | UNKNOWN | 293.256 |
| Seed2 / 4141 | TIMEOUT | 296.099 | UNKNOWN | 294.727 |

For seed1/4113, 27 pairs close in the incomplete bound phase and one uses nonlinear
BaB. For seed2/4113, all 28 close before BaB. For seed0/4129, 26 pairs close
before BaB and pairs {1,4} and {4,5} use it. The path is configured for alpha-CROWN
and GenBaB, but early completions do not establish that alpha optimization or
nonlinear search was needed.
The five complete external calls cost 930.438 seconds in total, each under its
own 300-second budget. Three establish complete numerical positives unavailable
to B1 online on these objects. Seed0/4140 stops at pair {3,6}, with 180 positive,
nine unresolved and 63 missing obligations. Seed2/4141 stops at pair {3,7}, with
189 positive, nine unresolved and 54 missing obligations. Partial positive
receipts, earlier local calls and ACT route exclusions cannot fill these gaps.
The respective terminal BaB records report 924 and 430 visited domains; their
scalar-max bounds remain approximately -0.092588444 and -0.105093709.

The first fifteen-call ledger charges 2,307.556 seconds including the local
cohort and retained failed startup, with 13,517,998 artifact bytes. Read-only
review checks all five complete terminals. This failure-selected prefix is not
a fair population win-rate estimate, and positive numerical filters are not
independently checked source-complete proofs. The frozen queue continues.

## Saved evidence limits for a candidate property transform

No B3 mechanism has been selected. A static screen considered preserving the
same-factor router/expert relation through a property-sign transform. For the
unchanged output threshold tau, set A=qEa+c-tau, B=qEb+c-tau and m=ra-rb.
If A>0 and B<0 throughout the domain, the output exceeds tau exactly when
`m + log(A) - log(-B) > 0`. A lower secant for log(A) and an upper tangent for
log(-B) would give an affine support objective on the same factors. This is
not yet a new support algorithm, implementation or measured benefit.

The saved evidence does not provide the needed bounded strict-sign premises.
For seed0/4140 property F5-F3, common intervals alone put expert3 across zero
and expert6 below zero. Tier1's numerical-policy result supports expert3's
raw margin >1e-7, but after subtracting the same tau=1e-7 it supplies no known
positive lower endpoint. There is no finite affine lower bound for log(A) on
(0,U], so inventing a small epsilon would be unsound. For seed2/4141 pair {3,7},
all nine expert7 intervals straddle zero. A negative candidate-point value
cannot replace a whole-domain sign bound.

These expert membership-domain facts can restrict to the pair domain, but
retain their input, network and guard-lowering assumptions. The records bind
the saved evidence; this screen made no model or solver calls and does not
justify new range searches, acceptance changes or a claim that the candidate
is empirically refuted. Finish B2 before selecting the next mechanism.

## Fixed order and semantics

Saved B1 dynamic float64 counterexamples are bound to the same B2 models,
materialized inputs and properties: seed0/4237 maps to GenBaB b2_025 and CROWN
b2_066; seed0/4341 to b2_042; seed1/4356 to b2_045. Their violating pair/property
indices are respectively {6,7}/5, {4,6}/8 and {2,3}/1. The latter two are outside
the frozen CROWN prefix. The records bind the three witness hashes; this is a
saved-record consistency check, not a new replay or exact real-network proof.

A future complete POSITIVE, or a positive receipt for that witness's same
guarded pair/property, would require a numerical-conflict review before being
used as a safety claim. Other-pair positives and UNKNOWN/TIMEOUT do not conflict.
The frozen external configuration disables attacks, so it need not find these
witnesses. Neither this mapping nor later external outcomes promote B1's
offline counterexamples into its online result table.

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
with the three original compatibility edits preserved; the suffix snapshot
adds the disclosed empty-coefficient heuristic fallback. For each property,
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
