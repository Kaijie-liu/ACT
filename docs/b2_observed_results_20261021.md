# B2 external paths on the observed unresolved cohort

The [frozen catalog](b2_observed_freeze_20261021.json) follows the completed
[B1 diagnosis](b1_observed_results_20261021.md). It is a failure-selected
development study, not an independent population comparison or a G4 win claim.
Execution has not started at this freeze. [Records](b2_observed_records_20261021.json)
will preserve every attempted and unexecuted entry, intersections and costs.

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

Run the immutable `b1_b2_20261021_47ce052fa` copy in the new
`b2_observed_20261021_r1` directory. Record full cost, missing obligations and
terminal intersections before drawing mechanism conclusions. Tooling errors
retain their attempts and costs; ordinary UNKNOWN and TIMEOUT continue the
catalog without a new positive-result gate.
