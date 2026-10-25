# Residual-family 1b: final pruned GenBaB results

The frozen queue completed **120/120** full requests and **13/13** auxiliary
calls: **6 numerical positives, 114 TIMEOUTs**, no UNKNOWN/ERROR/UNSUPPORTED or
unexecuted requests. This is the pruned-only follow-up, not a new ACT comparison
or a reopening of the stopped route-union mechanism.

| Model | Numerical positive / 20 | TIMEOUT / 20 | Public PGD replayed unsafe / 20 | Mean full-call seconds |
| --- | ---: | ---: | ---: | ---: |
| R2E8 seed0 | 1 | 19 | 2 | 280.06 |
| R2E8 seed1 | 3 | 17 | 4 | 255.24 |
| R2E8 seed2 | 1 | 19 | 9 | 280.14 |
| R3E8 seed0 | 0 | 20 | 6 | 294.36 |
| R3E8 seed1 | 0 | 20 | 3 | 294.40 |
| R3E8 seed2 | 1 | 19 | 2 | 282.16 |

Public PGD found26 replayed counterexamples on120 model-input domains; these
are not credited to the verifier. None overlaps the six numerical positives.
The114 nonpositive verifier requests include those26 known unsafe objects;
the fifth revision retains their labels in the diagnostic catalog.

The six positives are seed1 R2 at1259,1271,1317; R2 seed0 and seed2 at1351;
and R3 seed2 at1351. Their surviving histories numbered4,4,6,3,3,3 respectively.
All23 returned history results closed in the incomplete bound-propagation phase,
before nonlinear BaB. Independently recomputing the Cartesian partition checks
the excluded-prefix plus verified-history coverage, including all nine output
properties. It does not prove these surviving histories reachable, reprove the
numerical bounds, or establish source-complete network certificates.

## Costs and evidence

The full calls used33,727.074012s; auxiliary selection/PGD/census used206.921720s.
Final accounting including the online result audit is33,953.001589s
(9.431389h), within36,000s. All capped requests remain in costs and denominators.
The pre-execution control cost9.762978s is separate. The independent saved-record
review took5.906158s and does not supplement any online proof or deadline.

The [machine-readable final review](multilayer_stage1b_pruned_results_20261025.json)
binds immutable execution6568139e9797726a7e773f6dda1274e9e1574734,
the manifest, all final receipts,2,215 raw-file inventory entries and controller
console. Raw results occupy15,851,507bytes before the separately listed console.
The frozen auditor was recomputed, and an independent calculation checked the
positive coverage partitions, all120 PGD/request identities, zero positive/
counterexample conflicts and the full120 denominator. No model inference,
solver query or new counterexample replay was performed in this review.
An initial `python -S` review invocation stopped at the project's eager torch
import; the successful read-only audit used the existing act-py312 environment.

## Interpretation and next diagnostic

For R3, excluding11 public-PGD unsafe objects leaves49 requests.48/49 lack full
coverage; the registered telemetry groups44 as single-query dominated,1 as
many-history dominated,3 as unseparated and1 as traversal complete. The
many-history fraction1/49 has Wilson95 interval[0.00361,0.10694]. These are
operational categories, not proof that ReLU or gate coupling is the unique cause.
The inherited audit's `ENTER_STAGE2` label refers only to its old coverage
endpoint; the fifth revision does not authorize reopening the route-union line.

The interval-only census is inconclusive about exact legal-history counts:
candidate upper medians/p90 remain784 for R2 and21,952 for R3 at all three
radii. Observed lower medians are1; sampled histories are lower bounds, not an
exact count. These wide brackets cannot establish a route-combination bottleneck.

Stage4 now has55 prior1a +24 B2 +114 residual-family objects =193. Both variants
on all objects would need386 local calls; the registered cap remains250.
The frozen order and full193-object denominator retain any unexecuted suffix,
missing first-history identities and known original counterexamples explicitly.
Next: the already frozen R3 semantic controls, then final catalog/execution
freeze and V2/V3 diagnostics. No stage4 numerical call has run; G1–G6 remain open.
