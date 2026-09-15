# General evidence V1: implementation, controls and selection freeze

Completed 2026-09-16. Implementation publication: `e16f4c434` on
`feat/moe-route-verification`. This preparation delivers a generic optional
request interface and a clean-only experimental freeze, **not** the new
60-request experiment or an additional real-model certificate.

## Implementation and controls

The new `moe_evidence/` path accepts E/C, identities, represented bounds,
explicit linear properties and all legal pairs. No input98-specific capture,
expected pair list, nine-property loop, parent manifest or historical bound is
used. Original ACT sources/configurations and the 458 source identities of the
accepted input98 optional smoke remain unchanged. Original CROWN is deliberately
the frozen C10 comparator, not a newly generalized CROWN backend.

`general_evidence_v1_controls.json`: **36 tests PASS**, no failures, errors or
skips, 9.40s observed test-suite time. Fourteen new tests cover E/C variations,
multi-pair/tied routes, partial reuse, arbitrary rational properties and
offsets, missing/duplicate obligations, invalid request/source/property/frame
bindings, modified construction after rehash, checked nonpositive versus absent
evidence, proposal reserve, checker deadlines and real owned-child timeout.
An analytic E3/C3 all-tie model runs ACT capture, fresh LP proposals, all-six
obligation checking, and the matched/evidence subprocess pipelines with terminal
audits and equal independently generated source facts. The portable checker
runs outside the checkout under `-I -S`, without model/solver imports.
Legacy budget, rational construction and scheduling tests supply the remaining
regressions. These analytic controls are not convolutional task outcomes.

An early development assertion incorrectly expected the existing terminal
helper to raise when independent checking was absent. The assertion was
corrected to its actual fail-closed UNKNOWN contract, without changing the
acceptance policy. This test-development observation is recorded in the report;
no failed real experiment or denominator was replaced.

## New selection

Protocol: `general_evidence_v1_protocol.json`.
Selection: `general_evidence_v1_selection.json`, SHA-256
`db3043fb124703e8123e5326eda853dc0d67d45e9104ca316487a631603daea7`.
Independent reconstruction: `general_evidence_v1_freeze_review.json`, PASS,
0 issues; **execution_started=false**.

Twenty clean-correct inputs:
114,117,123,124,135,144,146,152,154,157,159,168,179,180,181,187,194,199,203,205.
The union of **5,696 historical source records / 878 previously used indices**
is excluded, including the complete previous convolutional roster and failed
or incomplete endpoints. Predictions/labels alone choose ascending indices;
there is no route-complexity, bound, attack or proof-success predicate.
Prior full-test telemetry remains disclosed. Freeze/materialization took4.42s;
the separate-process clean reconstruction took1.30s. Neither is verification
runtime or a speed measurement.

Same validation-selected convolutional epoch89 checkpoint, same2/255 clipped
CPU/float64 tensors, same300s per complete request. Three arms are original
matched V2, generic conditional evidence, and original plain CROWN. Method order
rotates by input rank. No input98/input16/ACT-only re-query, training, new gate
range, support setting,25% fraction, production numerical policy or external
backend option was changed. Raw tensors/exclusion inventory stay local, hash-bound;
no checkpoint, data or external repository is committed.

## Guarantee and remaining execution gate

The checker verifies all required scoped facts, projections, HZ→LP exports,
checked ranges, rational McCormick construction, dual lower bounds and final
coverage. It still trusts network/input→HZ and ordered expert sources, guard
lowering and route infeasibility exclusions. CHECKED_CONDITIONAL is not native
floating-point SAFE, and CROWN numerical filters remain in a separate column.

The per-request supervisor and terminal audit have analytic controls. The
**cohort-level supervisor, lock/resource/roster integration and final three-arm
aggregation have not yet been executed or registered** for this new selection.
They require a separately frozen execution identity, control tests, publication
and launch authorization; preserve this selection and protocol rather than
reselecting. No full-cohort verdict, rate, speedup or cross-architecture benefit
can be reported from this preparation. Planned queries:60; executed new
verification queries:0.
