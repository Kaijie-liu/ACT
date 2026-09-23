# Current-request feasibility and saved-query diagnosis — 2026-09-23

## Decision

One deterministic, solver-free assignment is feasible for the **stored HZ
model under its existing float policy**. This is a promising replacement for
redundant base existence search, not yet a production speedup. Independent
saved-row checking also places that assignment inside all 16 previously
unresolved expanded violation queries. However its recovered input is exactly
the original center, where the original model has no violated property. The
represented and original outputs disagree. Do not promote it to UNSAFE or
claim that longer solver time will close the output certificate.

The next source diagnostic must locate the first point-correspondence
divergence (conversion / propagation / factor map). No particular operator,
unsoundness, or unique relaxation cause has been established. A spurious
represented point by itself does not prove that a sound outer approximation
excludes real executions. The old request and all its 19 property results
remain unchanged; no additional certification queries were made.

## Fixed objects, controls and failed attempt

Parent: `configs/recent_moe/metamoe_protected_smoke_r1.json`, SHA256
`7a8e31de3412657a8d2d2fc9b5d19e5f7847c7aa3655e80f223da64710fe23e0`.
Exactly the old MNIST0 / selected expert1 / materialized float64 box, same
matrix, properties and 1e-7 feasibility policy. No old result was rewritten.

The proposal recognizes appended compressed ReLU equalities, uses exactly one
all-zero free-factor seed, and has no optimizer, cache, historical incumbent,
clipping or retry. Acceptance separately checks **all** constraints, domains,
integrality, model hash, request, input and evaluation nonce. Failure never
proves infeasibility. The production solver and scheduler were not changed.

Saved-only R1 rejected unsorted base CSR, then grouping raised an exception;
its partial `launch.json` and `assignment.json` remain. R2 changes storage
recognition only: locate new columns without assuming CSR index sorting, never
sort/coalesce/modify the original coefficients. R2 has its own protocol and
directory. The initial two unit-test fixture failures were fixed before the
R1 freeze; no scientific acceptance gate changed.

Controls: 22 assignment + 7 grouping + 6 replay + 10 independent-audit tests
pass in both ACT and pinned intake environments; another 10 existing outer
supervision tests pass in ACT (55 total focused tests). Coverage includes
ties, multiple ReLU layers, guard rejection, unsorted CSR, duplicate new
columns, request/nonce/model contamination, partial/nonfinite points, deadline,
exact query grouping, input/output map tampering, and abstract versus original
violations. Small controls include native feasibility differential; real saved
diagnostics made zero native queries.

## Assignment and query evidence

| Quantity | Observed value |
| --- | ---: |
| Variables (continuous / binary) | 6,918 (5,636 / 1,282) |
| Original constraints | 3,847 |
| Free-factor seed width | 3,072 |
| Construction | 0.016676 s |
| Full policy check | 0.001776 s |
| Proposal + check | 0.018591 s (3 s cap) |
| Largest recorded equality residual | 2.776e-17 |
| Variable-bound / integrality residual | 0 / 0 |
| Expanded properties / distinct stored queries | 19 / 10 |
| Historical UNKNOWN properties / distinct queries | 16 / 7 |
| UNKNOWN rows feasible at this HZ point | 16 / 16 |

The independent auditor evaluates each CSR row with scalar `math.fsum`, checks
the box and binary domains, and tests every saved expanded query. This does
not use the proposal recurrence. Its arithmetic is still floating point, not
an exact rational feasibility certificate. The three other expanded queries
(rows 11, 15, 18) reject this point and were infeasible in the historical run;
we do not independently re-prove their infeasibility here.

Rows 0–9 have byte-identical expanded query matrices and thresholds against
the same base model. They took **14.271729 s**, of which **12.839022 s** was
after the first call. This corresponds to the repeated zero-block obligations
in the class-separated output, but equality was checked on the actual saved
arrays rather than inferred from class semantics. This is an opportunity for
identity-bound deduplication, not a measured speedup of an implemented
scheduler. Do not merge near-equal rows or propagate UNKNOWN as a proof.

The 16 historical local timeouts have no completed native bound/primal record.
Native entry windows were approximately 1.245–2.021 s. Nothing here identifies
which internal HiGHS phase consumed that time. The new point observation is
additional evidence about the represented queries, not a retrospective change
to their frozen UNKNOWN status.

## Frozen one-point provenance replay

Implementation `0a1f871af`; freeze `751f58b76`; config
`metamoe_assignment_replay_r1.json`, SHA256
`192f773968577dd21c990ff3faae8e04587566c235f43bdd8e6bcf5e56aed5bc`.
One saved point, one fresh router + guarded expert propagation, 30 s outer cap,
8 GiB RSS policy, no native query and no new proposal. The output matrix must
hash-match before the saved point is accepted. It does; the complete binding is
`38538a4242c941a1221d7372956e9fcfee8bbf5c117f23c37fb9efa27eee51d3`.

| Observation | Value |
| --- | ---: |
| Fresh full matrix equals saved matrix | Yes |
| Recovered point equals original center, lies in box | Yes, exact array equality |
| Original full-model prediction / route | 17 / 1 |
| Original versus padded expert output difference | 0 |
| Minimum original point margin | 1.600795345925858 |
| Original violated properties | 0 / 19 |
| HZ represented violated properties | 16 / 19 |
| Largest HZ versus source output difference | 3.203507818200434 |
| Worker / outer charged / through terminal | 9.089 / 9.728 / 10.260 s |
| Peak sampled group RSS | 1,798,721,536 bytes |

Status: **HZ_SOURCE_POINT_MISMATCH**, not SAFE or source UNSAFE. Source versus
padded expert agreement rules out the final zero-padding wrapper at this point;
it does not validate the entire conversion. Independent saved-only audit
reconstructs the input and represented output from matrices, checks the physical
request and all property rows, and recomputes the discrepancy using hash-bound
forward arrays. It does **not** run another original forward or prove source
equivalence across the box.

## Evidence and cost boundaries

- Compact independently rebuilt audit: `metamoe_current_assignment_archive_20260923_r1.json`.
- Checker: `scripts/audit_metamoe_current_assignment.py` (no solving / model load).
- Raw failed/successful proposals: `/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_r{1,2}`.
- Raw bounded replay: `/data1/Kane/MOE/baseline_runs/metamoe_assignment_replay_20260923_r1`.
- Separate audit: `/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_review_r1.json`.

Raw matrices/arrays/checkpoints are not committed. The archive binds all raw
diagnostic artifacts and the unchanged parent archive. Proposal timing excludes
model loading, source validation and publication; outer replay and audit costs
are separately recorded. Comparing 0.019 s to the historical 3 s base timeout
does **not** establish an end-to-end speed ratio or extra SAFE.

No new cohort, extra solver time, support tightening, numerical threshold,
candidate exclusion or default backend change. A source-correspondence control
comes before production integration; exact-query deduplication remains a
separate single-factor optimization.
