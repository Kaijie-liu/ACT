# MetaMoE R4: complete old-input representation control PASS

Implementation `bf8296804`; pre-run configuration commit `a05d61400`.
Config SHA256:
`ba6884d5644b6f77e46868963dd076347c1651102bca75e1ef113e33c8d73fd3`.
Read the [frozen protocol](metamoe_csr_protocol_20260922_r4.md) and
[saved-only independent review](metamoe_csr_diagnostic_20260922_r4.json).
Raw logs,70 layer records and terminal receipt remain under
`/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_diagnostic`.

## Outcome

| Item | Observed result |
|---|---|
| Representation phases | full router, guarded expert0, guarded expert1: all COMPLETE |
| Layer records |16 router +27 expert0 +27 expert1 =70|
| Outer terminal / independent intake review |COMPLETED / passed=true|
| Total charged execution |8.509516366s of90s|
| Sampled own-group peak RSS |1,812,234,240 bytes, approximately1.688GiB of8GiB|
| Maximum accounted admission |670,799,716 bytes, approximately0.625GiB of2GiB|
|12 convolution plans total time |0.227862855s, INCLUDED in the total budget|
|Largest planner scratch reserve |117,360 bytes|
|Solver calls / dense retained HZ |0 / none|
|Separate review cost |0.265664957s; no propagation/solve in the reviewer|

No memory cap or reserve multiplier was raised/lowered. No factor, guard,
output coordinate or mathematical obligation was removed. The actual numeric
convolution builder is unchanged; the new policy only computes a sharper
integer nonzero upper bound before allocation. Old v1 and legacy defaults
remain available and the original R3 refusal is preserved.

## What changed at the previously blocked layer?

For guarded expert0 layer7, the same coarse formula still gives the old refusal.
The new structural count uses the union of factor supports in each actual
convolution receptive field rather than summing repeated factor occurrences.

| Quantity | Coarse v1 estimate | Spatial-union estimate / observation |
|---|---:|---:|
|Total generator+constraint nonzero upper bound|24,002,307|1,561,063|
|Workspace reserve|3,280,982,736 bytes|408,503,504 bytes|
|Total accounted admission, including same cache|3,329,680,336 bytes|457,201,104 bytes|
|Observed output nonzeros|not constructed in R3|1,561,063|
|Observed retained output arrays|not constructed in R3|18,910,796 bytes|

All12 actual convolution output nonzero counts equal their structural upper
bounds in THIS run; all retained-byte checks pass. This is not a general claim
of equality: zero weights and coefficient cancellation can make actual storage
strictly smaller. The planner deliberately does not depend on their values.
The numerical operator still executes the original multiplication and guards.

This establishes that the previous refusal was avoidable with tighter
structural accounting under the SAME resource contract. It does not establish
a runtime speedup: R3 stopped before finishing expert0, while R4 completed both
experts, so their total times are not a matched-work timing comparison.

## Controls, identity and limits

68 targeted controls pass:11 planner,7 new protocol/auditor,10 v1 sparse,
10 supervisor,21 existing MoE/conv and9 class-separated semantics. Tests include
matrix/center/RHS/frame/exact equality of both single convolution and a complete
guarded CNN chain. Failures found during development (unvisited malformed CSR
index validation, instantaneous-process RSS control, weak admission-slice
auditing) were fixed BEFORE the execution freeze, not by rerunning this input.

Driver/tests use existing act-py312; the frozen author-model dependency
subprocess is unchanged from R3. Model/NPZ/environment/source identities are
validated inside the90s budget and again by the independent saved-only reader.
No package install, author source change, checkpoint change or new sample.

Independent read-only reconstruction matched the saved review apart from its
own elapsed time:478 frozen source/input hashes,72 raw JSON hashes, three
repository/environment identities,70 layer events and224 accepted admissions
checked. No dense HZ or drop was recorded. Formal/smoke selection and execution
paths remain absent. This structural reread is not a second network execution
or an independent mathematical proof of propagated bounds.

**This PASS closes the OLD-input sparse representation/guarded-entry control,
not the complete verification task.** No candidate-feasibility, selected-score
nonzero or output-property solves were performed. No new SAFE, strict
certificate, route-changing result, full paired comparison or new20-input
experiment is claimed. Existing source-to-HZ trust gaps remain unchanged.

## Next gate

Freeze a separate full old-input paired smoke using the admitted R4 policy,
with300s per complete request, same model/tensors/properties and unchanged
author numerical sufficient path. Include imports, own route/guard analysis,
all solver work and terminal accounting; require independent roster/cost
review and original full-model witness replay. Only after that gate may a new
formal cohort be separately frozen. This R4 script has no paired/cohort entry
point and its audit explicitly records `opens_formal_cohort=false`.
