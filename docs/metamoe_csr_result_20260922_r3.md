# MetaMoE CSR R3: router passes; guarded-expert admission still blocks comparison

Execution code `25aa7c37e`, pre-run config commit `9b79f9ddb`.
Frozen config SHA256
`a7303a0535becd2915a575fffc775a9922227e758c2ff66779cec206d1d5bddf`.
The committed [protocol](metamoe_csr_protocol_20260922_r3.md) and
[independent structural review](metamoe_csr_diagnostic_20260922_r3.json)
bind the old MNIST0 run; raw progress/logs remain in
`/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r3_diagnostic`.
The driver uses act-py312; the child retains the previously pinned paired
execution's `moe-author-cpu-py312-20260921` interpreter/environment identity.
No dependencies, source author repositories, weights, inputs or acceptance
thresholds were changed. This is representation diagnosis, not a new bound run.

## Actual result

| Item | Observed result |
|---|---|
| Whole-process terminal | COMPLETED, exit0 (the diagnostic itself wrote its result) |
| Representation result | RESOURCE_REFUSED; only `router` completed |
| Charged time | 7.425767869s of90s, including setup/imports/propagation/cleanup |
| Sampled own-group RSS peak | 1,605,013,504 bytes, approximately1.495GiB of8GiB |
| Solver entry calls | 0, actual solver_hz aliases intercepted |
| CSR densification | forbidden; no retained dense HZ |
| Independent intake gate | **false** |
| Full smoke / new20inputs | **not launched / not selected** |

The old first router ReLU now completes with836 unstable binary factors,
4,744 continuous factors,147,271 total nonzeros across six matrices and
2,135,084 retained bytes. Its logical dense shape is irrelevant to actual CSR
storage. All subsequent router layers complete: final router HZ has5,592
continuous/1,260 binary factors,339,076 nonzeros and4,129,448 retained bytes.
These counts measure representation, not route feasibility or output safety.

The sparse guarded expert0 entry preserves the router factor space and
constraints and propagates Conv→Scale→Bias→ReLU→AvgPool. Its NEXT convolution
(layer7) is refused BEFORE native matrix construction:

| Admission component | Bytes/count |
|---|---:|
| Already accounted cache/slots |48,697,600 bytes|
| Predicted retained CSR upper bound |384,303,984 bytes|
| Predicted output nonzeros upper bound |24,002,307|
| Convolution operator nonzeros upper bound |1,290,240|
| Requested workspace reserve |3,280,982,736 bytes|
| Total accounted admission |3,329,680,336 bytes (about3.101GiB)|
| Frozen representation budget |2,147,483,648 bytes (2GiB)|

The refusal is primarily the deliberately conservative workspace/nonzero
estimate, NOT an observed process OOM. Neither the low preceding RSS nor the
predicted retained CSR size proves the refused operation would fit: it was
not executed. Expert0 did not complete; expert1 was not attempted after the
refusal. No negative property bound or model unsafety follows.

## Decision and preserved boundaries

This implementation fixes the demonstrated dense-shape gate and dense guarded
entry problems, but does NOT yet admit the full required expert chain under
the new frozen resource contract. Therefore the execution gate remains closed.
No300s smoke, candidate/feasibility query, new20-input selection, formal40-call
comparison, retry, cap increase, or numerical-policy change follows this run.
Old R2 results remain untouched. The author native positive is still a
numerical sufficient filter, not an ACT win or an independently checked proof.

The next limited engineering question is whether convolution structural
nonzero/workspace estimation can be made substantially sharper (or construction
made bounded-workspace) without deleting admission or touching mathematical
encoding. Controls must establish conservative estimates, sparse-product
equivalence, failure cleanup and no dense fallback before another OLD-input
version is frozen. It is not justified to raise2GiB from this record, call
3.10GiB an actual allocation, or consume the formal cohort to diagnose it.

## Controls and audit boundary

50 targeted controls passed:10 resource/guarded controls,10 supervisor/auditor
controls,9 class-separated semantics and21 existing MoE/conv regressions.
The separate historical Meta suite passed24/25; the old ONNX BN-preservation
control remains failed in the current ACT exporter. Native PyTorch intake is
the frozen path here. All initial control failures/corrections are disclosed
in the protocol; no dependency repair or relabeling of the ONNX failure.

The independent reader verifies layer progress, resource/terminal identity and
complete diagnostic failure record. It neither re-runs propagation nor proves
source→HZ inclusion or output bounds. Historical source-containment gaps and
submission-readiness limitations are unchanged.

A second read-only review checked all469 frozen file hashes,26 saved diagnostic
JSON hashes and three clean author/backend repository HEADs. Rebuilding the
diagnostic review from raw records produced identical content. A no-write
formal-freeze control rejected the false diagnostic gate before any cohort
directory creation; R3 full-smoke/formal output directories and formal config
are absent. This is an independent structural reread, not human review or a
second network/solver execution.
