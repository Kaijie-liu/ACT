# MetaMoE R3: opt-in CSR admission and sparse guarded intake

This is a new implementation/execution identity, not a repair of frozen R2
results. R2's MNIST0 error was the 64M logical affine-cell admission limit,
not an observed OOM or a negative property bound. Legacy defaults and that
limit remain unchanged.

## Implementation and resource contract

`HybridZConfig(sparse_resource_policy='csr_bytes_v1',
sparse_representation_bytes=2147483648)` opts into sparse-only sequential
affine/ReLU CNN propagation. Admission counts all SIX CSR arrays (including
both inequality matrices), centers/RHS, live cache entries and ReLU slots.
Integer combinatorial bounds estimate output nonzeros before construction;
workspace reserves include sparse products, COO/triplet arrays and padding.
Actual retained bytes are checked after construction. Unknown operators are
rejected BEFORE their registry handler executes. Failed ReLU construction or
post-admission cannot publish new factor slots. Guarded entry preserves its
original sparse relation/frame/constraints; it is never replaced by a dense box.

The reserve is deliberately conservative engineering accounting, NOT a proved
peak-process memory bound. Objects retained by the caller, models, Python,
native solvers and allocator overhead are additionally supervised by a sampled
8 GiB sum-of-own-process-group RSS gate (50 ms sampling; shared pages can be
double-counted). Sampling is NOT an instantaneous OS memory cap. Only groups
created by this supervisor are terminated. All real requests retain 300 s.

No ReLU encoding, support setting, numerical acceptance condition, tie policy,
model weight or scientific property is changed. Unsupported/missing sparse
relations fail closed. Capacity refusal is UNKNOWN, never SAFE or UNSAFE.
This does not repair the project's historical input-containment/source-proof
gaps; positive results remain conditional HZ-policy results.

## Ordered gates; no new cohort before success

1. CPU controls: CSR bytes, boundary capacity, old/new exact matrix differential,
   convolution/group/dilation/pool shapes, guards, factor-slot transaction,
   unsupported dispatch, no densification, full-entry resource refusal.
2. Outer controls: own child cleanup, timeout, RSS refusal, bad/partial/late
   candidate, exceptions, independent late-positive rejection, complete costs.
3. Commit implementation, then freeze/push R3 old-input configuration. Old
   MNIST0 diagnostic has 90 s INCLUDING imports/lowering/propagation/recording,
   no solver calls. Propagate the full router and BOTH guarded experts. Layer
   progress is durable before final output. Dense conversion and actual local
   solver entry points are intercepted. Failure does not open the formal gate.
4. If representation/entry diagnosis passes, execute the old CIFAR0/MNIST0
   four-call smoke, each whole request 300 s and 8 GiB sampled RSS. Preserve
   independent ACT and author execution; author native PyTorch backend and
   route-invariance sufficient adapter remain otherwise unchanged.
5. Independent roster/cost/identity audit plus original-full-model witness
   replay. A normal numerical UNKNOWN is allowed; resource/representation
   failure, timeout or execution error blocks the formal gate. Passing does
   not require a positive bound or select for easy certificates.
6. ONLY if all gates pass, freeze a NEW20-input/40-call directory: first10
   clean-correct per domain in raw test order, scan first1000, exclude index0.
   No route-count, margin or bound selection; no retry/expansion. Confirmatory
   implementation/environment/weight/resource identity must equal the passed
   smoke. Commit/push the manifest BEFORE launch. If a gate fails, archive it
   and stop; do not raise resources or consume the new cohort.

Input radius remains 2/255 IN NORMALIZED SPACE, clipped to [-10,10], not a new
pixel-space claim. Materialized CPU float64 tensors and all19 global class
margins are identical across arms. All tie-legal top1s are obligations; the
unselected expert blocks remain zero. Author positive numerical filters and
ACT policy positives remain separate. Only original-model replay is UNSAFE.

## Cost/terminal contract

Per-request charging begins before environment setup and includes spawn,
imports, input/model validation, own route analysis, translation, solves,
candidate write and process-group cleanup. Cleanup crossing deadline is TIMEOUT.
Malformed/partial candidate files retain hash/error and cannot crash terminal
accounting. Outer termination always dominates a late candidate. Batch ERROR
stops remaining calls with explicit NOT_STARTED rows; TIMEOUT is retained.
Logs/receipts/terminal audit and batch wall overhead are separately counted.
Batch clock includes top-level validation through summary write, excluding its
own final cost-file write and later independent audit. These are not free
per-request computations hidden inside the method.

## Validation history before execution

- New sparse controls10 PASS; new outer/auditor controls10 PASS.
- Existing class-separated semantics9 PASS on explicit CPU execution.
- Existing MoE/core convolution factory regressions21 PASS (CPU).
- Existing Meta tests24/25 PASS: the historical ONNX export BN-preservation
  control fails in ACT's current exporter (opset conversion/BN lost). It is not
  the native PyTorch path used here; no dependency change or hidden repair.
- Initial class-separated invocation without disabling auto-GPU selection
  failed its explicit CPU contract; CPU rerun passed. Two test-authoring errors
  (parenthesized context manager/import location) were corrected before freeze.
- Two read-only reviews identified/fixed dispatch-before-allocation, actual
  solver alias interception, partial JSON, late cleanup acceptance and smoke
  execution/replay cross-binding. Review is not independent SAFE reproof.

Actual diagnostic/smoke/formal outcomes must be appended separately. This
protocol document itself does not assert that any gate or cohort has completed.
