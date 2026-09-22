# R4: convolution support-union admission; one old-input control

User-authorized follow-up to the preserved R3 resource refusal. No new input,
property solve, smoke comparison or formal cohort is authorized by this file.
Old R3 configs/results and `csr_bytes_v1` behavior remain historical identities.

## Single engineering change

Opt-in `csr_spatial_v2` changes ONLY the convolution retained-nnz estimate.
For each batch, convolution group and output spatial site, collect the union
of continuous-factor indices of all input rows in the receptive field; repeat
for binary factors. Multiply each union size by output channels per group.
Every output row support is contained in this union, even with zero weights,
stored zeros, duplicate CSR indices or numerical cancellation. Batches are
counted separately; one batch's pattern is not multiplied by batch size.
Add all original constraint nonzeros unchanged.

This bound is checked against the old independent fanout bound. We do NOT
change the numerical convolution builder, multiply order, ReLU encoding,
shared factor frame, guard constraints, support policy or acceptance threshold.
The planner reads CSR index structure, not weighted floating-point values.
It makes no cache keyed merely by object/frame identity.

One reused bool marker of max(n_cont,n_bin) and4096-index chunks bound its
working storage. BEFORE any marker allocation, charge
`8*max(n_cont,n_bin) + 16*4096 + 4096` bytes on top of current cache/slot usage.
On completion discard the marker. No dense matrix, input-index concatenation,
unbounded set, sparse numerical product or model inference is used to count.
Incomplete planning never produces a usable partial bound.

The ORIGINAL operator reserve formula remains:

```
8 * retained_bytes_upper
+ 160 * (output_rows * convolution_fanin)
+ 16 * (n_cont + n_bin)
```

Only `retained_bytes_upper` uses the new structural nonzero bound. The2GiB
representation budget,8GiB sampled own-process-group RSS limit and90s TOTAL
old-input deadline stay fixed. These remain engineering resource policies,
not a proved process-memory bound or a guarantee against between-sample peaks.

## Controls before freeze

11 new planner controls cover independent Python-set reference counts,
actual nnz≤union≤coarse, B>1 with different patterns, groups/depthwise,
non-square kernels/padding/stride/dilation, zero weights, empty matrices,
duplicate/unsorted/stored-zero indices (including duplicates across4096-index
chunks), invalid indices/geometry, no partial
plan after exception, pre-marker and pre-builder refusal, no densification,
six-matrix/c/b/ub/frame/exact equality, and complete guarded CNN/ReLU-slot
differential.7 new protocol controls cover changed requests/resources/source,
missing phases, empty events, nonzero solver counts, dense/late/RSS failures,
admission slice/layer binding and exact reserve/total/limit arithmetic.
Existing10 sparse,10 supervisor,21 MoE/conv and9 class-separated controls pass.

Two initial failures were fixed before freeze: validation previously inspected
only visited indices (now validates all CSR indices in bounded chunks); an old
RSS test required a nonzero sample from an instantly exiting process (test
now sleeps0.2s to span polling windows). The latter changes TEST timing only;
supervisor and mathematical implementation are unchanged and explicitly hashed.
Read-only audit review also found that admission slices could point at another
layer; strict nonnegative ordered ranges, layer identity and resource arithmetic
checks plus mutation controls were added BEFORE freezing any execution.
Historical ONNX BN-export failure remains sealed; this is native PyTorch.

## Execution identity and cost

The R4 freezer inherits the exact old MNIST0 NPZ, checkpoint, author source and
already-approved author-intake dependency subprocess from R3. The driver and
all ACT development/tests use existing act-py312; no dependencies installed or
upgraded. The author-model subprocess environment is unchanged to avoid a
version/kernel confound. Only two core integration files plus the sampling-control test
are permitted old-source rebindings; new planner/driver/freezer/audit/tests and
this protocol are separately bound. Parent R3 and failure archive hashes remain.

Commit/push implementation, then freeze/push config BEFORE launch. Fixed run:
old MNIST0 → full router → guarded expert0 → guarded expert1;90s includes
imports/identity checks, lowering, ALL planning/propagation, partial progress
and final serialization, child cleanup. Parent log hashing/receipt write and
independent review are separate disclosed costs. The actual solver_hz entry
aliases and CSR `.toarray()` are intercepted. No implicit retry or second input.
No modification of bound files while active; preserve partial files on failure.

PASS requires all three phases complete, zero solver calls, no dense retained
HZ, valid retained estimates, within charged deadline and sampled RSS gate.
Review additionally checks per-layer planner admission and all frozen hashes.
Even PASS means representation/control completed, NOT router feasibility,
output safety, a source-complete certificate, or permission to launch a new
formal comparison. If refused or timed out, preserve that result without
raising budgets, decreasing reserve constants or using incomplete counts.

The next decision follows this ONE result. The source-containment proof gaps,
author numerical-filter distinction and human-review requirements do not change.
