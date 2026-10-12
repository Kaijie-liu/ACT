# Complete dense source capacity control for V3

Frozen on the clean parent `b7d35bad2b222828af96bc1284d52b28f47118fc`, after
[finite V3 controls](hz_scaled_batch_20261013.md). This admits exactly one
complete CPU query of the already captured synthetic dense source. It does
not admit a trained model, physical CUDA, an external comparison or another
source-only prefix. Capacity, not a positive safety margin, is the endpoint.

## Fixed source and algorithm

Read the existing source at
`/data1/Kane/MOE/runs/hz_tensor_rows_20261013_r1/sources/dense`, manifest SHA-256
`0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc`.
Copy its complete immutable bytes into the new request bundle inside the request
deadline; check them against this anchor. Do not regenerate weights or use old
HZ matrices or bounds. It has 3072 inputs, a 3072-to-2 router, expert 0 with
3072-to-256, ReLU, 256-to-2, and expert 1 with 3072-to-2. The captured domain
is center zero, radius 1/255, clipped to [-1,1], label 0, two classes and two
experts, selected-softmax top-2 with all tie-legal unordered pairs covered.

Use the committed V3 source producer, independent source checker and scaled
multi-target exact consumer without changing them: CPU float64, one thread,
tile width 8, 128 updates and 129 evaluations. Independently construct the
whole plan. Both raw proposed and explicit zero candidates for every endpoint
are mandatory. The intended inventory is one pair, one output property,
two gate endpoints and four candidates; any discrepancy refuses acceptance.
Do not expand the dense constraint matrix for an extra oracle. The finite
same-candidate differential controls already cover that implementation check.

## Limits and accounting

One parent hard deadline of 300 seconds covers process startup, imports,
source copying, source construction and checking, native preparation,
objective/dual generation, exact acceptance, serialization and result reception.
The original cumulative Work cap is 8,000,000. The reviewed complete static
upper bound is 7,963,310 (implemented allocation 7,963,292), not a runtime
guarantee. No count redefinition, raising caps, omitted checks or retry.

Use a separate child process group with 16 GiB address-space limit and sampled
8 GiB resident-set limit; record the worker's OS peak RSS and the parent's
sampled peak separately. Parent polling does not imply an exact instantaneous
RSS cap. Kill only this owned process group on deadline or memory refusal.
Inspect CPU jobs, available memory and disk before admission; require at least
16 GiB available RAM and 1 GiB free disk. Do not poll or retry CUDA.

Record all phase times, cumulative source/production Work, native nested times,
parent startup/wait/reception time, peak memory, bytes and terminal publication.
Save stage checkpoints after source publication/checking, and publish raw
candidate bytes before exact acceptance. A hard kill may leave only a prefix;
missing or late receipts never count as a completed query or positive result.
Errors, nonpositive complete results and resource failures remain archived.

The new archive is `/data1/Kane/MOE/runs/hz_scaled_dense_20261013_r1` with a
128 MiB total logical-byte cap, including code, source, proof, logs, relocated
copy and terminal. No existing directory may be reused. Keep failures; no
automatic restart. Tests and offline audit are separate recorded costs, not
free work folded into a production runtime claim.

## Execution and independent review

First test only the new supervisor's deadline, exception, partial-evidence,
memory and acceptance/cost checks; do not rerun or enlarge the six source
controls. Commit and push implementation before the single dense call.

If a complete query is received before its deadline, relocate its complete
bundle and the archived checker code. Run `python -B -I -S` under a separate
300-second bound; reconstruct source and all candidates without loading torch,
NumPy, SciPy, highspy, a checkpoint, solver or historical run. Check the exact
result agrees, including UNKNOWN. This is independent offline evidence checking,
not another optimization attempt; its cost is reported separately.

An independent stdlib audit binds protocol and complete execution-code inventory,
all terminal and phase costs, every source/pair/property/endpoint, all raw
candidates and the unchanged relocated bytes. For incomplete execution, audit
the failure/prefix and costs without inferring positivity. A complete result
requires all the obligations; a positive result additionally requires every
bound strictly greater than 1/10000000 under the unchanged policy.

## Continue or stop

Report whether the full source and full query fit, actual Work and phase/memory
costs, plus the sign of the checked conclusion. A negative or nonpositive lower
bound is not a counterexample. A synthetic positive is not a trained-model
certificate or evidence of GPU/external competitiveness. If capacity fails,
analyze only the saved prefix before any new mechanism decision; do not rerun,
increase the budget or reduce the fixed obligations. If it fits, inspect actual
trained-source compatibility read-only before a separately frozen next stage.
G1–G6 remain open; historical 23-gain source limitations are unchanged.
