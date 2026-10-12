# Request-store device interface: CPU continuity and fault controls

The [frozen finite protocol](hz_store_device_design_20261013.md) completed once.
The immutable common/private support algorithm now has an explicit optional
`cuda:0` candidate path; CPU remains the default. **No physical CUDA ran.**
All four new-interface CPU runs match the archived CPU implementation exactly,
including all 41 endpoint candidate records and original-row rational residuals.
28 named controls and the offline stdlib recheck pass. All G1–G6 remain OPEN.

## What changed

[Runtime](../act/back_end/moe/store_device.py) binds a private captured execution
context to the request, store, property contract, deadline, invocation, isolated
UUID and 1 GiB allocator policy before initialization. The initializer and
synchronization/readback primitives come from the earlier device module; the
flat-LP optimizer and auto-device manager are not used. A read-only review found
a possible double-serialization alias race before execution. It was fixed by
hashing the exact captured bytes, and both alias cases are controlled.

[Block support](../act/back_end/moe/store_endpoints.py) keeps one shared factor
space and the common/guard/private-a/private-b operation order. Common and
expert sparse blocks persist per request; guard blocks transfer once per pair,
objectives per tile. All best duals return through blocking readback before CPU
exact original-row acceptance and zero-dual comparison. The 128 updates, finite
checks, endpoint roster and positive threshold are unchanged. Invalid or late
results cannot fall back to a stale best or a different dtype/device.

This is an optional device port of an existing MoE-specific support algorithm,
not evidence that a GPU itself is an algorithmic contribution. Its hard native
interruption and driver-release contract still needs separate integration.

## Frozen differential results

Each row used one 300-second owned-process deadline for import, source creation,
both arms, original-row rechecks, publication, parent receipt and cleanup.
The CPU reference SHA is fixed in the protocol. Each arm uses the same bound
store, complete tie-legal pair inventory, label 0, margin 0 and tile width 8.

| Source | Endpoints per arm | Positive / required duties, both arms | Complete request | Parent launch to receipt |
| --- | ---: | ---: | --- | ---: |
| half | 5 | 3 / 3 | checked declared-source positive | 2.389656 s |
| negative | 6 | 3 / 6 | UNKNOWN_NONPOSITIVE | 2.354481 s |
| mixed | 6 | 0 / 3 | UNKNOWN_NONPOSITIVE | 3.068629 s |
| guarded affine | 24 | 12 / 12 | checked declared-source positive | 2.311480 s |

The new guarded source actually requires guard information: its nonconstant
experts have negative full-box margins, but legal-pair margins are at least 1/4.
The singleton tie pair is retained. Finite generated bounds are positive, not
claimed optimal. Both old and new CPU prove the same duties: this is a stronger
device continuity control, **not a new coverage gain or a trained-model result**.
The half fixture's empty guarded branch can have a vacuous lower bound; that
value is not an attainable network margin. Negative checked bounds are not
UNSAFE witnesses.

## Controls, costs and independent recheck

The 28 archived tests are 13 new runtime/publication controls plus 15 existing
endpoint-contract tests. They cover source/property/device identity, context
alias capture, explicit CPU allocation, initialization/OOM/transfer failures,
post-optimization and post-readback synchronization, late/partial/nonfinite duals,
invalid signs/query/slot binding, missing duties, and failure of final reporting
after a partial tile. All four full new CPU producers also run under a global
meta default. A CUDA initializer trap covers each numerical child; simulated
device failures are never reported as hardware observations.

The owned-group engine records all five terminals (unit invocation plus four
differentials), on-time publication observations, sampled combined RSS and
cleanup. The unit invocation takes 1.958428 s including startup. This is not
a CPU/GPU speed benchmark. Device initialization, transfer, optimization and
readback costs are nested inside API totals. Initialization is outside the older
`native_preparation_seconds` field; unnamed setup/synchronization overhead is
still in inclusive API/request totals. Do not sum nested fields or call them an
exhaustive decomposition.

The [offline auditor](../scripts/audit_store_device_controls.py), executed with
`python -B -I -S`, checks archived implementation hashes, the complete roster,
on-time parent terminals, both source/contract/proof chains and shared exact
residuals. It returns PASS, 0 issues, in 0.481396 s. It uses the same independent
stdlib mathematical checker, not a second mathematical implementation or human
review. The declaration-to-checkpoint and native floating-point claims stay false.

A further 76 existing supervision, endpoint, source, store, capacity, layout and
proof-closure regressions pass in 1.253 s. This was tool-observed, not a separately
archived test-outcome ledger; 15 endpoint tests overlap the frozen 28, so these
counts must not be added as distinct tests.

## Archive and next action

Protocol commit `832e8adce`; implementation/execution commit `3628c6eaf`.
Raw archive: `/data1/Kane/MOE/baseline_runs/hz_store_device_20261013_r1`, 512 files,
1,477,519 bytes after audit, below the frozen 64 MiB cap. No evidence was removed.
The [compact receipt](hz_store_device_20261013_r1.json) binds the archived freeze,
calls, test outcomes and audit. No checkpoint, raw source package or external
repository is committed. `/data1` still has about 87 GiB available; no cache
deletion was needed for this stage.

Stop this finite CPU-equivalence study. Next, bind this **store/block** device
interface to the existing hard-budget lifecycle: actual UUID admission,
pre-initialization recheck, completed-candidate reception, partial evidence,
driver-resource release and full accounting. First test the new wiring with CPU
and simulated faults. Only then separately freeze a small physical CPU/GPU
comparison with fresh resource admission. Do not retry the old eight GPU slots,
run sealed real requests, claim a speedup, or combine this synthetic positive
with the separate large given-HZ negative control.
