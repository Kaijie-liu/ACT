# Request-store device interface: finite CPU/reference controls

Frozen before implementation or numerical calls, 2026-10-13. Parent:
`857a96c6d2849c3a8b9ce75f4409621d2aca6662`. This is a device-interface factor,
not a new optimizer, timing study, real-model experiment or GPU admission.

## Algorithm and binding

Keep the immutable common/guard/expert-a/expert-b block order, shared residual
combination before absolute value, CPU float64 reference, one host thread, zero
initial multipliers, 128 projected subgradient updates (129 evaluations), rate
`0.125/sqrt(step+1)`, nonpositive inequality multipliers, and per-column best.
Keep every finite-iterate check. Each proposed dual is checked in CPU rational
arithmetic against the original stored rows and compared with the zero dual.
No altered threshold, factor identity, mathematical domain or dropped duty.

Add explicit `cpu` or isolated `cuda:0` only. A request-owned runtime initializes
once, transfers common/private blocks once and guards once per pair, transfers
objectives per tile, and synchronizes and reads back all best dual blocks before
exact selection. Its externally anchored context binds store, request, property
contract, absolute deadline, invocation, UUID and the existing 1 GiB allocator
policy. Context validation precedes initialization; no device/dtype fallback.
Initialization, transfer, optimization, readback, exact selection, serialization
and inclusive API cost are recorded; nested costs must not be added twice.
Cooperative API checks cannot interrupt a blocked native call. Full device
supervision/release controls and physical admission are separate next stages.

## Frozen CPU differential roster

Use the archived CPU implementation at
`baseline_runs/hz_store_supervision_20261013_r1/implementation/act/back_end/moe/store_endpoints.py`,
SHA-256 `1ad73af7dc2367efd036f1efe3e4afcdb38ff2aafe3be0d27b6e142d3ab6e57d`.
For each source, build one bound immutable store; independently run both old
CPU and new-interface CPU once on all endpoints, tile width 8. Compare exact
candidate records and independently recheck both complete inventories.

| Source | Recipe | Endpoints / output duties |
| --- | --- | ---: |
| half | Existing supervised half source | 5 / 3 |
| negative | Existing negative3 source | 6 / 6 |
| mixed | Existing 70-input mixed source, including private ReLU/equality/compensation | 6 / 3 |
| guarded | Below | 24 / 12 |

Guarded recipe: one input `[-1,1]`, router `(x,-x,0)`, three experts and five
classes, true label 0 and margin 0. Experts 0 and 1 use `z=x` and `z=-x`,
respectively, with output 0 `1/4+z` and output k `(k/8)z`, k=1..4. Expert 2
has output `(1/4,0,0,0,0)`. Retain all three tie-legal pairs, including the
singleton-domain pair `{0,1}`. All legal margins are at least 1/4; the full-box
nonconstant expert margins need not be positive. The finite optimizer is not
required to find a positive bound: retain any failure without tuning.

No CUDA initialization in this stage. Test an initializer guard and CPU explicit
allocation even under a global meta default. Simulated device fault controls:
missing/wrong context or hash; wrong store/request/property, deadline, UUID,
allocator and invocation; initialization/OOM/transfer/synchronization/readback
failure; late readback; nonfinite/partial/wrong-shaped duals; invalid multiplier
or query binding; caller context alias mutation; no fallback or partial positive.
Existing endpoint/partial/schema controls remain required. Simulations are not
claims that a real CUDA driver has exercised or released these failures.

## Execution and reporting

New directory `baseline_runs/hz_store_device_20261013_r1`; no old evidence edits.
Four once-only finite differential calls and one finite unit-control invocation,
each with one parent-owned 300-second deadline including import, source creation,
both proposals, checks, publication, observed receipt and owned-group cleanup.
Use the existing owned bounded engine (2 GiB sampled combined RSS), not a new
supervisor. Total archive cap 64 MiB. Keep failures, partial files and test names;
stop numerical calls after an unexpected failure. Offline stdlib audit may
recheck saved evidence, not generate new candidates. Reference and updated source
hashes and all terminal rows are mandatory. No speedup claim from these controls.

All G1–G6 stay OPEN. Success establishes only same-math CPU continuity and
fail-closed interface behavior. No real positive, full floating-point proof,
actual GPU speedup or external-baseline win is inferred.
