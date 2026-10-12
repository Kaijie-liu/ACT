# Request-store device lifecycle controls

Frozen before implementation/execution at parent `27be3c644`, 2026-10-13.
Reuse the five-phase owned-process engine; do not create another supervisor or
rerun the sealed flat-device experiment. This stage admits only CPU/simulated
controls. Physical execution requires a separate committed freeze, passing
control receipt and fresh resource observations; no such freeze is created here.

## Unchanged mathematics and new execution binding

Keep the half, negative and explicit partial declarations, all tie-legal pairs,
properties, gate endpoints, 128 updates, tile width 8, float64 and exact CPU
original-row acceptance from the checked store component. One cumulative 96M
work limit crosses producer/checker boundaries. The current device port, not
the older flat-LP optimizer, supplies candidates.

One 300-second request clock covers validation/imports, source creation,
planning, admission, device initialization/transfer/optimization/readback,
serialization, independent checking, reception and cleanup. Reserve 30 seconds
after production, 3 seconds after checking and 1 second for final reception;
admission/release have bounded 5-second phase caps. Fault delay targets get a
15-second cap within that same clock, not a new budget. Sampled combined RSS
remains 2 GiB; source/evidence/stage capacities remain unchanged (64 MiB archive).
Parent inventory/hashing is capacity-bounded, not independently preemptible;
late acceptance is denied and actual observed API return is recorded.

GPU intent is impossible during imports/source/planning. After actual store and
property hashes exist, derive `STORE_DEVICE_CONTEXT_V1` from those identities,
the source, producer deadline, invocation, UUID and fixed 1 GiB allocator policy.
Require two fresh initial read-only observations and one fresh pre-initialization
recheck. Before enabling CUDA, publish intent bound to producer PID and recheck.
Do not fall back to CPU or change dtype. Check every result against the launch
context, actual hardware UUID, environment, candidate and complete obligation
identities. The exact checker runs `-I -S`, without numerical dependencies.

Production failure still requires owned-group cleanup then release observation.
Two fresh inventories must show the owned PID absent. This is not proof of
driver correctness or all memory reclamation, and never substitutes for next
admission. A foreign tenant may remain. Partial tiles are evidence prefixes, not
a completed request. Missing/late/misbound checker or release cannot accept.

## Finite ten-call roster

| Call | Device requested | Expected execution |
| --- | --- | --- |
| half | CPU | complete, 3/3 declared-source positive |
| negative | CPU | complete execution, UNKNOWN 3/6 positive |
| partial | CPU | complete execution, UNKNOWN missing endpoint |
| admission_busy_stub | CUDA requested, initialization forbidden | RESOURCE_UNAVAILABLE, admit only |
| pre_cuda_busy_stub | CUDA requested, initialization forbidden | RESOURCE_UNAVAILABLE after source preparation and release |
| candidate_exception | CPU | ERROR at actual new candidate runtime seam, then release |
| readback_delay | CPU | TIMEOUT in second tile readback; retain first tile |
| release_owned_stays | CUDA requested, initialization forbidden | DEVICE_RELEASE_UNCONFIRMED after pre-CUDA refusal |
| release_delay | CUDA requested, initialization forbidden | DEVICE_RELEASE_UNCONFIRMED; release phase times out |
| wrong_candidate_metadata | CPU | ERROR after candidate generation; no checked acceptance |

Every fault must have a reached witness. Numerical fixtures/targets do not change.
Pure controls cover context/source/property/deadline/UUID and allocator mutation,
cost bounds/nesting, missing/foreign receipts, partial aggregation, late API return,
stale/malformed release and foreign-tenant observation. Reuse existing exact
endpoint/source tests, not a fresh optimizer grid. No simulated hardware result
may be counted as physical execution or speedup.

New archive `baseline_runs/hz_store_device_supervision_20261013_r1`. Save execution
identity, complete named tests, start/pending roster, all terminals, prefixes,
cumulative counters, costs and cleanup. Stop on unexpected status, API exception
or unknown real cleanup/release; continuing after the two expected simulated
release failures requires verified no-CUDA intent and confirmed actual group
cleanup. No hidden retry. Independent stdlib audit rechecks normal full evidence
and all fault/terminal/cost bindings without generating candidates.

After these controls, stop this lifecycle study and consider a separately frozen
small physical same-algorithm comparison. All G1–G6 remain OPEN. No trained-model
proof, external superiority, practical speedup or native floating-point proof
is established by this stage.
