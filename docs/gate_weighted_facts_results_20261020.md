# Gate weighted facts finite control results

The approved M1 implementation and one frozen synthetic control batch completed:
**19/19 control groups passed**, followed by saved-record reconstruction and a
separate read-only arithmetic review. No real model, native optimizer or GPU ran.
This closes the finite implementation/control stage, not G1-G6 or the real-request
efficiency hypothesis.

Execution commit: `a7eaf9c438a52eb06f45ab60c6033c3dd7707852`.
The [protocol](gate_weighted_facts_control_20261020.md) and
[freeze](gate_weighted_facts_freeze_20261020.json) were committed and pushed before
the single execution. Freeze SHA-256 is
`6a970c6e22f353105a42bdcbd3f9c60ea829be0d0a8c8c4c276b3beaf2437554`.
The result directory is `/data1/Kane/MOE/results/gate_weighted_facts_20261020_r1`.
There were no retries. This one execution authorization is consumed.

## Implemented mechanism and observed controls

`f0.gate_weighted_facts` is default off. After the existing gate computation, it
combines signed membership-domain expert facts with the canonical first-expert
weight interval. Both staged and matched monolithic can omit only those obligations
whose checked combination passes the unchanged positive-margin policy. The original
positive-fact fast path and original F0 fallback remain available. No gate/support
query or propagation was added or moved.

| Fixed function-level fixture | Off F0 stub calls | On F0 stub calls | Off and on gate calls | Off and on propagations | Off and on complete verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Staged, two pairs and two properties | 4 | 2 | 2 | 2 | UNKNOWN |
| Monolithic, same obligations | 2 | 1 | 2 | 2 | UNKNOWN |

These counts come from unchanged execution-function ASTs with local backend doubles.
They establish that the new rule skips the expected calls while leaving unproved
properties in the original path. They are not native MILP savings, model speedups
or newly certified requests. Both fixtures still have an unresolved output property.
The old all-positive fixture returns SAFE from given facts with zero propagation,
gate, F0 construction or solve calls in either function path.

The controls reject negative facts used alone, incorrect gate direction, changed
properties/constants/source scopes, nonfinite values, foreign conditioned-router
objects, missing obligations and late acceptance. They preserve fallback for missing
facts, nonpositive combinations and support metadata that M1 does not accept.
Injected native failures remain UNKNOWN; injected deadlines remain TIMEOUT with
partial rows censored. The deadline cases are controlled clock/exception tests,
not a 300-second soak of the unchanged watchdog.

## Source and mathematical checks

The [compact archive](gate_weighted_facts_records_20261020.json) losslessly reconstructs
both original runtime JSON files, including all fixed-case details. The
[audit summary](gate_weighted_facts_audit_20261020.json) records the source hashes,
terminal state, original controller stdout receipt and exact cost boundaries.

Within the batch, reception independently reconstructed four saved pipeline facts.
A separate read-only reviewer also used `Fraction` to recompute those four facts,
eight expert signed bounds and eight endpoint expressions, checking their saved
source/scope/frame/property bindings and reused/residual partitions. Each resulting
exact bound was

```
2251799797922647 / 4503599627370496
```

and the recorded accepted value `0.49999999649999943` did not exceed it.
Twenty frozen source hashes and the complete 19-group order matched.
The separate review was AI-assisted code/data review, not an independent human review.
It read the saved controls; it did not re-execute them.

The guarantee remains **numerical-policy acceptance conditional on the given
expert intervals and gate sources**. Downstream rational arithmetic does not prove
network lowering, input inclusion, guard generation or margin-to-sigmoid enclosure.
The request's numerical gate records remain trusted inputs to this check.
Consistent replacement of all upstream gate records is not detected as an invalid
network derivation by this arithmetic checker. Endpoint weights zero/one in the
controls are algebraic degeneracies, not finite-softmax source-validity claims.

## Complete control accounting

| Recorded quantity | Value |
| --- | ---: |
| Worker M1 Work | 71,416 |
| Reception check Work | 800 |
| Combined Work within the unchanged 8M cap | 72,216 |
| Worker JSON | 54,406 bytes |
| Terminal JSON | 2,770 bytes |
| Retained runtime artifacts | 57,176 bytes |
| Additional external-output reservation | 32,768 bytes |
| Observed session plus controller RSS peak | 45,473,792 bytes |
| File descriptors before and after | 4 → 4 |
| Before terminal-file export | 0.528670 seconds |
| Through terminal-file export | 0.530397 seconds |

The 8 GiB per-process AS cap, 2 GiB sampled RSS stop, 128 MiB artifact cap and
300-second batch deadline were unchanged. Reception continues the worker's Work
ledger rather than granting a second allowance. The Work metric covers M1
capture/arithmetic/check visits, not every interpreter or legacy checker operation;
all operations remain included in observed wall time.

The timing starts at the controller's first timestamp. It includes imports,
binding/admission, worker launch, controls, worker export, reception/checking,
owned-session cleanup and terminal-file publication. Full interpreter cold start,
the final stdout flush duration and final process exit are not in the quoted duration.
`full_cold_start_seconds` remains null; the controller returned exit code zero with
its hard deadline still armed. This is not a performance result.

The owned worker completed with no execution/cleanup error; its leader was reaped,
and no live member of its reserved SID remained. The cleanup snapshot contains
the exited leader before reaping. The record explicitly keeps
`zombie_reaping_complete=false` and `escaped_sessions_checked=false`; it does not
prove global zombie or escaped-process cleanup. Parent RSS was additionally sampled
around reception and audit. No other user's jobs were changed.

## Decision and next boundary

Continue this mechanism only to a **separately frozen and authorized** native
integration/development stage. The controls establish implementation readiness to
investigate, not that real requests improve. The next design must fix the input
selection, two opt-in/off configurations, actual request-level F0-call and total-time
criteria, SAFE/UNSAFE/solved gain-loss and the intermediate continue/stop band before
observing results. All creation, propagation, gate, fallback and checking costs count.

Do not repeat these small controls to accumulate progress, change the .25 schedule,
add support queries or infer that the historical 402/752 estimate is attained.
The missing-gap fallback is deliberately more conservative. Native package loading,
real HZ propagation and actual optimizer integration have not been exercised here.

This rule still uses an input-independent gate interval. It is a computation
optimization candidate, not a new HybridZ abstract domain, restored gate-input
dependence, an external-tool victory or a complete same-source proof. G1-G6 remain
OPEN, and the historical source gaps and sealed experiments remain unchanged.
