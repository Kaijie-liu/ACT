# Frozen F0 timing diagnostic R1

## Completed diagnostic and next decision

Execution `fd69daa1a4c80158d8c43d9142b304193a5ca160`, single request terminal
TIMEOUT at300.069s. Automatic, separate-process and fresh archival audits match
exactly: PASS,0 issues;2055 durable events, no partial JSONL tail. All40 old
smoke artifacts remain byte-identical. No worker/full90 remains running or queued.
Compact evidence:
`act/pipeline/moe/results/conv_f0_timing_review_20260915_r1.json`.

| Segment | Observed seconds | Scope |
| --- | ---: | --- |
| Outer startup and common prelude before F0 | 4.849 | Complete |
| Two guarded pair propagations | 45.017 | Complete; includes support |
| Gate ranges | 0.032 | Complete |
| Eighteen pair/property encodings | 8.341 | Complete;8.255 is difference support |
| Nine union/disjunction constructions | 14.014 | Complete |
| Eight returned property-native calls | 208.612 | All status1, solver-limit UNKNOWN |
| Last property-native call | unavailable | No return;19.138s exposure until outer terminal |

Support inside propagation is12.269s LP and29.889s MILP (12 batched sweeps each).
The12 LP sweeps are not wholly exact;8/12 MILP sweeps report exact. A sweep not
wholly exact can still improve some bounds; these counts are not binary
elimination counts or an argument for disabling support. Nine property problems
each have15207 variables,3021 integral entries,39484 rows and1101862 nonzeros.
Native time includes SciPy setup and HiGHS presolve/search; these are not
separately measured. Serialization/write/fsync observed cost1.101s is already
charged; metadata/wrapper overhead is additional, not independently isolated.

The last property obtains20.628568s before construction. Its native call starts
at280.931445s, when only19.068555s of the300s request remains, but still receives
20.628568s. The1.560013s excess is a **measured stale-budget contract mismatch**.
Its requested native deadline is after the external deadline. This explains why
the final allocated call cannot count on its entire allocation; it does not
prove that a smaller allocation would produce SAFE or even an earlier native
return. Eight earlier calls also exhausted their limits without reporting a
primal/dual/gap/node value. Missing fields are unavailable, not zero.

Next useful work is a **separately versioned budget/terminal repair**: charge
construction to the granted property slice, recheck the true request remainder
at native entry, reserve a declared terminal-publication margin, persist each
completed property before the next call, and retain the external watchdog.
Apply the same accounting principles to both comparison arms; do not tune
support,25%, numerical acceptance or the cohort at the same time. Such repair
targets complete UNKNOWN/TIMEOUT evidence first, not a promised certificate or
smoke pass. No repair or rerun is included in this diagnostic stage.

Two archival caveats are explicit. Generic caller locals can survive loops:
pair propagation's recorded `property_index=8` is not an active property, and a
union solve's caller `pair=[1,3]` is not its sole domain. The derived table uses
the validated two-pair inventory and actual property-loop index, not these
incidental locals. The first unpublished summary also read exactness from the
tuple-valued dispatch wrapper; that draft is retained locally and the final
summary uses nested `hz_support_bounds` returns. Raw execution/audit records
were not changed. Five read-only review controls check these distinctions.
Final combined observer/review/lifecycle/F0 suite:43 tests PASS.

Scope authorized after read-only review of the failed convolutional smoke:
one old request, **rank0 / dataset index0 / matched monolithic**, same frozen
epoch89 checkpoint, materialized CPU/float64 input, epsilon2/255, property,
support policy and total300 seconds. Protocol:
`scripts/conv_f0_timing_protocol_r1.json`. No full90 launch, adaptive rerun,
new input, retry, extension, support ablation or numerical-policy change.

The old smoke remains FAIL; its timeouts cannot be recategorized from a new
profiled run. The diagnostic is not a speed comparison or a positive-certificate
gate. Resource checks, shared lock and owned-process-group watchdog are reused
from the frozen supervisor. All427 frozen ACT Python files and the old four
wrapper files remain unchanged. New Python lives under `scripts/`.

## Why this diagnostic

The sole completed old adaptive F0 took34.435s: pair propagation21.615s
(support queries20.643s included), first property build/solve12.795s and
gate support0.013s. It exited on a full-model-replayed UNSAFE candidate. That
does **not** explain the two monolithic outer timeouts: their durable evidence
previously ended at the broad MONOLITHIC_F0_RUNNING stage.

Both smoke inputs have18 non-reusable pair/property obligations despite5/4
individual interval facts. Support quotas are per-layer calls, not an overall
F0 cap. Scheduled property calls use remaining-budget allocations, not a fixed
10-second ceiling. Some existing lowering/build work precedes native calls
without decrementing the already-granted native time limit. These are code
observations, not yet a causal timing attribution. This diagnostic changes none
of them.

## Instrumentation contract

`f0_timing_trace.py` wraps registered ACT function references by identity,
including pair/expert propagation, guarded support, gate/difference support,
F0 encoding, disjunction construction, HZ lowering, native SciPy `milp`, and
`RequestBudget.limit`. Arguments, objects returned, exceptions and solver
options are forwarded unchanged. No all-function profiler or solver callback.
An observer failure raises a BaseException-derived error rather than silently
being converted to an ordinary solver fallback.

BEGIN is appended and fsynced before calling the original function; END/RAISE
is appended afterwards. A hash chain, nested span IDs, request/runtime hashes
and monotonic time relative to the **outer request** bind all records. Metadata
records pair/property/layer where available, dimensions/nnz, actual limits,
returned status, primal/dual/gap/nodes. Large arrays are not serialized.
Aliases installed are themselves recorded.

Logging, imports and construction all remain charged to300 seconds. Tracing
can alter deadline-sensitive outcomes. `logging_seconds_observed` measures
serialization/write/fsync but not all metadata/wrapper overhead and omits the
last event's own write. Never subtract it to claim unprofiled speed.

Independent `audit_conv_f0_timing.py` checks hashes, time ordering, nesting,
request/parent/config identity, all terminals and existing snapshot/package
contracts. A final partial JSONL line is retained/countable only on outer kill.
Open spans are **right-censored**, not zero-duration successes: report known
elapsed lower bound and exposure to cutoff separately. Nested inclusive times
overlap; the native SciPy interval includes its Python setup/presolve/search,
not separately identified HiGHS branch-and-bound time. Structural audit is not
an independently checked SAFE proof.

## Execution and review

After tests, commit and push the observer/protocol; then, from a clean feature
checkout using act-py312, execute:

```
python -m scripts.run_conv_f0_timing
```

Only directory `data/moe/results/conv_f0_timing_20260915_r1` is accepted;
existing directories cannot be resumed/replaced. A separate final audit runs
automatically, including for retained failure. Repeat audit in another process
before compact archival. Preserve raw trace/logs locally; commit only compact
review with exact artifact hashes. Read the resulting durations before deciding
whether any new optimization is warranted. This stage does not authorize one.

Controls: argument/return/exception identity, aliases/restoration, unchanged
budget calculation, identical toy F0 matrices/verdict, actual tiny SciPy solve,
nonfinite metadata, corrupt trace rejection, and actual SIGKILL durability.
Prelaunch suite:38 tests PASS (`scripts.test_conv_f0_timing`,
`scripts.test_conv_three_arm_lifecycle`, `act.back_end.moe.test_weighted_top2`).
