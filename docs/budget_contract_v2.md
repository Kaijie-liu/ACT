# V2 budget and partial-terminal execution contract

## Completed implementation and controls

Implementation execution `e9cde69450df03d3ed74097011d7ce45d88a98ad`;
compact independent review:
`act/pipeline/moe/results/budget_contract_v2_controls_review_20260915_r1.json`.
Both source-defined three-expert/all-three-tie-pair controls retain their
baseline SAFE result. Four baseline/V2 packages pass the original structural
auditor. Separate `python -S` journal checks match the saved controls exactly:
adaptive80 events/2 property results/10 native calls, monolithic62 events/
1 property result/7 native calls; no unreturned calls in these complete toys.
The focused suite passes66 tests, including real SIGKILL retention, synthetic
stale-allocation reproduction and recomputed-hash budget tampering rejection.
All40 original smoke and13 timing-diagnostic artifacts remain unchanged.

These are correctness/conformance controls, not trained-convolutional smoke,
coverage or performance evidence. No new conv query or full90 has started.
Next implement/freeze the new outer supervisor and journal-plus-terminal audit
for the two ACT arms on the old smoke inputs, retaining all terminals and the
same300s outer cap. Test that integration before a separately recorded run.
Do not interpret this successful implementation stage as a passed R1 gate.

This is the bounded engineering follow-up to the measured1.560s stale native
allocation in `docs/conv_f0_timing_r1.md`. It does not change a solver gate or
claim that accounting repair resolves the convolutional verification problem.

## What changes, identically for both ACT arms

The explicit opt-in `scripts.budget_contract_v2.verify_v2` adapter applies to
scheduled, scoped-reuse, SciPy/CPU ACT adaptive and matched monolithic paths.
The300-second request start remains the outer supervisor's start, including
imports/model loading. Five seconds **within** that budget are reserved for
replay/evidence/terminal publication. This is a declared new execution policy,
not a295s uncharged run plus another5s outside the request. The25% allocation
rule is unchanged and operates on the remaining working budget.

Every granted local allocation has an absolute deadline. Wrappers around
expert solving, support sweeps, gate/difference support and weighted property
solving retain the tighter parent deadline. HZ lowering and union construction
therefore consume the property slice before the native call. Native SciPy
entry receives the minimum of its original requested limit, the live local
remainder and the request working remainder. A durable READY record is fsynced
**before** computing that final minimum, so logging cannot silently spend an
already-issued native quota. Less than1ms means no native launch, never zero
as a stand-in for exhaustion. No tolerance, presolve, gap or objective changes.

This is cooperative accounting, not a real-time guarantee. Python construction
and native presolve/search can overrun;5s is a fixed engineering reserve, not a
proof that serialization always finishes in time. The external300s watchdog
and late-result rejection are still mandatory. This version does not add a
new solver callback, sign-based early acceptance or partial MILP-proof reuse.

## Durable evidence and limits

The hash-chained, fsynced journal records local scopes, grants, native READY and
return/skip/error, property results, direct full-model replay observations,
and major stages. Property keys come from the **actual encoding(s)** and their
linear property, never stale caller pair/index locals. An unresolved property
cannot be promoted merely because another property completed.

Native READY contains the deadline, not a pretended exact launch timestamp;
returned calls add the actual entry clock and passed limit. If killed before
return, their exact passed limit/result are unavailable. A property's solver
result is durable before the following replay/next property; replay is a
separate record, not inferred from the existence of an expert candidate.
Journals alone can establish neither request SAFE nor independently proved
UNSAFE. Full coverage, complete package, numerical acceptance, full dynamic
witness replay and independent outer-terminal checks remain required.

`scripts/check_budget_contract_v2.py` uses only the Python standard library.
It checks sequence/hash, scope nesting, live-budget inequalities, completion
and incomplete-call accounting. It can run with `python -S`. It does **not**
prove HZ propagation, optimality, MILP arithmetic or deployment equivalence.

## Version identity and entry point

The frozen ACT Python files, old supervisor, old timing observer, configs and
results remain byte-identical. The new adapter deliberately changes execution
through a registered set of process-local ACT aliases, restored on exit;
it is **not** merely observational instrumentation. One main-thread worker is
allowed per process. Ordinary `verify_staged_linf` defaults are unchanged.

`scripts/conv_budget_contract_v2.json` declares this policy.
`scripts/conv_budget_worker_v2.py` requires a new request with
`execution_budget_contract=execution_identity()`, binding policy/source hashes
in addition to the old checkpoint/input/config identity. It refuses reused
result roots, non-ACT arms, changed base requests and new full-cohort inputs.
Packages explicitly include `execution_budget_contract` and a journal hash;
reusing only the old method-config hash would not identify this execution.

The worker is not an outer launcher. A new, separately frozen supervisor/audit
protocol is required before real-model smoke. It must validate both journal
and package bindings, retain every terminal, check snapshots on killed calls,
reject late positives, and use the same reserve on both arms. Existing smoke
FAIL and the single diagnostic TIMEOUT remain unchanged. Full90 is not
authorized by this implementation or by successful unit controls.

## Validation and next action

Controls cover clock-driven construction/serialization charges, stale grants,
global loading charges, nested deadlines, no-zero/no-launch exhaustion,
argument/result identity, both actual staged and monolithic toy API paths,
unchanged numerical policies and original package audits, replay separation,
tampering with recomputed hashes, and SIGKILL durability. An initial test mock
used a variadic signature unlike SciPy; it was corrected to the explicit
`options` signature. No production semantics were changed to satisfy that test.
The final focused suite also checks worker refusal of unbound/changed base
requests, existing journals and inconsistent advertised reserve policies.

After commit/push, `python -m scripts.run_budget_v2_controls` writes one new
source-defined toy control root and runs the journal checker in a separate
`python -S` process. This is not a trained conv query or a performance study.
Then freeze a bounded two-arm OLD-input smoke with a new runtime and terminal
auditor before execution. Do not rerun R1, adjust support or numerical gates,
extend the300s cap, or treat complete UNKNOWN as a new certificate.
