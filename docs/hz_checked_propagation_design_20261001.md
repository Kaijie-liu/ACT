# Checked multi-objective support inside HybridZ: bounded integration control

Frozen before implementation, from `98fd008a9`. This stage addresses G1's actual
propagation seam. It does not declare G1 complete and does not authorize real
models, sealed inputs, native optimization calls, CUDA or dependency changes.
The [control configuration](../configs/hz_checked_propagation_20261001.json)
fixes the finite roster and existing CPU algorithm. No new precision search.

## Implementation contract

Use an explicit additive `HybridzTF` subclass and a versioned guarded expert
propagation entry. Existing core/config/pipeline bytes and default dispatch stay
unchanged: historical audits bind their current bytes. The subclass participates
in the real analyzer's `apply → _propagate_sparse_hz → sparse ReLU` path, rather
than merely running a detached support script.

The entry constructs the tie-inclusive unordered top-k guard from the supplied
correlated input/router HZs, binds their snapshots, route/expert, network graph,
guarded entry and request, and checks these identities before and after use.
Their shared factor interpretation, network-to-HZ transformation, guard lowering
and inherited affine/ReLU arithmetic remain trusted. A hash is not their proof.
An expert invocation covers only its declared route, not all MoE obligations.

At each guarded ReLU select at most four unstable rows, closest to zero with
stable row-order ties. Query both sides over exactly the current sparse HZ.
Keep all original continuous/private binary columns and constraints. Use the
already-frozen 128-step multi-objective CPU proposal and exact residual checker.
Bind every query to the layer and the caller envelope. Require the full roster;
no partially checked batch changes propagation. Convert exact rational lower
and upper bounds to binary64 outwards, intersect with existing bounds, reject
contradictions and pass the result to the unchanged sparse ReLU encoder.

If proposals fail or the finite capacity is exceeded, record that outcome and
retain the existing bounds; do not truncate factors, constraints or requested
sides. Preserve native fallback where configured, with only the absolute
request budget remaining. Budget allocation to fallback is conservative: its
LP/MILP nominal limits jointly cannot exceed the remaining time. Calls may
overrun cooperatively, so no hard-deadline or full native-execution claim is
made until separate supervision is integrated. This control tests that seam
with an instrumented fallback, not a native solver.

Candidate work has a four-second local allowance within a single at-most-300s
request deadline. Source capture, binding, propagation, proposal, checking,
rounding, fallback and return are charged; proposal cost is nested rather than
added twice. Check absolute deadlines before accepting and on return. A late
result yields no accepted propagation. Global analyzer state is restored on
success and failure. One request owns its source objects; hashes detect ordinary
alias pollution, not concurrent mutate-and-restore attacks.

## Controls and comparisons

Run the 14 named control groups in the configuration on small constructed HZs
and small ACT networks. Never load a checkpoint or dataset. The mechanism example
has input `x∈[-1,1]`, router `(x,-x,0)`, tie-inclusive pair `{0,1}` and expert
`ReLU(x+1/4)`: its guard forces `x=0`. Checked guarded support can eliminate its
ReLU binary; the guard-discarded sound outer domain cannot. Include top-1 and
other legal pair identities, further ReLU layers and distinct private factors.

Compare disabled integration to unchanged propagation, and retained guard to
the explicitly guard-discarded outer domain. This is an integration/mechanism
control, **not** a claim of tighter bounds than existing native LP support or
lower end-to-end cost. Preserve weak/failed results instead of tuning steps.

Exercise outward rounding, missing and duplicate sides, valid-but-wrong source,
network/router/entry mutation, proposal exception, local/global deadlines,
capacity refusal, fallback allowance and restoration of global analyzer state.
Archive all attempted controls and complete accepted support packages. Recheck
stored exact bounds independently of the candidate generator and report the
given-HZ trust boundary. Full MoE aggregation, GPU speed and realistic capacity
remain separate tasks; all G1–G6 stay OPEN.
