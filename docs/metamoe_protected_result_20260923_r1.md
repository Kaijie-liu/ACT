# Protected expert queries: starvation removed, output certificate still open

## Decision

The fixed-budget execution control worked: the base query no longer consumes
the entire expert allocation. All19 expanded output-property queries reached
the native entry marker,3 returned infeasible under the **unchanged numerical
policy**, and16 were stopped at their individual deadlines. Base feasibility
also remains UNKNOWN. Full-request result is **UNKNOWN**, not a new SAFE.

This resolves the execution-access problem on ONE old input. It neither proves
the16 properties unsafe nor shows that their representation is too weak. There
is no output LP optimum or valid full-model violation in this run. Do not add
time, tune the relaxation, reuse a historical base witness, or launch a cohort
on the strength of this control.

Implementation `f2c872df7`, pre-run freeze `ffdf8008a`, config
`configs/recent_moe/metamoe_protected_smoke_r1.json`, SHA256
`7a8e31de3412657a8d2d2fc9b5d19e5f7847c7aa3655e80f223da64710fe23e0`.
The [protocol](metamoe_protected_protocol_20260923_r1.md) and
[compact archive](metamoe_protected_archive_20260923_r1.json) bind the execution.
504 source/input identities are retained, with no original source rebinding.
Raw evidence is at
`/data1/Kane/MOE/baseline_runs/metamoe_protected_20260923_r1`;
the full review is the adjacent `metamoe_protected_20260923_r1_review.json`.

## Fixed contract and actual outcome

Same oldMNIST0,checkpoint,physical normalizedfloat64 box at2/255,19 properties,
seed100,2threads,2GiB CSR admission,8GiB sampled own-group RSS and300s outer
deadline. Expert allocation stays30s,base capped at10%,remaining ordered rows
share the remainder. Candidate/guard/propagation/support,integer factors,
tolerance and point/infeasibility acceptance are unchanged. No default ACT
backend change; the protected entry is opt-in,one-lane LINEAR_LE only.

| Item | Earlier observation R1 | New protected control |
|---|---:|---:|
| Expert allocation |30s shared|30s shared|
| Base elapsed |78.313s native invocation|3.007s including transfer/startup/cleanup|
| Base accepted result |feasible validated incumbent|UNKNOWN after local deadline|
| Native output-property entries |0/19|19/19|
| Completed native property returns |0|3|
| Accepted expanded-region exclusions |0|3|
| Remaining properties |19 unproved|16 unproved|
| Full output/request |UNKNOWN|UNKNOWN|

These are sequential diagnostic runs, not a frozen paired speed comparison.
The base time columns have explicitly different instrumentation boundaries.
Historical feasible status is **not** imported into the new run. Unknown base
cannot license CERTIFIED, even if every property were eventually excluded.

The3 accepted exclusions are zero-based property rows11,15,18. All other rows
return UNKNOWN from their local deadline; no row is unvisited, no contracted
witness query is needed, no expert/full-model witness is reported.20 native
entry markers include the base; only3 native calls return a completed result.
Interrupted pre-call markers are not proof of completed native work. The raw
query records and matrix bindings, not a wrapper solve counter, define these
counts. Child workers are terminated and reaped; no live child remains.

Base/query timeouts retain all partial files. There are18 child launches because
an already-running healthy child is reused; a killed query is never retried.
The largest measured local overrun including polling,kill,receipt/hash work is
0.012704s and is charged. This is not a claim of instantaneous OS enforcement.
Expert evaluation finishes before its30s deadline.

Router coverage remains candidates[1],excluded[0],unresolved[]. Nonzero support
still accepts the unchanged fast-fallback range[3.345338179462453,
4.30685851511294]. The expert model retains5636 continuous +1282 binary factors,
3847 constraint rows and336069 nonzeros. Incoming guard factors are not
simplified or relaxed. Numerical/source trust remains network→HZ,input/guard
lowering and the HZ solver policy; this is not a source-complete certificate.

## Full cost and audit

| Recorded cost | Seconds / size |
|---|---:|
| Charged request including imports,supervision and cleanup |157.479581s|
| Batch through summary including postflight inventory |157.766841s|
| Candidate analysis,unchanged path |92.173107s|
| Protected expert evaluation,trace span |28.226183s|
| All20 query returns,including base |28.200758s|
| Child cleanup included in those queries |0.146369s|
| Selected-score support,unchanged path |30.056467s|
| All43 propagation layer spans |4.590948s|
| Four HZ→MILP lowerings |0.007507s|
| Trace writes observed before last event |0.072929s|
| Postflight protected-artifact hashing |0.008639s|
| Separate saved-only audit |0.306676s|
| Fresh saved-only reconstruction for archive |0.306669s|
| Sampled process-group peak RSS |2418720768 bytes|
| Preserved raw evidence |155 files /5651040 bytes|

Parent/child and nested spans overlap; do not sum every table row. Audit and
archive costs are separate from the request. Interpreter/interactive gaps and
final audit/archive JSON writes are not included in their function clocks.
Earlier request211.277s vs current157.480s is an observation, not an established
speedup: configuration,termination behavior and shared-machine load differ.

Outer terminal COMPLETED/exit0, no outer timeout/RSS refusal. All124 trace
events close without a partial tail. Frozen terminal/query auditor PASS;
archive reconstruction agrees except for its own timing. It checks actual
property projection,integer ordering,matrix/token/hash bindings,deadline and
outer precedence,complete ledger and original float-point validation. It does
not independently reprove HiGHS infeasibility or source lowering.

54 pre-freeze controls pass; new21+11 also pass in the pinned intake environment.
Two additional saved-only archive controls reject outcome drift but tolerate
the independent auditor's own different wall time. The first direct auditor
launcher failed before reading evidence with `ModuleNotFoundError: act`;
adding the project PYTHONPATH fixes invocation without editing frozen code or
rerunning the experiment. The old-suite device-initialization invocation issue
is preserved in the protocol. Neither failure is hidden as an experiment retry.

To reproduce the separate audit to a NEW output path:

```sh
PYTHONPATH=/data1/Kane/MOE/ACT /data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_metamoe_protected_smoke_r1.py --config configs/recent_moe/metamoe_protected_smoke_r1.json --output /data1/Kane/MOE/baseline_runs/metamoe_protected_saved_review_NEW.json
```

## Next boundary

Stop this R1; no retry,extra sample,time extension or changed acceptance gate.
The immediate bottleneck has moved from **no property access** to **base and16
property queries still unfinished under their protected slices**. Internal
native phase and representation sufficiency remain unseparated for those16;
calling their cause a weak LP relaxation would still exceed evidence.

Any next change should stay execution/evidence oriented: examine whether a
current-request,fully validated feasible factor assignment can avoid redundant
base search; do not equate a router-only point or historical feasibility result
with an expert-model witness. Candidate native supervision is a separate
remaining soft-limit problem (92.17s here); it must retain complete coverage and
cannot convert timeout into exclusion. These require separate controls/freeze,
not modification of this run. Singleton guard elimination and representation
changes remain separate worklines. No formal comparison gate is opened.
