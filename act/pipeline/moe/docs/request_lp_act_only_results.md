# ACT-only rational transfer R1 — completed, 0/3 complete positive requests

Execution `fbc48d6a9`; all three fresh generations finish normally under their
registered caps. Read-only reconstruction independently checks **all115 proof
exports**, including unused expert bounds, and reproduces all request verdicts.
The compact archive is `../results/request_lp_act_only_review_20260915_r1.json`;
reconstruct with `python -m act.pipeline.moe.review_request_lp_cases`.
Raw evidence stays in `data/moe/results/request_lp_act_only_20260915_r1`.

| Case | Exact legal pairs | Reused positive | Residual positive | Unresolved | Complete positive | Generation seconds | Separate request check seconds | Evidence MiB |
|---|---|---:|---:|---:|---|---:|---:|---:|
| seed0/4029 | {1,7} | 4 | 1 | 4 / 9 | No, UNKNOWN | 111.87 | 29.62 | 464.94 |
| seed1/4018 | {1,3}, {1,5}, {3,5} | 25 | 1 | 1 / 27 | No, UNKNOWN | 90.23 | 29.59 | 379.61 |
| seed2/4014 | {0,2}, {0,4} | 14 | 1 | 3 / 18 | No, UNKNOWN | 141.88 | 41.37 | 579.63 |

There are46 positive obligations of54, but zero completely discharged requests.
All remaining obligations have checked **nonpositive** residual bounds, not
missing arithmetic checks, proposal failure or a generation timeout. The
unresolved pair/property-index/bound entries are:

- seed0: {1,7}/3/-0.611485, /4/-0.811245, /5/-0.187306, /6/-0.481434;
- seed1: {3,5}/1/-0.604089;
- seed2: {0,2}/1/-0.609630, /8/-0.286429; {0,4}/1/-0.421648.

Property indices refer to the registered nine classification competitors, not
arbitrary class IDs. Exact rational bounds and source identities are in the
archive. Binary factors are explicitly relaxed to continuous boxes; source
binary counts range up to49/49/64 across the three cases. The evidence does not
show that integer reasoning is *necessary*, or that increasing its budget would
resolve these rows: gate enclosures and other relaxation gaps are not isolated.

## Costs and trusted assumptions

No old computed bound, route census, common fact or proof package is reused.
Generation includes each worker's startup, imports, checkpoint/input loading,
fresh ACT propagation/coverage, every LP proposal, inline construction checks
and evidence writes. The separate request-check times are additional. The
driver's one-time startup/source hashing and this later archive reconstruction
are not included in the per-request generation figure; these numbers are not
full CLI/process-to-process benchmark times. No staged/monolithic speedup is
inferred from them.

Only previously materialized **raw inputs** are reused. Their original total
preparation was0.2768000243231654s for ten files; this is disclosed separately,
not apportioned into fictitious per-case measurements. Historical training and
case-selection experiments are excluded and explicitly not claimed free.
Evidence MiB includes raw completed request directories, not model checkpoints.
Repeated serialized source matrices account for substantial storage; sparse
representation alone does not make a compact submission proof artifact.

All checks remain conditional on network/input-to-HZ propagation, membership/
pair-guard lowering and route-infeasibility exclusions. The floating F0
construction is not trusted or called. These three failures do not refute the
earlier HZ-policy SAFE results; they limit this frozen LP proof-generation path.
R1/R2/R3 on index3000, including its18/18 conditional R3 proof, remain unchanged.
No extra query, gate refinement, substitution or larger-budget retry is queued.

## Consequence

The requested transfer attempt is complete, with a negative complete-request
result. It is not a failure of the arithmetic checker and not a reason to claim
115 independently proved networks. The paper must report both the earlier
single-request success and this fixed three-case limit. Next effort moves to
the separately registered second model family and the reviewer workflow, not
to chasing the eight remaining LP signs on these selected cases.
