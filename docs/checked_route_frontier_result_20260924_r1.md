# Checked route frontier: useful-pruning signal and measured no-pruning overhead

Implemented and executed on `feat/moe-route-verification`, clean execution HEAD
`1f86072af`. [Protocol](checked_route_frontier_protocol_20260924_r1.md),
[73-control record](checked_route_frontier_controls_20260924_r1.json),
[frozen config](../configs/backend_controls/checked_route_frontier_synthetic_r1.json),
[saved-only exact review](checked_route_frontier_results_20260924_r1.json).
All515 source files in the sealed4096 comparison remain unchanged.

## Actual method change

The old source-complete construction path propagates every expert and builds
every unordered pair. The new opt-in path first checks the router source and
exact dual bounds for strict score dominance. An insider strictly dominated
everywhere by an outsider makes that pair guard empty. Only such checked-empty
pairs are discharged; unknown and tie-legal pairs remain. Experts unused by
any retained pair need no propagation. The final ledger still accounts for
EVERY original pair/property, not a reduced experiment denominator.

This is checked exclusion and lazy source construction, not another parse
cache, a stronger output relaxation or a changed solver/status tolerance.
Candidate duals use only the final affine definitions. No native LP/search
is needed for this fixed candidate rule; all evidence is checked independently
against the complete router LP with exact residual correction. This is NOT a
claim that candidate pruning itself is new: production ACT already analyzes
routes. The gap addressed here is its source-complete proof path, which did
not yet have checked exclusions and constructed everything unconditionally.

## Twelve frozen synthetic calls, no real-model or solver experiment

Both fixtures are E8/C10/width8/depth2, with nontrivial hidden layers, constant
safe expert final outputs, and identical router final weight rows. Only score
offsets distinguish strictly ordered versus all-tied routes. Three alternating
repetitions per method/fixture;30s total per call,2CPU, sampled8GiB. These are
explicitly designed mechanism controls, not representative trained models.

All12 calls completed and independently reconstructed the same positive
analytic request. Every retained expert trace, pair guard, output matrix and
property row matches the exhaustive reference exactly. No real checkpoint,
dataset, sealed proof, old positive bound or external backend was loaded.

| Fixture / method | Completed | Required output bounds / original obligations | Median build process | Median check process | Median total returned time | Evidence bytes |
|---|---:|---:|---:|---:|---:|---:|
| Strictly ordered / exhaustive | 3/3 | 252/252 | 1.1896s | 2.2806s | 3.5342s | 7,385,979 |
| Strictly ordered / checked frontier | 3/3 | 9/252 | 0.3080s | 0.3051s | 0.6200s | 626,875 |
| All tied / exhaustive | 3/3 | 252/252 | 1.1649s | 2.3075s | 3.5405s | 7,383,747 |
| All tied / checked frontier | 3/3 | 252/252 | 1.3837s | 2.4568s | 3.8906s | 7,446,131 |

The useful-pruning fixture propagates2 instead of8 experts and builds1 instead
of28 pair LP families. Its243 other output obligations are discharged by
checked route exclusions; they are NOT243 output lower bounds. Total median
cost decreases82.46%. In the all-tied control no expert/pair can be skipped;
total median cost increases9.89%. This counter-control is kept, not tuned away.
The frontier therefore stays opt-in, with no universal speedup claim.

Times include source generation/validation, candidate construction and checking,
propagation, LP materialization, exact output checking, serialization, imports,
receipt and owned cleanup. Return time includes ledger writing; ledger overrun
invalidates acceptance. Subsequent administrative review is separate. There
are no native solver calls in the12-call comparison, and no free preprocessing.
Sampled parent+worker RSS ranges: prunable exhaustive70.85–71.63MB, frontier
47.73–49.96MB; tied exhaustive72.43–81.90MB, frontier78.24–81.59MB (decimalMB).
Three repetitions support only this synthetic cost observation, not trained
model/general workload or external-tool performance.

## Independent checks and failure behavior

73 controls passed:20 new method/supervision controls and53 unchanged source/
proof regressions. New controls include multiple retained routes, boundary
ties, dimensions/depth/zero radius, exact retained-matrix differential, wrong
source/factor/property/run identities, missing/duplicate obligations and
partial/nonpositive output evidence. Standard-library `python -S` checking
does not import the producer, model runtime or solver.

The synthetic supervisor rejects hard deadline, worker exception, missing
receipt, a complete positive file published by a worker that then exceeds its
deadline, and final ledger overrun. Partial evidence survives but is never
promoted. The saved12-call review reruns source, exclusion and all-required-bound
checking rather than only trusting hashes. Additional in-memory mutations of
the actual saved package reject10 corruptions (router binding, removed pair,
removed property, changed exclusion, removed layer, inward input, bad guard,
wrong-sign inequality dual, output run, output LP identity); removing one
output candidate yields NOT_CLOSED with252 original obligations.

158 raw files,69,088,670bytes are retained locally in the frozen result root;
the compact committed review binds every raw file. No raw model/data/archive
is committed. Reconstruct using:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m checked_route_frontier.controls --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m checked_route_frontier.review --check
```

## Decision and what has NOT been achieved

Proceed only to separately versioned REAL full-proof supervisor integration,
reusing the existing hard budget/lifecycle: checkpoint/source intake, router
evidence receipt, checked exclusion, retained propagation, native output
candidates and all-original-obligation aggregation. Control partial route
evidence and whole300s costs BEFORE a separately frozen real comparison.
Real router bounds may prove no useful exclusions; that remains a stop/fallback
outcome, not a reason for adding time or tuning the sealed inputs.

This stage has ZERO new real positive certificates and changes ZERO external
comparison results. MetaMoE4-versus9 and the static CROWN11-versus13 findings
remain. There is no claim of high-accuracy/cross-family closure, native floating
execution correctness, route change established merely by retained pair count,
or a repair of the23 historical source-gap gains. Inputs98,4088,4096 remain
sealed. No production configuration,25% schedule or numerical gate changed.
