# Full V2: saved-log analysis of unresolved obligations

This is a post-run diagnostic, not a new experiment or verdict revision.
Parent: `act/pipeline/moe/results/conv_full_v2_review_20260915.json`.
Derived record: `act/pipeline/moe/results/conv_full_v2_obligations_20260915.json`.
The stdlib-only analyzer checks the parent's entire byte/hash inventory before
reading the 60 ACT journals. No GPU, model forward, support query, LP or MILP
solve is performed. All 90 original outcomes, including CROWN, remain frozen.

## What actually stopped

All 32 ACT timeout requests completed exact route analysis and entered F0.
Their final reason is `REQUEST_BUDGET_EXHAUSTED`, not candidate infeasibility
failure. Twenty-five stop after a monolithic property; four adaptive cases
stop at `f0_property_solve`, three at `f0_property_complete`. Final packages
censor the property arrays on timeout; empty arrays do **not** mean zero queries.
The durable journal recovers the actual attempts and their scopes.

| Timeout subset | Requests | Single / multiple pair | Property queries returned | Pair-property scopes queried | Eligible reused obligations | No property query observed | Required obligations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 13 | 6 / 7 | 192 | 192 | 1 | 14 | 207 |
| Matched monolithic | 19 | 6 / 13 | 170 | 314 | 1 | 0 | 315 |

Every returned query in this table is solver status1 / solver-limit UNKNOWN.
Across **all** 60 ACT requests, there are 419 property queries (261 monolithic,
158 branch-weighted), all status1. There are no completed nonpositive-relaxation
property records and no unreturned property calls. Expanding a union query
into its pair-property scope is coverage accounting, not multiplying the
number of independent solves. All32,681 native READY records have terminals:
32,640 calls returned and41 were skipped before entry for lack of budget.

The 14 unstarted property scopes are confined to four adaptive requests:

| Input | Legal pairs | Unqueried obligations | First F0 property at request seconds |
|---|---:|---:|---:|
| 26 | 3 | 7 | 105.05 |
| 48 | 3 | 2 | 112.03 |
| 95 | 3 | 3 | 98.65 |
| 106 | 2 | 2 | 148.16 |

The JSON lists exact pairs and competitors. No property-query event means
the solve itself was not observed; construction/preparation may already have
occurred. The remaining unqueried scopes across successful UNSAFE requests
are legitimate early termination, not additional solver failures.

This supports **unfinished search / budget allocation** as the observed stop,
not a demonstrated completed-relaxation bottleneck. It does not prove that
additional time would yield SAFE or identify a unique cause of difficult search.

## Cost and long-tail observations

Within the timeout subsets, native F0 entry-to-return observations consume
2,408.11/3,853.95 seconds (62.48%) for adaptive and 4,098.86/5,625.23 seconds
(72.87%) for matched. Adaptive additionally spends 588.34 seconds in native
Tier1 expert solves and 715.58 in native support; matched spends 1,086.88 in
support. These are disjoint native spans, not sums of overlapping wrapper
durations. Remaining time includes loading, propagation, construction, logging,
replay and terminal overhead; the journal cannot attribute every residual
second to one component. First-property median times among timeout requests
are 98.65 and 54.28 seconds respectively, with different route strata.

All recorded native allocations fit the active absolute deadline. Nevertheless
794 native returns are observed more than1ms after their local deadline across
the60 ACT requests; the largest is42.9999s in `_solve_output`, input26/adaptive.
The timeout matched subset's maximum is3.25385s in a monolithic property.
Observed return latency includes wrapper observation; V2 fixes passed-budget
accounting, not hard real-time enforcement inside the native solver. The outer
watchdog remains necessary. These observations do not authorize deleting the
25% slice or extending the300s request budget.

## Positive diagnostic duals: a concrete, but unaccepted, signal

Of419 returned property queries,131 have an available full-objective diagnostic
dual and54 have a value above1e-7;51 of these positive records occur in timeout
requests. All still have solver status1, which fails the frozen acceptance gate.
The analyzer does not substitute a primal incumbent or a missing value for a
dual. For branch-weighted calls it uses `solver_dual_objective`, which includes
the HZ center; for monolithic calls it reads the unique scoped native
`mip_dual_bound`, whose objective includes the selector-weighted centers
(`act/back_end/moe/monolithic_f0.py`, `_build_disjunction`). A raw weighted
factor-only objective cannot be interpreted as the full margin.

Two inputs are especially informative:

| Input / pair | Adaptive stored positive duals | Matched stored positive duals | Remaining interval fact | Original outcomes |
|---|---|---|---|---|
|16 / {0,3}|9/9, minimum0.333916|9/9, minimum0.317202|None needed for this diagnostic coverage|Both TIMEOUT; CROWN UNKNOWN|
|98 / {1,2}|8/8, minimum2.888291|8/8, minimum2.888291|Competitor8 eligible from both expert intervals|Both TIMEOUT; CROWN numerical POSITIVE|

For input16 the stored relative gaps are approximately0.26–0.78; for input98
approximately0.07–0.13. Thus the frozen zero-gap/optimal-status requirement
continues to reject the records even when the reported full-objective dual
has positive sign. The logs contain terminal duals, not trajectories: they do
not reveal when positivity was first reached or how much early stopping could
save. The positive records are **not independently checked lower-bound proofs**;
neither of these requests becomes SAFE. Both are single-pair, so they are not
new route-changing-certificate evidence either.

The34 individual expert facts available across60 requests yield only two
pair-property reuse opportunities (the same input98 obligation in both arms).
Having one expert's positive fact is not sufficient for a mixture. This is
a measured limit on reuse availability under the common no-support prelude,
not evidence that the containment theorem or reuse implementation is wrong.

## Cross-arm context prevents assuming every timeout is safe

The35 UNSAFE method runs cover18 distinct inputs. Among adaptive's13 timeouts,
input113 has a CROWN-path full-model witness. Among matched's19 timeouts,
indices8,20,47,56,69,75,113 have another arm's replayed witness. This retrospective
union is not a same-budget portfolio result. It demonstrates why counting all
timeouts as latent SAFE or predicting that more time will certify them is wrong.

## Next decision, not an automatic new experiment

1. Preserve the full V2 table and numerical gate. Do not increase sample count,
   reopen backend searches, or retune25% based on this diagnostic.
2. A separately scoped proof-evidence study can use inputs16/98 as observed
   controls to test **independently checkable sign-sufficient lower bounds**.
   It must retain the model/domain/guard/offset identity and cover every
   obligation. An LP dual check only certifies its LP relaxation; a native
   MILP dual scalar is not a rational certificate or a branch-tree proof.
   If an independently checked LP cannot reproduce positivity, report that
   limit rather than simply dropping status0. This study would need a new
   protocol; none was launched here.
3. Separately, the four partially unqueried multi-pair cases motivate a bounded
   scheduling/long-tail study on observed inputs, with matched changes and
   unchanged acceptance policy. They do not motivate a candidate-uperset
   fallback: all32 timeouts already had exact route coverage.

## Reproducibility and analysis correction

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scripts.analyze_conv_full_v2_obligations --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m unittest scripts.test_conv_full_v2_obligations
```

Nine stdlib-only controls exercise union-vs-query counting, wrong class/pair,
duplicate scopes, missing bounds, scope pairing, late-return vs overallocation,
both-expert fact eligibility, and cross-check all60 archived logs against the
frozen auditor's independent property counts and obligation partitions.
The first unpublished projection used
`F0_RUNNING` instead of the actual `TIER2_F0_RUNNING` state label, leaving13
entry timestamps null; this parser label was corrected before archival, with
no source-result or count change. That draft is retained at
`data/moe/results/conv_full_v2_analysis_drafts_20260915/first_projection.json`,
SHA-256 `c057ce67e772fb4f8350348787021384365bbdf72e175821ccba55d1c89ea101`.
Final checks reconstruct every field, verify parent raw hashes and do not
modify any frozen artifact. This is diagnostic consistency checking, not a
new independent proof of the solver's bounds.
