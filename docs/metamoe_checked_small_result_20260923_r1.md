# Frozen MetaMoE small comparison: stopped by the registered ERROR rule

Execution HEAD `6aba571243d4ddf056ccf30a520a465c43609874`, initially clean and
remote-synchronized. User authorized executing the frozen 20 requests. Config
SHA256 `97e9fcb8fb9556ecef4bf0287510eb7bde690b3eef0c9bb3799e69e4b73aa069`.
No source, selection, budget, tolerance, relaxation or method changed during
execution. No retries/resume or replacement inputs.

- [Frozen selection/protocol](metamoe_checked_small_protocol_20260923_r1.md)
- [Execution config](../configs/recent_moe/metamoe_checked_paired_small_r1.json)
- [Independent terminal/evidence audit](metamoe_checked_small_archive_20260923_r1.json)
- [Saved-only descriptive analysis](metamoe_checked_small_analysis_20260923_r1.json)
- [Analysis implementation](../scripts/summarize_metamoe_checked_small.py)

## Terminal accounting and interpretation

**14 normal returns + 1 ERROR + 5 NOT_STARTED_AFTER_ERROR = 20 registered
rows. This is NOT a completed 20-execution comparison.** Every row remains in
its registered denominator (10 per arm, 5 per dataset). The script's normal
exit means it published the fail-stop ledger, not that all requests succeeded.

Same checkpoint and physical boxes, normalized-space epsilon `2/255`, clipping
`[-10,10]`, CPU/float64, two threads, 300-second whole-request ceiling. This is
not a pixel-space radius claim. All tie-legal routes, selected-score
definedness and 19 global output margins are retained. The local expert
allocation remains 30 seconds. Both arms pay independent startup/load and
analysis; no historical census or cross-arm answers used.

Author path means unchanged author backend plus our frozen strict
route-invariance sufficient adapter. It is not a literal reproduction of
the paper's original aggregate table, nor a direct dynamic-dispatch encoding.

| Dataset / original index | ACT terminal / seconds | Author terminal / seconds |
|---|---|---|
| CIFAR10 / 1 | UNKNOWN / 13.855061 | BACKEND_POSITIVE / 8.410950 |
| CIFAR10 / 2 | UNKNOWN / 38.524710 | BACKEND_POSITIVE / 8.078231 |
| CIFAR10 / 4 | UNKNOWN / 38.862921 | BACKEND_POSITIVE / 8.642096 |
| CIFAR10 / 5 | UNKNOWN / 38.750653 | BACKEND_POSITIVE / 7.887553 |
| CIFAR10 / 7 | UNKNOWN / 39.906849 | BACKEND_POSITIVE / 8.338618 |
| MNIST / 1 | POSITIVE / 9.961124 | BACKEND_POSITIVE / 7.523245 |
| MNIST / 3 | POSITIVE / 10.301565 | BACKEND_POSITIVE / 7.528139 |
| MNIST / 7 | NOT_STARTED_AFTER_ERROR / not measured | ERROR / 252.922061 |
| MNIST / 9 | NOT_STARTED_AFTER_ERROR / not measured | NOT_STARTED_AFTER_ERROR / not measured |
| MNIST / 10 | NOT_STARTED_AFTER_ERROR / not measured | NOT_STARTED_AFTER_ERROR / not measured |

ACT positives have grade `HZ_POLICY_ACCEPTED`; author positives have grade
`AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER` (raw `safe-incomplete`, all seven).
These are different numerical contracts, not interchangeable formal SAFE.
All seven ACT executions found one candidate, excluded the other, and left
no unresolved route; there is **no new route-changing certificate** here.

| Full registered arm denominator = 10 | ACT | Author adapter |
|---|---:|---:|
| Policy positive / numerical filter (separate grades) | 2 | 7 |
| Full-model replayed UNSAFE | 0 | 0 |
| Completed UNKNOWN | 5 | 0 |
| Outer TIMEOUT / RESOURCE_LIMIT | 0 / 0 | 0 / 0 |
| ERROR | 0 | 1 |
| NOT_STARTED_AFTER_ERROR | 3 | 2 |
| Actually attempted | 7 | 8 |

For the **seven pairs with both arms returned normally**, the positive
intersection is MNIST `[1,3]`, ACT-only empty, author-only CIFAR10
`[1,2,4,5,7]`. Decided sets coincide with positive sets: no UNSAFE was accepted.
The other three inputs are not comparable, NOT negative or zero-cost examples.
The visible complete pairs favor the author sufficient path; this batch
supplies no ACT coverage/speed advantage. Fail-stop missingness prevents a
complete 10-input comparative conclusion. No significance or general
superiority claim; 20 scheduled calls are not 20 independent inputs.

## Error retained; no automatic retry

At roster position 15, MNIST/7 author returned `ERROR: backend_nonzero_exit`.
The worker published its error normally; the outer receipt is `COMPLETED`,
not TIMEOUT. The retained backend traceback terminates at
`complete_verifier/branching_domains.py:291`:

```python
assert len(self.all_lAs) == len(bounds['lAs'])
```

It occurs during BaB domain insertion. This identifies the failed internal
consistency check, not the unique upstream cause. No dimensions or offending
key sets were recorded there, and no crash reproduction was run. Backend
stderr/stdout and effective config remain hash-bound in the archive. We did
not suppress the assertion, downgrade ERROR to UNKNOWN, skip this input,
patch the frozen backend, or continue the five remaining requests.

## Cost (all attempted requests, including the ERROR)

| Scope | ACT total / mean seconds | Author total / mean seconds |
|---|---:|---:|
| All attempted (7 ACT / 8 author) | 190.162884 / 27.166126 | 309.330892 / 38.666362 |
| CIFAR10 (5 / 5) | 169.900194 / 33.980039 | 41.357447 / 8.271489 |
| MNIST attempted (2 / 3, includes author ERROR) | 20.262690 / 10.131345 | 267.973445 / 89.324482 |

Overall medians: ACT 38.524710 s, author 8.208424 s. These have different
attempted denominators and statuses; the author's long failed run is not a
solving-speed comparison. On the seven normally returned pairs the ACT minus
author time difference has mean **19.107722 s**, median **30.220825 s**;
on the two positive MNIST pairs the mean difference is **2.605653 s**.
Do not treat a fast UNKNOWN as a solved request, or unexecuted rows as zero.

Charged requests total **499.493776 s**. Batch wall through final summary is
**501.577576 s**, including parent postflight/hash inventories (not included
in individual request ceilings). Frozen offline input selection is separate.
Original-model replay costs 1.158158 s (zero accepted UNSAFE, explicit empty
ledger); independent structural audit costs 3.471774 s. Those reviews do not
enter performance times. Peak sampled group RSS is 2,580,246,528 bytes ACT,
2,334,547,968 bytes author; sampled peaks are not exact instantaneous maxima.

## Saved-only obligation diagnosis; no new solving

All seven ACT requests completed route coverage, fresh checked base
feasibility and selected-score nonzero checks. No native nonzero optimization
was needed. Base checks took 0.0311--0.1404 s; the old repeated-base bottleneck
does not explain these five incomplete outputs.

| ACT request | Output violations excluded / 19 | Output subquery TIMEOUT | Expert elapsed before publication |
|---|---:|---:|---:|
| CIFAR10/1 | 18 | 1 (property row 1) | 5.270876 s |
| CIFAR10/2 | 0 | 19 | 30.018854 s |
| CIFAR10/4 | 0 | 19 | 30.023773 s |
| CIFAR10/5 | 0 | 19 | 30.017487 s |
| CIFAR10/7 | 0 | 19 | 30.021290 s |
| MNIST/1 | 19 | 0 | 1.571118 s |
| MNIST/3 | 19 | 0 | 1.374942 s |

All 133 requested output rows have records: 56 excluded by the frozen native
infeasibility policy, 77 UNKNOWN with local query TIMEOUT. None was dropped.
The protected local deadlines for the unknowns expire after roughly
1.49--1.65 s per expanded query; native calls started, but no accepted native
result was available for those rows. This is a recorded execution limitation,
NOT a completed nonpositive optimum, a feasible violation, proof that the HZ
relaxation cannot certify, or proof the original model is unsafe. A query
deadline and a whole-request deadline are different. The 300-second outer
budget was not exhausted by any ACT request; no limits are changed here.

## Audit and next boundary

Frozen auditor **PASS, 0 issues** on all 20 terminal records, source/box/
property bindings, saved assignments, independent stored-HZ sign checks,
route/property aggregation and full costs. `smoke_gate_pass=false` is
intentional for a non-smoke protocol, not an efficacy failure or missing
positive gate. A second process disabled native MILP, support optimization
and the production sign helper; its recomputed archive was identical except
audit time. Seven saved-analysis controls check missing-data accounting,
costs, roster integrity and positive-grade/pair distinctions.

The original-model replay ledger has **zero witnesses**, not a new positive
proof. The audit does not independently prove source-to-HZ/guard lowering or
all native/backend numerical bounds. The historical BN source ledger and
all earlier successes/failures remain unchanged.

Raw root: `/data1/Kane/MOE/baseline_runs/metamoe_checked_paired_20260923_r1_small`;
1,243 files, 211,441,134 bytes after replay, all hash-inventoried. No raw model,
arrays or external source committed. New archive/analysis accompany this
report; original frozen manifests and failed run are unchanged.

**Decision:** seal this stopped execution. No expansion, retry, tuning or
resume is authorized by the old freeze. If continuing, first isolate the
author BaB interface failure in a separately scoped old-input compatibility
control, then register any repaired execution separately. ACT's 77 local
query deadlines justify reviewing saved per-property dispatch/budget costs,
not changing relaxation or increasing time on this run. No source-complete,
route-changing or broad author-superiority claim is upgraded.
