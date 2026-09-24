# Shared router residuals: same bounds, lower synthetic proof-segment cost

Status: frozen synthetic R1 completed and audited; opt-in implementation only.
No real request or native solver invocation; no new complete MoE certificate.

## Main result

The same fixed equality candidates can be represented as **8 score potentials**
and checked by exact shared-residual subtraction instead of preparing/checking
56 separate full router-margin LPs. This removes repeated matrix work without
weakening the lower-bound check or discarding shared-input cancellation.

All 18 registered calls completed. All 9 paired runs agree exactly on all 56
ordered bounds, residual box corrections, nonzero residual counts, and all 28
pair decisions. These are 3 synthetic fixtures with repetitions, not 9 different
trained models. No threshold, factor frame, query count or source matrix changed.

Whole router-segment medians (3 repetitions per arm):

| Synthetic router | Pairwise | Shared residual | Reduction | Pairs retained, both arms |
|---|---:|---:|---:|---:|
| Shared final weights, descending offsets | 2.6503 s | 0.4219 s | 84.08% | 1/28 |
| Shared final weights, all tied | 2.6667 s | 0.4465 s | 83.26% | 28/28 |
| Seeded non-identical final weights | 2.6965 s | 0.4480 s | 83.39% | 3/28 |

These times include source generation, router construction, candidate
generation, independent fresh-process source/margin checks, serialization,
imports, owned cleanup and terminal publication. They exclude **all expert and
output work**; these phases were not executed. Final cost-ledger publication and
later audit are identified separately, with late publication invalidating
completion. The ratios of medians are approximately 6.0–6.3; they are not speedup
claims for complete verification or real models, nor medians of paired ratios.

## Where the cost changed

| Router | Build process: pairwise → shared | Check process: pairwise → shared |
|---|---:|---:|
| Prunable | 0.9208 → 0.2049 s | 1.7286 → 0.2049 s |
| Tied | 0.9253 → 0.2003 s | 1.7290 → 0.2349 s |
| Random | 0.9354 → 0.2006 s | 1.7448 → 0.2362 s |

Recorded worker-operation medians place fixed candidate generation at about
0.739–0.747 s in the old path versus 0.0155 s in the new path. Common router
construction remains approximately 0.096–0.102 s. Fresh load/source/margin
checking decreases from approximately 1.677–1.700 s to 0.174–0.180 s.
The total-segment comparison, not one favorable sub-operation, is the result.

Candidate files decrease from 95,929 to 760 bytes per call (99.21% smaller).
This is **candidate payload only**: the source and router trace remain present
in both arms. It is not a 99% reduction of the complete evidence package.
Maximum sampled RSS is approximately 50–51 MiB for the old arm and 48–50 MiB
for the new arm; no memory-limit stop occurred. These are sampled parent-plus-
owned-process estimates, not exhaustive peak-memory proofs.

The new algebra reports one final-state parse, 8 potentials, 56 derived bounds,
208 nonzero equality products and 1,400 residual-difference coordinates in each
fixture. The parse count describes final residual algebra only: layerwise source
checks still parse their inputs and outputs. No checks were removed.

## Frozen execution and audit

- Freeze/launch commit: `fa19ce7afa0c68d50a475140f24edff678a195d2`.
- Protocol: [shared_route_residual_protocol_20260924_r1.md](shared_route_residual_protocol_20260924_r1.md).
- Config SHA256: `537ec8821b7fbb094d591929d7fa3d5d1c136842e4e53b236df314acfb8145fe`.
- 3 fixtures × 3 repetitions × 2 arms; arm order reversed on odd repetitions.
- 30 s per router segment, two-thread environment, sampled 8 GiB resource limit.
- 45 passing controls: 25 new mathematical/identity/supervision controls and 20
  unchanged route-frontier regression controls, 11.328 s total.
- Saved-only independent source and bound audit: PASS, zero issues; 16.696 s
  separately charged, not subtracted from or added to request budgets.
- All 551 frozen source/protocol bindings intact, including all 538 old bindings.
- All 236 raw files (31,353,534 bytes) retained outside Git; no failures dropped.
- Compact audit and file-hash archive are committed; raw source matrices are not.

Evidence:
[controls](shared_route_residual_controls_20260924_r1.json),
[audit](shared_route_residual_audit_20260924_r1.json),
[archive](shared_route_residual_archive_20260924_r1.json),
[fresh saved-only replay](shared_route_residual_replay_20260924_r1.json).
The replay runs the checker implementations again, without the candidate
providers, model libraries or native solvers. This is not an independent human
review or a third mathematical proof implementation. Existing old pairwise LP
checking supplies the exact arithmetic differential reference.

Read-only local reproduction (requires archived files but no checkpoint/data):

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/replay_shared_route_residual_study.py
```

## Continue/stop decision and scientific boundary

**Continue to a separately versioned full-proof integration.** There is a
specific measured cost signal, including the all-tied control where no route
can be omitted. The next integration must use one 300 s request budget, accept
only complete identity-bound route receipts, preserve all original output
obligations, and independently check the newly built retained expert/output
matrices. Test cutoff, errors, partial evidence and all-phase cost accounting
before any new real freeze. No real execution is authorized by this synthetic
config; choosing a new real comparison is a separate scoped decision.

Do not retroactively speed up the sealed input4098 run: its online receipt
still timed out and its 24 exclusions were checked only offline. The synthetic
factor sizes are much smaller, so linear extrapolation of these ratios to that
run is not justified. Inputs 98/4088/4096/4098 remain sealed. No production gate,
historical 23 main-table gains, external result or high-accuracy claim changes.

This is a MoE evidence-composition refinement using a standard equality-residual
bound, not a claim of new general LP theory. It improves a necessary preparation
step but has not yet increased complete output certificates. Stable external
advantage, high-accuracy source-complete route-changing certificates, human
technical review and clean-environment reproduction remain separate goals.
Those evidence requirements—not the number of controls or a synthetic ratio—
determine the strength of an eventual ISSTA submission.
