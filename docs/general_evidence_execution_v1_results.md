# General conditional evidence: frozen new-input result and cost review

The 20-input, three-arm convolutional study is complete: **60/60 terminal
records; final audit PASS, zero issues**. Execution source HEAD is
`a0248e697e0b34f381c4cb96bd4fa3396e0c1de8`; selection SHA256 is
`db3043fb124703e8123e5326eda853dc0d67d45e9104ca316487a631603daea7`.
The original 300-second request budget, 2/255 box, checkpoint, source hashes,
sample order, proposal cap, checking reserve and acceptance gates are unchanged.
This is a new endpoint cohort, not a rerun of input98 or the older30 inputs.
The separate archival re-review also passes with zero issues:60 fresh
per-request reviews,17 repeated full-model witness checks,105.6605s outside
the request budget, and no new solver queries.

## Complete-request outcomes

| Arm | Positive requests | Replayed UNSAFE | UNKNOWN | TIMEOUT | Mean observed seconds |
|---|---:|---:|---:|---:|---:|
| Matched V2 | 0 HZ-policy SAFE | 10 | 0 | 10 | 220.6804 |
| General evidence | 0 CHECKED_CONDITIONAL | 0 | 0 | 20 | 283.0893 |
| ACT route frontend + plain CROWN | 0 numerical positive filters | 7 | 13 | 0 | 4.0807 |

Evidence grades are **not** interchangeable formal-SAFE results. The evidence
arm seeks positive proofs, not a separately budgeted adversarial search. Its
zero UNSAFE count is not evidence of equivalent counterexample-search effort.
There are20 input blocks, not60 independent samples. The zero paired positive
difference and degenerate[0,0] bootstrap interval indicate no observed gain,
not population equivalence. No post-hoc rule or cross-arm witness sharing is
used to change an arm's original result.

Twenty of20 matched/evidence common-fact pairs agree; none are unavailable.
The cohort has15 single-pair inputs and5 multi-pair inputs. All17 UNSAFE run
records revalidated by the terminal auditors concern12 distinct inputs. Thus
at most8 of these20 inputs could still be safe, based on available witnesses;
this is not an achieved certified accuracy or free portfolio score.

**Verdict:** the input98 positive-control success did not translate to positive
complete-request coverage on this new cohort. This frozen configuration is
closed with a negative result. It does not establish cross-architecture
route-changing certificate gains or high-accuracy real-scale strict safety.

## Why "20 TIMEOUT" is not a single causal diagnosis

The official TIMEOUT terminals remain unchanged. Their saved intermediate
records provide separate, more limited observations:

| Evidence available before termination | Requests | Interpretation |
|---|---:|---|
| Local precheck covers all obligations, but has nonpositive bounds | 7 | Those recorded bounds cannot prove the request, even if final packaging/checking were free |
| Local precheck reports missing obligations | 6 | Saved evidence is incomplete; four also have nonpositive bounds |
| No saved complete local precheck | 7 | No complete local aggregate can be claimed |

The13 saved prechecks contain135 necessary obligations:3 positive,70
nonpositive/below-threshold and62 missing. These are saved local-check outputs,
not newly rerun LP proofs and not successful isolated request checks. Their
hashes, pair/property inventories and exact-rational count/sign accounting are
checked in the derived review. No claim that all missing/unchecked bounds would
be positive is justified. A nonpositive lower bound does not establish a
counterexample or the impossibility of a stronger proof.

These mathematical/evidence states must be distinguished from execution stop
locations:16 requests hit the whole-request watchdog (3 during precheck,
9 during package,4 during isolated check);4 returned internal proposal-budget
exhaustion. No whole-request isolated evidence check completed. In particular,
the4 fully packed bundles are not4 positive certificates.

## Cost accounting

| Phase | Completed phase records | Mean completed seconds |
|---|---:|---:|
| Capture, including loading/propagation/support/export | 20 | 53.5666 |
| Propose, including construction/local validation/writes | 16 | 160.5982 |
| Local whole-request precheck | 13 | 66.1612 |
| Portable package | 4 | 17.9495 |
| Isolated check | 0 | unobserved |

These means condition on phase completion; they are neither20-request phase
means nor values that can be summed into a representative successful request.
All requests, including internal exits and watchdog terminations, remain in
the complete-request cost table. The JSON keeps completed durations, internal
noncompleted elapsed times, censored phase windows, and unassigned outer time
separate. Censored windows include orchestration/cleanup and are not exact
operation times. Unobserved propagation-only, serialization-only and native
solver-only costs are **null**, not zero or the residual of a subtraction.

For the four completed packages only, serialized dependency sizes range from
733.03 to851.33MB, deduplicated contents from43.57 to45.78MB, and compressed
bundles from10.31 to11.66MB (decimal units). Content deduplication is already
active. These storage reductions do not eliminate all repeated decoding or
checking, and do not make the four isolated checks complete. Other requests'
missing package sizes remain null; the JSON separately records retained raw
artifact bytes, which must not be compared as though every request completed.

All519 recorded proposal-wrapper calls report PROPOSED; their recorded elapsed
sum is1348.4416s. This is proposal-wrapper time, including conversion and
certificate writing, not exclusively native LP time or a count of globally
optimal weighted MILP solves. Substantial proposal-phase time falls outside
those wrappers (source decoding, construction, exact checks and orchestration
among other work). The current logs cannot identify each exclusive component.
Thus "native solver timeouts explain all20 failures" is unsupported.

The4 internal exits occur around220s, in the proposal phase with return code3.
Source inspection exposes a reserve-boundary behavior: `propose_all` checks
remaining time before source decoding/construction/checking; a later
`grant(..., reserve=80)` can then raise `EvidenceBudgetExpired`. The worker
returns3, and the driver stops before precheck instead of using the remaining
reserve for a partial aggregate. A simulated-clock regression reproduces this
control-flow possibility with more than77 seconds of request work remaining.
This explains a reachable execution path consistent with these logs, not a
stack-trace-level attribution of every exit; the four workers emitted no such
trace. It does not imply that fixing the exit would yield positive certificates.

Completed local prechecks already cost about66s on average, before packaging
and the isolated checker. The80s reserve is not a guarantee that all downstream
work will fit. The proper conclusion is a budget/observability limitation, not
permission to expand the frozen reserve or remove an independent check.

## Archival review and reproducibility

`cohort_analysis/archive.py` performs a **separate read-only re-review** in
`data/moe/results/general_evidence_archive_20260916_v1`, never the original
run directory. It verifies all frozen identities and launches60 per-request
audit processes, repeats available full-model UNSAFE replays, compares the
resulting details and summary exactly with the original final audit, and
hash-binds the cost observations. No proposal, solver search, training or
real-model verification request is added. Seven accounting controls cover
missing/duplicate/wrong properties, state/count/route mutations, exact threshold,
right-censored and unmeasured costs, immutable output and reserve-boundary
behavior. Tests run in the existing act-py312 environment.
The seven new controls plus six existing cohort-supervisor regressions pass
together:13/13 in8.444s. The latter use analytic controls, not the new cohort.

The compact result is `docs/general_evidence_execution_v1_results.json`:
it includes original-record hashes, all60 terminal bindings, fresh review
identities, phase and saved-precheck observations for each of20 evidence
requests, and separate re-review cost. Raw inputs, checkpoint, HZ matrices,
portable bundles and external code remain outside Git. This is an auditable
result index, not a self-contained distribution of the empirical proof data.
The original audit's108.5902s and this new review's elapsed time are outside
request budgets; neither rescues a TIMEOUT. Upstream network-to-HZ, source
binding, guards and route exclusions remain trusted.

Commands, from the project root with the existing environment:

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest cohort_analysis.tests evidence_cohort.tests -v
# One-shot read-only archival review; refuses existing destination:
/data1/Kane/miniconda3/envs/act-py312/bin/python -m cohort_analysis.archive
```

## Disposition and next decision

Do not enlarge this cohort, tune the reserve/gate ranges on it, rerun input98,
or upgrade intermediate evidence into completed results. The completed20-input
study is now observed development data for any later engineering change.
Keep the earlier first-family positive confirmation and all convolutional
failures visible; they answer different questions.

If a new engineering revision is authorized, first address the reserve-boundary
handoff and measure repeated source decoding/construction/check costs using
analytic controls or saved evidence, **without** new bound searches or changes
to proof obligations. Any reuse optimization must retain independent bindings
and exact checks. Freeze and test it separately before deciding whether another
real-request comparison is warranted. There is no evidence here for changing
the25% scheduler, loosening numerical acceptance, increasing gate complexity,
or expecting packaging optimization alone to close these requests.
