# Complete-proof frontier: stop the timing line, preserve the research question

Saved-only synthesis at starting HEAD `6f35863b327b6bc74b970fe456d31f0afb00a37c`.
No experiment, bound query, propagation, model load or dependency change.
The [machine-readable ledger](proof_closure_20260925_r1.json) is reconstructed
from five pinned committed archives by `scripts/summarize_proof_closure.py`.
This is **archived accounting**, not another independent proof of their bounds.

## What the real requests reached

These are seven engineering calls on four previously observed inputs, not seven
independent examples, a fresh cohort, or a common-config success-rate estimate.
Every call retains all 252 original pair/property obligations and its 300 s cap.

| Input / method | Terminal | Recorded ledger time (s) | Online route-excluded duties | Unclosed duties | Last reached phase |
|---|---|---:|---:|---:|---|
| 4088 / exhaustive | TIMEOUT | 298.217 | 0 | 252 | construction |
| 4096 / uncached | TIMEOUT | 298.257 | 0 | 252 | construction |
| 4096 / cached | TIMEOUT | 298.377 | 0 | 252 | construction |
| 4098 / exhaustive | TIMEOUT | 298.185 | 0 | 252 | construction |
| 4098 / checked frontier | TIMEOUT | 298.049 | 0 | 252 | router check |
| 4099 / pairwise | TIMEOUT | 298.050 | 0 | 252 | router check |
| 4099 / shared residual | RESOURCE_LIMIT | 287.555 | 225 | 27 | construction/publication |

All seven have **zero output LP calls, zero output candidates, zero checked
output bounds, and zero complete source-to-output positive results**. No
retained expert/output construction was published. A source capture is not an
independently checked expert enclosure.

Later offline checks can exclude 24 pairs for 4098/frontier and 25 pairs for
4099/pairwise. Those are **not online discharges**: both executions stopped
before the checked-route receipt. Only 4099/shared accepted the 25 exclusions
within its request. The remaining `{4,7}`, `{5,7}`, `{6,7}` are potential
routes; retention is not a checked route-flip witness.

Costs retain the historical ledger convention: its own final write and later
audit are excluded; where present, the enclosing caller clock is also recorded
separately in the JSON. No missing downstream phase is recorded as zero time.
The later offline audit is not added as free online work or a budget extension.

## What the new four-call control does and does not add

Fresh synthetic upstream generation, propagation, construction, publication and
checking completed for two objects in two modes. Small: 12 checked retained
constructions; medium: 225 excluded duties plus 27 checked retained
constructions. **Neither is an output lower-bound certificate.** Source and
construction bytes match across arms, but these objects are not the real
4099 source and cannot supply its missing matrices or proofs.

Whole direct-to-readonly costs were 0.240712→0.221606 s and
0.406803→0.438045 s. One observation per arm gives one win and one loss, not a
reliable real-request speed forecast. Keep readonly optional/default-off and
the finite timing study closed.

The last real constructor returned in memory at 282.947834 s, leaving only
15.052166 s before the 298 s work cutoff. Publication, complete source
checking, output proposal and complete bound aggregation had not finished or
started. This arithmetic is **not** a lower bound on their future cost, nor a
proof that a better method cannot close them. It does show why fixing the
serialization peak alone was never sufficient evidence to promise success.

## Continue / stop decision

**Stop automatic performance reruns and further cache microbenchmarks.** This
analysis identifies no new isolated, data-supported intervention that warrants
another real-request freeze. Do not advance to the next dataset index merely
because a local control now passes. Keep inputs98/4088/4096/4098/4099 sealed;
no new sample, budget, numerical gate, relaxation or default is selected here.

The scientific frontier is still a **same-source complete output proof**. The
current failure is before output evidence generation. It is not evidence that
the output LP is too loose, that more solver time is necessary/sufficient, or
that the network is unsafe. Router checking is a demonstrated local advance;
real output utility is unestablished. Faster synthetic source checking does
not repair the historical 23 main-table source gaps.

Continue the already-supported manuscript/review work: incorporate this table,
retain the adverse external comparisons and conditional guarantees, and give
an uninvolved human reviewer the source/obligation boundary to challenge. This
turn does not contact a reviewer or claim that review has happened. PI-managed
access/licensing and separately scoped clean-environment empirical reproduction
remain open; the relocated accounting kit is not that reproduction.

If new research is authorized, first state **one concrete complete-obligation
mechanism**, why it addresses the measured pre-output bottleneck, what invariants
must remain (shared factors, all tie-legal obligations, exact source checks),
and what finite controls could falsify it. Only then freeze a separate real
request with a single total budget. No such protocol or run is silently
created by this decision, and no ISSTA/CCF-A acceptance guarantee follows.

## Reproduction

With the repository's existing Python (standard library only):

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/summarize_proof_closure.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/test_proof_closure.py
```

These commands read compact committed archives, not raw checkpoints, datasets,
large saved sources or solver state. Missing/changed archives, omitted calls,
offline-as-online substitutions, invented bounds, inconsistent coverage and
costs are rejected. Pinning bytes validates identity, not mathematical truth.

Underlying result explanations:
[4088](scoped_proof_execution_result_20260924_r1.md),
[4096](scoped_parse_proof_execution_result_20260924_r1.md),
[4098](frontier_proof_execution_result_20260924_r1.md),
[4099](residual_proof_execution_result_20260925_r1.md),
[four upstream controls](readonly_upstream_result_20260925_r1.md).
