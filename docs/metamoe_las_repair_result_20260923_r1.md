# Scoped author-backend lAs repair: compatibility passes, no new certificate

Original stopped20 batch `e2235df73` stays sealed. This is a separate,
user-authorized repair/control, not a retry merged into that comparison.
External author/backend repositories and dependencies remain unchanged;
the repair is an explicit, hash-bound runtime compatibility wrapper in ACT.

## Reproduced cause, not an assertion bypass

An observation-only request, frozen at `6057f90a2`, reproduced MNIST7 ERROR
in **253.254768 s** (batch253.579077s). Independent saved audit PASS0:
[observation archive](metamoe_las_observe_archive_20260923_r1.json).
No mathematical/backend option changed in that observation.

After initial verification pruned the completed properties, the remaining C
was `[-1,1,0,...,0]`, only the router-dominance obligation. Initial domain
storage retained9activation entries. Five expert entries (`/82`, `/88`,
`/94`, `/100`, `/input-40`) were exactly zero. The next backward computation
represented them by absent/None entries and returned only4router entries.
The domain insertion's original9!=4lAs assertion then failed. The observation
records the keys, shapes, zero counts and C; it does not independently prove
the original backend's bounds or source conversion.

[Repair implementation](../scripts/metamoe_las_repair.py) binds the initial
full schema, exact C and bound-graph identity. It may restore an explicit zero
ONLY when the entry was zero initially, is currently None, and its node is
not an ancestor of any concat block with a nonzero property coefficient.
It uses an exact zero test (including tiny nonzero refusal) and unions all
property rows. Shapes/dtype/node mappings must agree. Active missing entries,
unknown keys, changed graphs/C and unsupported layouts still fail closed.

This reconstructs irrelevant **branching metadata**, not a new safety bound.
All native nonmissing lAs and lower/upper bound tensors are untouched. The
original domain insertion and assertion still run. No properties, constraints,
subdomains or results are dropped or promoted. The scope is the frozen
float64, axis1 final-concat adapter/kfsb execution, not a universal backend fix.

## Controls and real old-input outcome

17controls passed in both ACT and pinned author environments, including
no-op/native object identity, justified restoration, active/nonzero/shape/
graph/C/key refusals, independently checked log mutations, exception restoration,
outer cutoff, partial/late-result precedence and cost recording. The unchanged
observed failure is preserved as a negative control.

Real repair control implementation `1581e3cd9`, freeze/execution `8d782b09d`.
Config SHA256 `6fa42920afd36b90fd88f84dd52b8a7de0b1f578b7d3e8765822d4dc8f23fe2b`.
Exactly2old author requests; same physical inputs/checkpoint/normalized2/255,
19global output rows plus route/nonzero obligations, CPUfloat64/two threads,
300s/8GiB each. No retries or option search.

| Old input | Terminal | Charged seconds | Restorations / original domain insertions |
|---|---|---:|---:|
| MNIST1 | BACKEND_POSITIVE, raw safe-incomplete | 7.059411 | 0 / 0 |
| MNIST7 | outer TIMEOUT | 300.063271 | 7 / 7 |

MNIST1 retains its numerical sufficient filter with no restoration invoked.
MNIST7 now passes the original failed insertion repeatedly and continues BaB,
then is killed at the registered outer deadline. It is **not** a new positive
or full-model UNSAFE. Compatibility gate passes because the registered condition
requires a checked restoration AND actual original insertion without ERROR;
it never required a positive certificate. This is not merely absence of a
traceback, and not evidence of general runtime improvement.

All attempted costs total **307.122682 s**; batch through final summary
**307.812343 s** includes parent inventory/postflight. The 0.063s over the
outer ceiling is observed termination/cleanup, not an extended solver grant.
Peak sampled group RSS:2,236,211,200bytes (MNIST1) and5,467,676,672bytes(MNIST7).
Separate original-model replay ledger is empty (zero accepted UNSAFE),1.470343s;
saved audit0.359359s. No new source or solver proof is implied by that replay.

## Independent audit and object/option equivalence

[Saved audit](metamoe_las_repair_control_archive_20260923_r1.json):PASS0,
`repair_control_gate=true`. It reconstructs structural reachability without
calling the repair's helper and checks every logged restoration, schema,
command binding, terminal precedence, input/property mapping and costs.
A second audit with production repair installation/reachability disabled is
identical except audit clock.

Separate saved checks also reconstruct the timeout request's VNNLIB, not
just the successful request's: all21violation disjuncts and endpoints are
byte-identical to their old input requests. YAML backend options are identical
after removing only output/input artifact paths, new request-config identity
and measured remaining time. No alpha, BaB, branching, tolerance or iteration
setting was tuned. The new launcher is explicitly disclosed, not called a
literal unchanged author-tool run.

Raw control root:
`/data1/Kane/MOE/baseline_runs/metamoe_las_control_20260923_r1`.
38files/1,285,534bytes retained and hash-inventoried; no checkpoints, raw tensors
or external repositories committed. Old crash and new timeout both retained.

## Next boundary: freeze only

Freeze a newfull20-call two-arm execution on the **same observed10inputs**,
not new selection/holdout. ACT unchanged; author uses this disclosed wrapper.
Same300s, all obligations, numerical grades and ERROR fail-stop. Do not splice
the old14normal returns into a mixed-version result. Do not launch the full
batch as part of this repair stage. Neither strict source-complete guarantees
nor performance/coverage claims are upgraded.
