# Real checked-frontier comparison — both budget stops, no new complete proof

The user explicitly authorized exactly the two frozen requests and saved-only
audit/archive, without expansion or added time. Execution started clean on
`feat/moe-route-verification` at
`de874d06ea3bf26094578f0961fdbb32b4967cc2`. The frozen config remains
`b640a366fdce1ec9a746956eb4532560b5a9907434c5b2129d87f17b6d56b012`.
No retries, alternative inputs, solver options, numerical gates or source-code
changes were made. The separately added archive script is not execution code.

## Budgeted outcome

Same seed0/rank2/CIFAR4098,label9, exact2/255, original252 pair/property duties
per arm,300s/2threads/sampled8GiB each. Fixed order exhaustive then frontier.

| Measurement | Original exhaustive | Checked frontier |
| --- | ---: | ---: |
| Terminal | TIMEOUT | TIMEOUT |
| Full returned-call cost (s) | 298.1853 | 298.0495 |
| Intake process (s) | 8.5335 | 8.6283 |
| Complete construction process / censored (s) | 289.6466 / cutoff | Not entered |
| Router proposal process (s) | Not applicable | 104.3811 |
| Independent router check / censored (s) | Not applicable | 185.0334 / cutoff |
| Sampled parent+worker peak RSS (GiB) | 5.4894 | 0.8705 |
| Complete expert/output construction published | No | No |
| Native output LP calls / output candidates | 0 / 0 | 0 / 0 |
| Checked output bounds | 0 / 252 | 0 / 252 |
| Budgeted complete positive requests | 0 | 0 |

Both work deadlines were298s, reserving2s INSIDE300 for terminal publication.
Small polling/owned-process cleanup overruns of the work deadline did not
exceed300s or promote a late proof. Neither stop was caused by the8GiB RSS gate.
The total returned-call cost is596.2348s; admission/setup and subsequent offline
audit are separate. The frontier's lower observed RSS is NOT equal-work memory
savings: it never reached expert/pair construction. Both times are censored;
no complete-request speedup follows from their small difference.

Captured source bytes are identical across arms, SHA-256:
`d55b1e73f00889d5475a2cfbc09305d3a2f0fa2bb83d72b995412530beed5e9c`.
This is source-artifact identity, not a full source derivation. Neither arm
published a complete output construction; real retained-matrix equality could
not be checked, and is not inferred from synthetic controls.

## What was actually saved

The exhaustive trace enters `source_guard_output_construction` at8.8424s and
never exits before cutoff. This frozen worker does not record finer internal
layer/pair milestones. Do not borrow the previous4096 parsing trace as if it
described this run or infer a precise layer where this request stopped.

The frontier arm saves its router prefix and all56 prescribed ordered-margin
candidates, plus the source/prefix/run-bound completion receipt. Within the
proposal process, router construction costs15.7921s and the fixed final-affine
dual computation87.7849s; these are suboperations, not extra costs to add to
the104.3811s process total. Router-prefix serialization costs0.2032s.

The independent router phase receives/binds files in0.6940s and enters
`independent_router_source_and_bounds` at113.7323s. It does not publish a
`route_check.json` before cutoff. Consequently no route exclusion is accepted
by the online pipeline, no expert is omitted on that authority, no lazy expert
construction begins and no output query is launched. Candidate files and a
complete proposal manifest are NOT equivalent to completed proof checking.

### Saved-only exact route check: useful exclusions, not an online certificate

The subsequent frozen independent auditor checks all56 saved router candidates
against the stored source/prefix and finds **24 of28 pairs excludable**. Thus a
checked router frontier can, on this real source object, retain4 pairs rather
than28 (36 rather than252 output duties after logical route discharge). This is
an actual partial-proof finding, not a claim that every candidate bound is
positive. Retained does not mean reachable;4 retained pairs do not establish
route change. No expert/output matrix or bound for those36 duties was generated.

The first saved-only batch audit costs186.5758s, including186.2262s on the
frontier arm, outside both request budgets. It is never backdated into the run.
The raw run has no checked-route receipt and the archive's online evidence
fields correctly record zero accepted exclusions and252 duties without online
positive evidence. The independent audit's `checked_excluded_pairs:24` refers
to **later mathematical rechecking**, not the stopped execution. These two
counts measure different events and must not be conflated.

In particular, the router check being near its cutoff does not make this an
almost-complete MoE proof: all retained expert propagation, guarded output LP
construction, source checks, proposals and full aggregation remain to be done.

## Interpretation and stopping decision

This comparison adds **zero complete certificates**. It does not establish a
real-model speedup, external-tool advantage, high-accuracy/cross-family result,
route-changing guarantee, or repaired historical23 source-gap gains.

The budgeted bottleneck moved from exhaustive source construction to the added
router evidence preparation/checking. Offline24-pair exclusion supports this
mechanism's logical usefulness on this object, but does not establish practical
complete-request benefit. Lazy retained construction was never reached, so its
real cost/benefit was not measured. No output LP was queried:
these logs cannot diagnose output relaxation precision, solver failure, model
unsafety or impossibility of a certificate. An offline check of saved candidates
does not retroactively turn the timed request into success.

Seal both calls and input4098 under this protocol. Inputs98/4088/4096 remain
sealed. No additional query, time, sample, tuning or relaxed acceptance follows.
Any further research would need a separate controlled decision based on these
saved costs, not a retry of this failed request.

## Audit and archive

- [Frozen saved-only batch audit](frontier_proof_execution_audit_20260924_r1.json):
  PASS/0 issues, both original terminals, full measured costs, same declared
  source,56 router candidates rechecked and24 offline pair exclusions. No
  completed output proof, no real retained-matrix comparison, no new solves.
- [Hash-bound archive](frontier_proof_execution_archive_20260924_r1.json):
  all89 raw files totaling219,641,063bytes retained locally, plus phase costs,
  completed/censored operation observations, evidence absence and original
  denominators. All538 frozen source/protocol bindings remain unchanged.
- [Archive/check script](../scripts/archive_frontier_proof_comparison.py):
  fresh `python -S`, forbidden model/solver/producer imports and process/network
  execution. Twelve actual receipt corruptions and four trace corruptions are
  rejected; two unchanged actual cost records and open/partial trace controls
  pass. These controls concern identity/cost accounting, not new output proofs.

The initial archive costs186.6572s including the batch audit; these overlapping
costs must not be added. A second fresh saved-only reproduction passed, comparing
all deterministic archive/audit fields (excluding review-time measurements).
Saved-only reproduction is separately performed and
does not alter any run file or issue a new optimization query. No checkpoint,
training data, model inference or solver is loaded. Software audit is not an
independent human review or a native floating-execution proof.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_frontier_proof_comparison.py --check
```

The pre-execution absence gate is no longer applicable. Use the saved-only
archive command above. Do not run the execution command again.
