# BN repair + checked base closes one old MetaMoE request under HZ policy

Implementation `b348e0254`, frozen execution `754a43ef3`; exactly two old
MNIST0 requests, native then checked. Both use repaired BN and the same
checkpoint, materialized tensor, 2/255, 300 s outer/30 s expert budgets,
3 s base cap, 19 ordered properties and unchanged numerical acceptance.
No deduplication, more time, relaxation change, new image, or retry.

[Independent saved audit](metamoe_checked_base_archive_20260923_r1.json):
PASS, 0 issues; identical expert matrix/properties; all 19 obligations present.
[Saved-only diagnosis](metamoe_checked_base_diagnosis_20260923_r1.json) adds
no solving or forward passes. Raw artifacts stay outside Git under
`/data1/Kane/MOE/baseline_runs/metamoe_checked_base_control_20260923_r1`.

| Same corrected request | Native base | Checked current base |
| --- | ---: | ---: |
| Full request result | UNKNOWN | POSITIVE / HZ_POLICY_ACCEPTED |
| Base feasibility | UNKNOWN at cap | feasible, full matrix checked |
| Base observed call cost | 3.004700 s | 0.036863 s |
| Output violation queries excluded | 19/19 | 19/19 |
| Expert result | UNKNOWN: nonvacuity unproved | CERTIFIED under frozen policy |
| Expert elapsed before publication | 4.703765 s | 1.741840 s |
| Native calls (base + properties) | 20 | 19 |
| Full charged request cost | 135.027110 s | 130.967952 s |

Both completed router coverage with candidate `[1]`, excluded `[0]`, no
unresolved candidates. Selected-score nonzero enclosure in both arms is
`[3.345338179462453, 4.30685851511294]` (recorded `fast_fallback` support
enclosure, not a completed optimal support claim). Batch cost 267.035807 s;
charged requests 265.995062 s. Postflight inventory/hash and separate audit
are not hidden in an alleged solver speedup; see receipts and cost manifest.

## What this resolves

The correct dataflow is BN input -> SCALE -> BIAS, rather than BIAS bypassing
SCALE. Prior [same-object point control](metamoe_bn_corrected_result_20260923_r1.md)
checks the source/IR/HZ discrepancy after that repair, with a fresh matrix.
Historical defective outputs remain sealed, not relabelled.

On the repaired matrix, BOTH arms exclude all output violations. The native
arm's *only remaining expert gate* is the base feasibility/nonempty-set check.
A current assignment independently checked against all stored rows, variable
bounds and integrality supplies precisely that missing fact. It does not
prove safety by itself; all 19 output queries still run. Under this frozen
policy the full request is now positive. Thus this instance supplies no
remaining output-relaxation deficit that warrants a tighter representation.

## Remaining costs, not imaginary relaxation gaps

| Observed phase | Native | Checked |
| --- | ---: | ---: |
| Candidate/router analysis | 92.981225 s | 92.262044 s |
| Selected-score nonzero support | 30.053963 s | 30.049217 s |
| All property query calls | 1.672570 s | 1.679424 s |

Router/candidate plus score support account for about 93.4% of the checked
request's charged cost. The source repair did not optimize these phases.
Rows 0..9 remain byte-identical expanded queries, but all 19 queries now cost
only about 1.68 s. Neither deduplication nor further expert relaxation work
is the principal observed opportunity here. A next separately scoped control
could study current-request checked feasibility in the router/support path,
including each query's additional constraints and native-call deadline; no
such change or follow-up experiment is included or queued in this result.
Do not bypass guard checks or reuse a base point as a margin certificate.

## Claim limits

- One old route-STABLE request, not a new route-changing certificate/cohort or
  evidence of population/cross-model speedup. Fixed native-first order.
- 90 focused ACT tests and 51 overlapping pinned-environment tests passed.
  Tests include native differentials, contamination, invalid/partial points,
  missing obligations, timeout/publication, exception and fallback controls.
- HZ_POLICY_ACCEPTED, NOT source-complete or independently proved native
  infeasibility. Full stored-matrix feasibility uses the existing float policy;
  the source/IR/HZ point check is not an all-domain lowering proof.
- The original BN-containing source-claims need identity-specific review.
  Old experiments are not repaired retrospectively, and the finding should
  not be indiscriminately extended to unrelated BN-free MLP/CROWN paths.
- Fast base remains OPT-IN. Default solver, frozen old manifests and all
  proof/witness/numerical gates remain unchanged.

Reproduction commands (after checking the frozen environment and sources):
the run command refuses an existing result root; audit/analysis commands read
the saved data without optimization. Use new output filenames for re-audits:

```
python scripts/audit_metamoe_checked_base_control.py --output /data1/Kane/MOE/new-audit.json
python scripts/analyze_metamoe_checked_base_control.py --output /data1/Kane/MOE/new-analysis.json
```
