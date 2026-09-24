# Scoped proof R1: one real request, construction timeout, no output certificate

Executed once after explicit authorization, from clean
`feat/moe-route-verification` at
`bc1716e706cb0ded60d6c941bc3aadabe3946520`.
No retry, resume, added time, sample substitution or historical positive-bound
reuse. The execution implementation and its 494 frozen source files were
unchanged; only a separate saved-only archive script and documentation were
added in this stage.

## Identities and actual outcome

- [Frozen request](../configs/backend_controls/scoped_proof_execution_r1.json),
  SHA256 `0f2ada11d86ab74f923b42a8a057716aed2fa7b4ee31e26f06f0510168aeadc5`.
- [Execution contract](scoped_proof_supervision_protocol_20260924_r1.md) and
  [historical freeze](scoped_proof_execution_freeze_20260924_r1.md).
- [Independent saved-only audit](scoped_proof_execution_audit_20260924_r1.json).
- [Hash-bound compact archive](scoped_proof_execution_archive_20260924_r1.json).
- Raw root: `data/moe/results/scoped_proof_source4088_20260924_r1`.
  Invocation `e7edb5f07b5b447d9a1cd897b508e5ae`.

One seed0/rank0 CIFAR4088/label7 request, the same frozen checkpoint and stored
float64 center, a NEW exact rational2/255 clipped domain, all28 unordered pairs
and all9 competing classes: **252 required output obligations**. No pair
exclusion or old certificate is used. Whole-request budget300s, CPU2, sampled
parent plus owned worker-group RSS8GiB, with the registered2s terminal reserve
inside the budget.

**Effective terminal: TIMEOUT. Complete checked output proof: false.**
The parent terminated its own construct worker at the hard work deadline;
worker return code-9 is consistent with that action, not evidence of an OS
OOM. Parent and owned worker were absent after completion; no other job was
interrupted.

| Stage | Observed seconds, cleanup included | Outcome |
| --- | ---: | --- |
| Intake, imports, hashing, model/center load, capture/binding, source serialization | 9.078395 | Completed |
| New source/guard/output construction | 289.132778 | TIMEOUT; no bundle published |
| Independent construction check | Not started | No checked conversion receipt |
| LP proposals | Not started | Zero native LP calls / candidate files |
| Exact output-bound aggregation | Not started | Zero checked bounds |

Charged terminal/receipt elapsed: **298.216830s**. Stage sum298.211173s plus
supervisor/publication overhead0.005658s closes that ledger. Supervisor stdout
returned298.217247s including the final cost-ledger write/check; the additional
approximately0.000417s is not hidden in the stage totals. The original console
record was:

```json
{"status":"TIMEOUT","complete_output_positive_proof":false,"seconds":298.21724740229547,"root":"/data1/Kane/MOE/ACT/data/moe/results/scoped_proof_source4088_20260924_r1"}
```

Peak sampled parent+worker RSS **5,656,584,192 bytes (5.268GiB)**, below8GiB.
This was a time cutoff, not a resource-limit verdict. Sampling is not an
instantaneous peak-memory proof.

## What was actually retained and checked

The captured declared source is74,304,755bytes, SHA256
`052dd5ecf0e8118b425da2cd74ce18f6555b085ee2a296ced98823a668133464`.
All12 raw files, totaling74,365,366bytes, are retained and inventoried by hash.
Raw source arrays and checkpoints are not committed to Git.

There is **no** `construction.json`, `source_check.json`, LP candidate or
`evidence_check.json`. None of the252 required output bounds was independently
checked, positive or nonpositive. The audit therefore keeps `checked_bounds`
as `null` (no bound-check phase), while the archive counts zero produced bound
certificates and252 obligations without one. These are not252 completed
negative LP results and not evidence that the LP relaxation cannot certify.

The frozen auditor, run under `python -S`, checks invocation/spec, complete
denominator, terminal/receipt hashes, phase order and timing, total-cost closure
and the nonpositive-proof flag. It returned PASS/zero issues. Because no bound
evidence exists, this audit does **not** rerun propagation, check a complete
conversion, or independently prove any output margin.

The separate archive rechecks all494 bound implementation files, the executed
spec against the frozen config, all252 required identities, recorded operation
events and every raw file hash. Imports of model/array/solver libraries and
external execution are prohibited. Re-reading the archive reproduces the same
result, excluding separately measured administrative audit time.

Receipt controls, using only disposable copies of small saved ledgers, accept
the unchanged receipt and reject all six corruptions: budget, invocation,
cost accounting, terminal hash, receipt hash and a false positive-proof flag.
No real or synthetic LP solve is launched by these controls. The initial
saved-only audit costs0.000530s, and initial archive collection0.038811s; both
are administrative costs after the failed request, not proof-generation work
charged a second budget.

## Bounded interpretation and disposition

The last recorded operation is
`source_guard_output_construction`, entered at9.422783s and still open when the
construct phase ended at298.214518s. The288.791735s interval includes cutoff
and cleanup; it is a censored observation, not the completed operation time.
The read of the source took0.136721s. Construction serialization was never
entered. Finer layer/algebra/copy/allocation costs were not instrumented, so the
saved record cannot select one of them as the unique root cause.

This attempt establishes an **upstream execution-capacity failure before LP
generation**, not a failed solver search, inadequate McCormick precision,
non-certifiable LP, network counterexample, or completed source-proof repair.
It provides no source-complete SAFE, no route-changing witness, and no native
floating-point proof. The historical23 policy gains still have their audited
source-containment limitation; input98 stays sealed.

**Seal this attempt.** Do not rerun, expand time or rows, reuse old matrices,
or tune output relaxation from this outcome. If a new research scope is later
authorized, first investigate source-construction cost with explicit phase/layer
instrumentation and controls, retaining every mathematical obligation and
check. That would be a new engineering study, not continuation of this run.

## Saved-only reproduction

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_proof.audit data/moe/results/scoped_proof_source4088_20260924_r1
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_scoped_proof_execution.py --receipt-controls
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_scoped_proof_execution.py --check
```

The prelaunch freeze check required an absent output directory; it is a
historical prerequisite, **not** a post-execution test to rerun. The explicit
execution entry also refuses the now-existing directory. Do not delete it to
force another attempt.
