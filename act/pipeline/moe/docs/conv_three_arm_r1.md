# Frozen convolutional-family three-arm experiment R1

Status: **six smoke calls completed and independently re-audited; structural
audit PASS, zero issues; smoke gate FAIL**. The 90 full requests have NOT run.

## Completed smoke R1

Execution commit `c192bca4d3c3161abe3ae816471894be24e3d908`, unchanged epoch89,
inputs 0/1, radius2/255 and method configurations. Compact archive:
`../results/conv_three_arm_smoke_review_20260915_r1.json`; raw directory below.

| Input | Adaptive | Matched monolithic | ACT-fronted plain CROWN |
|---|---|---|---|
| 0 | UNSAFE, 113.04s | TIMEOUT, 300.07s | UNKNOWN, 4.43s |
| 1 | TIMEOUT, 300.06s | TIMEOUT, 300.05s | UNKNOWN, 4.32s |

Six of six terminal slots are present; no ERROR or unattempted slot. All three
outer timeouts retain their partial evidence. Both inputs have two exact legal
pairs. All two adaptive/monolithic common-fact comparisons agree, including
the requests killed before final packages. One complete HZ package and two
complete external records pass structural checks; the one UNSAFE replays on
the full dynamic model in the represented box. Both external calls finish
all18 margins (two pairs each), but do not establish a positive numerical
filter. No SAFE certificate or numerical positive was obtained.

The independent second audit exactly reproduces the automatic audit. Total
observed request time is1021.96s (17.03min), including timeout cleanup and
terminal submission; audit and resource wait are separate. This is a
conformance smoke, not a powered performance comparison.
An additional independently timed re-audit takes1.80s and returns the same
summary. Automatic-audit wall time was not separately captured; the archive
records it as null, not zero. This does not change any request's charged cost.

A post-archive repeat of the original30 tests exposed a test-only /proc race:
the killed child's entry vanished between exists() and read(). Preserve the
frozen wrappers/tests and the recorded failed repeat in
`../results/conv_smoke_posttest_review_20260915_r1.json`. The subsequent
`scripts.test_conv_three_arm_lifecycle` suite inherits the same controls but
handles ESRCH/ENOENT and brief kernel termination latency, and tests that
unrelated IO errors still fail. Use that module for subsequent test runs;
it changes neither the executor nor the six archived requests.

**The registered smoke gate fails** because monolithic produced zero complete
non-error records across the two inputs. A complete terminal ledger and audit
PASS do not override this requirement. The supervisor returns nonzero and
`STOPPED_REVIEW_REQUIRED`; no full90 is started or queued. Do not extend time,
replace inputs, retune the model or relabel snapshots as complete packages.

Last preserved timeout stages: monolithic F0 for both monolithic calls; Tier2
F0 for adaptive input1. These are observed stop positions, not proof of a
unique solver/encoding cause or evidence that longer runs would prove SAFE.
Further performance/implementation work requires its own bounded follow-up;
the failed R1 remains sealed and the original full cohort remains unqueried.

## Smoke execution wrapper (2026-09-15)

The new `scripts/run_conv_three_arm.py` is deliberately smoke-only: no full,
pipeline or resume option. It runs exactly the six frozen jobs, acquires the
existing shared timing lock, checks a clean pushed feature checkout, resource
availability and every frozen algorithm identity, and binds four wrapper source
hashes in `runtime.json` before launching any query. ACT Python files, selected
checkpoint, tensors, three arms and thresholds are unchanged.

The parent owns each process group, enforces the 300-second request cap through
terminal publication, preserves late packages without accepting them, retains
common-fact/route snapshots, and writes a separate unattempted roster after an
error. Resource waits are recorded separately. SIGTERM/interrupt cleanup kills
only the owned request group. A machine-wide SIGKILL/power failure cannot execute
Python cleanup; missing run terminals must never be interpreted as completion.

`scripts/audit_conv_three_arm.py` runs in a separate interpreter. It checks the
frozen requests and E4 route universe, HZ package structure and full-model
witness replay, cross-environment input/property/backend identities, all nine
external margins per reachable pair, timeout non-promotion, missingness and
actual common-fact agreement. An intact failed run can have audit PASS while
its smoke gate is FAIL. These are distinct statuses; neither is independent
re-proving of solver bounds. All six terminals and at least one complete
non-error record per arm are required; no positive count is required.

Validation before launch: 30 focused tests pass, including real child-process
deadline cleanup and mutations of route coverage, properties, dtype and proof
level. Separate clean-only freeze reconstruction still passes unchanged. Its
historical `NOT_YET_INTEGRATED` field describes the original selection event,
not the newly integrated wrappers, and is not rewritten.

Authorized command, from the clean committed checkout using act-py312:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -m scripts.run_conv_three_arm
```

New raw root: `data/moe/results/conv_three_arm_smoke_20260915_r1`.
After automatic audit, perform a separate read-only re-audit, archive compact
results and update this document. Do not automatically run the 90 full calls.

Authoritative machine-readable records:

- `../configs/conv_three_arm_protocol_r1.json`: design and acceptance rules.
- `../configs/conv_three_arm_selection_r1.json`: checkpoint/config/source hashes,
  exact input identities, exclusions, smoke and full job lists.
- `../results/conv_three_arm_freeze_review_20260915_r1.json`: separate-process
  clean-only reconstruction, PASS with zero issues.

## Subject and selection

Use the validation-selected seed 17 convolutional checkpoint, epoch 89,
SHA-256 `f5781a792f844a68de941f1a6b314d0e627ad5dd30e262088e6c1864d6bc5289`.
It has four experts and ten classes, not the former eight-expert topology.
The 67.06% training-stage test accuracy does not select a verification checkpoint.
No additional training, checkpoint sweep or architecture change is part of R1.

The main cohort consists of 30 clean-correct CIFAR-10 test inputs, selected in
ascending dataset order from index 0 after excluding the recorded historical
verification-index union. The frozen inventory contains 5,416 hash-bound
artifacts and 832 distinct indices, including failed/timeout terminals and
later external/proof requests. It identifies **recorded prior endpoints**, not
a claim that all forms of earlier image inspection are absent. Whole-test
training telemetry is explicitly disclosed. Selection performs only CPU/float64
batch-one clean forwards; no route multiplicity, bounds or solver outcomes are
selection predicates. No verification result on the new model was inspected.

Full indices: 4, 8, 16, 17, 20, 24, 26, 39, 42, 47, 48, 51, 56, 62, 63, 66,
69, 75, 78, 83, 84, 89, 95, 96, 98, 101, 106, 107, 111, 113.
Smoke indices: 0 and 1, the first two clean-correct indices in the exclusion
union. They are separate from the full cohort and do not enter its statistics.
They are previously observed images, not previously verified conv requests.

Float64 is initialized **before ToTensor**, matching the verification contract;
this differs from training's float32 preprocessing. The same materialized
center/lower/upper tensors must be loaded by every arm and both environments.
Radius is exactly the registered Python value `2/255`, with clipping to [0,1].
No boundary-adaptive radius or route-instability prefilter is used. The six
possible unordered pairs are a universe, not a claim that all are feasible.
Each arm must independently establish complete coverage of all tie-legal
reachable pairs and all nine classification margins. Incomplete enumeration
cannot establish a positive complete result.

## Arms and scope of the comparison

| Arm | Frozen path | Evidence level |
|---|---|---|
| Adaptive | `route_complexity_reuse_v1.json`, scoped reuse and 25% multi-pair Tier 1 allocation | HZ policy accepted |
| Monolithic | `monolithic_matched_reuse_v1.json`, same independently computed common facts | HZ policy accepted |
| External | ACT route frontend plus whole-box variable-weight static pair, plain CROWN, matrix convolutions | Numerical filter only |

The two HZ configurations differ only in `comparison_method`; neither receives
the other's facts or a historical route census. No extra support tightening or
new gate partition is introduced. The monolithic arm is the matched-reuse
reference, **not** the historical legacy configuration and not a claim to cover
every possible strong monolithic implementation. This keeps one interpretable
internal comparison while including the executable external path in three arms.

The external arm retains the router, both experts, shared input and actual
variable selected-softmax weights. It verifies each static pair on the full box,
a sufficient condition stronger than the guarded obligation. It is neither
full alpha-beta-CROWN/BaB nor independent direct dynamic-model verification.
External commits and Python path are bound in the selection manifest. ACT uses
the existing act-py312 environment, Python 3.12.12, Torch 2.9.1+cu128,
SciPy 1.16.3 and HiGHS 1.14.0; no installation or upgrade is authorized.
The external worker already checks Python 3.11.16 and Torch 2.11.0+cu130.
CPU/float64 and one numerical-library/solver thread apply to all arms.

## Execution and failure contract

The 30 input blocks each run all three methods; cyclic method order balances
each of the three positions ten times per arm. There is only one timed request
at a time. Each complete request has a 300-second outer cap, including process
startup, loading, route analysis, common facts, support, graph building,
cross-environment handoff, all property calls and terminal submission.
Input materialization is recorded separately and equally excluded; subsequent
loading is charged. Audits are separately timed. The full worker ceiling is
27,000 seconds (7.5 hours); six smoke calls add at most 1,800 seconds. These are
caps, not runtime forecasts. Resource waits/audits are additional and reported.

Before timed work require at least 16 GiB available RAM, 5 GiB free disk and
load per logical CPU at most .5; wait at most 24 hours without killing others.
The orchestrator must kill the owned process group on deadline, retain the
terminal and any already-written common-fact snapshot, and never promote a
late package to success. Execution errors stop the run; timeouts and ordinary
UNKNOWN are scientific terminal outcomes and do not trigger sample replacement.
No resume, overwrite, time extension or effect-dependent expansion is allowed.
The separately authorized single-request timing follow-up is completed and
documented in `docs/conv_f0_timing_r1.md` at the repository root; its retained
TIMEOUT and measured stale native-budget allocation do not change this smoke
gate or authorize the full cohort.
The subsequent opt-in V2 budget/partial-terminal implementation and toy
controls are documented in repository-root `docs/budget_contract_v2.md`.
Real-model V2 smoke is still unexecuted; it needs a new frozen outer protocol.
Neither those controls nor the worker adapter replace the R1 conformance gate.

Six smoke calls must be independently structurally audited. Each arm must
produce at least one complete non-error package/record across the two inputs;
there is no positive-count or improvement requirement. If this gate fails,
retain the attempt and stop for an implementation review; do not replace smoke
inputs or silently enlarge the budget. Full results always have denominator 30
per arm. An interrupted/error-stopped run reports unattempted slots separately
and cannot be called a completed 30-input experiment.

## Interpretation and planned audit

The primary metric is the paired SAFE-indicator difference, adaptive minus
matched monolithic, across 30 inputs. Report gained/lost SAFE and gained/lost
solved independently. Against CROWN, compare complete-request positive-result
sets with evidence levels visible; do not compare HZ SAFE+UNSAFE with external
positives as though both count certificates. Negative bounds are not witnesses.
Any UNSAFE requires domain-valid replay on the full dynamic model, regardless
of which pair or relaxation suggested it.

Use 10,000 input-block bootstrap replicates, seed 20260915, unadjusted descriptive
2.5/97.5-percentile intervals. A degenerate [0,0] interval is not a population
equivalence result. Show all five terminal categories, all-terminal costs and
single/multiple/unresolved-route strata; strata are explanatory, never selected
for the primary headline. A successful structural audit is not independent
re-proving of all bounds. One model cannot establish broad cross-architecture
or high-accuracy strict-certification superiority. Do not pool this cohort with
bal010, the historical 100-input confirmation or old AdvMoE controls.

## Historical registration: delivered interface and remaining launch work

`conv_three_arm_worker.py` accepts a materialized ordinary request, checks the
E4/C10 contract and dispatches adaptive/monolithic through the common production
API or CROWN through the pinned external worker. It does not depend on the
absent `dataset` checkpoint field. Mocked dispatch and topology rejection tests
pass; no full-size trained-model solver call has run in this stage.

The old batch scripts hard-code three MLP models and are **not** this experiment's
runner. A dedicated outer orchestrator and final three-arm auditor remain to be
integrated/tested, with the execution wrapper hashes frozen before smoke. New
orchestration-only wrappers can live under `scripts/` and bind their own source
identity separately from the frozen ACT Python-file inventory. They must not
modify these model/input/method decisions. The existing ACT
source files are hash-bound; any proposed algorithm change needs a new version,
not silent replacement. The current manifest explicitly records
`NOT_YET_INTEGRATED` and `execution_started=false`.

Recheck the frozen selection in a separate process (clean forwards only):

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.moe.freeze_conv_three_arm audit
```

This turn freezes the experiment; it does not launch the smoke/full pipeline,
open an old holdout or start request-level rational proof queries.
