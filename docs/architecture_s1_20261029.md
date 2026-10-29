# S1: capacity-matched architecture screen (revision 8)

Stage: literature review and **training freeze**, before dense outcomes.
The [five-work review](architecture_verifiability_literature_20261029.md) found
close precedents but no identical combined study; no novelty stop was triggered.
This is ordinary-training S1, not the S2 certified-training experiment.

## Frozen models and training

Existing L1E8/L2E8, seeds 0/1/2, come from the original
[training record](multilayer_training_records_20261022.json). No retraining,
best-seed selection or checkpoint substitution. L1E8 has router
3072→128→8 and eight experts 3072→256→128→10, ReLU hidden layers.
The router has 394,376 parameters; each expert has 820,874. Logical active
parameters include the whole router and two whole experts. Biases count.

| Dense group | Matched L1 target | Dense widths | Actual parameters | Relative mismatch |
| --- | ---: | --- | ---: | ---: |
| Total | 6,961,368 | 3072→1758→879→10 | 6,957,295 | −0.05851% |
| Active | 2,036,124 | 3072→602→301→10 | 2,034,469 | −0.08128% |

Rule fixed before results: preserve two ReLU layers and the expert's 2:1 hidden
width ratio; choose the integer width with minimum absolute parameter-count
error, lower width on ties. No unused weights/padding. This is **approximately**
matched capacity, not exact equality. L2E8 is a depth/context extension, not a
capacity-matched comparison to these L1 controls. Active parameters describe
conditional graph work, not whole training/verification cost: existing training
materializes every expert before gather.

Six new dense models, group order total then active, seed order 0,1,2. Shared
CIFAR-10 45k/5k split, seed0, exact original split hash; ToTensor only, no
normalization/augmentation. 50 epochs, batch256, AdamW lr0.001/wd0.0001, no
scheduler. CE only; the MoE's original 0.1 Switch balance term is architecture
specific and has no dense counterpart. This objective difference is disclosed,
not described as an isolated architectural intervention with identical losses.
Earliest highest validation accuracy selects the checkpoint; held-out test
accuracy evaluated once after selection. Retain every seed and trajectory.
No verification outcome selects training parameters or checkpoints.

CPU float32 deterministic training, single native/interop/BLAS thread and zero
data-loader workers. Original MoE training used four native threads; record this
execution difference, never compare historical training wall times as a paired
speed claim. Current queue uses one nice10 worker because host load already
exceeds its 20 allowed CPUs. User's ceiling is eight workers, not a requirement
to occupy eight immediately. No GPU while another task uses it. Each managed
process has 8GiB RLIMIT_AS. Training duration/weights are separately charged;
300 seconds remains the complete verification request limit, not total training.

Source/data/seed/checkpoint identities and control receipts are bound in
`architecture_s1_training_freeze_20261029.json`; real training runs only from an
immutable committed copy. Resume checkpoints include optimizer, best state,
model/shuffle RNG and split identity. An interrupted attempt remains recorded;
recovery must retain completed epochs, not quietly restart a seed.

Necessary new-code control, one supervised batch≤300s: actual L1 parameter
count, both full dense shapes/outputs, finite effective training step, exact
save/load output and optimizer-next-step equality, shuffle recovery, wrong
parameter identity rejection, earliest validation tie. It uses synthetic inputs,
no CIFAR training and no verification queries. Existing supervision is reused.

## Evaluation design fixed now; numerical runner freeze follows controls

Twelve models (six old MoE and six dense), ε=2/255 and8/255, inputs clipped to
[0,1]. User's fixed dataset-order prefixes: indices0–999 for incomplete bounds,
0–199 for complete verification. Include clean-wrong examples in denominator;
do not select common-clean-correct. These prefixes overlap historical exposure,
including index98. Revision8 explicitly designates them for this **new** study;
this is disclosed development, not reopening an old frozen proof attempt or
claiming a fresh holdout. Old records and conclusions remain unchanged.

Incomplete CROWN-IBP means backward CROWN using interval intermediate bounds,
without α optimization/BaB. It is an evaluation method, not evidence of CROWN-IBP
training. For MoE it must cover every tie-legal route, using the registered
route reduction/pruning; failure to cover is unproved, never clean-route SAFE.
Exact support for weighted nonlinear graphs, missing/timeout handling and
full-request runtime policy will be bound to the evaluation runner before any
certification results. The training freeze does not claim this runner is ready.

Complete verification: coupled GenBaB route reduction/pruning for MoE, same
engine/version and comparable settings for dense ReLU. No M-A cascade or stopped
M-B/M-C mechanism. Each complete request receives300s for startup/loading,
conversion/pruning/all histories/bounds/branching/evidence/cleanup/terminal.
Cold/hot boundary and concurrency are paired, all failures/errors retained.
Public PGD and full-model replay run separately on all designated objects; costs
and overlaps reported, no free positive evidence and no independent-route
counterexample labelled full-model UNSAFE. Numeric filters stay distinct from
source-complete rational proof or deployed floating-point guarantees.

The complete-call arithmetic is **12×200×2=4,800**, worst-case400 CPU-hours;
eight perfectly utilized workers would take50h before extra work/contention.
This is not a one-week completion promise. Incomplete evaluation has24,000
model/input/radius items, separately metered. The S2 cap3000 is not borrowed.

Primary matching comparisons use L1E8 against each dense group at2/255;
8/255 is the preregistered larger-radius result, not a rescue radius if the
primary screen fails. Report complete coverage/certified accuracy on200 and
incomplete certified accuracy on1000 separately. User's S1 decision uses the
point estimates: at least one matching group has nonnegative complete certified
accuracy difference and strictly lower all-request median cost. Report paired
95% bootstrap intervals alongside, without silently adding a new CI acceptance
gate or relabelling this finite screen independent confirmation. Publish both
matching comparisons, both radii, all seed results and all gain/loss. Resample
input blocks preserving three paired seeds (10,000 resamples, seed20261029);
three seeds are fixed replication units, not600 independent inputs. L2E8 is
secondary and cannot rescue a failed L1 matched screen.

S1 success authorizes S2 as the user requested; failure stops and reports.
S2 needs its own finite allocation within3000 complete calls, τ/schedules and
baseline controls frozen before new outcomes. No post-hoc architecture search.

## Current execution

At freeze preparation: zero new training runs and zero S1 verification calls.
Resource observation: approximately100GiB memory available,275GiB disk available,
20 allowed CPUs, load27; GPU80% utilized by existing work. One CPU worker chosen.
The dated goal and handoff preserve old G1 closure and all previous failures.

The single new-code control completed under existing development supervision:
3.355s execution plus0.040s cleanup, exit0 and no live owned session members.
All listed assertions passed; six old MoE checkpoint hashes independently match
the archived training record. The [training freeze](architecture_s1_training_freeze_20261029.json)
binds the control/source/data/parameter identities. These checks are preparation,
not a scientific result or a full numerical proof.

Training launched from committed source `f619c24bd7480edb3e50e47413c9048230a4f721`
in `/data1/Kane/MOE/executions/architecture_s1_training_20261029_r1`;
PID3300583/start1212371695, detached session, nice10. Results:
`/data1/Kane/MOE/results/architecture_s1_training_20261029_r1`.
[Start record](architecture_s1_records_20261029.json) binds launch and split
receipts; 8GiB AS observed in process limits, first epoch recorded, no startup
error. Six models are queued, not finished; no S1 numerical bounds or complete
verification calls have run. Current evaluation adapters remain the next task.

## Common evaluation adapters and input preparation

The opt-in [adapter](../act/pipeline/moe/architecture_evaluation.py) now supplies
both methods. Incomplete uses the pinned auto-LiRPA `CROWN-IBP` mode for prefix
margins and final guarded properties, without alpha optimization or branching.
Complete MoE retains the coupled R7 GenBaB configuration and original full-CROWN
prefix pruning. Dense complete verification uses the same API/CPU/float64,
batch1/attack/deadline settings, with the engine's native kFSB ReLU branching.
The existing historical `verify_graph` defaults are unchanged; the optional
configuration is explicit. No stopped M-A/M-B/M-C path is enabled.

Incomplete traversal can stop at its first unproved history, always UNKNOWN.
The unvisited/returned-unproved counts remain visible, and incomplete work never
becomes a positive or a solved-time saving. All complete-request comparisons
still report full costs and states. Report clean-wrong and shared-clean-correct
cost strata as interpretation, because a cheap wrong prediction is not useful
certification performance. The registered main denominator remains all inputs.

The fixed synthetic adapter batch passed18tests in18.105s under the existing
supervisor: dense positive/zero/witness cases, nonzero property constants,
variable gates, nine tie-legal two-layer histories, missing/invalid domains,
deadline rejection, UNKNOWN short-circuit versus historical full traversal,
and loading the ACT-created dense checkpoint in the isolated verifier environment.
Nine tests are relevant traversal regressions; nine are new adapter controls.
The alternate-tie negative example remains UNKNOWN despite a positive native
clean route. This tests coverage, not strict certification of a trained model.
[Record](architecture_s1_records_20261029.json) binds source hashes and cases;
raw control evidence is `results/architecture_s1_evaluation_control_20261029_r1`.

[Common input preparation](../scripts/architecture_s1_inputs.py) materialized
indices0–999 as one shared float32 tensor plus labels, with both float64 boxes
and all nine label-versus-competitor rows fingerprinted per input. No model was
loaded or used for selection. The source CIFAR test hash matches training.
Preparation took3.628s including supervised cleanup, about13.14MB including
catalog/receipts, charged to the S1 artifact account, not free per-arm answers.
Per-request reading/materialization will still be charged. Binary64 box
construction is explicitly the numerical policy, not a repaired exact-input
source proof. All historical exposure remains disclosed.

The full-request worker/reception/parallel catalog freeze is still pending;
adapter controls do not authorize a result before that existing S1 requirement
is implemented. No new per-interface approval is required. Training continues
from its separate immutable source, unaffected by these code changes.

## Complete-request execution integration

[S1 execution](../scripts/architecture_s1_execution.py) now connects the
adapters to the existing owned-session supervisor. Each request independently
checks its committed source, external engine identity, checkpoint, materialized
input and all property rows. Loading the entire model, including inactive expert
weights, is charged. Full-model replay and structural coverage checks precede
publication; the controller independently checks the published identity and
coverage. This remains numerical evidence, not a new source-complete proof.

The proposed production timing is300s total with nested worker/supervisor/cleanup
cutoffs at288/290/295s. The final five seconds cover reception and terminal
publication. Online overruns cannot be accepted from a later offline review.
Returned TIMEOUT, completed UNKNOWN, clean-error witnesses and outer cutoffs
remain distinct records. Partial candidate files without a valid completed
publication are retained and rejected. Phase counters cover engine startup,
model/input loading, verification and coverage/replay; the complete request
clock additionally includes serialization, reception, cleanup and terminal
writing. Unallocated overhead is not silently omitted.

The finite queue rotates the model order by input/radius, without model-based
input selection. It contains4800 complete or24000 incomplete items. At most
eight single-thread workers have separate output directories; only the caller
publishes the shared stage ledger. Controller/worker writes use one locked
128MiB artifact account, with the common-input preparation bytes charged when
the stage account is created. There is no per-worker quota reset. A512KiB
terminal reserve is part of, not additional to, that cap. Reservations precede
writes; interruption can conservatively overcount, never undercount. Only live
progress may be replaced; failure and terminal records are immutable. Unexpected
errors stop new admission while already launched requests finish under their
own deadlines. Tool repairs retain the prefix and require a separately recorded
continuation of the unexecuted queue.

One new integration control completed in24.184s: five standard-library tests
and seven synthetic native calls, including four normal requests with two
concurrent workers and exception/deadline/partial-publication faults. All owned
sessions were reaped, no late or partial positive was accepted. Physical native
artifacts19,477bytes are covered by the conservative23,415byte account. A later
static boundary review added acceptance of an early returned TIMEOUT as an
explicit partial terminal; its single differential control passed in2.396s,
without repeating the seven native calls. Both source versions and receipts are
retained in the [stage record](architecture_s1_records_20261029.json).

No trained verification was run. Public PGD integration, full numerical-run
manifest and training completion audit remain before the real queue launch.
The finite runner refuses an unfrozen real queue or a mutable checkout.

## First completed training model (prefix audit)

Dense_total_seed0 completed all50epochs. Earliest highest validation accuracy
selects epoch37 (53.24% on5000validation examples); the selected checkpoint
reports52.66% on10000test examples. A read-only audit checked all50epoch records,
the earliest-maximum rule, architecture/seed/count, checkpoint hash and metadata,
shared split and original immutable source identity. It did not rerun inference
or training. Five models remain; this is not a completed seed comparison or
evidence for/against C2 or C3. Training continues from its existing process.

## Public attack and final execution preparation (2026-10-30)

Public PGD uses the same full model and physical input/box as verification,
independently of verifier answers:20 updates per restart,2 uniform random
restarts, minimum true-label margin over all nine competitors, step sizeε/4.
The seed is20261029+100000×model_seed+10000×1[ε=8/255]+input_index, shared
across paired architectures. Clean-error centers are checked first. Search
stops only after a valid full-model counterexample or its frozen work/deadline
limit. Failure to find a witness is **not** evidence of robustness.

All1000inputs,12models and2radii receive this auxiliary attack:24000domains.
For loading efficiency, ten consecutive inputs share one auxiliary process,
with20s internal per-domain deadlines and the same300s hard process budget.
Loading, input checks, all ten searches, witness replay/serialization and cleanup
are charged to the auxiliary batch. No cold per-input timing is inferred from
this shared load. A native operation can overrun an internal domain deadline;
its late result is rejected and the outer hard limit remains in force. These
2400auxiliary batches are not4800complete verification slots or evidence
supplied to either certification path. Complete/incomplete verifier requests
remain individually cold, independently loaded and budgeted.

Non-center witnesses use a lossless representation: each coordinate refers to
its bound endpoint or retains its exact binary64 interior value. Decoding is
checked against the original point and replayed before accepting the PGD
witness. Clean-center witnesses reference the bound common input. No attack
point quantization, radius reduction or extra acceptance margin is introduced.
The compact ledger now references the immutable terminal and its measured
total cost, instead of storing a second copy of the full terminal; every
underlying source/session/outcome remains retained and charged.

The first attack-control batch stopped on an incorrect test expectation: a
fixed20×2search was assumed to find a known negative region of a linear toy
model. Both original/default-step and explicit-step searches actually returned
no witness. R1 remains FAIL. R2 keeps the **same model, seed, domain and attack**,
checks agreement of the two searches, and correctly treats no witness as attack
failure. It runs only the corrected and previously unexecuted unit cases; the
five unchanged passing cases are retained from R1. A four-request native
integration batch then checks PGD and both verification families under two
concurrent workers. One non-center counterexample is actually encoded and
replayed; three auxiliary domains return no witness. R2 passes in18.935s, with
17206bytes retained. No trained-model certification or attack was run.

[Final study preparation](../scripts/architecture_s1_study.py) requires all six
dense completions and validates every50epoch trajectory, earliest validation
choice, weights, recipe/split and the six existing MoE identities before it can
write the numerical-run manifest. A standard-library control checks all31200
unique native batch/request IDs:2400publicPGD,4800complete,24000incomplete;
there are no accuracy/route-based omissions. It checks the50epoch selector and
missing-epoch rejection. The source-bound manifest will freeze eight workers,
both actual engine configurations, request deadlines, shared artifact account,
catalog hashes and the already registered S1 analysis. The launch repeats a
read-only CPU/memory/disk admission check and uses an immutable committed copy.
No numerical-run manifest or real verification queue has yet been launched.

Dense_total_seed1 is now also complete: selected epoch25, test53.29%. Its
separate prefix audit checks all50epochs and checkpoint/source identities
without another forward pass. Four models remain; full paired certification
results, not these partial clean accuracies, will decide S1.

## Paired analysis and terminal audit prepared before verification outcomes

[Statistics](../act/pipeline/moe/architecture_statistics.py) implement the
registered input-block bootstrap:10000resamples, seed20261029, keeping all
three fixed seed pairs together in each input block. Complete200-input and
incomplete1000-input results are separate, as are2/255 and8/255 and the two
capacity matches. The main time comparison uses all-request medians, including
UNKNOWN, TIMEOUT and ERROR, with every executed cost retained. Missing jobs or
missing costs prevent a final S1 decision; they are not silently excluded.

The point gate is unchanged. Its literal rule can authorize S2 when both arms
have zero numerical positives but MoE's median cost is smaller. The report
explicitly flags this degeneracy; it does **not** establish that MoE certifies
more inputs. No new positive-rate/CI threshold is silently imposed. Both
matching groups and all confidence intervals remain visible. Shared
clean-correct cost medians are explanatory strata, obtained from the separately
recorded same-object clean predictions, never a filter on the main sample.
This helps expose speed differences caused by more cheap clean-error returns.

[Terminal audit](../scripts/architecture_s1_audit.py) checks frozen request and
input/engine identities, immutable file links, online deadlines, complete
property/history coverage and full-model replay records. It preserves a late
terminal as TIMEOUT even if an earlier candidate file contains a positive.
Accepted public counterexamples conflicting with numerical positives trigger
an evidence error. This is a structural audit of numerical evidence, not a
recomputation of every CROWN bound or independent model replay. Auxiliary
attacks do not replace or upgrade the verifier's recorded terminal.

Routing tables distinguish observed numerical exclusion counts from unknown
route counts. A single remaining candidate under recorded CROWN exclusions is
not called an exact feasible-route census or a checked route-flip witness.
Clean-error early exits and incomplete records cannot manufacture single-route
evidence. Costs of shared public-PGD batches, cold verifier requests and the
offline audit are separate. Parallel queue wall time is not the sum of request
CPU/wall costs and is reported separately.

Eight necessary standard-library/numpy controls passed in2.593s without any
model or solver call: block versus call resampling, gains/losses, frozen point
gate, explicit zero-certificate flag, missing denominators/costs, route-unknown
counts, early-controller-error charging, and rejection of missing properties,
wrong identity and late positives. All raw controls remain in the single
[S1 record](architecture_s1_records_20261029.json). Real numerical execution is
still waiting for the six-model training completion and final manifest binding.


## Architecture manuscript working draft

The new [architecture draft](../paper/architecture_main.tex) reuses three
complete-history/guard/endpoint propositions and states the full cost
accounting: loading, router pruning, every visited selected-expert graph,
proof and evidence work. Per-history active parameters alone do not determine
that sum. The routing training loss is marked conditional future work; ignored
unstable routes remain a training approximation, never an evaluation proof.

The draft binds the two S1 parameter matches, all-input denominators,
three-seed paired-input analysis and the unchanged point gate, including its
zero-certificate degeneracy. S1/S2 result sections are explicitly pending.
The previous verifier paper and all its failed decisions remain intact;
revision8 uses that material for motivation/appendices, not a separate submission.

Two existing-compiler passes and text extraction completed in0.787s without
undefined references or overfull boxes; page4 was visually checked. Six generic
pages are a working draft, not an official ICML template, final eight-page paper,
human review or evidence that C2/C3 hold. The existing S1 record binds the
source/PDF/build receipt; no model or solver call was added by writing.


## Six dense models completed; numerical evaluation frozen

The original immutable training queue completed all300epochs/six seeds in
7998.453s (2.222h), without restart or checkpoint replacement. Selection remains
earliest maximal validation accuracy, not test accuracy or certifiability.

| Group | Seed | Selected epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Total | 0 | 37 | 53.24% | 52.66% |
| Total | 1 | 25 | 53.40% | 53.29% |
| Total | 2 | 28 | 53.04% | 53.73% |
| Active | 0 | 42 | 52.72% | 52.78% |
| Active | 1 | 40 | 53.28% | 53.57% |
| Active | 2 | 40 | 52.84% | 53.47% |

The independent record pass checked all300epoch records, integer validation
counts/earliest-best selection, actual capacity formulas and all12 final model
hashes in0.242s. Final preparation also loaded checkpoint metadata and checked
training/split/source identities before binding them. Neither action reran
accuracy or model verification. Original failures and all selected weights stay
outside Git and retain their immutable hashes.

[Final evaluation freeze](architecture_s1_evaluation_freeze_20261030.json)
binds the12models, actual source/engine versions, materialized domains,
all31200unique native request identities, timing and numerical policy, controls,
statistics and the original decision gate. Counts remain4800complete calls,
24000incomplete items and2400publicPGD batches covering24000domains. Eight
single-thread CPU workers,8GiB AS each, shared128MiB artifact policy and300s
whole-request limits are unchanged. The run starts from a committed immutable
copy after fresh resource admission; this freeze contains no verification
outcomes or certification claim.


## Numerical queue launch and observed artifact capacity

The immutable evaluation source is `d139f9335ad9e780d42a12212f85784e4702316c`
at `executions/architecture_s1_evaluation_20261030_r1`; results have the same
basename under `results/`. Detached controller PID1692415/start1213209188,
eight CPU request workers, nice10, no overlap with training. Fresh admission
observed20CPUs/load6.16/about99.7GiB available memory/about312GiB ordinary-user
disk. Structured session/error/terminal records are retained; redundant console
output is discarded. Launch/start receipts live inside the shared byte account.

A first independent structural review accepts408publicPGD batch terminals in
0.798s, with3411.679s charged request time. All are AUXILIARY_COMPLETE; that
means the auxiliary batch was recorded, not that all ten domains are safe or
that fixed-search completion occurred on a counterexample's early return.
No complete/incomplete verifier outcome or S1 decision is available yet.

Actual non-center PGD witnesses expose insufficient artifact planning:
3030domain records used22,504,308bytes, mean7427bytes. A descriptive projection
for24000domain records alone is178,251,945bytes, above the frozen134,217,728byte
stage cap, before most verification evidence. This is not a full upper bound.
The original128MiB policy remains enforced; neither precision nor evidence
is deleted. The pending user decision proposes512MiB for the same fixed S1
objects and execution, with an explicit new freeze and only unexecuted requests
continued after any original-cap stop. It does not change300s/8GiBAS, methods,
seeds, acceptance, sample counts or old records. The capacity estimate was
inadequate; it is an execution-storage issue, not evidence against an architecture.


## Original-cap prefix stopped and archived

The original controller exited after1617registered requests:1609accepted public
PGD batches and8ERROR. One ledger records an explicit shared-artifact-cap
BufferError before child launch; seven native workers exited1 without accepted
publication during the same exhausted account. Their failure/partial records
remain, without claiming a separately demonstrated unique cause for each child.
The normal-write cutoff reserves512KiB for terminals within the128MiB total;
no file/precision/threshold was removed to extend execution.

Queue wall time1687.434s (28.124min); summed complete request cost13476.846s
(3.744h), no missing request cost. All1616spawned sessions report
LEADER_REAPED_NO_LIVE_SESSION; the original controller is gone and no old live
worker was observed. The frozen-source independent audit takes5.546s and
retains1609AUXILIARY_COMPLETE/8ERROR/29583NOT_EXECUTED, S1=PENDING_S1_COMPLETION.
It checks records/coverage/online reception, not fresh PGD or model proofs.

Unexecuted:783public batches plus4800complete and24000incomplete verifier
requests. Failed batches remain failed; partial domain records are not upgraded
to accepted whole batches offline. The same31,200-request denominator and
fixed sample/seed/configuration/decision rules remain. No scientific S1 stop or
S2 authorization follows from this storage stop.

The existing512MiB resource-amendment question is still pending. Current quota
has not changed; no continuation has launched. An approved continuation must
bind the archived prefix and all its spent costs, preserve the failures and
continue only the unexecuted roster under one aggregate artifact account.


## Approved storage-only continuation, 2026-10-30

User answer: **批准共享上限 512 MiB**. The old128MiB stop and all1617
attempts remain immutable, including8ERROR. The continuation runs only the
29583unexecuted IDs, in the original order:783publicPGD batches,4800complete
and24000incomplete requests. No failed or accepted request receives another
attempt. The full31200-item denominator and all spent request costs remain.

A new result directory carries the old account's entire charged value, including
failed reservations, plus the continuation manifest into one512MiB account.
Old inputs are not charged a second time, and old evidence is not copied or
rewritten. The original numerical source d139f9335 and its engine, model and
input bindings remain unchanged. A resource-only manifest changes the capacity
and execution remainder, adds the original root/tree/manifest identity, and
binds the small continuation controller and combined auditor. The old frozen
queue/worker is invoked directly from its immutable execution directory; no
numerical code, global patch, timing rule or acceptance check changes.

Combined analysis reads each request at exactly one bound location, rejects
retries, retains original online terminals, sums both segments' request costs,
and reports the original model/input denominators. New-root storage includes
all carried charges; neither a new directory nor a new manifest resets quota.
Continuation controls cover failed-ID exclusion, exact order, orphan/duplicate
rejection, source/deadline/capacity changes, prior-record tampering, carried byte
limits, and an actual saved-prefix analysis differential with no model calls.
Fresh resource admission and committed immutable controller precede launch.


Continuation controls PASS: six bounded standard-library cases and one
saved-prefix aggregate differential,21.278s including supervised startup and
cleanup,0model/solver calls. The differential preserves all oldstates, full
roster, chargedcosts, modeltables, pairedcomparisons andS1=PENDING.
The committed controller37479ccf3 launched detachedPID2157745/start1216372585
from `executions/architecture_s1_evaluation_20261030_r2`; results use the same
basename. Its first independent terminal-prefix audit is bound in the shared
record. The512MiB account starts with133858729chargedbytes including allold
charges and the newmanifest; no oldaccount/evidence was rewritten. Numerical
workers still execute the original d139f9335 source. FullS1 results are pending.


## Public PGD complete; full verification started

All2400auxiliary batches are settled:2392AUXILIARY_COMPLETE and8original ERROR.
The completed continuation contributed783batches without a newerror. The full
24000-domain denominator includes80unaccepted domains from the oldfailures;
partial saved records are not promoted offline. A9.661s independent record
pass validates the immutable terminal/domain bindings, true labels and recorded
online full-model replay. It does not rerun attacks or independently reprove
witness arithmetic. Total charged request cost is21107.093s (5.863h), not
parallelwall time; no attack outcome is supplied to verification workers.

| Radius | Model group | Replayed counterexample | Fixed search: no witness | Unaccepted | Denominator |
|---|---|---:|---:|---:|---:|
| 2/255 | L1 | 2161 | 819 | 20 | 3000 |
| 2/255 | L2 | 2272 | 708 | 20 | 3000 |
| 2/255 | total | 2099 | 881 | 20 | 3000 |
| 2/255 | active | 2174 | 806 | 20 | 3000 |
| 8/255 | L1 | 2921 | 79 | 0 | 3000 |
| 8/255 | L2 | 2953 | 47 | 0 | 3000 |
| 8/255 | total | 2944 | 56 | 0 | 3000 |
| 8/255 | active | 2972 | 28 | 0 | 3000 |

These are descriptive counts on1000shared inputs and3fixed seeds pergroup,
not3000independent images. No-witness is not a positive certificate. At8/255,
only0.93%–2.63% were not disproved by the fixed attack, limiting the observed
ordinary-trained models' robustness, without establishing any certification
advantage. The2/255 groups also retain their20missing domains each.

The same controller now executes the4800complete verification requests,
followed by24000incomplete CROWN-IBP items. Initial accepted numerical terminals
were checked for all-property/full-route coverage, budget and reception; they
are a startup prefix, not a comparison or S1decision. Model/seed/input/epsilon,
all-denominator timing and frozen decision rules remain unchanged.
Reproduce this public summary with `scripts/architecture_s1_audit.py ROOT
MANIFEST OUTPUT --public-only`; its output is charged to the current shared
account and includes the auditor and input/ledger identities. Full combined
analysis remains the same CLI without `--public-only`, after queue termination.
