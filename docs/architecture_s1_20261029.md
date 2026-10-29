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
