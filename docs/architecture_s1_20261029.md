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
