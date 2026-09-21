# Author baseline deployment and ACT intake: execution ledger

## Latest consolidated state (supersedes historical preparation below)

| Work | Deployment / executed control | Current ACT original-semantics support | Next uncompleted gate |
|---|---|---|---|
| Dual RS |Both exact resume controls PASS; frozen90epoch training running (epoch51 saved at this update); real certification runner/6 controls prepared|Separate statistical L2 path, not deterministic MoE HZ SAFE|Finalepoch90 audit, final-weight binding, then frozen2-input certification|
| MetaMoE |Original full public CIFAR checkpoint loaded; frozenindex0 full-model counterexample independently replayed|NEW original class-separated top1 box entry,9 semantic controls; conditional HZ policy|MNIST ONNX/source mismatch and full author-table / paired experiment|
| RoME |Public trained checkpoint strict prediction load; native standardAA control and independent replay complete|Original dense multilayer routing forward/gradient intake; whole-box UNSUPPORTED|Three-norm paper-scale evaluation and complete dependent-history lowering|
| Robust Experts |Real native train/clean/PGD20/APGD20 R2 completes but BN state drifts; separate eval-mode R3 completes13.21s with147/147 checkpoint tensors preserved; new full-size E4/k1 history compiler compatibility PASS|NEW complete-history HZ entry with conditional numerical policy and prefix-STE checks; analytic multi-layer proofs, NOT trained full-size certificates; original non-STE raw dispatch fail-closed|Frozen trained-model whole-box experiment, nonlinear joint-HZ coverage/scale, full-state continuation and long-training recipe|
| J-TLAT |Pinned author repository README-only|No runnable author implementation substituted|Public code/weights or PI-mediated artifact access|
| Feature Noise |Paper studied; author implementation not identified|No invented implementation counted as deployment|Identify experiment source and freeze task|

New multilayer implementation/evidence: [contract and controls](multilayer_history_verifier.md).
The two fixed full-size compiler probes are not a trained-model certificate or
a paper comparison. RoME continuous dense routing is not reinterpreted as top-k.

Robust Experts source-corrected R3 is explicitly NOT byte-identical native R2.
Independent saved-only audit checks147 tensors,74 finite SGD buffers and both
PolyLR child epoch positions. The first checker incorrectly assumed a top-level
last_epoch; its frozen code remains, a separate v2 parser checks the actual
ChainedScheduler schema with mutation controls. This is our checker-format fix,
not a new run, changed result or relaxed equality gate. Full training continuation
is still unproven. See [workflow and failure analysis](robust_experts_native_workflow_20260921.md),
[R3 execution](robust_experts_workflow_archive_20260921_r3.json) and
[independent state audit](robust_experts_saved_state_audit_20260921_r3.json).

Latest execution update: Dual RS90epoch training is RUNNING, launched from
826e8e94b after both real resume controls passed. Fresh run directory:
`/data1/Kane/MOE/baseline_runs/dual_rs_selector_training_20260921_r1`.
No automatic certification, retry, or Git write. Epoch1 total70.43s; this is
an early timing observation, not a guaranteed completion time. Its34.83% test
metric measures sigma-selector labels, NOT CIFAR classification accuracy.

## Completed in this stage

- Dual RS numerical-compatibility step/resume R2: PASS, exact all-state equality.
  [Step archive](dual_rs_training_control_archive_20260921_r2.json).
- Native train→test→scheduler→atomic completed-epoch save, then independent
  process continuation: PASS on the separately frozen two-epoch prefix. All
  model, AdamW, scheduler and RNG state and epoch diagnostics agree exactly.
  Shared outer300s, execution12.831s, with postflight12.877s.
  [Epoch archive](dual_rs_epoch_control_archive_20260921_r1.json).
- Ninety-epoch recipe is now frozen: ALL47,302 author-eligible train rows and
 8,719 eligible test rows; native batch256/two128-original updates, AdamW.01,
  wd.01, milestones30/60/1000 gamma.5, λ40/η.5, seed1, workers0. Only numerical
  change is the named differentiable log-domain consistency evaluation. Final
  epoch90, never best certification/accuracy.12h hard outer deadline includes
  train/test/save and final saved-state audit. No automatic retries or git writes.
  Config `configs/recent_moe/dual_rs_training_r1.json`; launch is now recorded.
  The two-input RS certification *scientific recipe* is fixed there, but the
  trained-weight certification execution manifest remains a subsequent gate.

## Original MetaMoE top-1 intake

`ClassSeparatedTop1.from_metamoe(original_model)` accepts the pinned author
class, not a remapped shared-class surrogate. The separate box entry preserves
zero-filled other classes, requires selected score nonzero before the score/
score reduction, covers all legal ties and global output inequalities, and
replays possible violations on the original complete model. Existing weighted
top2 entry and frozen results are unchanged.

R1 failed before model load (missing child import path). R2 uses an explicit
ACT import root and separately frozen CPU environment. Original full public RT
checkpoint, CIFAR raw-order test index0, normalized-space2/255,300s:
**UNSAFE_REPLAYED at the center**, global label3/prediction5, in BOTH source
float32 and explicit float64 snapshot. No replacement sample. Float32↔float64
probe max4.10e-6 is descriptive, not a full-domain equivalence test.
[Compact record archive](metamoe_full_intake_compact_archive_20260921_r2.json) and
[independent original-module replay](metamoe_full_witness_replay_20260921_r2.json).
The zero-filled-source interface is validated; this particular request exits
on a real counterexample, so it is NOT a real positive HZ bound demonstration.
Nine analytical controls exercise the positive/negative/undefined guard cases.
The first Git summary accidentally inlined one full input witness. The current
tree replaces it with a hash reference, preserving all original server records
and the prior summary hash; historical commits are not rewritten.

All HZ positives from the new entry remain conditional on network/input/guard
lowering and the frozen solver numerical policy. This does not solve the older
MNIST folded-ONNX probe mismatch or the project's source-complete proof gap.

## Multi-layer authors: original execution intake, not fake box support

`LayeredAuthorIntake` runs the original complete model and observes gate events
in that same forward. Full-size output and input gradients are bitwise equal
with/without the trace. Source mutation and train-mode controls reject.

| Model | Actual calls | Verified execution semantics | Box verification |
|---|---:|---|---|
| RoME CIFAR ViT,145,547,434 parameters |24|Dense ALL-LoRA-expert mixture, native float32 local/global softmax and depth β|Explicit UNSUPPORTED|
| Robust Experts ResNet18 ConvMoE layer4,E4,k2,36,394,168 parameters|5|Intermediate top-k dispatch with native +1e-5 normalization|Explicit UNSUPPORTED|

Robust Experts' missing SyncBN import is fixed only in a separate compatibility
checkout. If the optional source is available its original initializer remains;
if absent the empty type tuple does not pretend to implement SyncBN. The tested
architecture is restricted to ordinary BatchNorm2d. Two files change, with an
exact archived patch; original author checkout remains clean. Missing-einops
failure is retained. [Source/controls archive](author_adapter_controls_archive_20260921.json).

These are initialized full-size models, not their trained accuracy. RoME's
public CIFAR10 MAX checkpoint is also now downloaded and its prediction path
has passed a separately labeled trained-state intake. Default strict R1 failed
on48 absent auxiliary global_proj tensors. Source inspection shows they feed
only returned diversity telemetry. R2 builds the author-supported zero auxiliary
projection variant and STRICTLY loads all440 prediction-state tensors. Output
and input gradients equal a full original module with synthetic auxiliary
heads; this is NOT a resumed-training or full auxiliary-state equivalence.
Code defaults s4,b6,alpha=rank remain explicit, not claimed as recovered recipe.
Native AutoAttack dependency and author's evaluate.py help pass in a dedicated
CPU environment. Frozen index0 standardLinf8/255 control COMPLETED in18.559s
including postflight. Clean3→adversarial5; independent full-model replay PASS.
The saved-input exact distance exceeds8/255 by2.79e-8, within the registered1e-7
numerical attack tolerance. EMPIRICAL attack only, NOT an exact-box UNSAFE.
Native AA found the attack in APGD-CE and skipped later attacks on this already
broken input. No attack parameter search or table accuracy from this one input.
[Intake archive](rome_deployment_archive_20260921.json),
[attack and replay archive](rome_autoattack_archive_20260921_r1.json).
A clean
trace is NOT enumeration of all dependent histories. Neither family is sent
through normalized output-layer F0 with an incorrect semantic label.

## Remaining gates — NOT marked complete

R1 native Robust Experts workflow has now FAILED before training because
Lightning1.9 ModelCheckpoint references np.Inf, absent in NumPy2. The failure
is archived unchanged. Separate R2 environment pins NumPy1.26.4/OpenCV4.11/
tifffile2024.8.30; pip check and both controls pass. R2 retains the identical
worker, author model/attack/transform settings and600s budget. Its new execution
config is frozen before launch; no outcome-dependent changes to the model.
[R1 failure archive](robust_experts_workflow_archive_20260921_r1.json).

Robust Experts now also has a dedicated workflow environment, passing native
Lightning/Hydra/attack imports and two configuration/terminal controls. Public
CIFAR100 is integrity checked; serial timeout and range-connection failure are
retained. Its600s real control is frozen but not executed at this preparation
entry: native E4/k1/layer4 ConvMoE, one batch2 per phase, PGD7 training and
PGD20/APGD20 evaluation; original SGD and transforms. Location-only data_dir
forwarding is needed because the author subclass swallows that argument. All
logging is local CSV and network connections are prohibited in the child.
Hydra1.3.2/Python3.12/CPU Torch2.9.1/Lightning1.9.5 is an explicitly named
compatibility stack, not a claim of identical author environment.
[Preparation archive](robust_experts_workflow_preparation_20260921.json).

| Work | Remaining requirement |
|---|---|
| Dual RS |90epoch training running; certification execution recipe/6 controls now passed, actual final-weight binding and real two-stage pilot pending; then paper-scale replication|
| MetaMoE author-table reproduction |MNIST conversion/source adapter control; router and expert tables; matched complete-request comparison|
| RoME |Standard AutoAttack deployment control and independent replay completed; paper-scale evaluation still separate; no multi-layer HZ lowering yet|
| Robust Experts |R2 native workflow and R3 eval-mode variant completed; full-state continuation control then separately frozen long training/attack protocol; no paper-trained weights or multi-layer HZ lowering yet|
| J-TLAT |Pinned author checkout still README-only; no runnable implementation to deploy|
| Feature Noise |Author experiment implementation not identified; do not substitute an invented implementation|

Human expert review, final venue/claim signoff and author contact remain PI tasks.
Neither these controls nor successful JSON audits create new source-complete
certificates or update historical competition tables.
