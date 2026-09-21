# Author baseline deployment and ACT intake: execution ledger

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
  Config `configs/recent_moe/dual_rs_training_r1.json`; launch is a separate step.
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
[Record archive](metamoe_full_intake_archive_20260921_r2.json) and
[independent original-module replay](metamoe_full_witness_replay_20260921_r2.json).
The zero-filled-source interface is validated; this particular request exits
on a real counterexample, so it is NOT a real positive HZ bound demonstration.
Nine analytical controls exercise the positive/negative/undefined guard cases.

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
public CIFAR10 MAX checkpoint has separately been located and downloaded; its
trained-state loading and author attack runner are subsequent gates. A clean
trace is NOT enumeration of all dependent histories. Neither family is sent
through normalized output-layer F0 with an incorrect semantic label.

## Remaining gates — NOT marked complete

| Work | Remaining requirement |
|---|---|
| Dual RS |90epoch training completion; freeze actual final weight identity; supervised two-stage Monte Carlo pilot; then paper-scale replication|
| MetaMoE author-table reproduction |MNIST conversion/source adapter control; router and expert tables; matched complete-request comparison|
| RoME |Public trained-weight strict intake; AutoAttack dependency/CLI and supervised attack control; no multi-layer HZ lowering yet|
| Robust Experts |Training/attack workflow with public CIFAR100, source-vs-paper recipe separation; no trained weights or multi-layer HZ lowering yet|
| J-TLAT |Pinned author checkout still README-only; no runnable implementation to deploy|
| Feature Noise |Author experiment implementation not identified; do not substitute an invented implementation|

Human expert review, final venue/claim signoff and author contact remain PI tasks.
Neither these controls nor successful JSON audits create new source-complete
certificates or update historical competition tables.
