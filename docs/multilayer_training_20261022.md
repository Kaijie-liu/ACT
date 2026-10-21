# Frozen CPU training for the multilayer bottleneck experiment

Authorized by the user's fourth revision, after successful stage0 semantics.
Twelve models: L1E8,L2E8,L3E8,L2E16, each seeds0/1/2. The order is configuration
then seed, sequential CPU only, four intra-op threads, one inter-op, zero loader
workers, nice10. Existing act-py312; no installs, GPU or shared-user job changes.
Each process retains8GiB RLIMIT_AS (address space, not physical-memory limit).
Training time and checkpoints are separate from verification slots/artifact cap.

The original recipe is `experiment1_multiseed_training_r1.json`: CIFAR10
ToTensor only, no normalization/augmentation,50epochs, batch256, AdamWlr0.001,
weight_decay0.0001, switch balance coefficient0.1, no scheduler. New common
45k/5k split uses seed0 for every model, unlike the old per-model-seed split.
Random initializations and shuffle generators use each model seed. CPUfloat32,
deterministic operations; verification will explicitly materialize a separate
CPUfloat64 declaration and never conflate its guarantee with training execution.

Architecture: each router is Linear(input,128)-ReLU-Linear(128,E); each expert
is Linear(input,256)-ReLU-Linear(256,128)-ReLU-Linear(128,output).
First input width3072, intermediate MoE outputs128, final output10. Every MoE
selects top2 with selected softmax; no intervening residual, BN or added
activation. Average the per-layer switch balances then multiply by0.1, so
increasing depth does not silently multiply the regularization coefficient.
L1 retains the old architecture/recipe, with the explicitly changed CPU/shared
split execution conditions. Intermediate width is frozen before any training.

Fixed50epochs for every seed, earliest epoch on validation-accuracy ties.
Select only on validation accuracy; evaluate the test set once after selection,
report every seed regardless of accuracy or verification. No verification query
is used to select training/checkpoint. All per-epoch metrics are append-only.
An active `resume.pt` atomically replaces its own preceding unselected training
state and includes model/optimizer/best state/RNG/shuffle identities; final
`selected.pt` is never overwritten by another run. Failures remain in place.
Estimated parameter bytes for all selected models453.4MB; optimizer/best/resume
snapshots add several GB and are tracked separately, not committed to Git.

Before real training, one necessary synthetic control batch≤300s tests the four
depth/expert configurations on tiny dimensions: dynamic/training forward,
finite nonzero optimization, checkpoint exact reload and second-step optimizer
resume differential. This is an implementation control, not new model evidence.
The manifest then binds control result, code and original CIFAR batch hashes.
The twelve-model run uses an immutable git snapshot and survives terminal close;
the controller stops on failure and saves a traceback, never substitutes seeds.
