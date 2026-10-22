# Residual multilayer model validity study

## Launch 2026-10-23

Frozen six-model CPU queue launched at00:29:33AEDT,PID3997912/PPID1/SID3997912,
nice10,two numerical threads,8GiB RLIMIT_AS read back. Immutable execution
`executions/multilayer_stage1b_training_20261023_r1` is commit
`21a21f2559803063b5b9ac3dea724e80b76654ca`; result root has the same basename.
Manifest SHA256`6b3a401dcb87fb4e5e6e289e4f7174e9b8abc119cfd2349189e2ae800c57702a`;
all eight source hashes,seven data files,and prior control/receipt match.
New split matches frozen original
`92a4da411c6d6c731da72203665b2533b0fb91b37f7d7f0c82e82d8ace0f1e5d`.
Startup record SHA256`a4ea04ce2f8764cf61507e58c773cd44d165648fd7adbe916cd6ff5e552de7a8`.
Admission:load12.72/20CPU,available91GiB,disk80GiB;other jobs untouched,no GPU.
First model is training,not final selected checkpoint; no accuracy/proof claim.
After completion independently audit all six models/300epochs and earliest
validation selection before freezing the240-request verification study. Apply
the[post-prefix mechanism supplement](multilayer_stage1_20261022.md) to1b too.

Stage1b is the user-authorized model-validity follow-up, not a replacement for
the twelve frozen stage1a models. The old low accuracies and concentrated routes
remain results. The new recipe changes a declared bundle of training/architecture
factors; it is not an individual residual-versus-balance causal ablation.

Six CPU models: L2E8 and L3E8, seeds0/1/2. A linear3072→128 stem is followed by
two or three128→128 selected-softmax top2 MoE blocks, each `x + MoE(x)`, then one
linear128→10 classifier. The final head makes all residual additions dimensionally
defined. No extra stem activation, normalization, skip projection or scaling.
Routers retain128 hidden units and experts retain256/128 hidden units.
Loss is cross entropy plus0.1 times the **sum** of per-layer switch-balance terms.
Original-family execution remains mean reduction and its original build/RNG path.

Other recipe items are unchanged: common seed0 45k/5k CIFAR split,50epochs,
AdamW.001/weight_decay.0001,batch256,ToTensor only,no scheduler/augmentation,
earliest validation-best checkpoint,one final test report per seed,all seeds kept.
CPUfloat32 training and separately materializedfloat64 verification are distinct.
Two CPU numerical threads bound concurrent demand;8GiB per-process AS and nice10,
no GPU. Training and storage are charged separately. Resource checks precede launch;
do not interrupt others or alter the immutable running1a experiment.

Before training, one new synthetic≤300s batch covers L2/L3 forward equivalence,
finite update, exact checkpoint/optimizer/shuffle restoration, sum-balance loss,
and hand-computed static/ACT/external prefix/head/residual graphs. Census applies
the signed affine stem bound and adds both residual input endpoints; observations
must not be excluded. No trained model or native solver is used in these controls.

After all six models finish, freeze fresh common inputs in ascending order outside
the full historical exclusion and stage1a selected cohort, not by route count or
bounds. PublicPGD and30×3 census follow1a. Verification is6models×20inputs×2arms
(ACT v1,GenBaB pruning),240calls,20h,300s/request. All auxiliary cost, incomplete
coverage,errors,UNSAFE sources and guarantee levels remain explicit. Freeze the
external r3 compatibility and plain-CROWN prefix/alpha-GenBaB leaf configuration
used by resumed1a; the alpha-CROWN prefix entrance experiment remains separate.
Apply the same L3 primary definition and Wilson rule. Agreement between1a/1b
supports the corresponding next step; disagreement is gray/report. The separate
alpha-CROWN entrance test precedes new mechanism design if1a finds a bottleneck.
Neither successful training nor graph compatibility establishesG1/G3/G5.

## Control and training freeze

`results/multilayer_stage1b_control_20261022_r1` passes both L2/L3 controls in
8.504584s including existing owned-session cleanup (body7.102405s). Training
forward/dynamic forward, sum-balance, serialized model/optimizer/shuffle restore,
ACT/static full and prefix graphs, guard-max and residual census checks pass.
No trained checkpoint or solver was used. The training/propagation sources are
exactly the controlled versions; a subsequent freeze-helper metadata correction
clarified that the classifier head is applied once, with no implementation change.

[Six-model training freeze](multilayer_stage1b_training_freeze_20261022.json)
binds data, original split hash, all source files and control receipt. No real1b
training has started at this snapshot: load38.49 on20 logicalCPUs warrants not
adding a second numerical job. This is resource scheduling, not a scientific
stop or a request for new authorization. Do not alter1a while it executes.
