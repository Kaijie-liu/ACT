# Residual multilayer model validity study

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
