# Residual multilayer model validity study

## Fifth-revision change 2026-10-25

User cancels the unexecuted ACT arm: the next version is **GenBaB pruning only,
120 complete requests,10h total,300s/request**, with the same six checkpoints,
selection rule, public PGD and census. The old two-arm240-call R4 freeze below is
retained unexecuted; it must not be edited or passed to the new launch. At this
registration the new queue is0/120,not launched. Nonpositive outcomes join the
rule-defined hard-core diagnostic pool; they do not reopen the stopped route-union
mechanism. New execution identity/freeze must bind the single-arm catalog first.

## Training archive and verification freeze 2026-10-24

Real training is complete: **6/6 models, 300/300 epochs**. The independent
[record audit](multilayer_stage1b_training_records_20261024.json) checks the
frozen code/data/weight identities, common 45k/5k split, all epoch records and
earliest validation-best checkpoint selection. It does not rerun accuracy or
establish an independently checked network proof.

| Model | Selected epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: |
| R2E8 seed0 | 19 | 50.78% | 50.79% |
| R2E8 seed1 | 13 | 50.54% | 50.91% |
| R2E8 seed2 | 21 | 51.26% | 51.15% |
| R3E8 seed0 | 22 | 50.40% | 50.97% |
| R3E8 seed1 | 13 | 51.28% | 50.79% |
| R3E8 seed2 | 13 | 51.40% | 52.18% |

Queue time was 4,111.390998s; retained training files are 252,333,975B plus
113,729B controller log, separate from verification's artifact budget. No
zero-use expert occurs in the reported test routing counts. The two most-used
experts account for 28.68%–36.43% of selections, rather than the old first-layer
99% concentration. This is a model-validity observation, not a claim that the
architecture change alone caused it, that accuracy satisfies G5, or that
verification is easier.

[Execution freeze R4](multilayer_stage1b_verification_freeze_20261024_r4.json)
binds all six weights, 24 source files and the existing external r3 compatibility
configuration. It retains epsilon 2/255, 300s/request, 240 complete slots,
20h overall including auxiliary work, and 128MiB new artifacts. Two arms are
ACT multilayer v1 and GenBaB with plain-CROWN prefix pruning/alpha-GenBaB leaves;
the alpha-CROWN prefix experiment is not silently enabled. The R3E8 primary
endpoint, public/full-model witness union and registered A/B/C mechanism rule
are inherited. There is no invented L1 cost anchor or excluded collapsed seed.

Selection is frozen as ascending common clean-correct inputs outside 1,855
historical identities plus all 30 stage1a selected identities (1,885 total).
The first 20 of 30 selected inputs form the paired verification cohort; all30
enter the census. Actual indices and float64 tensors will be materialized by the
first budgeted selection call, not selected here using routes or positive bounds.
No model inference or stage1b verification was performed while freezing:
**0/240 calls, not launched**. Existing stage1a runs from its immutable copy.

Two distinct standard-library catalog/terminal-audit controls pass. Catalog
equivalence preserves the original600-call order, rejects invalid arms/slot
counts, and the full synthetic240-call ledger checks UNSAFE union (n'=59),
paired counts, missing-terminal rejection and wrong-arm rejection. Two ledger
fixture errors (duplicate directory creation, then intentional overwrite through
the production no-overwrite writer) are retained; only the unfinished test was
rerun. Cumulative controlled elapsed time is23.219515s and retained control files
total19,341B including complete receipts. The earlier path-resolution preparation
error did not launch a worker; its empty root is retained. These are test-harness
failures, not real verification outcomes. See the compact
[freeze/control record](multilayer_stage1b_verification_records_20261024.json).

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
