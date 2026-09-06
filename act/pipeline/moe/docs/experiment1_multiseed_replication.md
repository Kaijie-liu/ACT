# Experiment 1 multi-seed formal replication

## Question and scope

The current formal Route A evidence is tied to one verification-scale training
artifact, `cifar10_top2_e8_seed0_bal010.pt`. Official-scale RT-ER and AdvMoE
experiments establish numerical conformance and specialization, but their CROWN
backend is not outward rounded and therefore does not independently replicate
the formal certificate effect.

This bounded replication addresses the red-team question:

> Are the verification-scale candidate, width, and route-changing certificate
> effects specific to the seed-0 trained model?

It replaces further Lagrangian multiplier/CROWN effect search. The Lagrangian
holdout remains locked.

## Stage R0: frozen training

Train seeds 1 and 2 with the exact accepted `bal010` recipe:

- CIFAR-10 from the project-local TorchVision root;
- selected-softmax weighted top-2, eight experts;
- router hidden width 128 and expert widths 256/128;
- Switch-style balance loss with coefficient 0.10;
- deterministic seed-specific 90/10 train/validation split;
- 50 epochs, batch size 256, AdamW learning rate 0.001 and weight decay 0.0001;
- save the best-validation checkpoint and evaluate test only after selection.

Both registered seeds are retained regardless of accuracy, balance, or later
verification yield. A failed run is preserved and is not replaced by another
seed. Checkpoints, logs, and raw summaries remain under `/data1/Kane/MOE`.

The tracked configuration is
`act/pipeline/moe/configs/experiment1_multiseed_training_r1.json`. The runner
requires a clean `feat/moe-route-verification` worktree and the existing
`act-py312` interpreter, refuses overwrite, records the CIFAR-10 file manifest,
and writes its summary incrementally.

After both runs terminate, `audit_experiment1_multiseed_training.py`
independently checks the dataset/config/artifact hashes, reconstructs each
seed-specific validation split, replays validation and ordered-test metrics,
checks every saved tensor for finiteness, and requires exactly 50 epoch log
rows. This audit does not query a verification endpoint.

## Stage R1: formal endpoint freeze after R0

R1 is not queried during training. Once both training outcomes are frozen, a
separate commit must bind, before execution:

- an ordered cohort rule that is independent of verification outcomes;
- the fixed-radius and route-boundary obligations;
- the existing outward-safe HZ/HiGHS numerical policy;
- a per-model budget and failure rules;
- input-clustered statistics and model-level aggregation.

The primary interpretation is model-level, not row-level. Replication on both
registered models supports seed-robust wording; one positive and one null model
supports model-dependent wording; no replication restricts the formal effect to
the original seed-0 artifact. No model may be excluded for low accuracy or
unfavorable routing geometry.
