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

R1 uses the same 40 CIFAR-10 images for both models. The cohort is the first 40
ordered test indices at or after index 1000 that are clean-correct for both
models and absent from the union of the seed-0 development and confirmatory
cohorts. This rule uses no router candidate, route-boundary, property-bound, or
certificate outcome. `experiment1_multiseed_selection_r1.json` records the
indices, both checkpoint hashes, and both excluded-cohort hashes; a separate
auditor must reconstruct it before execution.

For each model R1 runs the unchanged four fixed radii
`{0.25,0.5,1,2}/255`, followed by exactly one route-boundary endpoint at
`1.05 * route_upper` per input. Candidate, guarded-support, gate-elimination,
F0, tie, witness-replay, and numerical SAFE semantics are identical to the
accepted seed-0 confirmatory protocol. The hard deadline is 300 seconds per
boundary input; there is no closure rerun.

The model-level replication conditions are frozen as:

- exact-HZ candidate reduction versus zonotope at least 20% among route-
  unstable fixed-radius rows;
- route-unstable width-ratio median below 0.7 and p90 below 1;
- at least one full-denominator route-changing unique SAFE certificate;
- every UNSAFE witness replays in the full weighted model; and
- independent audit reports zero issues and no silent numerical fallback.

Both models must be reported separately. Only 2/2 model-level successes support
seed-robust wording; pooled row counts cannot substitute for that criterion.

The independent selection audit reconstructed all 40 indices exactly, verified
both checkpoint hashes and the 200-index excluded union, and reported zero
issues. Its raw SHA-256 is `839d586f...461abb`. R1 was still unqueried when
this audit completed.

## R0 result

Both registered runs completed at implementation HEAD `db4f34b4f` in 135.41
seconds total and passed the independent replay audit with zero issues.

| Seed | Best epoch | Best validation | Test | Effective experts | Max load | Min load |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 48.36% | 48.47% | 7.7635 | 17.27% | 6.93% |
| 2 | 35 | 47.92% | 47.63% | 7.7745 | 17.11% | 6.79% |

The checkpoint SHA-256 values are `cfd5fb07...641139a` and
`a60517c7...e97758`. Both models are retained. These are training and model-
balance results only; R1 has not yet queried a formal endpoint.

The tracked compact result is
`results/experiment1_multiseed_training_20260906_r1.json`; raw summaries,
checkpoints, logs, and the full audit stay below `data/moe`.

## R1 result

Both model runs completed at implementation HEAD `6d8864501` on the same 40
registered images. The two independent audits reported zero issues, closed both
guard-accounting identities, found no silent numerical fallback, and replayed
all 16 `UNSAFE` witnesses through the full selected-softmax model.

| Endpoint | Seed 1 | Seed 2 |
|---|---:|---:|
| Route-unstable fixed-radius rows | 32 | 18 |
| Exact smaller than IBP | 20/32 (62.50%) | 17/18 (94.44%) |
| Exact smaller than ordinary zonotope | 6/32 (18.75%) | 14/18 (77.78%) |
| Conditional width median / p90 | 0.352 / 0.477 | 0.352 / 0.447 |
| Guard binaries before -> after | 4,726 -> 3,876 | 4,491 -> 3,767 |
| Boundary SAFE / UNSAFE / UNKNOWN / TIMEOUT | 13 / 9 / 16 / 2 | 6 / 7 / 24 / 3 |
| Unique route-changing SAFE | 13/40 (32.5%) | 6/40 (15.0%) |
| Boundary applicability within 4/255 | 28/40 | 26/40 |
| Solved among applicable | 22/28 (78.6%) | 13/26 (50.0%) |
| F0 resolved base semantic incompleteness | 17/22 (77.3%) | 10/22 (45.5%) |

The executable frozen configs retained inherited GO thresholds in addition to
the condensed model-level prose above. To avoid relaxing a condition after
observing results, the final aggregation requires every independently audited
`go_condition`. Neither model passes that full conjunction: seed 1 misses the
ordinary-zonotope candidate threshold by one row (`18.75% < 20%`), while seed 2
misses the registered overall solved-rate threshold (`32.5% < 50%`). Thus the
complete registered bundle has `0/2` passes and does not support seed-robust
full-bundle wording.

The mechanism-level result is mixed and must not be replaced by pooled rows.
Route-changing unique SAFE certificates and route-unstable width separation
replicate on both retained models. Candidate reduction relative to IBP passes
on both, but reduction relative to ordinary zonotope passes on only one of two
models and is therefore model-dependent. Solver coverage also passes on only
one model. Descriptively, there are 19 unique-SAFE and 16 replayed-UNSAFE
model--sample pairs across 80 paired model inputs, and 35/54 applicable
model--sample pairs are solved; these pooled counts are not primary endpoints.

The compact, reproducible aggregate is
`results/experiment1_multiseed_replication_20260906_r1.json`. It binds the
selection manifest and both raw config, census, boundary, and independent-audit
hashes. No closure rerun, model replacement, or threshold change follows this
mixed endpoint.
