# ICML 2025 RT-ER B1 reproduction diagnostics

This record separates execution success from scientific reproduction. The
seed-0 Blackwell-compatible run completed all 130 epochs, landed its ordered
full-test endpoint, and passed independent replay. It did not enter either
frozen paper-reference interval: clean accuracy is `34.22%` versus `77.81%`,
and PGD-50 accuracy is `32.70%` versus `69.09%`. Per the preregistered
asymmetric rule, this remains a seed-0 outcome until the frozen seed-1 run
lands.

## Tier 0: trajectory and executed augmentation path

The complete local W&B compatibility log contains exactly 130 training rows;
the immutable checkpoint schedule contains exactly 13 test rows. Clean test
accuracy peaks at `37.40%` at epoch 30. From epochs 20 through 130 it remains
within `32.96%--37.40%`, with a fitted slope of `-0.0231` percentage points per
epoch. The endpoint robust-to-standard ratio is
`32.70/34.22 = 0.95558`. This combination is diagnostic of a run that learned
little; it does not establish why.

The released, pinned source configures horizontal flip, two-pixel random
translation, and cutout size eight on the FFCV training loader. The disclosed
launcher did not pass `--noaug`, and the official source remained unchanged.
No tensor-level augmentation trace was retained during the completed run, so
the evidence is source-to-execution identity, not replay of sampled transforms.

The publication-ready SVG keeps text as text:
`paper/figures/icml2025_rt_er_b1_seed0_training_trajectory.svg`.

## Tier 1: paper/source configuration comparison

| Field | Paper | Released/executed path | Classification |
|---|---|---|---|
| epochs | 130 | script default 200; launcher passes 130 | reconciled by disclosed paper-config argument |
| optimizer | not located | Adam | paper underspecified |
| cyclic LR | described as starting at `1e-4` | `base_lr=5e-5`, `max_lr=1e-4`, `step_size_up=500`; PyTorch initializes at `5e-5` | text/code semantic ambiguity |
| weight decay | not located | Adam default zero | paper underspecified |
| RT-ER beta | reported beta 6 | launcher beta 6 | consistent |
| augmentation | generic statement and citation | flip, translate 2, cutout 8 | paper underspecified; source path confirmed |
| adversary | epsilon 8/255, PGD-10/50 | same; code additionally fixes step 2/255 | consistent with code-only detail |
| mixed precision | not located | enabled by default | paper underspecified |
| router scope | objective written over robust-MoE parameters | released hard dispatch leaves router fixed | parameter-scope gap; separately supported by tensor and optimizer-state evidence |

The optimizer, weight decay, AMP, and exact augmentation rows are not called
paper/code contradictions. The LR wording does not uniquely specify whether
`1e-4` is the initial, maximum, or nominal rate, so it remains an ambiguity.

## Seed-1 follow-up

Seed 1 reuses the unchanged official script, model, objective, author
arguments, data, endpoint attack, and frozen accuracy intervals. Only the
previously unspecified stochastic seed changes from 0 to 1. A 30-GiB GPU and
60-GiB disk launch gate prevents unsafe co-scheduling. The resource watcher
also requires a clean, remote-synchronized feature branch immediately before
launch. It then runs the same 13-checkpoint telemetry schedule, epoch-50
rehearsal, endpoint replay, audit, documentation, commit, and push sequence.

## Compatibility consolidation

Python 3.11 `typing.override` support now lives only in
`act.util.typing_compat`. Exact environment tests import every Blackwell B1
entry point and, separately, every `act-py312` control/telemetry entry point.
The split is intentional: Blackwell runs the official model and endpoint,
while SciPy/HiGHS telemetry remains in `act-py312`.

## Retained failed generation

The first Tier-0 report-generation attempt failed closed because two-column PDF
extraction separated a long paper-text anchor. Its partial JSON and SVG are
retained with `.failed_attempt001` names. The successful rerun uses two local,
nonambiguous phrase anchors; no scientific value or threshold changed.

## Evidence

- Tier 0: `results/baseline/icml2025_rt_er_b1_seed0_tier0_diagnostics.json`
- Tier 1: `results/baseline/icml2025_rt_er_b1_hyperparameter_audit.json`
- independent audit: `results/baseline/icml2025_rt_er_b1_diagnostics_audit.json`
- seed-1 telemetry config: `configs/icml2025_route_telemetry_blackwell_seed1.json`
- retained failed landing protocol: `configs/icml2025_b1_landing_protocol_seed1_r1.json`
- repaired landing protocol: `configs/icml2025_b1_landing_protocol_seed1_r2.json`
- clean-clone retry protocol: `configs/icml2025_b1_landing_protocol_seed1_r3.json`

The first seed-1 attempt trained through epoch 10 and then failed closed before
telemetry read the checkpoint. Its config used the semantically preregistered
status `PREREGISTERED_BEFORE_SEED1_EXECUTION`, while the telemetry entrypoint
accepted only the canonical `PREREGISTERED_NOT_RUN`. The failed run root,
checkpoint, metrics, and logs are retained. The repaired r2 attempt changes
only that protocol spelling and attempt/log identities; model, data,
objective, seed, epochs, telemetry, endpoint, thresholds, and tolerances are
unchanged.

Before relaunch, the repaired entrypoint replayed the retained epoch-10
checkpoint over all 10,000 ordered test inputs. The affine-oracle reference
crosscheck covered 100 inputs with maximum radius error `6.94e-17`, and all
200 concrete route-boundary witnesses replayed. The six-second integration
result and hashes are tracked in
`results/baseline/icml2025_rt_er_b1_seed1_attempt1_failure_and_repair.json`.

The explicit replay was launched without the supervisor's
`PYTHONDONTWRITEBYTECODE=1` environment and generated three untracked Python
bytecode files in the official clone. The r2 supervisor correctly failed its
clean-clone gate before creating a training run. Those three replay-generated
files were moved, with hashes, to a recoverable quarantine under
`/data1/Kane/MOE/baseline_runs`; no source or user artifact was deleted. The
official clone is clean again. The r3 protocol retains both failed attempts
and changes no scientific field.

The independent diagnostic audit reports zero issues.
