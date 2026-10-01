# Real model intake limits for checked HybridZ

The new source-checked HybridZ path cannot yet accept either existing real model
family. The [metadata audit](hz_real_intake_20261001_r1.json) separates definite
dimension/file refusals, unsupported operators, a coefficient representation
gap, and unmeasured execution capacity. It loads no checkpoint, input, tensor
value, model, HZ or optimizer. Eight standard-library diagnostic tests pass.

This is not another version of the old direct-node capacity experiment. No old
input, full-size source, GPU preflight or solver was rerun. All goal gates remain
OPEN. The concrete next mechanism is checked binary64 outer-enclosure lowering;
changing constants alone cannot establish real intake.

## Registered models and required obligations

| Family | Registered structure | Full all-pair duties |
|---|---|---:|
| bal010 MLP | 3,072 inputs; eight experts, ten classes; router 3,072–128–8; each expert 3,072–256–128–10 | 28 pairs × 9 properties = 252; up to 504 endpoint targets |
| convolutional | 3×32×32 inputs; four experts, ten classes; pooled affine router; two strided convolutions, pooling and two affine expert layers | 6 pairs × 9 properties = 54; up to 108 endpoint targets |

The MLP recipe is registered for seed1/2; seed0 matches registered parameter and
tensor counts, **not a newly loaded topology**. Each MLP has 6,961,368 parameters
and 52 parameter tensors. Checkpoint and state hashes are retained as metadata
labels, not verified against checkpoint contents in this study. The convolutional
model's recorded test accuracy is 67.06%, not a newly evaluated accuracy.

Sources: [MLP recipe](../act/pipeline/moe/configs/experiment1_multiseed_training_r1.json),
[factory](../act/back_end/moe/factory.py),
[registered model identities](../act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json),
[convolutional review](../act/pipeline/moe/results/conv_training_review_20260915_r1.json)
and [convolutional factory](../act/back_end/moe/conv_factory.py).

## Definite current refusals

| Interface | Current restriction | Consequence |
|---|---|---|
| Source properties | experts ≤4; classes ≤5 | MLP exceeds both; convolutional exceeds classes |
| Every checked HZ state | factors and output rows ≤128; equality plus inequality rows ≤256 | Both 3,072-dimensional inputs fail before propagation |
| Endpoint roster | at most four properties | Ten-class full classification needs nine |
| Support batch | at most eight queries | Nine properties need nine even with a degenerate gate, otherwise eighteen |
| Source graph/capture | Flatten, Linear, ReLU only | Conv2d and AvgPool2d remain unsupported in this new path |
| Portable saved bundle | member ≤4 MiB; whole bundle ≤32 MiB | MLP parameter base64 alone is at least 74,254,592 bytes |

The new source supervisor also limits several individual input/prefix/result
messages to 4 MiB. These are file contracts, not a memory proof. Do not substitute
the old direct-node checker's 64 MiB member limit or infer that its chunked format
has already been integrated here. Direct mathematical checking has no universal
package byte cap; that does not establish usable capacity.

The [audit script](../scripts/audit_hz_real_intake.py) checks named literal
dimension contracts by AST without importing production modules. Source hashes
bind its observations. The input constructor assigns one factor per coordinate,
including zero-radius coordinates; hence this refusal does not depend on image
values or the number of unstable ReLUs.

## Growth in this HybridZ representation

In the existing construction, an unstable ReLU adds two continuous factors, one
binary factor, one equality and two inequalities. A nonzero affine compensation
adds one continuous factor per affected output row. A pair adds `2(E−2)` guards.

For the MLP pair, the all-unstable/every-affine-error-nonzero scenario gives 896
unstable ReLUs and up to 924 affine error factors. Counting a shared input/router
once gives 5,788 continuous plus 896 binary factors, or **6,684 total**, with 896
equalities and 1,804 inequalities (**2,700 constraint rows**). These are recipe
scenarios, not measured factor counts, nonzeros, memory or running time. The old
4,892 direct-node LP variables are a different representation and not this count.

The current all-pair producer separately propagates two experts per pair: 56
expert propagations and 340 layer records including the single router trace.
It retains all pair traces and live pair objects until endpoint preparation.
This is a concrete repeated-structure opportunity, not proof that it dominates
time. Since this producer's affine transforms and ReLU box ranges do not use
guard constraints to tighten ranges, shared expert templates with pair-specific
guard views may admit a renaming-equivalence proof. That is a **separate future
factorization experiment**, not permission to omit guards or merge private factors.

## A numerical representation gap independent of dimensions

The new `exact_float` insists that every rational coefficient round-trip exactly
through binary64. Finite binary64 source parameters are not closed under exact
arithmetic. The diagnostic checks five algebraic witnesses without constructing
or executing a network:

- Let `w=1+2^-52`, with generator coefficients `w` and `w·2^-96`. Rounding the
  two products `w*g` leaves exact errors `2^-104` and `2^-200`. Each error is
  representable, but the required compensation sum is not.
- A ReLU preactivation with center 1 and generators 1 and `2^-54` has exact
  upper endpoint `2+2^-54`, which is not representable.
- Two representable router centers, 1 and `2^-54`, have a nonrepresentable
  exact guard RHS difference `1−2^-54`.
- A ReLU with center `2^-56`, generator `1/2` and valid supplied range `[-1,1]`
  has equality RHS `2^-56−1/2`, which is not representable.
- Half the smallest positive binary64 subnormal is not representable. Blind
  half-scaling can therefore lose a nonzero coefficient even at tiny magnitude.

These establish that unrestricted no-loss lowering is not generally valid. They
do **not** establish which stored trained coefficients encounter it, how often,
or the numerical size of the resulting loss. We did not inspect those values.
Overflow, coefficient underflow, equality rounding and inward guards must not be
silently accepted. The old checker correctly rejects rather than hides the gap.

## Next decision

Proceed with a separately versioned [binary64 outer-enclosure control
design](hz_binary64_enclosure_design_20261001.md). Preserve an exact rational
reference and prove its inclusion in the actual stored HybridZ using explicit,
independently checked output and constraint compensation. Keep old equality
contracts, limits, proofs and production defaults unchanged.

After this mechanism is sound in finite controls, address shared/private block
representation, complete duty streaming and realistic capacity separately. The
GPU candidate kernel cannot solve a failure to create/check its source. No new
real execution or physical GPU retry is authorized by this report.

The eight diagnostic tests and 22 navigation/handoff regressions pass; a fresh
read-only regeneration matches the saved report. Separate read-only reviews
checked the metadata counts and the proposed inclusion algebra. They are not
independent human review or an executed implementation proof. The compact report
is 8,823 bytes. Workspace logical size was 223,832,599,996 bytes at handoff, about
125 KB above the preceding recorded snapshot; no file was deleted.

Recheck without site packages:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_hz_real_intake
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S scripts/audit_hz_real_intake.py --check docs/hz_real_intake_20261001_r1.json
```
