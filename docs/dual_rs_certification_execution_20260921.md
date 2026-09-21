# Dual RS: controlled training → final-epoch certification

Training is already launched, with its own12h supervisor and final90epoch
saved-state audit. Its source/configuration remains immutable while running.
Certification preparation is a separate stage; it has NOT run real Monte Carlo
queries at this entry. Nothing automatically starts on training exit.

## Frozen execution recipe

`configs/recent_moe/dual_rs_certification_recipe_r1.json` binds implementation,
native control receipts and public base-weight files. Fixed raw CIFAR test0,1;
N0=100 and N=10,000 for EACH executed stage; alpha=.0005 each; selector noise1;
classifier sigma candidates.25,.5,1; batch16; source-native CUDA autocast;2h
shared outer budget for setup, all samples, saves, child imports and count audit.
The small source postflight is separately reported, not hidden as free execution.
GPU gate24GiB, memory fraction.25, no fallback/retry or sample replacement.

`Smooth.certify` and its sampling method are the original implementations. A
subclass saves their completed selection/estimation counts without extra draws.
Six CPU controls in the frozen Dual RS environment pass, including identical
native results AND RNG after instrumentation, independent numerical Clopper–
Pearson computation, first-index ties, abstention, corrupt counts and failure
after partial evidence. ACT environment runs the non-native controls but lacks
statsmodels; these two native-specific tests are explicitly run in the author
environment, not treated as passing skips. [Archive](dual_rs_certification_controls_archive_20260921.json).

The first stage selects a sigma; only its chosen classifier is certified using
a different predetermined noise stream. If either stage abstains, composed
radius is zero. Otherwise the composed L2 radius is the smaller stage radius.
Correct-class radius is zero for a wrong predicted class. Never select another
sigma/checkpoint because it yields a better radius.

This is a probabilistic guarantee for smoothed functions under the author's
implementation/numerical assumptions: per-input failure probability at most
.001, or at most.002 for both pilot inputs together by a union bound. It is
NOT a deterministic HZ L-infinity SAFE, an independently checked floating-point
proof, or a paper-scale CRA estimate. Count auditing still trusts model/noise
execution and numerical probability tails.

## Final weight gate and supervised execution

After the training outer terminal is TRAINING_LANDED and its saved-state audit
passes, run:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/prepare_dual_rs_certification.py --bind-landed
```

This creates a NEW execution manifest binding epoch090.pt, its metadata,
training terminal and audit. It cannot accept early/best/resumed substitute
weights; an actual attempt before landing was rejected with no execution
manifest. Review/commit/push that manifest before launching:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/dual_rs_certification_pipeline.py --config configs/recent_moe/dual_rs_certification_execution_r1.json
```

All child execution and final count audit share the one outer process group and
deadline. A late/missing success cannot override TIMEOUT/ERROR. Completed counts
and partial stage files remain, but are not complete certificates. This launch
is still pending final weight availability and is not an automatic background
promise. Full real end-to-end certification remains an execution gate.
