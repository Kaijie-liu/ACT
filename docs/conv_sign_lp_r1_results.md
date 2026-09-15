# Sign-sufficient LP evidence: two checked positive controls

Execution `e269821cf` completed the two frozen post-selected property controls
from `docs/conv_sign_lp_r1.md`. Results, not new complete-request certificates:

| Input / pair / property | Exact-checked lower bound (decimal display) | Continuous factors after relaxation | Relaxed binaries | Capture / proposal / check / second check seconds |
|---|---:|---:|---:|---|
|16 / {0,3} / class5 − class0|3.690808268875|8,078|1,668|29.32 / 3.37 / 1.67 / 1.67|
|98 / {1,2} / class0 − class1|5.712379468352|6,803|1,243|24.61 / 2.52 / 1.02 / 1.02|

The JSON stores exact fractions, not just these rounded decimals. Both fresh
checker processes use Python `-S` and load neither Torch, NumPy nor SciPy.
A third archival reconstruction agrees exactly. Checks independently rebuild
the continuous LP from each supplied scalar F0 HZ, then verify signed dual
multipliers, objective center and finite-box residual using exact rationals.
These are more than a scalar-dual or JSON consistency check.

Before capture the registered recipe performs route/guard/support work:368
and336 native calls, recorded in801 and737 journal events. Each journal ends
with the intentional `CapturedProperty` exception at exactly one scope,
**before any weighted-property MILP solve**. No verification verdict is emitted
by these capture workers. Resource use is CPU-only, one worker/thread; no
dependency or production configuration changes occurred.

All-stage totals are36.03 and29.16 seconds. They include loading, paid support,
export, proposal and both isolated checks, but not the later archival review.
The raw proof-control directories occupy23,773,603 and14,713,616 bytes. This is
one-property generation cost, not a nine-property complete-request timing or
a speed comparison with the300-second full verifier. No raw arrays/checkpoints
are committed.

## What this establishes

For these **newly generated supplied F0 HZ objects**, the continuous LP already
has a sufficient positive bound. Integer branch-and-bound is therefore not
necessary to prove these two stored obligations. The exact checker accepts
the bound because of the multipliers and residual correction, not because a
native solver announces an optimal status. It does not need a zero gap.

The existing proposer still asks HiGHS for a completed LP to obtain candidate
multipliers; no time-to-first-positive trajectory was measured. This experiment
does not implement or measure early stopping. Nor is the new lower bound a
reconstruction of the historical MILP dual: full V2 did not save those matrices,
and regenerated support under finite time budgets can differ.

This is a constructive reason to pursue independently checked sign evidence,
instead of simply increasing the MILP time limit. It is **not** evidence that
all convolutional timeouts are caused only by the optimal-status acceptance
gate, or that all nine properties of these inputs can already be checked.

## Evidence tier and trust boundary

The result tier is `CHECKED_POSITIVE_SUPPLIED_F0_LP`. Complete requests
certified: **0**. The two controls cover one property each; both original
full V2 requests remain TIMEOUT, and neither is a new route-changing example.

Trusted components remain:

- Network/input-to-HZ propagation and its floating coefficients.
- Guard/route lowering and exclusions.
- Floating construction of the supplied F0 HZ, including its gate/product
  outer relaxation, and its association with the requested property.

Checked components are the supplied-HZ→continuous-LP transformation and the
signed-dual, exact-rational lower-bound arithmetic. This does not inherit the
stronger pre-F0 rational-McCormick contract of the earlier MLP study. Checking
a nonpositive lower bound would not prove a violating point or UNSAFE.

## Declared publication-order deviation

The freeze commit existed locally before any control executed. The first
`git push` was rejected by GitHub with Internal Server Error (reported UTC
2026-09-15 09:27:14). The run nevertheless launched before successful remote
acknowledgement. Immediate retry pushed the **same** commit while the controls
ran, with no code, configuration or sample change. Exact remote publication
time was not captured by the runner; this account is from the operator tool
transcript.

Thus evidence auditing passes, but the specified **push-before-launch** order
was not satisfied. The archive explicitly records
`DECLARED_REMOTE_PUBLICATION_TIMING_DEVIATION`, not an unqualified protocol
PASS. This post-selected feasibility study is not a confirmatory efficacy
experiment. Neither rerunning nor backdating is used to hide the deviation.

## Reproduction and next bounded decision

Archive: `act/pipeline/moe/results/conv_sign_lp_review_20260915_r1.json`.
It binds all proof artifacts, the original archives, frozen code, model and
input identities, stage terminals and capture journals. Fifteen focused
producer/checker tests and four capture-review controls pass. Recheck without
new model or solver calls:

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scripts.review_conv_sign_lp --check
```

The next useful separately frozen stage is **all necessary properties of these
same two observed requests**, not more samples or a broader backend search.
Each property must either have an independently checked positive bound or a
checked scoped reuse fact; route coverage and final aggregation must also be
accounted for. Report partial/unknown results if any obligation fails. Only
complete coverage could support a request-level conclusion, still conditional
on the explicitly trusted lowering. Keep this evidence path opt-in and leave
the production status0 gate unchanged. A later, separate construction check
can transfer the pre-F0 rational-McCormick machinery to reduce the trusted base;
it must not be silently assumed from this result. No follow-up is queued.
