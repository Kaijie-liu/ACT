# AdvMoE official-code seed-0 training numerical exclusion

## Completed execution, rejected checkpoint

The completed run is `seed0_r3`, labeled **official-code,
Blackwell-compatible dependency reproduction**. It executes the unmodified
official source at commit `c50796fb8284512b6f6ad8e843f95182cec527cf` with the
frozen configuration
`act/pipeline/moe/configs/advmoe_training_seed0_r3.json`. The only change from
the two excluded predecessors is operational: r3 runs the ACT supervisor in a
persistent user service so that its lifetime is independent of a Codex/tmux
execution scope. All scientific arguments are identical across r1--r3.

The supervisor completed with return code zero after 6,546.53 seconds. It
preserved exactly 100 consecutive, loadable, epoch-addressed checkpoints and
reported no missing checkpoint epoch. The official source clone remained
clean at the pinned commit and tree. The first independent audit,
`act/pipeline/moe/results/baseline/advmoe_training_seed0_r3_audit.json`,
reports structural `PASS` with zero issues, but it did not check tensor
finiteness and is superseded for scientific acceptance by the numerical audit
described below.

The strengthened audit reloads, rehashes, and checks every floating tensor in
all 100 immutable checkpoints. It reports `FAIL`: from snapshot 1 through 100,
all 270,578 floating elements in the standalone router are NaN, all 269,202
floating elements in its optimizer state are NaN, and all 4,599,826 embedded
router-reference elements in the model state are NaN. In contrast, all
5,570,378 non-router model elements and all 5,565,450 main-optimizer elements
are finite at every checkpoint. The authoritative record is
`act/pipeline/moe/results/baseline/advmoe_training_seed0_r3_numerical_audit_r2.json`.
Consequently, `seed0_r3` is execution-complete but scientifically excluded as
a learned-router MoE checkpoint.

## Metrics and checkpoint identity

The released evaluation path reports the following endpoints, retained only
as diagnostics of the finite main network under the invalid router state:

| identity | saved epoch | clean accuracy | released 10-step adversarial accuracy | SHA-256 |
|---|---:|---:|---:|---|
| best adversarial checkpoint | 90 (after training epoch 89) | 93.79% | 91.34% | `2b764ba110d110a6e9f17e8ff76f4dfd951c5947d84bb4fde865eab997e642c1` |
| final checkpoint | 100 (after training epoch 99) | 93.88% | 90.59% | `4ba196d18044b23716ca884c763bd1322154cb0d6375af520371a3d57ba8365a` |

The best-checkpoint hash matches exactly one immutable snapshot, epoch 90,
and the final live checkpoint matches the epoch-100 immutable snapshot. These
numbers are not accepted learned-MoE metrics or formal robustness
certificates. With both route scores NaN, PyTorch `argmax` deterministically
selects index 0, so the reported accuracy is effectively the performance of a
single selected path rather than evidence of a functioning learned router.
The released code selects the best checkpoint using the test-set adversarial
accuracy; this artifact limitation remains attached to every diagnostic use
of that checkpoint.

Structural audit attempt 001 is retained because it correctly rejected a
missing `size_bytes` field in the supervisor's final "existing snapshot"
record. The checkpoint itself was complete, loadable, and hash-correct. The
subsequent structural audit independently reconstructs that one metadata
field, records it under `recovered_metadata`, and verifies all 100 contents.
That structural result did not establish numerical validity. Future
supervisor records include the size for both new and existing snapshots, and
future landing audits fail closed on non-finite model, router, and optimizer
tensors.

## Excluded endpoint telemetry

Endpoint telemetry r1 is retained without scientific interpretation. Its
independent audit finds 724 non-finite JSON values, nine non-finite raw arrays
for each trained endpoint, and non-finite checkpoint routers. The apparent
`10,000:0` route counts and zero PGD flips are artifacts of NaN `argmax`
behavior and are not route-collapse or stability measurements. The record is
`act/pipeline/moe/results/baseline/advmoe_training_endpoint_telemetry_seed0_r1_audit.json`.

## First-nonfinite diagnosis

Three bounded reproductions use the unchanged released trainer and stop within
the first three real CIFAR-10 batches. Batches 0 and 1 remain finite. On
zero-based batch 2, all 269,202 router gradient elements are NaN immediately
before the router optimizer step, while every router parameter and buffer and
every existing router-optimizer state element is still finite. The non-router
model remains finite. This excludes the optimizer step itself as the first
source.

PyTorch anomaly tracing localizes the first invalid derivative to
`XlogyBackward0` in the released router KL term at `train_moe.py:143`, reached
by `loss.backward()` at line 156. Every recorded router input and output before
that backward call remains finite. The maximum within-example router-score gap
grows from 0.862 on the first batch to 25.827 on the second and 320.282 on the
third. On the third batch, the float32 softmax of the clean router scores
contains 16 exact zeros; no earlier recorded call contains a zero. Because the
KL target remains differentiable, its `xlogy` target derivative is active at
those underflowed zeros and emits NaN.

The independent r4 audit binds all configs, raw results, logs, official source
identity, batch position, forward finiteness, first softmax underflow, and
autograd traceback. It reports `PASS` with zero issues at
`act/pipeline/moe/results/baseline/advmoe_router_nonfinite_diagnosis_seed0_r4_audit.json`.
The earlier r2 audit failure is intentionally preserved because that auditor
omitted the console log required to bind the anomaly's forward traceback. No
official-source file was modified.

## Excluded predecessors

- `seed0_r1` stopped after 27 snapshots because the original monitor read and
  hashed a checkpoint while the released writer was replacing it. The stable
  copy protocol and a forced-race regression close that defect.
- `seed0_r2` stopped after 70 snapshots when the enclosing Codex-created tmux
  systemd scope was externally removed. It has no Python traceback, reboot,
  kernel OOM, CUDA Xid, disk failure, or checkpoint corruption.

Neither predecessor supplies a scientific endpoint. Although their
checkpoints contain model, router, and optimizer states, the released format
omits Python, NumPy, PyTorch CPU, and CUDA RNG states. Resuming would therefore
change shuffled data order and adversarial random starts, so r3 restarted from
epoch zero instead of presenting a non-equivalent continuation as one run.
They have not been promoted by the r3 numerical failure.

## Scope and next gate

The upstream repository has no license file. ACT therefore records checkpoint
hashes and reproduction instructions but does not redistribute the checkpoint
pending legal review. This execution does not unlock trained-checkpoint router
telemetry or the five-radius two-path staged-verification table. The bounded
diagnostic has located the first invalid derivative. Before any replacement
training, a separately labeled compatibility variant must express the same
router KL without the underflow-singular target derivative, prove value and
gradient agreement on the finite regime, and pass a finite-state smoke gate
beyond the third real batch. It must not be described as unchanged official
training. No current AdvMoE checkpoint establishes route
stability, Route A coverage, learned-router external validity, or any formal
SAFE result.

## Compatibility-bridge finite-state smoke

The compatibility bridge leaves every finite native softmax gradient unchanged
and replaces a non-finite incoming gradient only where the corresponding
softmax probability is exactly zero. A non-finite gradient at any positive
probability fails closed. Unit tests establish exact finite-regime value and
gradient agreement, agreement with the stable logit-space KL expression in an
extreme underflow case, an unbridged NaN mutation control, and the positive-
probability failure control.

Smoke r1 is preserved as a harness interaction failure: anomaly mode aborts at
`XlogyBackward0` before the downstream tensor hook may run, so it applies zero
replacements. R2 disables only that diagnostic mode and retains all four
per-batch optimizer-stage finiteness checks. It completes 16 main and 16 router
updates, including the original failure batch, with 64/64 finite phase checks.
The bridge is genuinely exercised: 23 underflowed-gradient elements are
replaced in 32 gradient-hook calls. Despite a maximum router pair gap of
62,961.53, all 269,202 final router parameters, gradients, and optimizer-state
elements remain finite. The independent audit reports `PASS` with zero issues
at
`act/pipeline/moe/results/baseline/advmoe_router_finite_smoke_seed0_r2_audit.json`.

This unlocks configuration of a from-scratch compatibility-variant run, not
acceptance of the excluded official run. The variant must retain its explicit
label, preserve the official clone, fail on any non-finite state outside the
narrow bridge condition, and undergo checkpoint-by-checkpoint numerical audit.

## Accepted numerical-compatibility endpoint

The separately labeled `seed0_compat_r1` run completed all 100 epochs in
7,128.33 seconds. It is an **official-code numerical-compatibility variant;
softmax-underflow gradient bridge**, not an unchanged official-code result.
The bridge leaves finite native gradients unchanged and was exercised during
the run: 18 non-finite incoming gradient elements at exact-zero softmax
probabilities were replaced across 78,200 gradient-hook calls and 391,000
softmax calls. The official source remained clean at commit `c50796fb8`.

The independent endpoint audit reloads and rehashes all 100 consecutive
snapshots. It reports `PASS` with zero issues and no recovered metadata. Every
floating tensor in the non-router model, model-embedded router, standalone
router, main optimizer, and router optimizer is finite in every snapshot. The
final checkpoint is snapshot 100, SHA-256
`e2d93896e5be1fdbb1c9538f9f09014bdc6d3067ac5501352f61475b5294b49e`.
It reports clean accuracy 85.67% and released 10-step adversarial accuracy
61.14%. The released best-adversarial checkpoint is snapshot 95, SHA-256
`143f5d0db191dc74be5d77d5ff10d0e320709a0718a11ac0e3c6668e382a63db`;
its clean and released adversarial accuracies are 85.36% and 61.80%.

These are empirical released-path metrics, not formal certificates. The
released test set participates in checkpoint selection, and the repository's
missing license keeps checkpoint redistribution disabled. The accepted audit
is `act/pipeline/moe/results/baseline/advmoe_training_seed0_compat_r1_audit.json`.
This result unlocks trained-router endpoint telemetry and the frozen two-path
evaluation; it does not predetermine either result.

Endpoint telemetry is preregistered separately in
`act/pipeline/moe/configs/advmoe_training_endpoint_telemetry_seed0_compat_r2.json`.
It binds the accepted training audit and exact best/final hashes, retains the
compatibility label, and compares initialization, best, and final states under
both registered BatchNorm semantics. Its strong PGD row remains empirical
witness search and cannot establish route stability.

The r2 telemetry run completes and independently audits with `PASS` and zero
issues. Under deployment/eval BatchNorm semantics, initialization routes all
10,000 ordered test inputs to expert 0. The accepted best checkpoint routes
4,718/5,282 inputs and the final checkpoint routes 5,012/4,988; their effective
route counts are 1.997 and 2.000. Train-mode ordered co-batch diagnostics are
also balanced at 4,782/5,218 and 4,889/5,111, respectively. Thus the released
supervised router objective escapes the initial one-route collapse under both
registered semantics.

On the frozen 20-input diagnostic subset, the 10-restart, 100-step attack at
8/255 finds route changes for 7 best-checkpoint inputs and 8 final-checkpoint
inputs. These are concrete attack discoveries, not route-stability estimates
or formal certificates. The ordered-archive clean accuracies recomputed by the
telemetry path are 85.37% and 85.67%. The audit recomputes all route counts,
attack-success counts, and perturbation norms from finite raw arrays and
rehashes both checkpoint routers. The tracked record is
`act/pipeline/moe/results/baseline/advmoe_training_endpoint_telemetry_seed0_compat_r2_audit.json`.
