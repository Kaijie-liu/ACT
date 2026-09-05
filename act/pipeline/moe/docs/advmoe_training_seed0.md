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
telemetry or the five-radius two-path staged-verification table. Before any
replacement training, a bounded diagnostic must locate the first non-finite
router update and establish a finite-state smoke gate spanning enough real
batches to cover that failure. No current AdvMoE checkpoint establishes route
stability, Route A coverage, learned-router external validity, or any formal
SAFE result.
