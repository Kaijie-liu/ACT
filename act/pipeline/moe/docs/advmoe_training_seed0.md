# AdvMoE official-code seed-0 training landing

## Accepted run

The accepted run is `seed0_r3`, labeled **official-code, Blackwell-compatible
dependency reproduction**. It executes the unmodified official source at
commit `c50796fb8284512b6f6ad8e843f95182cec527cf` with the frozen configuration
`act/pipeline/moe/configs/advmoe_training_seed0_r3.json`. The only change from
the two excluded predecessors is operational: r3 runs the ACT supervisor in a
persistent user service so that its lifetime is independent of a Codex/tmux
execution scope. All scientific arguments are identical across r1--r3.

The supervisor completed with return code zero after 6,546.53 seconds. It
preserved exactly 100 consecutive, loadable, epoch-addressed checkpoints and
reported no missing checkpoint epoch. The official source clone remained
clean at the pinned commit and tree. The accepted independent audit is
`act/pipeline/moe/results/baseline/advmoe_training_seed0_r3_audit.json`; it
reports `PASS` with zero issues.

## Metrics and checkpoint identity

The released evaluation path reports the following endpoints:

| identity | saved epoch | clean accuracy | released 10-step adversarial accuracy | SHA-256 |
|---|---:|---:|---:|---|
| best adversarial checkpoint | 90 (after training epoch 89) | 93.79% | 91.34% | `2b764ba110d110a6e9f17e8ff76f4dfd951c5947d84bb4fde865eab997e642c1` |
| final checkpoint | 100 (after training epoch 99) | 93.88% | 90.59% | `4ba196d18044b23716ca884c763bd1322154cb0d6375af520371a3d57ba8365a` |

The best-checkpoint hash matches exactly one immutable snapshot, epoch 90,
and the final live checkpoint matches the epoch-100 immutable snapshot. These
numbers are empirical model metrics, not formal robustness certificates. The
released code selects the best checkpoint using the test-set adversarial
accuracy; this artifact limitation remains attached to every use of that
checkpoint.

Audit attempt 001 is retained because it correctly rejected a missing
`size_bytes` field in the supervisor's final "existing snapshot" record. The
checkpoint itself was complete, loadable, and hash-correct. The accepted audit
independently reconstructs that one metadata field, records it under
`recovered_metadata`, and verifies all 100 contents. Future supervisor records
include the size for both new and existing snapshots.

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

## Scope and next gate

The upstream repository has no license file. ACT therefore records checkpoint
hashes and reproduction instructions but does not redistribute the checkpoint
pending legal review. The accepted training landing unlocks the preregistered
router telemetry and the five-radius two-path staged-verification table. It
does not by itself establish route stability, Route A coverage, or any formal
SAFE result.
