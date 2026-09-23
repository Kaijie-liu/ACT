# Scoped proof R1 — controls passed, real request frozen, NOT RUN

Implementation commit: `fe722caf33c7a22286084e51bdfd995206a3adae`.
Frozen after a clean worktree check on `feat/moe-route-verification`.
User authorized integration, controls and **freeze**, not execution.

- [Execution config](../configs/backend_controls/scoped_proof_execution_r1.json)
  SHA256 `0f2ada11d86ab74f923b42a8a057716aed2fa7b4ee31e26f06f0510168aeadc5`.
- [Control gate](scoped_proof_supervision_controls_20260924_r1.json): **53 PASS**,
  18 new supervision/evidence tests +35 unchanged source regressions;
  17.568s test-process wall time, not a real-model proof-generation benchmark.
- [Independent freeze review](scoped_proof_freeze_review_20260924_r1.json):
  PASS/zero issues, 494 tested source files checked, original scope and assets
  rehashed without torch/pickle/input decoding; output directory absent.
- [Execution and guarantee protocol](scoped_proof_supervision_protocol_20260924_r1.md).

## Fixed request and limits

One previously selected engineering request only: seed0/rank0, CIFAR4088,
label7, original frozen checkpoint and stored float64 center. New requested
rational2/255 domain clipped [0,1], unchanged exact binary64 margin1e-7.
All28 unordered pairs ×9 classes =252 obligations, no feasibility exclusions,
old matrices/facts/positive bounds or selected-row shortcuts. No input98.

300s end to end, two CPU threads, sampled parent+owned-group RSS8GiB.
The new owned-process supervisor includes loads, identity checks, declared
source conversion/check, candidate LP work, final exact checking, serialization,
receipt and cleanup. Its fixed budget partition and final-ledger disclosure
are in the protocol. No change to the old production numerical gate or25% rule.

New output root (must not yet exist):
`data/moe/results/scoped_proof_source4088_20260924_r1`.
No automatic resume, retry, time increase, sample substitution, gate tightening
or follow-up search. A nonpositive or incomplete bound is NOT_CLOSED, not a
network counterexample or an identified relaxation limitation.

## What the controls actually establish

Synthetic checkpoint-to-terminal paths completed for both positive and
nonpositive examples. Their evidence was independently reread with `python -S`,
without loading a checkpoint, solver or data. Wrong invocation/source/LP/property
bindings, missing/partial evidence, invalid dual signs and residuals are rejected.
An exception after the first candidate preserves and checks that prefix but
cannot form a full proof. Hard cutoff cleans up owned descendants and refuses a
late full positive file. Memory refusal, missing final result, parent dispatch
exception, delayed publication and complete cost accounting are exercised.

This is evidence for the implementation/contract, **not proof that the real
4088 request will complete, yield positive bounds or fit memory**. All252 fresh
positive bounds are still uncomputed. Even a later successful result would be
for the declared real graph under stated graph/program and preprocessing
assumptions, not native floating-point execution; enumerating all pairs alone
does not establish route change.

## Commands after a later explicit execution authorization

Use the unchanged act-py312 interpreter from the repository root:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_proof.validate --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_proof.review_freeze --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -m scoped_proof.run execute --acknowledge-frozen-execution
```

The first two commands are checks only. The third is **not run in this stage**.
The actual entry requires a clean registered branch and resource admission;
timed intake rechecks frozen implementation, environment and asset identities.
After execution, audit the resulting terminal/receipt/cost and new evidence:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_proof.audit data/moe/results/scoped_proof_source4088_20260924_r1
```

That is saved-only exact rechecking, separately costed, with no new LP solve.
It cannot recover missing execution evidence or turn a late/failed request
into a budget-compliant success. Preserve every partial artifact and failure;
archive the actual outcome rather than rerunning a difficult obligation.
