# Real frontier comparison R1 — frozen, NOT executed

Implementation/control commit:
`8154a62810fc04c9e3037e20770418584c1d1fd4`.
Configuration: `configs/backend_controls/frontier_proof_compare_r1.json`.
SHA-256: `b640a366fdce1ec9a746956eb4532560b5a9907434c5b2129d87f17b6d56b012`.
Independent stdlib freeze review: `frontier_proof_freeze_review_20260924_r1.json`,
PASS,0 issues. All538 source/protocol identities and bound asset raw hashes
match. Control archive has94/94 passes. Review read no real checkpoint/tensor
objects and ran no inference or verification. The output directory is absent.

| Frozen item | Value |
| --- | --- |
| Object | seed0, old manifest rank2, CIFAR4098, label9 |
| Domain | exact2/255, clip[0,1], same center/checkpoint/property |
| Arm1 | original exhaustive proof pipeline |
| Arm2 | checked strict router exclusions before lazy source construction |
| Calls | one per arm, fixed exhaustive then checked_frontier order |
| Resource | each300s end-to-end,2threads,sampled8GiB |
| Original duties | all28pairs x9properties =252 per arm |
| Output proposals | unchanged native proposer; half remaining work time, equal remaining share |
| Acceptance | each original duty discharged by exact checked exclusion or fresh positive output bound |

No automatic execution, queued job or real positive certificate. These are
observed engineering inputs, not new holdout/high-accuracy/route-changing claims.
Never reopen98/4088/4096, borrow old source/bounds, add retries/samples/time or
modify numerical/production acceptance. Retained output matrices stay unchanged.
The original exhaustive path lacks checked infeasibility certificates; this
comparison tests new checked discharge plus avoided work, not speed alone.

## Later explicitly authorized execution

Use a clean `feat/moe-route-verification` checkout and the existing environment.
Inspect other jobs/resource use first; the runner also applies resource gates.
The fixed output root must remain absent before launch. It cannot resume.

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m frontier_proof.controls --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m frontier_proof.review_freeze --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -m frontier_proof.run execute --acknowledge-new-frozen-comparison
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m frontier_proof.batch_audit /data1/Kane/MOE/ACT/data/moe/results/frontier_proof_source4098_compare_20260924_r1
```

The pre-execution freeze review requires output absence; after execution use
the saved-only batch audit, not the absence gate. Archive audit stdout with a
new exclusive record and a compact raw-file hash inventory, leaving all failed
attempts locally intact. No raw checkpoints/data or large evidence go into Git.

Report complete positive/NOT_CLOSED/TIMEOUT/ERROR/resource states for both calls,
full costs (and missing cost explicitly), checked exclusions, retained output
bounds, source/retained matrix identity, and source/router/check/proposal phase
costs. A batch audit gap is not PASS; a partial proof is not complete. Positive
means declared-real-graph proof under the explicit program/preprocessing/runtime
trust boundary, not native float certification. Stop after the fixed two calls
and saved-only archive, then interpret whether any complete proof was added.
