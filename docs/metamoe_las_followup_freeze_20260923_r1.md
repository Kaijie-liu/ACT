# Repaired MetaMoE full-cohort followup: frozen, NOT executed

[Config](../configs/backend_controls/metamoe_las_followup_r1.json), SHA256:
`af662082da17e953960ba1127d0f0fcf8783f0dd5810ce9b0bd8f43ab6aa031c`.
Execution ancestor `e73830e3b`. [Independent freeze review](metamoe_las_followup_freeze_20260923_r1.json)
PASS0, including an identical saved control re-audit with production repair
helpers disabled. The output directory did not exist at freeze. Zero new
comparison executions, no automatic launch/followup/resume.

Original20-request R1 remains sealed after14normal/1ERROR/5notstarted. The
separate [repair control](metamoe_las_repair_result_20260923_r1.md) validates
compatibility, not efficacy: MNIST7 ends TIMEOUT after7successful original
domain insertions. All historical error/timeout records remain visible.

## Exactly what is frozen

- Same10previously selected inputs: CIFAR10 `[1,2,4,5,7]`, MNIST `[1,3,7,9,10]`.
  Same stored checkpoint and materialized center/endpoints/properties.
- Full two-arm roster20, original rotating order; no partial suffix splice.
  This is an already-observed-cohort repaired-version followup, not new holdout.
- ACT unchanged: checked expert base, routing assignment and nonzero precheck.
- Author path explicitly labelled author backend + lAs compatibility wrapper
  + strict route-invariance sufficient adapter. No external repo/dependency
  change; no alpha/branching/bound option tuning. Do not call it an unchanged
  literal author execution or full dynamic-dispatch verifier.
- CPUfloat64/two threads, normalized-space2/255, clamp[-10,10], margin1e-7,
  all19global output properties plus routing/nonzero. Not a pixel-space claim.
- Each complete request300s/8GiB sampled group RSS. ACT sparse2GiB policy and
  original30s expert allocation remain. Startup/loading/repair checking/logging/
  solving/publication charged; parent inventories/postflight and independent
  audit/replay separately recorded.20calls give at most6000s registered solver
  ceilings, not an end-to-end completion promise.
- ERROR/SOURCE_CHANGED fail-stop; later rows NOT_STARTED_AFTER_ERROR. TIMEOUT
  retained and next registered row continues. No automatic retries or overwrite.
  Preserve positive grades, replayed UNSAFE, completed UNKNOWN, outer timeout,
  ERROR and missing rows separately; no missing cost treated as zero.
- Report pair intersections/gains/losses and complete attempted cost; no
  outcome-dependent gate or claim that repair must improve coverage. Author
  numerical filters and HZ policy acceptances remain distinct from source-
  complete proof. If interrupted again, retain the full registered denominator.

## Commands for a separately authorized execution

Check branch/worktree/resources first; run ONCE only after launch approval:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/metamoe_las_paired.py --config configs/backend_controls/metamoe_las_followup_r1.json
```

Then separate replay and independent saved audit (no solving/retries):

```bash
/data1/Kane/MOE/envs/moe-author-cpu-py312-20260921/bin/python scripts/replay_metamoe_paired.py --config configs/backend_controls/metamoe_las_followup_r1.json --output /data1/Kane/MOE/baseline_runs/metamoe_las_followup_20260923_r1/original_replay.json
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_metamoe_las_paired.py --config configs/backend_controls/metamoe_las_followup_r1.json --replay /data1/Kane/MOE/baseline_runs/metamoe_las_followup_20260923_r1/original_replay.json --output docs/metamoe_las_followup_archive_20260923_r1.json
```

`repair_control_gate` is intentionally false for a non-control protocol; do
not turn it into an efficacy threshold. Archive results, full costs and failures,
update paper/handoff and commit/push. No additional samples, time, relaxation,
backend options or new source-complete claims are authorized by this freeze.
