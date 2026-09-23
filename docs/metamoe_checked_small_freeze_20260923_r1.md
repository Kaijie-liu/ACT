# Small paired comparison frozen; NOT executed

Selection implementation `e03777b08`, prior-to-forward plan `a19ca4bc9`.
[Selection plan](../configs/recent_moe/metamoe_checked_small_plan_r1.json)
SHA256 `d216ee49aaf45d3abfd9eaff35cd2da4960ace8d344597b08f03f67fe59074d9`.
[Execution manifest](../configs/recent_moe/metamoe_checked_paired_small_r1.json)
SHA256 `97e9fcb8fb9556ecef4bf0287510eb7bde690b3eef0c9bb3799e69e4b73aa069`.
[Saved-only final audit](metamoe_checked_small_freeze_20260923_r1.json): PASS,
0 issues, `FROZEN_NOT_EXECUTED`. [Protocol](metamoe_checked_small_protocol_20260923_r1.md).

| Dataset | Selected original indices | Scanned raw prefix | Excluded prior index |
|---|---|---|---|
| CIFAR10 | 1, 2, 4, 5, 7 | 0 through 7 | 0 |
| MNIST | 1, 3, 7, 9, 10 | 0 through 10 | 0 |

All 10 are original full-model clean-correct. The first eligible raw-order
prefix was reproduced in a separate process from original raw data, author
preprocessing and frozen model; every physical float64 center/lower/upper
was checked exactly. No routing complexity, margin bound or verification
endpoint was computed for selection. The forward necessarily executes the
model's router, but router output is not a selection criterion or saved census.

Selection cost 3.181478 seconds; separate source replay 2.126220 seconds;
orchestration through review 6.536737 seconds. Each was independently supervised
at300s/8GiB; these are offline preparation costs, ZERO verification queries.
Raw tensors/scans/receipts retained under
`/data1/Kane/MOE/baseline_data/metamoe_checked_small_20260923_r1`, not in Git.
The future20-call result directory does not yet exist. No background job.

Later execution uses the EXACT successful paired smoke runner, ACT options,
original model and author backend: 10 inputs x2 arms,300s per complete request,
normalized2/255,CPU/float64/two threads, same numerical/UNSAFE replay gates.
Method order rotates per input. All failures and costs stay in the denominator.
Report per-dataset and pooled descriptive counts, positive/solved gained and
lost sets; HZ policy and author numerical filters remain separate evidence
grades. This is not10 independent models,20 independent images, or general
competition supremacy. No extra samples if a result is negative.

The selection plan/prefix and cost auditor add no solver calls. Final focused
ACT regression:139 tests PASS; pinned worker regression:104 tests PASS.
Historical BN exposure review is separate and does not change this protocol.

## Next execution (separate from this freeze)

After checking branch/clean status and resources, run once:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/metamoe_checked_paired.py --config configs/recent_moe/metamoe_checked_paired_small_r1.json
```

Then replay every accepted UNSAFE using `scripts/replay_metamoe_paired.py`
in the pinned ACT worker environment and audit using
`scripts/audit_metamoe_checked_paired.py --config ... --replay ... --output ...`.
Use new output filenames; no resume/retry/source rebinding. Archive all20
terminal rows, even after an ERROR makes later rows NOT_STARTED. The auditor's
`smoke_gate_pass` is only a smoke field and deliberately false for the small
cohort; it is NOT an efficacy criterion. No code/config tuning during run.
