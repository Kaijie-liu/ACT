# B1 historical identity inventory — bounded, not yet complete

This read-only investigation protects the proposed new-input comparison from
reusing previously diagnosed endpoints. It ran no model, clean-forward, solver
or attack and selected no new inputs. [Source identities and counts](b1_history_inventory_20261021.json)
bind the records below. This is **not** a final exclusion manifest or permission
to launch selection. Existing full-test telemetry must continue to be disclosed.

## Concrete omissions from the old selector

The902-entry exclusion list in `docs/reuse_supervised_v1_freeze.json` is historical,
not current. The following identities are absent from it:

| Source family | Previously observed CIFAR-10 test inputs | Meaning |
| --- | --- | --- |
| That freeze's own new samples | 220,222,230,232 | New at that old freeze, not new now |
| Router boundary figure | 3444 | Explicitly selected maximum route-boundary example |
| RT-ER conversion differences | 659,2316,4604,9957; route difference9604 | Particular anomalies selected from full-test telemetry |
| RoME frozen comparison | 6044,9399 | Higher-index endpoints already used; registered queue also contains1917,2890 |

RoME's configuration binds the same CIFAR-10 `test_batch` SHA256
`f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831`.
Its original control also used0. Separately, the RT-ER cross-conversion cohort
declares `ordered_test_indices="0--999"` for clean and PGD endpoints. This is a
schema-specific selected cohort, not a generic matrix `indices` field and not
merely a full-test aggregate. **An index above4389 is not evidence of novelty.**

Other schemas requiring explicit handling include `line_sample_indices`,
`sample_index`, `request_index`, `ordered_dataset_indices`, `probe_dataset_indices`,
and nested `requests[].{dataset,index}`. Generic `index/indices` also denotes LP
columns, CSR, layers, properties and slices; blindly unioning it is incorrect.

## Coverage actually checked

The initial bounded metadata scan covered55,336 JSON/JSONL filenames in the
five ACT results/config/docs and MOE-results roots;49,405 small files were read,
3,577 oversized files and2,354 source copies were skipped. Its six parse failures
were one historical mixed stdout file and five journal-fault fixtures; they were
not silently admitted as valid input records.

A follow-up read20,680,524 bytes, below its20MiB cap, including5,352 small
`baseline_runs` files,10 `baseline_data` manifests and selected larger ledgers:

- Four-arm `rows.jsonl`:1,200 rows/100 inputs, no bad tail; identities match frozen
  jobs and all1,200 independent terminals, including125 outer timeouts.
- AdvMoE Lagrangian `rows.jsonl`:100 rows/20 inputs0–19, no bad tail; matches its
  ordered selection. These identities were already observed despite failure.
- MetaMoE's selected CIFAR-10 inputs are1,2,4,5,7; MNIST inputs1,3,7,9,10 remain
  a separate dataset. Later paired/follow-up summaries reuse that selection.
- Dual RS certification uses CIFAR-10 inputs0,1. Training indices are not test
  endpoints, and clean-forward scan rows are distinguished from verification.

## Remaining limits and next use

This scan does not yet close all historical identity sources.125 large baseline
files mostly have smaller queue summaries, but not every link was verified;
76,040 other-family files (predominantly HZ/H1/H2 controls) were not individually
proved synthetic. Dual RS's first256 eligible-test selector identities were not
reconstructed from their hash/label references. A truncated Robust Experts
prepared record belongs to CIFAR-100, not CIFAR-10.

Do not convert the findings into a final count by appending only these12 omissions.
The next selector needs dataset/source-aware provenance and explicit treatment
of remaining schemas/coverage. It must exclude previously selected anomalies
without calling all10,000 full-test telemetry images new or, conversely, claiming
that every telemetry row was a verification endpoint. Fresh selection forwards
must be frozen and charged; no free128×3 preparation or previously consumed
control is authorized by this inventory. This closes specific selection-design
mistakes, not a new algorithm result or a completed B1 experiment.
