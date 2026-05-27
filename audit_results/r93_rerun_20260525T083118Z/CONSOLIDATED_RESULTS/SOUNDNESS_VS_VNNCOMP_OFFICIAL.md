# ACT vs VNN-COMP 2025 Official Report - Label Audit

**Updated**: 2026-05-26 UTC  
**Official inputs**: `/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/{zero_tol,small_tol}/longtable.tex`  
**ACT inputs**: the designated source rows in each `CONSOLIDATED_RESULTS/<bench>/per_instance.csv`

## Interpretation

This audit tests report-label agreement:

- ACT `CERTIFIED` is compatible only with official `UNSAT`.
- ACT `FALSIFIED` is compatible only with official `SAT`.
- ACT `UNKNOWN`, `UNKNOWN_TIMEOUT`, and `UNKNOWN_RESOURCE_LIMIT` make no
  safety claim and are counted as `no_claim`.

It is not correct to say there are no report discrepancies.  The audit finds
one localized discrepancy family: `collins_rul_cnn_2022` FAL outcomes versus
official labels.  For all other comparable sources, including the new Round 2
`cora_2024` and `metaroom_2023` CERT rows, there are no label disagreements.

## Machine-Readable Artifacts

- Rebuild archived rows: `build_csvs.py`
- Re-run this audit: `soundness_check.py`
- Summary: `OFFICIAL_CROSSCHECK_SUMMARY.json`
- Disagreement rows: `OFFICIAL_CROSSCHECK_DISAGREEMENTS.csv`

`build_csvs.py` includes watchdog synthetic outcomes and gives them precedence
over in-flight child output for the same iid.  Consequently the counts below
include bounded attempts rather than silently dropping them.

## Summary

Rows include CPU and GPU separately where both designated sources exist.
`D` is label disagreement, not automatically an ACT soundness failure.

| Benchmark | rows compared | CERT agrees | FAL agrees, zero/small | no claim | D zero/small |
|---|---:|---:|---:|---:|---:|
| `acasxu_2023` | 372 | 146 | 30 / 30 | 196 | 0 / 0 |
| `cersyve` native Round 2 | 12 | 0 | 0 / 0 | 12 | 0 / 0 |
| `cifar100_2024` smoke | 5 | 0 | 0 / 0 | 5 | 0 / 0 |
| `collins_rul_cnn_2022` | 124 | 78 | 4 / 16 | 24 | **18 / 6** |
| `cora_2024` smoke | 7 | 1 | 0 / 0 | 6 | 0 / 0 |
| `dist_shift_2023` | 72 | 0 | 0 / 0 | 72 | 0 / 0 |
| `linearizenn_2024` | 120 | 26 | 0 / 0 | 94 | 0 / 0 |
| `malbeware` | 300 | 246 | 26 / 26 | 28 | 0 / 0 |
| `metaroom_2023` smoke | 5 | 2 | 0 / 0 | 3 | 0 / 0 |
| `nn4sys` | 107 | 2 | 0 / 0 | 105 | 0 / 0 |
| `safenlp_2024` | 1080 | 333 | 10 / 10 | 737 | 0 / 0 |
| `sat_relu` | 200 | 2 | 39 / 39 | 159 | 0 / 0 |
| `soundnessbench` smoke | 10 | 0 | 0 / 0 | 10 | 0 / 0 |
| `tinyimagenet_2024` smoke | 4 | 0 | 0 / 0 | 4 | 0 / 0 |
| `tllverifybench_2023` | 32 | 1 | 2 / 2 | 29 | 0 / 0 |
| **Total** | **2450** | **837** | **111 / 123** | **1484** | **18 / 6** |

Some attempted smoke iids have no official `SAT`/`UNSAT` result label in
`longtable.tex`; they remain archived in their CSVs but are not counted in
this comparison.  Scored
categories absent from the official longtable used here, such as
`relusplitter` and `ml4acopf_2024`, cannot be label-compared by this script.

## New Round 2 CERT Check

| Benchmark | ACT CERT iid | Official zero_tol | Official small_tol | Result |
|---|---:|---|---|---|
| `cora_2024` | 8 | UNSAT | UNSAT | agrees |
| `metaroom_2023` | 1 | UNSAT | UNSAT | agrees |
| `metaroom_2023` | 4 | UNSAT | UNSAT | agrees |

These are soundness checks of the returned CERT labels, but the runs are
bounded smokes; they do not make these benchmarks formal-qualified.

## Collins RUL Disagreements

The discrepancy is present in both CPU and GPU sources, hence each distinct iid
appears twice as a row disagreement.

| distinct iid | ACT result | official zero_tol | official small_tol |
|---|---|---|---|
| 0, 22, 47 | FALSIFIED | UNSAT | UNSAT |
| 4, 5, 13, 26, 27, 35 | FALSIFIED | UNSAT | SAT |

For small-tolerance residual iids `0`, `22`, and `47`, their CPU FAL receipt
JSON records all have:

```text
input_box_holds=true
spec_zero_tol_holds=true
spec_small_tol_holds=true
```

The receipts also bind model/spec/witness by SHA-256.  An earlier independent
ORT spot audit recorded:

| iid | safe band | sampled ORT output range | unsafe samples |
|---:|---|---|---:|
| 0 | `(196.977, 240.750)` | `[164.93, 165.06]` | 1000 / 1000 |
| 22 | `(196.977, 240.750)` | `[164.93, 165.06]` | 1000 / 1000 |
| 47 | `(8.240, 10.071)` | `[12.67, 12.76]` | 1000 / 1000 |

Thus the archived evidence supports ACT's FAL results, but the proper reporting
statement is: **ACT and the official labels disagree on these iids, with
strict ACT receipts and ORT replay evidence supporting the ACT side; official
reconciliation is required.**  Do not erase this discrepancy or silently
upgrade it to settled ground truth.
