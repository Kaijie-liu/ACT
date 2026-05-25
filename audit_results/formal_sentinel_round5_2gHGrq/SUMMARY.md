# ACT Formal Sentinel - Round 5

Date: 2026-05-24

## Configuration

- Canonical root: `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks`
- Formal reporting: `ACT_FAL_RECEIPT_FORMAL=1`
- Receipt directory: this directory
- Backend: ACT CLI `--verify vnnlib --solvers hybridz`
- Strict witness backend root: `/data1/Kane/HyZor`

## Formal Sentinel Results

| Benchmark | Official iid | Official zero | Purpose | Formal CLI result |
|---|---:|---|---|---|
| sat_relu | 0 | sat | strict-SAT positive receipt | FALSIFIED |
| sat_relu | 1 | unsat | negative control | UNKNOWN |
| safenlp_2024 | 100 | sat | known boundary-candidate case | UNKNOWN |
| safenlp_2024 | 1 | unsat | negative control | UNKNOWN |
| acasxu_2023 | 181 | unsat | prop_6 multi-query (8 queries) negative control | UNKNOWN |
| acasxu_2023 | 0 | unsat | negative control | UNKNOWN |
| tllverifybench_2023 | 0 | unsat | negative control | UNKNOWN |

No official-UNSAT sentinel was reported as `FALSIFIED`.

## Receipt Audit

The formal run produced exactly one FAL receipt:

- `sat_relu_0_q0_small_dense_lp_witness_0.json`
- Identity: benchmark `sat_relu`, official iid `0`, query index `0`
- `spec_zero_tol_holds=true`
- `spec_small_tol_holds=true`
- Independent checks: model SHA, spec SHA, and x-star SHA match; fresh
  CPU ONNX Runtime output matches the stored `y_ort` array.

`MANIFEST.csv` contains exactly the same one receipt entry.

## Error-Accounting Smoke

`receipt_missing_smoke.log` is intentionally not a formal positive result.
It re-runs the strict-SAT `sat_relu` instance with formal mode enabled and
without `ACT_FAL_RECEIPT_DIR`. The CLI reports:

```text
ERROR_RECEIPT_MISSING ... A=0 ... R=1
```

This confirms the two-channel contract: internal SAT is preserved, while
an unreceipted SAT is excluded from formal FAL counts.

