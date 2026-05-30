# E1 progressive wall probe — CLOSED NEGATIVE

**Date**: 2026-05-27
**Question**: do TO-bound benchmarks (cora, ml4acopf, relusplitter) migrate to CERT/FAL when given 5× wall (300s vs baseline 60s)?

## Result

| Bench | iids | wall=300s migrations | Net |
|---|---|---|---|
| cora_2024 | 10/10 | 0 → all UNKNOWN_TIMEOUT | 0 |
| ml4acopf_2024 | 3/10 (partial) | 0 → all UNKNOWN_TIMEOUT | 0 |
| relusplitter | not run | — | — |

cora: 10/10 still UNKNOWN_TIMEOUT after 315-369s wall. Identical state as 60s baseline.
ml4acopf partial: 3 iids ran 60-260s before each TO.

## Verdict

E1 is **NEGATIVE** on cora at 5× wall. The TO classification in
INSIGHTS_AND_NEXT_STEPS.md §T4 was wrong for cora — those instances
are not wall-bound but **encoding-bound** (T3 cliff): cora-mnist-set is
the only cora variant ACT decides; other 8 variants stay UNK regardless
of budget.

Implication for ROI ranking: **down-weight E1 (progressive wall)** as
a precision lever and **up-weight T2 (sparse-Gc representation)** —
because the only remaining decisive lever is reducing the per-iteration
cost (representation), not extending wall.

## Action

Direction E1 archived. Direction T2 (sparse-Gc HZ representation)
elevated to primary active line.
