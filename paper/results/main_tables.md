# Rebuilt main outcome tables

Generated from committed reviews; accounting reconstruction, **not independent reproof**.
Positive columns retain their evidence grade. Costs include unsuccessful requests.
Cohorts are separate; model–input pairs are not independent images. No pooled success rate.

| Cohort | Model | Arm | Positive evidence grade | N | Positive | UNSAFE | UNKNOWN | TIMEOUT | Mean seconds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| MLP confirmation | seed0 | adaptive | HZ_POLICY_ACCEPTED | 100 | 59 | 30 | 9 | 2 | 70.38 |
| MLP confirmation | seed0 | matched | HZ_POLICY_ACCEPTED | 100 | 50 | 26 | 13 | 11 | 101.79 |
| MLP confirmation | seed0 | legacy | HZ_POLICY_ACCEPTED | 100 | 46 | 22 | 2 | 30 | 153.61 |
| MLP confirmation | seed1 | adaptive | HZ_POLICY_ACCEPTED | 100 | 57 | 24 | 9 | 10 | 97.53 |
| MLP confirmation | seed1 | matched | HZ_POLICY_ACCEPTED | 100 | 47 | 18 | 12 | 23 | 131.74 |
| MLP confirmation | seed1 | legacy | HZ_POLICY_ACCEPTED | 100 | 45 | 18 | 4 | 33 | 162.77 |
| MLP confirmation | seed2 | adaptive | HZ_POLICY_ACCEPTED | 100 | 63 | 23 | 8 | 6 | 77.21 |
| MLP confirmation | seed2 | matched | HZ_POLICY_ACCEPTED | 100 | 59 | 19 | 7 | 15 | 101.17 |
| MLP confirmation | seed2 | legacy | HZ_POLICY_ACCEPTED | 100 | 50 | 18 | 1 | 31 | 143.44 |
| MLP external | seed0 | adaptive | HZ_POLICY_ACCEPTED | 10 | 4 | 2 | 2 | 2 | 139.00 |
| MLP external | seed0 | crown | CROWN_NUMERICAL_FILTER | 10 | 5 | 0 | 5 | 0 | 4.27 |
| MLP external | seed1 | adaptive | HZ_POLICY_ACCEPTED | 10 | 3 | 3 | 2 | 2 | 152.64 |
| MLP external | seed1 | crown | CROWN_NUMERICAL_FILTER | 10 | 4 | 0 | 6 | 0 | 4.23 |
| MLP external | seed2 | adaptive | HZ_POLICY_ACCEPTED | 10 | 4 | 4 | 1 | 1 | 122.68 |
| MLP external | seed2 | crown | CROWN_NUMERICAL_FILTER | 10 | 4 | 0 | 6 | 0 | 4.18 |
| Conv V2 | seed17 | adaptive | HZ_POLICY_ACCEPTED | 30 | 0 | 17 | 0 | 13 | 188.31 |
| Conv V2 | seed17 | monolithic | HZ_POLICY_ACCEPTED | 30 | 0 | 11 | 0 | 19 | 217.50 |
| Conv V2 | seed17 | crown | CROWN_NUMERICAL_FILTER | 30 | 1 | 7 | 22 | 0 | 4.40 |
| Conv evidence | seed17 | matched | HZ_POLICY_ACCEPTED | 20 | 0 | 10 | 0 | 10 | 220.68 |
| Conv evidence | seed17 | evidence | CHECKED_RATIONAL_CONDITIONAL | 20 | 0 | 0 | 0 | 20 | 283.09 |
| Conv evidence | seed17 | crown | CROWN_NUMERICAL_FILTER | 20 | 0 | 7 | 13 | 0 | 4.08 |

## Committed source identities

No raw checkpoints, private paths or numerical libraries are read. Confirmation counts
come from archived per-model aggregates with paired-delta checks; external/Conv V2
counts are recomputed from committed rows. This is not a raw-run re-audit.

- `act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json` — SHA-256 `d2941013813c034f93a407fb334c636c140e8ae50f58f377377bae4c8e70f5d5`
- `act/pipeline/moe/results/conv_full_v2_review_20260915.json` — SHA-256 `52039b3c425571ad12a72adfb40ad10cea028d6af6d39d8c60cf8f78c189c8a1`
- `docs/general_evidence_execution_v1_results.json` — SHA-256 `67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe`
- `act/pipeline/moe/results/external_pair_comparison_review_20260914_r1.json` — SHA-256 `9d54c961ab0b23e5f63189ed556514b1c69ce350eb770f6478b6a450632e7985`
