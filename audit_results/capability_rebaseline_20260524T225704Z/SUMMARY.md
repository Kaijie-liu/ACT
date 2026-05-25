# Capability rebaseline summary
`run_root` = /data1/Kane/ACT/audit_results/capability_rebaseline_20260524T225704Z

| run_label | benchmark | config | CERT | FAL | UNK | ERR | R | wall_min | legacy CERT | delta |
|---|---|---|---|---|---|---|---|---|---|---|
| acasxu_A_base | acasxu_2023 | base (GlobalLP) | 61 | 0 | 125 | 0 | 0 | 7.4 | 61 | 0 |
| acasxu_B_specaware | acasxu_2023 | auto/specaware | 73 | 0 | 113 | 0 | 0 | 12.0 | 74 | -1 |
| acasxu_C_auto | acasxu_2023 | ? | 73 | 15 | 98 | 0 | 0 | 22.9 |  |  |
| linearizenn_witness | linearizenn_2024 | witness | 0 | 0 | 0 | 60 | 0 | 0.0 | 13 | -13 |
| safenlp_A_passes3 | safenlp_2024 | passes=3 | 284 | 10 | 786 | 0 | 0 | 6.2 | 284 | 0 |
| safenlp_B_auto | safenlp_2024 | auto shallow_20 | 333 | 10 | 737 | 0 | 0 | 8.1 | 333 | 0 |
| sat_relu_witness | sat_relu | witness | 1 | 49 | 50 | 0 | 0 | 0.4 | 0 | 1 |
| tllverify_witness | tllverifybench_2023 | witness | 1 | 2 | 29 | 0 | 0 | 8.7 | ? |  |

## Formal result distribution (per-instance)

| run_label | REPORTABLE_FALSIFIED | ERROR_* | None / other |
|---|---|---|---|
| acasxu_A_base | 0 | 0 | 186 |
| acasxu_B_specaware | 0 | 0 | 186 |
| acasxu_C_auto | 15 | 0 | 171 |
| linearizenn_witness | 0 | 0 | 60 |
| safenlp_A_passes3 | 10 | 0 | 1070 |
| safenlp_B_auto | 10 | 0 | 1070 |
| sat_relu_witness | 49 | 0 | 51 |
| tllverify_witness | 2 | 0 | 30 |

## Soundness gate
**VIOLATIONS**:
  - linearizenn_witness: ERROR=60
