# HybridZ Frontier Assessment 2026-06-24

Scope: pure exact Hybrid Zonotope only. No Gurobi-counted proof path, no input
split, no sampling, no LP-witness promotion, no CROWN/triangle decision.

## Frozen Baseline

Current soundfix ICSE export:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`

The previous 2026-06-24 metaroom `100/100` row is superseded by the
2026-06-25 soundfix. The runner now treats split VNNLIB disjuncts correctly:
any `ADV` disjunct makes the instance `ADV`; `CERT` requires all disjuncts
certified.

Historical frozen ICSE export:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/`

Key current results:

| Benchmark | V+A | Current status |
| --- | ---: | --- |
| metaroom_2023 | 95/100 | #1, soundfix freeze |
| sat_relu | 100/100 | #1 tie, freeze |
| malbeware | 150/150 | #1, freeze |
| cersyve | 11/12 | #1, one hard UNKNOWN remains |
| dist_shift_2023 | 70/72 | #1, two timeout rows remain |
| cora_2024 | 25/180 | #1 after clean model-family exact-HZ portfolio |
| safenlp_2024 | 1079/1080 | #2, one hard UNKNOWN remains |
| cgan_2023 | 13/21 | #2, sparse-system/MIP wall |
| linearizenn_2024 | 40/60 | #3, binary-MIP wall |
| tllverifybench_2023 | 17/32 | #3, sparse-system/MIP wall |
| relusplitter | 43/220 | Current clean main result; old 102 candidate not reproducible |
| acasxu_2023 | 120/186 | Dense exact-MIP wall |

## Relusplitter 102 Candidate Audit

Old candidate:

`audit_results/hz_relusplitter_eqsubst_tail_MERGED_20260621/relusplitter.jsonl`

This is a branch merge, not a fresh frozen run. The tail branch reported many
`sparse_tail_eqsubst` CERT rows. Example iid140/q2 reported HiGHS
`Infeasible` in about 5.7s with:

- compressed exact ReLU enabled
- equality-substitution presolve enabled
- cutoff row enabled
- `n_cont=86842`, `n_bin=43413`, `n_eq=43413`
- `base_hz_feasible=True`

Current-code direct q2 repro uses the same HZ size and same sparse exact-HZ
semantics, but does not reproduce the proof:

| Probe | Artifact | Result |
| --- | --- | --- |
| HiGHS, no singleton elim | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_nosingleton.json` | UNKNOWN, root timeout |
| HiGHS, singleton elim | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_singleton.json` | UNKNOWN, root timeout |
| HiGHS + FBBT | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_fbbt3.json` | UNKNOWN, FBBT fixes many vars but no proof |
| HiGHS + connected presolve | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_connected.json` | UNKNOWN, no reduction |
| HiGHS objective-target | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_objtarget.json` | UNKNOWN |
| SCIP 60s diagnostic | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_scip60.json` | UNKNOWN/timelimit |
| Gurobi 60s diagnostic, not counted | `audit_results/hz_relusplitter_repro_diagnose_20260624/iid140_q2_gurobi60.json` | UNKNOWN/timelimit |

Current judgment: do not count the relusplitter 102 candidate. The current
open-source exact-HZ formulation cannot reproduce iid140/q2 even as a single
query. SCIP and Gurobi diagnostics also do not support the old 5s infeasible
claim. Treat the old candidate as a formulation/version gap until a clean
current-code frozen run proves otherwise.

## Small Tail Probes

cersyve iid11 was also checked as a low-cost possible +1:

`audit_results/hz_cersyve_iid11_followup_20260624/`

HiGHS and SCIP sparse exact-HZ profiles with compressed ReLU, ReLU cuts, FBBT,
relaxation precheck, and base-binary start both hit the external timeout without
a JSON verdict. This is not a cheap clean gain under the current path.

Follow-up dense exact-HZ worker probes:

`audit_results/hz_tail_probe_20260624/`

| bench | iid | diagnostic result |
| --- | ---: | --- |
| safenlp_2024 | 454 | CERT in about 34.5s with compressed exact ReLU and valid cuts; not official-wall countable. |
| safenlp_2024 | 454 | UNKNOWN under 20s with compressed-only, cuts-only, and compressed+cuts. |
| cersyve | 11 | UNKNOWN after 120s with compressed exact ReLU and valid cuts. |

Interpretation: safenlp has a near-wall open-source MILP scheduling opportunity,
but current evidence does not justify changing the frozen headline. Cersyve
iid11 is not a short cleanup target.

## Next Rational Work

Freeze the current ICSE artifact for reporting. Continue research only on
structural solver improvements that are benchmark-wide and sound:

1. Exact sparse MILP presolve: stronger FBBT/probing that can certify EMPTY, not
   just reduce variables.
2. Robust row/objective scaling in the sparse probe, validated on toy exact-MILP
   oracles before production.
3. Open-source solver portfolio scheduling at the benchmark/profile level.
4. Relusplitter current-code reproduction only if a sound formulation change
   explains the old tail gap.

Do not promote old branch-merge artifacts or per-iid rescue paths into headline
numbers.
