# Neural-HZ v2 (sessions of 2026-10-02 and 2026-10-03): navigation

Start here when resuming. `LOG.md` is the append-only record of every
experiment: hypothesis, files, numbers and decision (N000-N066+).
`THEORY.md` holds the definitions, lemmas and proofs (Sections 1-14). This
directory is isolated. Nothing here is imported by ACT. No production file,
frozen archive or HyZor file was changed, and nothing was committed or pushed.

Official scores are unchanged: formal **1870/2413** and E0 **61/400**. The
complete domain-only replay N039 v14, after re-verification of every MILP CERT
with numerically well-posed plans (LOG N111-N114), retained 1746/1870 with 38
gains and 0 conflicts; it does not promote. See `FINAL_REPORT_20261003.md`.

## Current candidate (what the replays run)

| component | file | what it is |
|---|---|---|
| engine n020.1 | `nhz_sound_v20.py` (= n017 + pooling + LP tightening and polishing stop at 40 percent of the row budget) | rigorous projection-aligned HZ: float32 storage, float64 arithmetic, rounding radii. Exact ReLU phases; smooth units as DeepZ shadow plus rigorous chord and tangent rows, recorded for the plans. Softmax per coordinate as tangent plane or interval, with bounds from score differences (n011.2 + n014 + n016) |
| terminal v8.1 | `nhz_terminal_v8.py` (on v7.8: HiGHS seeds 0-2 + SCIP member; dual bounds only from valid statuses; tiny coefficients folded) | MILP plans: disaggregated (Balas) ReLU rows; smooth units written sparsely; optional smooth phase segments; sign rule (Prop. 1'). Seed portfolio, watchdog, presolve guard |
| path v7.2 | `nhz_path_v7_2.py` (domain-only verdicts, LOG N099; numerically well-posed plans via terminal v7.11/v8.5, LOG N112; `nhz_path_v7.py` is the frozen v14 version). Next candidate engine: n021 `nhz_sound_v21.py` (row-coupled softmax, LOG N116; recovers PGD ViT 78) | stages A propagate, B rigorous LP (CERT only), D last-layer MILP, E all-layers MILP, F E plus smooth segments. ADV only from MILP incumbents of these plans, validated on ORT (rounded inward and snapped to bounds, checked S2 then S1). Early-stop target only on exact plans |
| runners | `run_n039v7_full_replay.py`, `run_n068_e0_path_v6.py` | one configuration for every row, plus one retry after an out-of-memory error inside the row deadline |

## Single-path replays

| run | path | state |
|---|---|---|
| `n039_full_replay_v1` to `_v6` | v3 to v5.5 | stopped for recorded reasons (LOG). v6 kept 1319/1320 over SafeNLP, sat_relu, malbeware and Cora, losing one row to an out-of-memory error |
| `n039_full_replay_v7` | v6.1 + n017 | stopped (fail-closed fix, LOG N070) |
| `n039_full_replay_v8` | v6.2 + n017 | stopped (centre witness, LOG N072) |
| `n039_full_replay_v9` | v6.3 + n017 | stopped (SCIP member, LOG N075) |
| `n039_full_replay_v10` | v6.4 + n017 | stopped (soundness review, LOG N085) |
| `n039_full_replay_v11` | v6.5 + n017 | stopped after 1,743 rows (diagnostic; LOG N087-N094) |
| `n039_full_replay_v12` | v6.7 + n019 | stopped at 2,277 rows: used LP-dual corners, centre point, centre MIP start (not allowed, LOG N099); diagnostic only |
| `n039_full_replay_v13` | v7.0 + n020.1 | stopped after 20 min (five workers exceeded the baseline's four-way concurrency, LOG N101) |
| `n039_full_replay_v14` + `n113_reverify_v14_milp_certs` | v7.0 + n020.1, MILP CERTs re-verified with v7.2 | **final re-verified vector: retained 1746/1870, 0 conflicts, 38 gains (28 CERT, 10 ADV); 6 CERTs withdrawn; gate FAIL** (LOG N114) |
| `n015_e0_replay_v4`, `n061_e0_replay_v5` | v1, v5.5 | stopped (v4 lost anchor 83) |
| `n068_e0_replay_v6` | v6.1 + n017 | stopped (LOG N070) |
| `n070_e0_replay_v7` | v6.2 + n017 | stopped (LOG N072) |
| `n072_e0_replay_v8` | v6.3 + n017 | stopped (LOG N075) |
| `n076_e0_replay_v9` | v6.4 + n017 | stopped (LOG N085) |
| `n084_e0_replay_v10` | v6.5 + n017 | stopped (LOG N094) |
| `n092_e0_replay_v11` | v6.7 + n019 | stopped (LOG N099) |
| `n100_e0_replay_v12` | v7.0 + n020.1 | stopped (concurrency, LOG N101) |
| `n102_e0_replay_v13` | v7.0 + n020.1 | terminated externally after 20 rows (LOG N109) |
| `n109_e0_replay_v14` | v7.0 + n020.1 | stopped at 170 rows after the false CERT on anchor 164 (LOG N111/N112); re-verified partial vector: 31 CERT gains, 16 anchors kept, 5 lost, 0 conflicts |
| `n115_e0_replay_v15` | v7.2 + n020.1 | **paused by the user** (LOG N122): CIFAR100 half complete and audited (39 CERT gains, anchors 18/25); TinyImageNet half partial |

Every replay directory has a PREREG and a FREEZE_SHA256SUMS. Partial runs are
never merged.

## Main findings (details in LOG.md)

- **Encoding matters as much as the domain.** The baseline's HZ MILP was faster because its
  ReLU encoding is disaggregated. Adopting that in latent coordinates and
  skipping integrality for sign-proven last-layer units recovered all SafeNLP
  budget losses and sat_relu (N044-N057).
- **MILP plans dropped every non-ReLU row** (sigmoid lines, softmax rows) before
  terminal v8. Writing the smooth rows sparsely, with explicit pre-activation
  columns, takes dist_shift row 12 from +1.86 to -3.02 with zero sigmoid
  binaries, in 0.6 s against 13 s for the baseline (N065, validity test N066).
- **ViT:** the attention branch is the loss, about 2.6x at the softmax and 1.75x at P@V (N063).
  n011.2 and n014 fix the PGD-model blow-up (LP bound 292 -> 0.30 on row 15).
  IBP-model rows are still about 7x looser than the baseline.
- **Witnesses:** ADVs from the LP box maximiser or the input-box centre are NOT gains or
  retained solves under the user rule of 2026-10-03 (LOG N099). That withdraws the Cora,
  MetaRoom and cgan ADV gains reported earlier. Only MILP incumbents of the element's own
  plans, validated on ORT, count.
- **E0:** the 61 anchors are attack-found ADVs. CIFAR 83 and 114 are not
  recovered without an attack (N033, note after N048).

## Decisions only the user can make

1. Decided by the user (2026-10-03): no attack or sampling helper may produce gains; path v7.0
   removes them (LOG N099).
2. Does skipping integrality for units where Prop. 1' proves the optimum unchanged
   satisfy hard limit 2? The binaries stay in the element; only the query skips them.
3. May the candidate path carry the baseline's attack stage only to retain the
   attack-found E0 anchors, with no gain credit from it?

## How to resume

1. Read the newest LOG.md sections and the replay directories' logs.
2. `summarize_n039.py "n039_full_replay_v6/w*.jsonl"` gives the per-family retention and gate.
3. Audit before reporting: `audit_n039_witness.py` for ADVs (S1/S2), and
   `audit_n017_cert_sampling.py` for CERTs.
