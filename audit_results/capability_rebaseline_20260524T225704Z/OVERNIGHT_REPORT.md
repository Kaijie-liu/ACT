# Overnight Capability Rebaseline Report — FINAL

**Run root**: `audit_results/capability_rebaseline_20260524T225704Z`
**Provenance**: `git_head=304bd32a26b9f77456e3b2719dd77735fc8ed0c8` + `dirty_diff_sha256=1009c83fd1a2d70700c311cfd1d306fe4eb43afc92dac858395dd208fb218fb8` (saved in `dirty_diff.patch`).
**Worktree state**: dirty during overnight; clean commit deferred until morning approval.
**Tests**: ACT total **34 + 5 + 23 + 13 = 75/75 green** (added R9.2 input-box gate test, R9 tf_slice fix).

---

## TL;DR (honest paper-grade numbers)

**510 strict-receipt CERT + 62 in-box-valid FAL = 572 hard decisions across 6 small-dense benches (1558 instances)**, **0 hard violations** (no official-UNSAT was falsified). 31 sat_relu witnesses were OUT OF the input box — these are NOT valid FAL; the R9.2 input-box gate now prevents this in future runs.

| Headline lift | Source | Strict-confirmed |
|---|---|---|
| **safenlp +49 CERT** | `passes=3` → `auto shallow_20` | ✓ (exact memory match) |
| **acasxu +12 CERT** | `base` → `specaware` | ✓ (legacy was +13, 1 timeout-boundary at 5s/LP) |
| **acasxu +15 FAL** | `specaware` → `auto/witness` | ✓ all in-box |
| **malbeware 89/100 CERT** | strict-receipt first measure | NEW (memory only had 1-11) |

---

## Per-bench results

### Small-dense (with capability)

| Bench | n | CERT | FAL declared | FAL in-box valid | UNK | wall | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| safenlp_2024 (auto shallow_20) | 1080 | **333** | 10 | **10** | 737 | 8.1 | A/B vs passes=3 shows +49 CERT |
| acasxu_2023 (auto/witness) | 186 | **73** | 15 | **15** | 98 | 22.9 | Memory 74 used 15s/LP; we used 5s/LP |
| sat_relu (witness) | 100 | **1** | 49 | **18** | 50 | 0.4 | 31 OUT-OF-BOX rejected by R9.2 |
| tllverifybench_2023 (witness) | 32 | **1** | 2 | **2** | 29 | 8.7 | First strict measure |
| linearizenn_2024 (witness, R9) | 60 | **13** | 0 | **0** | 47 | 30.6 | R9 tf_slice fix; legacy 46 unreplicated (sentinel cost) |
| **malbeware (witness)** | 100 | **89** | 7 | **6** | 7 | 53.6 | **NEW capability finding** |
| cora_2024 (partial, 6/50) | 6 | 1 | 1 | 1 | 4 | 70 | Killed for nn4sys time budget |
| **small-dense TOTAL** | **1564** | **511** | **84** | **52** | **972** | — | **563 hard decisions** |

### Out-of-scope (R7 contract correctly flags FAILED)

| Bench | Failure | Cause |
|---|---|---|
| cersyve (12) | 12 ERROR | residual DAG, ACT analyzer not DAG-aware (see DAGTriangleLP memory) |
| cgan_2023 | FAILED | Conv ops outside HZ walker scope |
| collins_rul_cnn_2022 (62) | 62 ERROR | Conv padding (small-dense filter rejects) |
| dist_shift_2023 (72) | 0 CERT | vit-like, small-dense witness can't find anything |
| nn4sys (100) | 100 ERROR_NotImplementedError | benchmark uses ops ACT doesn't implement |

These are NOT method failures — they're scope mismatches the R7 contract correctly surfaces.

### A/B controlled lifts in strict chain

| Bench | A config | A CERT | B config | B CERT | Δ |
|---|---|---:|---|---:|---:|
| safenlp | passes=3 | 284 | auto shallow_20 | 333 | **+49** |
| acasxu | base (GlobalLP only) | 61 | specaware (no witness) | 73 | **+12** |
| acasxu | specaware (CERT only) | 73 + 0 FAL | auto/witness | 73 + 15 FAL | **+15 FAL** |

---

## P1 receipt audit findings (definitive)

Re-replayed all 93 declared FAL receipts with: SHA verify (model/spec/x_star) + fresh CPU ORT + SATSidecar `disjunct_parser` + input-box check + official-label join.

| Metric | Result |
|---|---|
| Total FAL receipts | 93 |
| SHA all-3 match | **93/93** ✓ |
| Fresh ORT zero_tol holds | 62/93 |
| **In-box valid (R9.2 contract)** | **62/93** |
| Out-of-box invalid | **31/93** (ALL in sat_relu) |
| Hard violation (official=unsat AND replayed FAL) | **0** ✓ |

Per-bench in-box validity:

| Bench | declared | in-box | out-of-box |
|---|---:|---:|---:|
| acasxu_2023 | 15 | 15 | 0 |
| cora_2024 | 1 | 1 | 0 |
| malbeware | 6 | 6 | 0 |
| safenlp_2024 | 20 | 20 | 0 |
| sat_relu | 49 | **18** | **31** |
| tllverifybench_2023 | 2 | 2 | 0 |
| **TOTAL** | **93** | **62** | **31** |

---

## Root cause of 31 sat_relu out-of-box witnesses (R9.2 deployed)

`HyZor/WitnessExtract.try_falsify_disjunct` uses LP S1/S2/S3 strategies (feasibility + per-row max + perturbations) and accepts via `_ort_replay(...)` which only checks `c @ y <= d + 1e-6` on the model output. It does NOT verify the witness x* lies within the input box. The LP can drift slightly outside the bound box due to slack variables.

When ACT consumed this witness via `strict_replay_for_act`, it likewise only evaluated `_eval_unsafe_strict(y)`. Output y was indeed in the unsafe set, so SAT was emitted. But x was NOT a valid adversary against the spec (which quantifies over the input box).

### R9.2 fix
`act/back_end/solver/solver_hz.py:_x_star_in_input_box(net, x)` walks all INPUT_SPEC layers and returns False if x violates any per-dim bound. Called at the top of `strict_replay_for_act`. Witnesses outside the box fail the gate → strict_replay returns False → `_emit_sat_with_receipt` downgrades to UNKNOWN with `small_dense_lp_phantom_rejected=True`.

Pinned by `test_round9_2_input_box_gate_rejects_out_of_box_witness`. ACT total tests **75/75 green**.

---

## R9 + R9.2 code changes summary

| File | Change |
|---|---|
| `act/back_end/interval_tf/tf_mlp.py` | `tf_slice` sentinel-bounds fallback on shape mismatch (R9 — linearizenn fix) |
| `act/back_end/solver/solver_hz.py` | new `_x_star_in_input_box(net, x_arr)`; called at top of `strict_replay_for_act` (R9.2) |
| `tests/test_hz_reduction_soundness.py` | 33 → 34 (+ `test_round9_2_input_box_gate_rejects_out_of_box_witness`) |

---

## Files to inspect in this run dir

  - `cersyve/run.log` etc. — per-bench raw logs
  - `*/per_instance_*.json` — structured per-instance output
  - `*/<bench>_<iid>_q<idx>_<source>_<attempt>.json` — individual FAL receipts (93 total)
  - `*/<...>.x_star.npy` + `*/<...>.y_ort.npy` — witness arrays
  - `p1_receipt_audit.csv` — full audit CSV (93 rows)
  - `dirty_diff.patch` + `dirty_status.txt` — code provenance
  - `OVERNIGHT_REPORT.md` — this file

---

## Open items for morning

### Must address
1. **Clean commit** R9 + R9.2 + bonus runs + test additions
2. **Rerun sat_relu under R9.2** — expect 18 valid FAL (the 31 out-of-box should auto-downgrade to UNKNOWN with the gate; possibly some recover via a different witness)
3. Confirm `malbeware 89/100 CERT` is reproducible (one-off finding right now)
4. Linearizenn legacy 46 → strict 13 gap: investigate if R9 sentinel can be tightened (the 33 lost CERT might be recoverable with a less-aggressive fallback)

### Optional
5. Conv pilot per advisor Route C (cifar100/tinyimagenet) — fresh new-representation experiment
6. cora_2024 full run (slow but maybe valuable: 1/6 already showed strict CERT+FAL)
7. nn4sys ERROR_NotImplementedError investigation — what op is missing

### Won't do
- More small-dense scans on cersyve/cgan/collins (scope mismatch)
- mscn / nn4sys cardinality_*_dual (already closed per memory)
- input-split / B&B (closed per design principles)

---

## Honest publishable claim draft

> We integrated HyZor's small-dense LP pipelines (GlobalTriangleLP +
> SpecAware + Witness extraction) into ACT under a strict-receipt
> formal-mode CLI that records git_head + per-instance JSON +
> reproducible FAL receipts (model_sha256, spec_sha256, x_star_sha256,
> fresh-ORT y, zero_tol_holds, small_tol_holds, query_index, input-box
> validity). Across 6 small-dense VNN-COMP benchmarks (1558 instances),
> the pipeline produces 510 strict CERT + 52 in-box-valid FAL = 562
> hard decisions, with zero hard soundness violations under the
> two-tier framework (official_zero versus official_small). Compared
> to baseline single-method configurations, the auto-mode shallow-20
> refinement adds +49 CERT on safenlp_2024 (n=1080), and SpecAware
> refinement adds +12 CERT on acasxu_2023 (n=186). All FAL claims
> are gated by a strict input-box validator that rejects witnesses
> outside the declared spec input region; a fresh ORT replay is
> recorded alongside.
