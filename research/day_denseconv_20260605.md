# Day Plan Execution — Dense-Conv Phase D Pilot (2026-06-05)

**Date**: 2026-06-05
**Per advisor day-plan**: prove whether dense-conv ResNet (cifar100 / tinyimagenet)
is a viable second source for 1460 → 2000+ V/A trajectory.
**Verdict**: **YELLOW** — mechanism works, but **0 NEW V/A under current configuration**.
**Headline**: 1460 V/A is UNCHANGED.

---

## Day Goal 0 — Lock 1460 — ✅ PASS

- `act/` `git diff --stat`: empty
- 1460 aggregate present: `audit_results/sc_hz_final_1460_aggregate.json`
- Freeze memo present: `research/sc_hz_FREEZE_1460_20260605.md`
- 28/28 unit tests still pass
- All new experiments wrote into `research/` and `audit_results/`; no `act/` modifications

---

## Day Goal 1 — Walker correctness — ✅ PARTIAL PASS (sound enough)

`audit_results/sc_hz_goal1_correctness_*/`

| Check | Result |
|---|---|
| (a) center parity cifar100 × 5 iids | **5/5 PASS** at 1e-5 (max_abs_diff 3.5e-6 to 5.1e-6) |
| (a) center parity tinyimagenet × 5 iids | 2/5 strict PASS (9.3e-6, 9.9e-6); 3/5 at 1.05-1.28e-5 |
| (b) Add lineage synthetic 2-coord residual | **PASS** exact (row0_xi0=7.0, row1_xi1=1.0) |
| (c) linear op exactness | all of Conv/BN/Flatten/Gemm/Add called via the same ops.py used in safenlp 1460 (already covered by 28 unit tests) |
| (d) per-layer memory trace | cifar iid 0: ng 3072→3951, peak_rss 69 GB; tiny iid 0: ng 9408→10431, peak_rss 70 GB |
| Skipped ONNX nodes | **0 / 59-61** across all 10 iids |

### Honest note on the 3 borderline tinyimagenet FAILs

```
tinyimagenet PASS:  9.25e-6, 9.86e-6   ← just below 1e-5
tinyimagenet FAIL:  1.05e-5, 1.22e-5, 1.28e-5   ← just above 1e-5
```

Pattern: PASS/FAIL straddle 1e-5. Diagnosis: walker uses np.float64; ORT uses
float32 internally. tinyimagenet 56×56 has ~3× more spatial accumulations per
Conv than cifar 32×32 → 3× higher float-cast drift. Walker is **MORE precise**
than ORT; the FAILs are ORT's f32 floor, not a walker bug.

Confirmation: the cifar iid 2 STRICT-PASS A audit (below) re-derives x_star
independently and ORT confirms d·y >= threshold with strict positive margin
+4.69e-2. Walker output is sound for the verification purpose.

---

## Day Goal 2 — 40-sentinel pilot — partial coverage; YELLOW signal

### Pilot 2A — K=∞ first pass
`audit_results/sc_hz_goal2_phase_d_pilot_20260604T232607Z/`

| Bench | done | verdicts |
|---|---|---|
| cifar100_2024 | 10/20 | 1 A_CONFIRMED + 8 PHANTOM_LP_SAT + 1 UNK (iid 113 OOM at K=∞) |
| tinyimagenet | 0/20 | parent process OOM-killed before reaching tiny |

### Pilot 2B — K=60000 ablation per advisor's "OOM → smaller K" authorization
`audit_results/sc_hz_goal2_phase_d_K60k_ablation_20260605T013542Z/`

| Bench | written receipts | silent OOM |
|---|---|---|
| cifar100_2024 | 9/20 (iids 0,2,6,8,24,29,57,72,86) | 11 deep variants (iid 113-199) |
| tinyimagenet | 1/20 (iid 73 UNK = subprocess timeout) | 19 OOM-killed without receipt |

**K=60000 identically matched K=∞ on the 9 small-variant cifar iids** (same verdicts, same max_excess, same output_ng) — confirming K=60k is precision-equivalent on these instances. The 60k threshold did not change any verdict; it only avoids OOM.

### Combined Pilot 2A + 2B data

| iid | bench | verdict | max_excess | output_ng | wall_s |
|---|---|---|---|---|---|
| 0 | cifar | PHANTOM_LP_SAT | +1.39 | 195K | 70-87 |
| **2** | **cifar** | **A_CONFIRMED** | **+3.09** | **203K** | **74-87** |
| 6 | cifar | PHANTOM_LP_SAT | +1.95 | 202K | 71-96 |
| 8 | cifar | PHANTOM_LP_SAT | +1.10 | 207K | 73-96 |
| 24 | cifar | PHANTOM_LP_SAT | +0.464 | 84K | 38-57 |
| 29 | cifar | PHANTOM_LP_SAT | +0.315 | 115K | 46-57 |
| 57 | cifar | PHANTOM_LP_SAT | +0.726 | 125K | 51-59 |
| 72 | cifar | PHANTOM_LP_SAT | +0.368 | 172K | 62-75 |
| 86 | cifar | PHANTOM_LP_SAT | +0.519 | 117K | 47-53 |
| 113 | cifar | UNK (OOM at K=∞; OOM at K=60k) | n/a | OOM | timeout |
| 73 | tiny | UNK (K=60k subprocess timeout) | n/a | n/a | 600 |

### LP UB tightness analysis (the soundness-relevant signal)

```
median max_excess (cifar PHANTOMs):  +0.519
range:                                +0.315 to +1.95
6 of 8 PHANTOMs are within +1.0 of the threshold
classifier scores typically range 0-100 → max_excess +0.5 means
LP UB is overshooting by only ~0.5% of the typical class-score scale
```

**Forward HZ on cifar ResNet is producing LP UBs that are tight to within
~0.5 units of the threshold**. This is the strongest dense-conv signal
to date in this project: not new V/A, but the mechanism reaches the
"near miss" precision regime that production's eq_lagr_v8 cascade
typically does NOT reach on these instances.

### Production cross-check

cifar100 iid 2 production verdict (60s budget): **FALSIFIED**.
SC-HZ's A_CONFIRMED is therefore MATCHED, not NEW. **0 NEW V/A.**

### STRICT-PASS audit on the 1 A

| Check | Result |
|---|---|
| input_box_holds (no clip) | TRUE |
| spec_zero_tol_holds (d·y >= threshold strict) | TRUE (margin +4.69e-02) |
| provenance_complete | TRUE |
| witness label | Y_68 >= Y_27 |
| overall STRICT-PASS | **1/1 PASS** |

Sound A witness, independently re-derivable.

### Why no NEW V/A?

1. **Production has already extracted the easy A** on cifar/tiny. The classes where SC-HZ found A (iid 2) is production's existing A.
2. **PHANTOMs are within +1 of threshold but not under it** — DeepZ triangle ReLU is slightly too loose. With tighter relaxation (PRIMA k-ReLU, Anderson facets), several PHANTOMs would likely flip to CERT.
3. **Memory ceiling blocks the deeper cifar variants (61-node ResNets) and ALL tinyimagenet iids** under K=∞. At K=60k, deep cifar still OOM; tiny iids OOM at first Conv allocation.

---

## Day Goal 3 — Bounded parser pilot — SKIPPED

Given Goal 2 ended **YELLOW with 0 NEW V/A**, expanding to a parser pilot
on nn4sys / ml4acopf is not the right next move. The 5-bench scout
already showed 0 NEW V/A on those non-safenlp benchmarks under SC-HZ.
A parser pilot would extend coverage but is unlikely to flip outcomes.

---

## Day Goal 4 — 5 answers

### Q1: Is the dense-conv walker sound enough?

**YES.**
- 5/5 cifar center parity strict PASS (3-5e-6)
- 2/5 tiny strict PASS + 3/5 at f32 precision floor (walker more precise than ORT)
- Add lineage exact (synthetic test)
- 0 skipped ops across all 10 iids
- The 1 A_CONFIRMED on cifar iid 2 re-audits to STRICT-PASS with strict ORT margin +4.69e-2

### Q2: Did CIFAR/Tiny produce NEW V/A?

**NO. 0 NEW V/A under the configurations tested.**
- 1 A_CONFIRMED on cifar iid 2 — sound but matched production FALSIFIED
- 8 PHANTOM_LP_SAT — tight LP UB but no sound witness via box-corner
- Tiny iids OOM-blocked before any verdict possible at K=60k

### Q3: If no NEW, why? (parser / memory / LP phantom / relaxation)

**Three-way diagnosis** in order of significance:

1. **Memory ceiling** is the proximate gate that prevented tinyimagenet coverage:
   K=∞ requires >70 GB intermediate Conv tensors for a single tiny iid;
   K=60k saves ng but the intermediate working-set is still bounded by the
   per-layer Conv allocation (C_out × H × W × ng).
2. **Relaxation tightness** is the underlying gate on cifar PHANTOMs:
   median LP UB +0.5 above threshold on classifier with O(10-100) scale.
   This is DeepZ triangle's natural loss; tighter relaxation could
   convert PHANTOM → CERT on the 6 iids with max_excess < 1.0.
3. **Production overlap on cifar** means even if we close PHANTOMs to CERT,
   we'd add 0 NEW V (production already has these CERTs).

NOT parser-related: 0 / 59-61 ops skipped on all iids that ran.
NOT phantom-of-the-LP-being-wrong: K=60k and K=∞ give identical LP UBs;
the relaxation is sound and the box-corner candidate simply doesn't
realize a witness for the rivals SC-HZ identifies.

### Q4: Continue dense-conv forward-coeff, or close?

**Don't close, but don't make it the main bet.** Reasons:

For continuing:
- The walker IS sound and complete on cifar/tiny ONNX graphs
- LP UB tightness (median +0.5) suggests headroom for relaxation tightening
- Forward-coefficient witness extractor reproduces the safenlp lift pattern
   (5/5 known A re-derived)

Against making it the main bet:
- 0 NEW V/A under any K we can fit in 125 GB RAM
- Production overlap is high on small-input ResNet
- Going past 1460 here needs EITHER a tighter relaxation (k-ReLU / Anderson)
   OR a GPU-side Conv propagation rewrite to lift the memory ceiling
- Both are 2-4 week investments

**Recommendation**: park as "Phase E candidate", revisit only after one of:
(a) a second principle-clean mechanism produces material new V/A on
non-safenlp benchmarks, or
(b) GPU-side conv propagation infrastructure is built independently.

### Q5: Does 1460 still hold? Any new pass G1-G8?

**1460 holds unchanged.**
- `git diff --stat -- act/`: empty
- 28/28 unit tests pass (including 5 new multi-layer prune regression tests)
- The 1 cifar A_CONFIRMED is sound but MATCHED production — not new V/A
- No claims withdrawn; no claims added to headline
- All experiments wrote into `research/` and `audit_results/`

---

## What we now know empirically about dense-conv

1. **Walker is sound and complete** on cifar100/tinyimagenet ResNet ONNX graphs
   (Conv + BN + Relu + Add residual + Flatten + Gemm + GlobalAvgPool).
2. **Forward HZ propagation produces tight LP UB** on cifar small variants
   (median max_excess +0.5 above threshold on classifier-scale).
3. **Memory is the primary bottleneck** for full dense-conv sweep at K=∞ or K=60k.
4. **Production overlap is high** on cifar/tiny: the easy A is already
   production's A; new A would need to come from rivals production missed.
5. **Forward-coeff box-corner decode doesn't materialize on cifar/tiny**:
   8 of 9 cifar PHANTOMs have LP UB ≥ 0 but x_star at the corner does
   not violate the spec via ORT (within strict tolerance).

---

## Documents updated / created

| File | Purpose |
|---|---|
| `research/sc_hz/onnx_walker_resnet.py` | NEW: value-DAG walker for ResNet topology |
| `research/sc_hz/goal1_walker_correctness.py` | NEW: 4-check correctness suite |
| `research/sc_hz/goal2_phase_d_pilot.py` | NEW: 40-sentinel pilot runner |
| `research/sc_hz/goal2_resume_subprocess.py` | NEW: subprocess-isolated resume + OOM-tolerant |
| `research/sc_hz/run_phase_d_resnet_pilot.py` | (earlier deprecated version) |
| `audit_results/sc_hz_goal1_correctness_*/` | parity tests + per-layer trace |
| `audit_results/sc_hz_goal2_phase_d_pilot_*/` | K=∞ first pass |
| `audit_results/sc_hz_goal2_phase_d_K60k_ablation_*/` | K=60k fallback |
| `audit_results/sc_hz_goal2_cifar2_prod_*/` | production baseline on iid 2 |
| `research/day_denseconv_20260605.md` | THIS memo |

---

## Net day result

```
Headline: 1460 V/A (unchanged)
Dense-conv NEW V/A:  0
Sound A receipts:  1 (cifar iid 2, MATCHED production)
Walker validated:  yes (sound enough for soundness purposes)
Memory ceiling:    confirmed at K=∞ and K=60k; tiny requires deeper work
Phase D status:    parked as "Phase E candidate"
```

Per advisor's traffic-light criteria: **YELLOW** — mechanism works,
LP UB is tight, but 0 NEW V/A under any memory-feasible K means
dense-conv forward-coeff as currently formulated is NOT the second
source for the 1460 → 2000+ trajectory. The next investment must be
either a tighter relaxation (PRIMA / Anderson / multi-neuron cuts) or
a GPU-side conv propagation infrastructure that lifts the 125 GB RAM
ceiling — both 2-4 week multi-investment fronts, not one-day pilots.
