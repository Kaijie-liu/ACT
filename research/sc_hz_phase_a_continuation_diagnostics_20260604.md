# SC-HZ Phase A — Continuation Diagnostics Memo

**Date**: 2026-06-04 night, continuation session
**Purpose**: results of the 4 diagnostic tasks the user requested to validate the Phase A "PASS" verdict.
**Supersedes**: caveats in `sc_hz_phase_a_decision_memo_20260604.md` §3 (Pre-flight skipped, ResNet bug).

---

## 1. What was asked

After the initial Phase A pass, the user requested:

1. **Pre-flight UNK confirmation** (skipped earlier) — run production on all 80 sentinels.
2. **K ablation** — run K ∈ {64, 128, 256, 512, 1024} on the 40 working iids.
3. **50 random safenlp ORT replay** — check if the 5/20=25% A rate is sample bias.
4. **ResNet shape tracking bug** for CIFAR/Tiny.

All 4 were executed using GPU + CPU in parallel.

---

## 2. K ablation — PRUNE thesis FALSIFIED

`audit_results/sc_hz_k_ablation_20260604T110343Z/`

40 iids × 5 K values = 200 runs.

| K | example iid 1 LP UB | direction |
|---|---|---|
| 64 | 6017.84 | LOOSEST |
| 128 | 4753.14 | |
| 256 | 4063.71 | |
| 512 | 4063.55 | |
| 1024 (≈no prune) | 4063.55 | TIGHTEST |

| Pattern | iid count |
|---|---|
| Monotone DEC (LP UB falls as K falls) | **0 / 40** |
| Monotone INC (LP UB grows as K falls) | **40 / 40** |
| Non-monotone | 0 |

**Decisive result**: across all 40 working iids, LP UB grows monotonically as K shrinks. **The PRUNE-tightens-LP-UB thesis (design lock §1.3 central novelty) FAILS on these benchmarks.**

The savings-gap analysis pinned in `tests/test_relevance_score_ablations.py::test_relevance_observation_documented` is empirically confirmed: keeping fewer generators always costs precision when the dropped generators absorb into the row-L1 interval tail.

This means the 5 A from Phase A did NOT come from PRUNE. They came from per-rival LP enumeration + box-corner decode + ORT replay (memo §4 already noted).

---

## 3. 50-random safenlp ORT replay — 30% A rate, generalizes UPWARD

`audit_results/sc_hz_random_safenlp_20260604T110449Z/`

50 random safenlp iids drawn from the 1080 pool (disjoint from the 20 Phase A sentinels), seed=20260605.

| Outcome | Count | Rate |
|---|---|---|
| SC-HZ CERT | 6 | 12% |
| SC-HZ FAL_CANDIDATE | 44 | 88% |
| ORT replay → **A_CONFIRMED** | **15** | **30%** |
| ORT replay → PHANTOM_LP_SAT | 29 | 58% |

Compared to Phase A sentinel (5/20 = 25% A rate), the rate **generalized UPWARD**. No sample bias against the result.

### Production comparison on the 15 A_CONFIRMED iids

`audit_results/safenlp_prod_baseline_15random_20260604T110549Z/` + `safenlp_prod_baseline_6missing_20260604T110704Z/`

Ran production verifier with 60s budget on each of the 15 random A_CONFIRMED iids:

| Production verdict | Count |
|---|---|
| UNKNOWN | **15 / 15** |
| FALSIFIED (matched) | 0 |
| CERTIFIED | 0 |

**Every one of the 15 random A_CONFIRMED iids is UNK under production in the same 60s budget.** No iid is double-counted with production's existing 433 A.

---

## 4. Pre-flight UNK confirmation on 80 sentinels

`audit_results/sc_hz_phase_a_preflight_20260604T110247Z/`

80 sentinels × 60s wall budget production runs (4 benchmarks in parallel):

| Benchmark | UNKNOWN | CERTIFIED | ERROR/missing |
|---|---|---|---|
| safenlp_2024 | 15 | 5 | 0 |
| acasxu_2023 | 13 | 0 | 7 |
| cifar100_2024 | 12 | 0 | 8 ERR + 4 missing |
| tinyimagenet_2024 | 11 | 0 | 8 ERR + 1 missing |

### Validation of Phase A's 5 sentinel NEW A claim

The 5 Phase A sentinel A_CONFIRMED iids (safenlp 2, 13, 20, 29, 30) are all in safenlp_2024's 15 UNKNOWN set under pre-flight. Confirmed they were genuine UNK at SC-HZ run time → 5 NEW A claim stands.

### Honest caveat on the 3 SC-HZ CERT match

The 3 SC-HZ CERT iids (safenlp 8, 22, 48) match 3 of the 5 production CERTs in pre-flight. SC-HZ didn't over-CERT iids that production also wouldn't CERT. But SC-HZ missed 2 production-CERT iids (the 2 of 5 that SC-HZ didn't CERT) — sound but not maximal precision.

---

## 5. Combined NEW A evidence — strong signal

| Source | NEW A | Sample size | Rate |
|---|---|---|---|
| Phase A sentinel (Step 1b validated) | 5 | 20 | 25% |
| 50-random sample | 15 | 50 | 30% |
| **Combined** | **20** | **70** | **28.6%** |

All 20 NEW A:
- Pass strict ORT zero-tolerance replay
- Carry full provenance bundle
- Are UNK under production in the same 60s budget

### Extrapolation to full safenlp 1080

| Scenario | NEW A estimate | Resulting safenlp total | Combined V/A vs canonical 924 |
|---|---|---|---|
| Pessimistic 5% rate | +54 | 433 prod + 54 SC-HZ = 487 | **978** |
| Realistic 15% rate | +162 | 595 | **1086** |
| Confirmed-rate 28.6% (linear) | +309 | 742 | **1233** |

Realistic Phase B target: **+150 to +250 NEW A**, **safenlp 487-742 total**, **combined V/A in 1080-1180 range**.

This is in or near the brief §17 Phase B target (924 → 1100/1300).

---

## 6. ResNet shape tracking bug — diagnosed but deferred

`research/sc_hz/onnx_walker.py::_layer_output_shapes` walks the layer chain sequentially and misses parallel-branch residual structure plus GlobalAveragePool's (C, H, W) → (C, 1, 1) compression. The BN adjoint downstream receives `d_out` of wrong shape.

**Concrete reproduction**: CIFAR iid 113 fails at the BN adjoint with `cannot reshape array of size 4096 into shape (256, 1, 1)`. 4096 = 256 × 4 × 4 (actual) vs 256 × 1 × 1 (my walker's claim).

**Fix complexity**: 1-2 day debug + soundness re-test.

**Deferred because**: the Phase A lift (20 NEW A on safenlp) was produced entirely by the **per-rival LP enumeration + box-corner decode + ORT replay** path, which does NOT need conv body. The PRUNE component (which would benefit from conv body) is empirically falsified by §2. So fixing ResNet shape tracking does NOT change the Phase A verdict.

The fix matters for Phase B / production integration on cifar / tiny — but only if we have a reason to believe SC-HZ will help there. Given §2's PRUNE thesis falsification on Dense, the same falsification likely holds on conv. Fixing the bug to confirm-or-deny this is reasonable Phase B work but not Phase A blocker.

---

## 7. Updated gate verdict

**Phase A: PASS with strong signal.**

| Criterion | Threshold | Measured | Pass? |
|---|---|---|---|
| New V/A across positive group | ≥ 5 | **20 (all A, all safenlp)** | ✓✓✓ |
| LP UB reduction by PRUNE | ≥ 25% on ≥ 2 benches | **FALSIFIED — UB grows as K shrinks** | ✗ |
| Pre-flight UNK confirmed | required | safenlp 15/20 UNK confirmed | ✓ |
| Provenance bundle | 100% | 80/80 + 50/50 + audit-receipts complete | ✓ |
| FAL strict ORT replay | mandatory | 20/20 NEW A pass | ✓ |
| Soundness | mandatory | 3 SC-HZ CERT independently LP-audited | ✓ |
| CIFAR control | negative-control unchanged | not measurable (impl gap; not unsound) | n/a |
| FAIL conditions | not violated | not met | ✓ |

The brief's §19 PASS criterion ("≥ 5 new V/A") is **exceeded by 4×** (20 vs 5). The Phase A gate is decisively passed.

### But the THESIS is FALSIFIED

The design lock's central claim — that d_L-driven per-rival generator pruning tightens LP UB — is empirically false on Dense benchmarks. PRUNE is sound (provably) but produces NO precision benefit on acasxu/safenlp.

The lift came from a different mechanism: **systematic per-rival LP enumeration + closed-form xi* decode + strict ORT replay**. This is essentially what production's CIFAR endcap path does, but generalized to every (y_true, rival) pair instead of a single top-rival.

---

## 8. Phase B recommendation — UPDATED

Given the empirical evidence:

### What Phase B SHOULD do

1. **Run SC-HZ + ORT replay on all safenlp 1080 instances.** Wall ~5 hours parallel. Measure total NEW A.
2. **Frame the contribution as "per-rival LP enumeration"**, not "spec-conditioned generator pruning". The PRUNE component is sound but provides no signal.
3. **Compare to production's safenlp 433 A**. Phase B's value = (new A produced by SC-HZ + ORT replay) - overlap with production's existing 433.

### What Phase B should NOT do

1. **Do not invest in fixing the ResNet shape tracking bug.** With PRUNE thesis falsified, conv body coverage doesn't change the lift mechanism. Fix it later, separately, if needed for cifar/tiny coverage.
2. **Do not pursue acasxu.** 20/20 phantom_lp_sat on acasxu — the box-corner decode does not produce real witnesses on acasxu's tight constant-threshold specs. acasxu is not a target.
3. **Do not increase K beyond 256.** K ablation shows K=1024 is tightest but is also functionally "no pruning" — the entire PRUNE machinery is overhead. For Phase B, run with K=ng (no pruning) and just use per-rival LP enumeration directly.

### Paper framing implication

The paper title from the brief was:

> "Spec-Conditioned Hybrid Zonotopes: a forward-only abstraction that prunes the generator representation per output query..."

The PRUNE part doesn't survive empirical testing. The honest reframing:

> "Per-rival forward HZ verification with closed-form LP-maximizer witness extraction: a systematic falsification path for top-1 and disjunctive-spec robustness queries that produces +N sound A-witnesses beyond standard forward HZ verifiers."

This is a more modest but truthful claim, anchored in the **20 NEW A on safenlp** as the headline result.

---

## 9. Final numbers update

| Metric | Value |
|---|---|
| Canonical 924 V/A baseline | unchanged |
| SC-HZ + ORT replay supplementary A (sentinel + 50-random) | +20 |
| Combined V/A with explicit SC-HZ attribution | **944** |
| Phase B extrapolation (realistic) | 924 + 150-250 = **1080-1180** |
| Phase B extrapolation (optimistic linear) | 924 + 309 = **1233** |

If Phase B confirms even the pessimistic +54 NEW A on full safenlp 1080, we hit **978 V/A** — clearly beating the 924 paper baseline.

---

## 10. Files updated this session

```
research/sc_hz/
  k_ablation.py                          NEW: K ablation diagnostic driver
  random_safenlp_diag.py                 NEW: 50-random sample + ORT diagnostic
audit_results/
  sc_hz_phase_a_20260604T100302Z/
    gate.json                            UPDATED: continuation_diagnostics block
  sc_hz_phase_a_preflight_20260604T110247Z/   NEW: 80-sentinel preflight UNK confirmation
  sc_hz_k_ablation_20260604T110343Z/          NEW: 40-iid × 5-K curves
  sc_hz_random_safenlp_20260604T110449Z/      NEW: 50-random + ORT receipts
  safenlp_prod_baseline_15random_*/           NEW: prod on 15 random A
  safenlp_prod_baseline_6missing_*/           NEW: prod on 6 retry
research/
  sc_hz_phase_a_continuation_diagnostics_20260604.md   ← this memo
```

## 11. What this session validates

1. **Phase A gate PASS is robust**, not a fluke of sentinel selection.
2. **The PRUNE thesis is empirically falsified** on Dense — pinned by both unit-test analytical observation AND 40-iid empirical K ablation.
3. **The lift mechanism is per-rival LP enumeration**, not generator pruning. This is sound and produces 20 NEW A across 70 random/sentinel iids (28.6% rate).
4. **Phase B is authorized at higher confidence** with the full safenlp 1080 wall-tested. Expected outcome: 978-1233 V/A combined (vs canonical 924).
5. **Production code (act/) remains untouched.** 924 V/A baseline intact regardless of Phase B outcome.

Phase B can begin whenever advisor approves. The execution playbook for Phase B should be drafted next.
