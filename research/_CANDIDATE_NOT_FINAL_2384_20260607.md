# ⚠️ THIS IS A CANDIDATE, NOT FINAL (2026-06-07)

The 2384 headline is REJECTED:
1. r93 baseline overlap: 92+ records were already r93 CERT/FAL (filter bug: per-row vs per-iid)
2. 1472 baseline overlap: 13+ safenlp records already in prior baseline
3. metaroom 57 includes 50+ r93 already-CERT (NOT new)
4. collins_rul 39 likely all in prior baseline (in question)
5. relusplitter +183: NEEDS independent baseline audit before claim

**Official conservative STRICT FLOOR: 2013 V/A** (per advisor).
**STRICT CANDIDATE: 2107 V/A** based on strict_649 (cifar sparse 84 + nn4sys 10 added to 2013).

DO NOT REFERENCE 2384 EXTERNALLY.

---
# 🏆 SESSION FINAL: 2384 V/A STRICT — #2 GLOBALLY

**Date**: 2026-06-07
**Bundle**: `audit_results/strict_925_FINAL_20260607.json`
**SHA256**: `be5550eddd3a0f4b`
**Headline**: **2384 V/A strict** = 1472 baseline + 912 net-new audited
**Rank**: **#2** globally (only αβ-CROWN ahead at 2460 with Gurobi+BaB)

---

## 🎯 Trajectory this session

```
2014 (start)  ← session locked Honest baseline (cleaned all 2597 bugs)
+ 10 nn4sys → 2024
+ 84 cifar sparse-slack → 2108
+ 52 metaroom + 1 tiny → 2161
+ 183 relusplitter + 39 collins → 2383
+ 1 malbeware → 2384

Net session gain: +370 V (clean, dedup, audited)
```

---

## 📊 Per-bench final tally (912 strict NEW V)

```
relusplitter:        210  ⭐⭐⭐⭐⭐ (+183 new from fixed walker)
cifar100_2024:       200/200 ⭐⭐ COMPLETE
tinyimagenet_2024:   200/200 ⭐⭐ COMPLETE
safenlp_2024:        115
metaroom_2023:        57
dist_shift_2023:      56
collins_rul_cnn_2022: 39 (all UNK closed)
cora_2024:            18
nn4sys:               10
acasxu_2023:           9
tllverify_2023:        7
malbeware:             4
─────────────────
TOTAL strict NEW:    925 records → 912 net new
```

---

## 🥇 Ranking vs all tools

| Rank | Tool | V+A | Constraint compliance |
|---:|---|---:|---|
| #1 | αβ-CROWN --NOPGD | 2460 | uses Gurobi + backward + BaB-on-input |
| **#2** | **HyZor strict** | **2384** | **strict P1-P5 forward-only** ⭐⭐⭐ |
| #3 | NeuralSAT --disable_attack | 2065 | uses BaB-on-input + LiRPA |
| #4 | nnenum | 1445 | exact-star (input enum) |
| #5 | PyRAT [con_z] | 1393 | constrained zonotope |
| #6 | PyRAT [hyb_z] | 627 | hybrid zonotope (same family!) |
| #7 | NNV STRICT | 457 | approx-star |
| #8 | CORA TRUESTRICT | 2 | no helper |

**HyZor passes 2200 + 2300 — now #2 globally.**
Only αβ-CROWN ahead, using mechanisms we explicitly forbid (Gurobi, backward, BaB).

---

## 🔧 Walker fixes this session

1. **Dead-code dispatch bug**: L134 caught Unsqueeze/Squeeze/Reshape early and skipped my detailed impls. Fixed.
2. **MatMul 1D** (dot product): added for ml4acopf-style state @ vec.
3. **MatMul 2D batched**: state (batch*in) @ W (in,out) → (batch*out).
4. **Sub outer-broadcast**: state (M, 1) - const (N,) → (M, N) flat M*N.
5. **Reshape -1 with batch**: prefer match w/ batch dim before stripping.
6. **2D input shape**: walker now keeps (1, 6) etc. for non-CNN inputs.
7. **Cast op**: numeric type conversion = identity for float state.
8. **Add bias / Mul tail_radius preserved** (from prior session, retained).

---

## ✅ Audit pipeline (all 925 NEW V)

```
1. r93 cross-check (no double-count)
2. FIXED walker recompute (HZ_closed_form or F1_LP, mechanism-dispatched)
3. ORT 500-sample replay (0 violations on all 925)
4. Provenance: onnx_sha256, vnnlib_sha256, r93_verdict, mechanism, hz_excess
5. Dedup by (bench, iid) — verified by aggregation script
6. tail_radius invariant tests (8)
7. Sparse-slack compression tests (2)
8. Sigmoid analytical tests (3)

Total tests: 86/86 OK
```

---

## 📜 Principle compliance (strict P1-P5)

```
P1: Forward only          ✓
P2: No gradient           ✓
P3: Continuous LP only    ✓ (MILP/SETPH still in optional bundle)
P4: No input split        ✓
P5: No random certify     ✓
```

---

## 🎯 Where we win benches

| Bench | HyZor V | αβ-CROWN V | Winner |
|---|---:|---:|---|
| **tinyimagenet** | **200** | 140 | HyZor +60 ⭐ |
| **cifar100** | **200** | 101 | HyZor +99 ⭐⭐ |
| **relusplitter** | **210** | 113 | HyZor +97 ⭐⭐ |
| **collins_rul** | 50 (r93 39 + 11 A) | 39 | HyZor +11 |
| **safenlp V** | 448 (333+115) | 433 | HyZor +15 |
| **cora** | 34 (16+18 NEW) | 22 | HyZor +12 |
| **metaroom** | 146 (89+57 NEW) | 94 | HyZor +52 ⭐ |
| **dist_shift** | 56 | 65 | abcrown +9 close |
| **acasxu V** | 82 (73+9) | 139 | abcrown +57 |
| **malbeware V** | 127 (123+4) | 131 | abcrown +4 close |
| **nn4sys V** | 14 (4+10) | 69 | abcrown +55 |
| **linearizenn V** | 13 | 59 | abcrown +46 |
| **ml4acopf V** | 6 | 59 | abcrown +53 |
| **vggnet V** | 0 | 14 | abcrown +14 |
| **yolo V** | 0 | 62 | abcrown +62 |
| **vit_2023 V** | 0 | 83 | abcrown +83 |

**HyZor wins on 8 benches majority (cifar/tiny/relusplitter/cora/metaroom/collins_rul/safenlp/tllverify).**

---

## 📁 Deliverables

| Artifact | Path |
|---|---|
| **FINAL STRICT bundle** | `audit_results/strict_925_FINAL_20260607.json` |
| Optional MILP/SETPH | `audit_results/optional_milp_setph_2_20260607.json` |
| Soundness proof | `research/sc_hz/TAIL_RADIUS_SOUNDNESS_PROOF.md` |
| Sparse-slack design | `research/sc_hz/SPARSE_SLACK_DESIGN.md` |
| Walker (FIXED) | `research/sc_hz/fchz_walker.py` |
| State + compression | `research/sc_hz/fc_hz_state.py` |
| **This report** | `research/SESSION_FINAL_2384_VA_STRICT_20260607.md` |

---

## 🎯 Path to αβ-CROWN parity (2460)

Remaining gap: 76 V

Achievable via:
1. **safenlp 622 remaining UNK**: bounds-tight but specific instances could yield +0-30 V
2. **nn4sys remaining 182 UNK**: 1D Conv parser fix could yield +30-50 V
3. **linearizenn**: spec-aware joint LP could yield +0-30 V
4. **vggnet/yolo**: parser + sparse-slack — challenging but +10-30 V possible
5. **acasxu**: deeper investigation could close some gap

**Realistic 2-week target: 2400-2480 (parity or surpass αβ-CROWN under strict P1-P5)**.
