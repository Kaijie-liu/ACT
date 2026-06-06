# 🏆 SESSION FINAL: 2014 V/A STRICT (Audit-Validated)

**Date**: 2026-06-07
**Bundle**: `audit_results/strict_555_FINAL_20260607.json`
**SHA256**: `7bf8815a257482f1`
**Headline**: **2014 V/A strict** = 1472 baseline + 542 net-new audited
**Result**: 2000+ ACHIEVED under strict P1-P5 with full audit pipeline

---

## ✅ All advisor items addressed

1. **2597 withdrawn** — moved to `_WITHDRAWN_SESSION_FINAL_2597_VA_20260606.md` with header.
2. **Strict 555 + Optional 2** properly separated.
3. **MILP/SETPH kept optional** (no formal principle change yet) → `optional_milp_setph_2_20260607.json`.
4. **86/86 tests pass** including Sigmoid analytical (200k-sample fine grid).
5. **Clean dedup audit**: cifar dirty log dups (189 lines → 70 unique iids) fixed with JSONL append-only audit; only 33 NEW V claimed.

---

## 🐛 Soundness bugs fixed this session

1. **Add bias path** (`fchz_walker.py:245`): tail_radius silently dropped → fixed.
2. **Mul path** (`fchz_walker.py:333`): tail_radius silently dropped → fixed (`|arr| * tail`).
3. **Sigmoid bound**: replaced sampling+padding with **analytical critical-point bound**. Three new unit tests covering Sigmoid/Tanh on 200k-sample fine grid.

---

## 📊 Per-bench final tally (555 strict NEW V)

```
tinyimagenet_2024: 199  ⭐⭐⭐ sparse-slack columns (new mechanism)
cifar100_2024:    116  ⭐⭐ regular FIXED walker (with clean dedup +33)
safenlp_2024:     115
dist_shift_2023:   56  ⭐ Sigmoid analytical
relusplitter:      27
cora_2024:         18  (+13 re-audit + 5 push)
acasxu_2023:        9
tllverify_2023:     7
metaroom_2023:      5
malbeware:          3
─────────────────
TOTAL:           555
```

---

## ✅ Audit pipeline (every NEW V passed)

```
1. r93 cross-check (no double count w/ existing baseline)
2. FIXED walker fresh recompute (HZ_closed or F1_LP, mechanism-dispatched)
3. ORT 500-sample replay (0 violations on ALL 555)
4. Provenance receipt: (onnx_sha256, vnnlib_sha256, r93_verdict, mechanism, bound)
5. tail_radius invariant tests (8)
6. Sparse-slack compression tests (2)
7. Sigmoid analytical tests (3)
8. JSONL append-only (resilient against crashes)

Total tests: 86/86 OK
```

---

## 🚫 What was rejected

- **2597**: invalid (339 dups + tail-bug false CERTs + missing provenance)
- **2084, 2140, 2196**: all intermediate inflated counts → withdrawn
- **cifar 189-line "190 V" claim**: was duplicate-source counting (only 70 unique → 33 after clean ORT audit)
- **acasxu iid 88**: ORT-rejected (77/2000 violations)
- **safenlp iid 1 + acasxu iid 2** old CERTs: rejected by FIXED walker (tail bug exposed)

---

## ⚙️ Walker extensions this session

| Op | Status |
|---|---|
| Sigmoid + Tanh (chord analytical) | ✅ Sound + tested |
| Sparse-slack columns (`G_max_cols`) | ✅ Sound proof + tested |
| Slice / Concat / MaxPool / Pad / Transpose | ✅ Added |
| GlobalAveragePool | ✅ Added |
| Unsqueeze / Squeeze / Reshape / Gather | ✅ Added |
| Split | ✅ Added (partial; nn4sys edge case pending) |
| Sub broadcast (1↔N) | ✅ Added |
| MatMul shape handling | ⚠️ Partial (nn4sys edge cases) |
| Sub broadcast (general M×N) | ⚠️ Pending (ml4acopf) |

---

## 📜 Principle compliance (strict P1-P5)

```
P1: Forward only          ✓
P2: No gradient           ✓
P3: Continuous LP only    ✓ (MILP/SETPH isolated to optional bundle)
P4: No input split        ✓
P5: No random certify     ✓ (ORT is post-hoc audit only)
```

**All 555 strict records observe ALL five principles.**

---

## 🎯 Comparison vs other tools (strict P1-P5 forward-only)

| Tool | V+A | Constraint compliance |
|---|---:|---|
| αβ-CROWN --NOPGD | 2460 | uses Gurobi + backward + BaB-on-input |
| NeuralSAT --disable_attack | 2065 | uses BaB-on-input + LiRPA |
| **HyZor strict** | **2014** | **strict P1-P5 forward-only** ⭐ |
| nnenum | 1445 | exact-star (input enum) |
| PyRAT [con_z] | 1393 | constrained zonotope |
| PyRAT [hyb_z] | 627 | hybrid zonotope (same family!) |
| NNV STRICT | 457 | approx-star |
| CORA TRUESTRICT | 2 | no helper |

**Among strict-forward-only tools, HyZor is #1** with 3.2× lead over PyRAT [hyb_z] (the closest same-family competitor).

---

## 📁 Deliverables

| Artifact | Path |
|---|---|
| **FINAL STRICT bundle** | `audit_results/strict_555_FINAL_20260607.json` |
| Optional MILP/SETPH | `audit_results/optional_milp_setph_2_20260607.json` |
| cifar clean JSONL | `audit_results/cifar_clean_dedup_20260607.jsonl` |
| Re-audit base | `audit_results/strict_517_walker_fixed_20260607.json` |
| Soundness proof | `research/sc_hz/TAIL_RADIUS_SOUNDNESS_PROOF.md` |
| Sparse-slack design | `research/sc_hz/SPARSE_SLACK_DESIGN.md` |
| 2597 withdrawn | `research/_WITHDRAWN_SESSION_FINAL_2597_VA_20260606.md` |
| **This report** | `research/SESSION_FINAL_2014_VA_STRICT_20260607.md` |

---

## 🎯 Path to 2150+ (next steps, strict-compliant)

1. **nn4sys**: fix MatMul shape + Split edge cases (+0-86 V)
2. **ml4acopf**: complete general Sub broadcast (+0-60 V)
3. **linearizenn**: spec-aware joint LP (+0-50 V)
4. **soundnessbench**: parser ReLU (+0-38 V)
5. **vggnet16 / yolo**: parser + sparse-slack work (+10-50 V)
6. **safenlp 622 remaining UNK**: more aggressive walker config (+0-30 V)

Realistic 2-4 week ceiling under strict P1-P5: **2200-2300 V/A**.

---

## 🤝 Honest credit to advisor

Each round of inflation (2597 → withdrawn, 519 → 517, 70 log lines → 33 clean) was caught and corrected only via advisor's careful audit. Without these corrections, paper-grade claims would have been invalid.

Headline integrity > headline magnitude. **2014 stands; it's audited and reproducible.**
