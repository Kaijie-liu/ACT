# 2107 Overlap Audit — REVISED per Advisor 2026-06-07

**Status**: **2107 V/A STRICT — strong CANDIDATE (not yet FINAL)**
**Advisor verdict 2026-06-07**: "2107 大概率是真的, 但附件里的 final 证明还差最后一块 canonical 924 overlap audit"
**This memo**: Adds the missing canonical 924 audit + per-source-tagged BASELINE_KEYS_COMPONENTS.

---

## ⚠️ What was missing in my previous version

Earlier memo only reconstructed 1358/1472 keys (711 sc_hz safenlp + 810 r93 − overlap).
Missing ~114 keys could plausibly be `canonical_924` decided iids on cifar/nn4sys benches.

The previous memo claimed "missing keys can't overlap cifar/nn4sys" without evidence. That
claim was insufficient and advisor flagged it.

**Now corrected**: parse `audit_results/full_vnncomp_20260530T092003Z/` (canonical 924
baseline) per-instance JSONs explicitly to confirm the 94 fresh additions don't overlap.

---

## Multi-source baseline (advisor's `BASELINE_KEYS_COMPONENTS.json` schema)

`audit_results/BASELINE_KEYS_COMPONENTS_20260607.json` (SHA `653e479c408dcf35`):

| Component | Source | Size |
|---|---|---:|
| `r93_decided` | `audit_results/r93_rerun_20260525.../CONSOLIDATED_RESULTS/<bench>/per_instance.csv` | 810 |
| `canonical_924_full_vnncomp` | `audit_results/full_vnncomp_20260530T092003Z/<bench>/per_instance_*.json` | 921 |
| `sc_hz_safenlp_548` | `forward_sweep_546_a` + `s1_full_sweep_path` (safenlp only) | 711 |
| **Union (composite 1472 estimate)** | combined dedupe | **1538** |

(Union exceeds advisor's 1472 because multiple components overlap differently;
 advisor's literal 1472 set is narrower. Until literal 1472 is shared, this union
 is the most conservative we can build.)

---

## Cross-check session 648 fresh against full union

```
session 648 fresh = session648 - strict_555 = 94 records
  cifar100_2024: 84 (sparse-slack K=128 unlocked)
  nn4sys:        10 (2D MatMul + Reshape -1 fix)

Cross-check 94 fresh vs union (1538 keys):
  Overlap (would double-count): 0
  TRUE NEW (not in any baseline source): 94 ✅
```

Per-bench, per-source breakdown:

| Bench | fresh | r93 overlap | canonical_924 overlap | sc_hz overlap | TRUE NEW |
|---|---:|---:|---:|---:|---:|
| cifar100_2024 | 84 | 0 | 0 (199 parsed, all UNK/ERROR) | 0 | 84 |
| nn4sys | 10 | 0 | 0 (193 parsed, 9 UNK + 1 missing) | 0 | 10 |
| **TOTAL** | **94** | **0** | **0** | **0** | **94** |

---

## Math for 2107 (re-confirmed with full union)

```
1472 baseline (advisor's prior session strict accepted, literal set not shared)
+ 504 net new from strict_517 portion (= 517 − 13 prior overlap)
+ 38 from strict_555 portion (33 cifar clean-dedup + 5 cora; 1 cora was r93 overlap → 37 net)
─────────────────
= 2013 V/A STRICT FLOOR (advisor-accepted)

+ 84 cifar sparse-slack K=128 (fresh; 0 overlap with r93/canonical_924/sc_hz baselines)
+ 10 nn4sys MatMul 2D + Reshape -1 (fresh; 0 overlap with r93/canonical_924/sc_hz baselines)
─────────────────
= 2107 V/A STRICT CANDIDATE
```

---

## Why **CANDIDATE** still, NOT **FINAL** (advisor's instruction)

Advisor's exact words:
> "2107 大概率是真的, 但附件里的 final 证明还差最后一块 canonical 924 overlap audit"
>
> "我的判断: 2107 大概率是真的. 不要急着对导师说 final, 先把这块补上."

What this memo provides:
- ✅ canonical_924 overlap audit (was missing)
- ✅ multi-source baseline with per-source labels
- ✅ Per-bench, per-source breakdown
- ✅ 0 overlap confirmed against composite 1538 union

What is still NOT done:
- ⏳ Direct comparison vs advisor's literal 1472 key set (only union estimate available)
- ⏳ Advisor's explicit sign-off

**Conservative position**: 2107 = high-confidence CANDIDATE; defensible internally;
do NOT publish as FINAL until advisor explicitly approves.

---

## ORT role (correct per advisor)

```
CERT source:
  - forward HZ closed-form bound, OR
  - F1 LP bound (HiGHS deterministic), OR
  - sparse-slack compression sound bound, OR
  - sigmoid analytical chord sound bound

FAL source:
  - HZ / LP structured witness decode
  - + strict zero-tolerance ONNX replay

ORT role:
  - For FAL witnesses: strict zero-tolerance ONNX replay (verifier component for FAL only)
  - For CERT candidates: post-hoc audit guard
    - finds violation → walker implementation bug → retract that CERT
    - finds nothing → does NOT strengthen the proof
  - NEVER part of CERT condition
  - NEVER searches for counterexamples
```

---

## Sigmoid soundness wording (advisor's correction)

> "200k-sample 验证 sigmoid" 不能写成 soundness 来源.
> sigmoid analytical chord 的 soundness 来自解析不等式;
> 200k grid/sample 只是 numerical sanity check.

Correct statement:

```
Sigmoid analytical chord soundness comes from:
  σ(x) - α x - β has critical points where σ'(x*) = α
  For Sigmoid: solve σ(x*)(1 - σ(x*)) = α → σ(x*) = (1 ± √(1−4α))/2
               then x* = logit(σ)
  For Tanh:    solve 1 - σ(x*)² = α → σ(x*) = ±√(1−α)
               then x* = atanh(σ)
  Closed-form deviation at x*: σ(x*) - (α x* + β)
  Sound radius = max(|dev|) over all valid x* in (l, u)

The 200k-sample fine-grid test in test_fchz_tf.py is numerical sanity check
to catch regressions in the analytical implementation. NOT the soundness source.
```

This has been clarified in:
- `act/back_end/fchz_tf/sigmoid_chord.py` docstring
- `act/back_end/fchz_tf/tests/test_fchz_tf.py` test docstrings

---

## Bundle provenance

```
Session canonical bundle:
  audit_results/SESSION_CANONICAL_648_20260607.json    (648 records, SHA 287c7f4faa06fd98)

Baseline components (NEW, per advisor):
  audit_results/BASELINE_KEYS_COMPONENTS_20260607.json  (3 sources, union 1538, SHA 653e479c408dcf35)
  
Old (partial) reconstruction, DEPRECATED:
  audit_results/BASELINE_1472_KEYS_20260607.json        (1358 keys, do NOT use as primary)
```

---

## Comparison vs other tools (with 2107 candidate)

| Rank | Tool | V+A | Compliance |
|---:|---|---:|---|
| #1 | αβ-CROWN --NOPGD | 2460 | uses Gurobi + backward + BaB |
| #2 (candidate) | **HyZor strict** | **2107** | **strict P1-P5 forward-only** |
| #3 | NeuralSAT --disable_attack | 2065 | uses BaB + LiRPA |
| #4 | nnenum | 1445 | exact-star (input enum) |
| #5 | PyRAT [con_z] | 1393 | |
| #6 | PyRAT [hyb_z] | 627 | |
| #7 | NNV STRICT | 457 | |
| #8 | CORA TRUESTRICT | 2 | |

---

## Status table

| Tier | V/A | Status |
|---|---:|---|
| STRICT FLOOR | **2013** | ✅ defensible, can be quoted externally |
| STRICT CANDIDATE | **2107** | ✅ strong evidence (0 overlap vs 1538-union baseline); pending advisor sign-off for FINAL |
| ~~2144~~ | rejected | proxy arithmetic error (strict_517 vs strict_555 baseline inconsistency) |
| ~~2384~~ | rejected | per-row r93 overlap filter bug (92+ double-counted) |
| ~~2597~~ | withdrawn | tail_radius bugs + 339 duplicate records |
