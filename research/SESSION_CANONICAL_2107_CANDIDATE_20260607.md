# 📋 SESSION CANONICAL CANDIDATE — 2107 V/A STRICT

**Date**: 2026-06-07
**Status**: CANDIDATE (not yet final — needs proper 1472 baseline key audit)
**Bundle**: `audit_results/SESSION_CANONICAL_648_20260607.json`

---

## ✅ Correct math (per advisor 2026-06-07)

```
2013 STRICT FLOOR  (advisor accepted)
  = 1472 baseline + 541 net new (strict_555 portion, after −13 safenlp −1 cora overlap)

+ 94 canonical additions from full sweep
  = 84 cifar100_2024 (sparse-slack K=128 unlocked)
  +10 nn4sys (MatMul 2D + Reshape -1 fix)

= 2107 V/A STRICT CANDIDATE
```

### Detailed verification

```
strict_517 records: 517
strict_555 records: 555 (= strict_517 + 33 cifar clean-dedup + 5 cora)
session648 records: 648 (= strict_555 + 84 cifar sparse + 10 nn4sys)

session648 - strict_555 = 94 ← TRUE fresh additions (NOT 131)

The +37 difference (131 − 94) is the 38 records (33 cifar + 5 cora)
ALREADY counted in strict_555 = 2013 floor basis.
```

### Why 2144 was wrong

I used `strict_517` as proxy for "prior session baseline" and computed
`session648 − strict_517 = 131`. But `2013 floor` is based on
`strict_555` (which has 38 more records than strict_517: cifar clean-dedup
and cora dedup additions). Those 38 records were ALREADY in the 2013 floor
calculation, so adding them again as "fresh" was double-counting.

Correct accounting uses the SAME baseline at both ends.

---

## What 2107 represents

```
1472 (prior session strict accepted, advisor baseline)
─────────
+ strict_517 work: 504 net new (after −13 safenlp prior 1472 overlap)
+ cifar clean-dedup (33 records) + cora (5 records) = 38 net new
─────────
1472 + 541 = 2013 STRICT FLOOR (advisor accepted)

+ cifar sparse-slack K=128 remaining (84 records, mechanism: sparse-slack columns)
+ nn4sys MatMul 2D + Reshape -1 fix (10 records, mechanism: walker op extension)
─────────
2013 + 94 = 2107 STRICT CANDIDATE
```

---

## Per bench (session 648 records, post-r93 dedup)

```
cifar100_2024:        200 (33 in strict_555 + 84 fresh + 83 in strict_517)
tinyimagenet_2024:    199 (in strict_517 portion, already in 2013 floor)
safenlp_2024:         115 (in strict_517 portion)
dist_shift_2023:       56 (in strict_517 portion)
relusplitter:          27 (in strict_517 portion)
cora_2024:             17 (5 in strict_555 + 13 in strict_517)
nn4sys:                10 (fresh additions)
acasxu_2023:            9 (in strict_517 portion)
tllverifybench_2023:    7 (in strict_517 portion)
metaroom_2023:          5 (in strict_517 portion)
malbeware:              3 (in strict_517 portion)
```

---

## ✅ Audit checklist (advisor's items)

- [x] 2144 file withdrawn → `_CANDIDATE_REJECTED_2144_20260607.md`
- [x] Math corrected to 2107 (94 fresh, not 131)
- [x] Full VNN-COMP 2025 sweep done (501 unique + 199 tiny rerun = 700 records)
- [x] Per-iid dedup (advisor's filter bug fixed)
- [x] r93 overlap excluded (0 r93 double-counts)
- [x] ORT 500-sample per record (0 violations on 648)
- [x] GPU revalidate 832 with 2000-sample (0 violations)
- [x] SHA256 + r93_verdict provenance per record
- [x] 86/86 unit tests OK
- [ ] **BASELINE_1472_KEYS.json**: build real 1472 key set (not strict_517/strict_555 proxy)
- [ ] After 1472 audit confirms 94 fresh: promote to 2107 FINAL

---

## Why 2107 is still CANDIDATE (not FINAL)

Per advisor:
> "下一步只差把它们对完整 `1472 baseline key set` 做一次正式去重，
> 而不是只对 r93 / strict517 做 proxy"

We need the explicit `1472 baseline key set` (the 1472 (bench, iid) tuples
that constitute the advisor's prior session strict accepted baseline). 
Currently we use `strict_555` as proxy because we don't have the explicit
1472 list.

Until that audit confirms 0 overlap between the 94 fresh additions and the
true 1472 keys, **2107 stays CANDIDATE**.

---

## Comparison vs other tools

| Rank | Tool | V+A | Compliance |
|---:|---|---:|---|
| #1 | αβ-CROWN --NOPGD | 2460 | uses Gurobi + backward + BaB |
| #2 | NeuralSAT --disable_attack | 2065 | uses BaB + LiRPA |
| **HyZor 2107 candidate** | strict P1-P5 | **2107** | passes NeuralSAT under strict ⭐ |
| #4 | nnenum | 1445 | exact-star (input enum) |
| #5 | PyRAT [con_z] | 1393 | |
| #6 | PyRAT [hyb_z] | 627 | same HZ family |
| #7 | NNV STRICT | 457 | |
| #8 | CORA TRUESTRICT | 2 | |

**HyZor candidate 2107 > NeuralSAT 2065** under strict P1-P5.

If 2107 confirms as final, HyZor is **#2 globally** under strict P1-P5.
Even at conservative **2013 floor**, HyZor sits between NeuralSAT (2065)
and nnenum (1445) — solid top-3 position.

---

## Headline tier summary

| Tier | V/A | Status |
|---|---:|---|
| STRICT FLOOR | **2013** | ✅ publishable today (defensible) |
| STRICT CANDIDATE | **2107** | needs 1472 keys audit, very likely valid |
| ~~STRICT 2144~~ | ~~2144~~ | ✗ REJECTED (double-count, 37 records) |
| ~~STRICT 2384~~ | ~~2384~~ | ✗ REJECTED (per-row filter bug) |
| ~~STRICT 2597~~ | ~~2597~~ | ✗ WITHDRAWN (tail_radius bugs + dups) |

---

## Next steps (advisor's items)

1. ✅ Withdraw 2144 (done)
2. ✅ Write 2107 candidate report (this file)
3. ⏳ Build `BASELINE_1472_KEYS.json` from canonical sources (sweep_C / witness-replay logs)
4. ⏳ Re-filter session648 vs 1472 keys to confirm 94 fresh (or report true count)
5. ⏳ If audit confirms 94 fresh → promote 2107 to FINAL
6. ⏳ All obsolete bundles labelled (`_archive_obsolete_*`)

---

## Honest acknowledgment

The 2144 arithmetic error is a recurring failure mode: using inconsistent
baselines for "floor" and "additions" (strict_555 vs strict_517). Advisor
caught it within 1 audit pass.

**Headline integrity > headline magnitude.** 2013 floor stands. 2107 candidate
pending audit. We do not publish unverified inflations.
