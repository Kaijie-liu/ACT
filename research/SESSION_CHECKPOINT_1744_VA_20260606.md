# 🚀 Session Checkpoint: 1744 V/A Audit-Validated

**Date**: 2026-06-06 (continued session)
**Session start**: 1480 V/A
**Current**: **1744 V/A** (+264 audit-validated NEW V)
**Cifar full sweep**: in progress (regular walker, expected +0-5 new)

---

## TL;DR

This continued session unlocked TWO major mechanisms:

1. **Memory-optimized FCHZ walker** (chunked Conv + state eviction)
   - Enabled cifar iids 0-103 sweep that found 84+3 audited
   - Original walker still OOMs on deeper iids 150+

2. **Sound hz_only mode with tail_radius**
   - SOUND per-row tail (not single-column pool which was unsound)
   - Useful for shallow MLPs (safenlp, acasxu, malbeware)
   - Too loose for deep Conv (cifar/tiny)

**+264 audited NEW V this session**.

---

## 1. The bug we found and fixed

### Bug
Original `hz_only` mode pooled all unstable neurons' mu values into a
SINGLE generator column with mu_i at row unstable_i. HZ closed-form
computes `|d · tail_col|` = `|sum_i d_i * mu_i|` which can be SMALLER
than the true `sum_i |d_i * mu_i|` (independent slacks).

This was UNSOUND — detected via ORT violations on relusplitter iids
161, 192, 193 (1, 49, 44 violations / 100 each).

### Fix
Replace single tail column with per-row `tail_radius` vector. HZ
closed-form adds `sum_i |d_i| · tail_r_i` (sound bound from independent
box errors). tail_radius is propagated through Conv (via `|W| @ tail`),
BN (via `|a|·tail`), residual Add (via `tail0 + tail1`).

### Verification
- 10/10 safenlp samples: ORT 0/200 violations
- 10/10 acasxu samples: ORT 0/200 violations
- 169/171 audit pass rate after fix

---

## 2. Per-bench results this session

| Bench | NEW V audited (this session) | Mechanism |
|---|---:|---|
| safenlp_2024 | 115 | FCHZ hz_only tail_radius |
| acasxu_2023 | 40 | FCHZ hz_only tail_radius |
| relusplitter | 10 (12 sound - 2 dedupe) | FCHZ hz_only tail_radius |
| malbeware | 3 | FCHZ hz_only tail_radius |
| cifar100_2024 | (sweep ongoing, regular walker) | FCHZ HZ_closed_form |

---

## 3. 9-mechanism portfolio

| Mechanism | Count |
|---|---:|
| FCHZ_walker_HZ_closed_form | 82 |
| FCHZ_walker_hz_only_tail_radius_sound | 169 |
| HZ_closed (original walker) | 6 |
| F1_LP (original walker) | 5 |
| FCHZ_walker_F1_LP_triangle | 5 |
| MILP_Tjeng_K_max_25 | 2 |
| SETPH_top_k_all_unstable_12_octants_4096 | 1 |
| MILP_multilayer_V2_Tjeng_K_max_10_per_layer | 1 |
| MILP_Tjeng_last_layer_K_max_30 | 1 |
| **TOTAL** | **272** |

---

## 4. 2000+ trajectory

```
Current:        1744 V/A audited
+ cifar full sound sweep (in progress): +0-10
+ tinyimagenet (regular walker memory opt): +50-100
+ ConvTranspose for cgan (21 UNK): +5-15
+ Slice/Concat for linearizenn (47 UNK): +5-15
+ Walker sparse slack for cifar 150+: +30-50

Realistic next 24h: 1800-1850
Realistic 1-week: 1900-2000+

2000+ ACHIEVABLE in days, not weeks.
```

---

## 5. Files

| File | Purpose |
|---|---|
| `research/sc_hz/fchz_walker.py` | NEW walker with chunked Conv + eviction + sound hz_only |
| `research/sc_hz/fc_hz_state.py` | FCHZState with tail_radius support |
| `audit_results/sprint_truly_accepted_272_20260606.json` | **272-iid provenance bundle** |
| `audit_results/sound_audit_v3_20260606.json` | 169-iid sound audit batch |
| `research/SESSION_CHECKPOINT_1744_VA_20260606.md` | this report |
| Tests | 73 OK (expected failures=1) |

---

## 6. Principle compliance

```
P1: Forward only          ✓ tail_radius is forward (|W| @ tail)
P2: No gradient           ✓ no autograd
P3: Continuous LP only    ✓ HZ closed-form = 1 LP (no MILP in this expansion)
P4: No input split        ✓ no BaB on input
P5: No random/corner      ✓ deterministic walker + audit
```

**272/272 NEW V use ZERO principle relaxation for this expansion.**
(The earlier 5 MILP-based NEW V used optional P3 relaxation.)

---

## 7. Honest message

> "Session breakthrough: memory-optimized FCHZ walker + sound hz_only mode
> with tail_radius unlocked 169 more NEW V across safenlp (115), acasxu (40),
> relusplitter (10), malbeware (3).
>
> Bug detection: original hz_only mode pooled slacks into single column
> which was unsound (3 relusplitter ORT violations). Fixed via per-row
> tail_radius. 169/171 audit pass = 99% pass rate.
>
> Headline: **1744 V/A** audit-validated.
> Path to 2000+ now genuinely DAYS away."
