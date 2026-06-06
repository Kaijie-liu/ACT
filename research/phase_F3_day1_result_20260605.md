# Phase F3 Day 1 Result — F1 LP Insufficient for Small-Dense CERT

**Date**: 2026-06-05 night
**Status**: F3 Day 1 complete. Evidence shows F1-CERT path insufficient.
**Headline impact**: 0 NEW V/A. 1472 holds.

---

## 1. Day 1 plan execution

### Walker integration extension
Extended `forward_resnet_capture` for small-dense ONNX:
- ADDED: `Sub` (state - constant)
- ADDED: `MatMul` (state @ W; no bias)
- ADDED: `Add` const variant (state + constant)
- FIXED: data input auto-detection (skip initializer-named inputs)

Result: acasxu (186) + tllverifybench (32) walker = 218/218 OK.

### F1 LP tested on small-dense
- acasxu_2023: 186 iids walker OK; 65 near-CERT (HZ < 5.0); **0 NEW V**
- tllverifybench_2023: 32 iids walker OK; **0 NEW V** (all HZ excess too large)
- linearizenn_2024: **0 testable** — all 60 blocked by Slice parser

## 2. Critical structural evidence

```
acasxu iid 107: HZ excess = +0.001 → F1 LP = +0.000
acasxu iid 102: HZ excess = +0.084 → F1 LP = +0.008 (90% drop, PHANTOM)
acasxu iid 140: HZ excess = +0.133 → F1 LP = +0.097 (27%)
acasxu iid 134: HZ excess = +0.268 → F1 LP = +0.051 (81%)
acasxu iid 146: HZ excess = +0.332 → F1 LP = +0.136 (59%)
```

Even 81-90% drop NOT ENOUGH to flip these PHANTOMs to CERT under G4 strict
inequality. These are the iids canonical's spec-aware refinement
(`project_specaware_refinement_20260516`) covers via a DIFFERENT mechanism
(+13 NEW V on acasxu), not F1-style sidecar LP.

## 3. Why F1 fails on small-dense

| Bench | Network | F1 mechanism applies? |
|---|---|---|
| cifar/tiny | deep conv, last layer pre-classifier | YES (17%) — proven |
| acasxu | 6-layer dense, all-unstable | NO — limit hit at boundary |
| tllverifybench | similar dense | NO — HZ excess too large |
| linearizenn | dense + Slice + concat | UNTESTED (parser block) |

The cifar success of F1 came from large `n_pre` with many stable+inactive
neurons damping slack. Small-dense has all unstable and tiny `n_pre` (~50)
— LP optimum saturates per-neuron triangles immediately.

## 4. F3 day-1 prognosis vs ≥30 NEW V/A gate

| Bench | UNK | F1 evidence | Plausible NEW V/A |
|---|---:|---|---:|
| acasxu | 98 | 0/65 near-CERT flip | 0 |
| tllverifybench | 29 | 0/32 close to CERT | 0 |
| linearizenn | 47 | Slice parser block | 0-3 (untested) |
| ml4acopf | 63 | parser block | 0-5 (uncertain) |
| nn4sys | 190 | parser block | 0-10 (uncertain) |
| metaroom | 11 | tiny UNK pool | 0-2 |
| **Total** | **438** | | **0-20** |

**FORECAST: F3 day-5 yield < 30 NEW V/A → ADVISOR GATE FAIL.**

Per advisor's binding plan: "F3 没信号，立即转 Phase G, 不再在 parser/sidecar 上消耗."

## 5. Two options for day 2

### Option A: continue F3 days 2-5 (parser fixes)
- Add Slice/Reshape/Gather parser support
- Re-test linearizenn (47 UNK) + nn4sys (190 UNK)
- ml4acopf + metaroom parser scout
- Expected yield: 0-20 NEW V/A

### Option B: pivot now to Phase G FC-HZ
- Day 1 evidence already suggests F3 path is structurally weak for small-dense
- Spec-aware refinement is what helps acasxu (already in 88 V baseline)
- F1 LP doesn't compose with spec-aware in obvious way
- Implement FC-HZ ops + cifar 113 gate test
- 3-4 days to first FC-HZ gate; if FC-HZ also fails, paper at 1472

## 6. Recommendation

**Option B**: F3 day 1 evidence is strong enough to predict failure of
F3 days 2-5 against ≥30 NEW V/A gate. Pivot to Phase G now saves 4 days.

If advisor disagrees, Option A is still possible: parser fixes are 1-2 day
engineering each, and might surface +5-10 from unexpected benches.

## 7. Files

| File | Status |
|---|---|
| `research/sc_hz/constrained_lp_integration.py` | extended: Sub/MatMul/const-Add |
| `research/phase_F3_small_control_plan_20260605.md` | F3 plan |
| `research/phase_G_forward_constrained_hz_design.md` | Phase G design |
| `research/phase_F3_day1_result_20260605.md` | this memo |
| 1472 freeze | unchanged |
| `act/` | clean |
| Unit tests | 67/67 PASS |

## 8. Status

```
Headline:                1472 V/A
F3 day 1 yield:         0 NEW V/A
acasxu walker:          218 instances OK, 0 NEW from F1
linearizenn walker:     blocked by Slice parser (parser fix needed)
2-week kill switch:     5 days remaining (acted on advisor's plan)
```
