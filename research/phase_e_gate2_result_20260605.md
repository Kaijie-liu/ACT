# Phase E Gate 2 — Streaming-Prune Memory Gate RESULT

**Date**: 2026-06-05 evening
**Status**: **PASSED 40/40 OK, 0 OOM** (vs Day-of pilot 18 OOMs)
**Headline impact**: NONE yet (memory only); Gate 3 next

---

## 1. Gate 2 acceptance — all criteria met

| Criterion | Required | Measured |
|---|---|---|
| peak RAM per iid | < 80 GB | 62.8 GB max (cifar 130) |
| OOM count | 0 | **0** |
| TIMEOUT | 0 | 0 |
| RC_KILL | 0 | 0 |
| Coverage | 40/40 iids | **40/40 OK** |
| `act/` modification | none | clean |
| G10 pre-flight | available ≥ 90 GB | 95 GB at launch |
| Unit tests | 47/47 pass | 47/47 pass |
| V/A scoring | NOT a gate criterion | not measured |

`audit_results/sc_hz_gate2_memory_pilot_K5000_20260605T051640Z/`

## 2. Detailed numbers

### cifar100_2024 (20 iids)

| metric | value |
|---|---|
| OK | 20/20 |
| OOM | 0 (was 11+ in Day-of pilot at K=∞) |
| max peak RSS | **62.8 GB** (iid 190 — deep variant) |
| median wall | 105 s |
| median max_excess | +1.33 |
| 8 tightest excesses | 29:+0.32, 24:+0.46, 72:+0.51, 86:+0.52, 180:+0.60, 57:+0.73, 145:+0.92, 113:+0.98 |

### tinyimagenet_2024 (20 iids)

| metric | value |
|---|---|
| OK | 20/20 (was OOM-blocked entirely in Day-of pilot) |
| OOM | 0 |
| max peak RSS | **37.3 GB** (very stable across all 20) |
| median wall | 52 s |
| median max_excess | +298 (very loose due to tail compression at K=5000) |

## 3. What gate 2 unlocked

1. **Memory ceiling broken**: streaming-prune + RLIMIT_AS 80 GB keeps every
    iid under budget. The Day-of pilot had 11+ OOMs across the same 40 sentinel
    set; we now have 0.
2. **Tinyimagenet measurable**: previously OOM-blocked entirely at K=∞;
    now 20/20 OK at K=5000.
3. **Cifar deep variants (iids 113, 118, 130, 145, 156, 168, 180, 185, 190, 195, 199)** — these were the 11 deepest variants that Day-of pilot OOM-killed; now all 20/20 OK at 59-63 GB.
4. **Per-iid subprocess isolation**: OOM-kill on one iid no longer crashes the parent. The pilot completed without intervention.

## 4. Trade-off: precision lost to tail compression

K=5000 streaming-prune is the memory knob. It folds dropped generator columns
into per-row tail; this is sound but loose.

Effect on cifar PHANTOMs vs Day-of K=∞ baseline:

| iid | Day-of K=∞ max_excess | K=5000 streaming max_excess | delta |
|---|---:|---:|---:|
| 29 | +0.315 | +0.317 | +0.002 (essentially same — small variant) |
| 24 | +0.464 | +0.464 | +0.000 |
| 72 | +0.368 | +0.510 | +0.142 |
| 86 | +0.519 | +0.524 | +0.005 |
| 57 | +0.726 | +0.727 | +0.001 |
| 113 | n/a (OOM in Day-of) | +0.977 | newly measurable |

Cifar SMALL variants barely change (their natural ng was already ≤ 5000).
Cifar DEEP variants are now measurable for the first time.

Tinyimagenet excess is hugely loose (+93 to +383). This is the price of
K=5000 on a 56×56 ResNet — the tail picks up the very-many ReLU slacks at
each layer. To regain precision on tiny, Gate 3 must add tighter relaxation.

## 5. Gate 3 plan — final-tail k-piece ReLU

Target metric: median LP UB excess on cifar's 8 tightest PHANTOMs:
```
Current K=5000 streaming:  +0.605 (median of 8)
Advisor target ≥30% drop:  ≤ +0.42
Stretch goal (any → CERT): one iid's max_excess < 0
```

Mechanism (per advisor Phase E spec):
- Apply k-piece relaxation to LAST 1-2 ReLU layers ONLY
- k=2 → 1 extra column per coord; k=3 → 2 extra columns
- Soundness invariants (G9 binding):
   - LP UB after cut ≤ LP UB before cut (triangle baseline)
   - Box range no widen
   - nb (binary count) no increase (we are continuous-only anyway)
   - FAL via strict ORT replay (G4)
   - CERT via independent LP audit (G2)

If any of the 8 PHANTOMs drops to max_excess < 0 → NEW V (production-UNK at
canonical r93, so likely NEW vs production).

## 6. Files

| File | Status |
|---|---|
| `research/sc_hz/onnx_walker_resnet.py` | streaming_K_target param added to forward_resnet |
| `research/sc_hz/conv_streaming_prune.py` | live in walker Conv branch |
| `research/sc_hz/gate2_memory_pilot.py` | Gate 2 driver with subprocess isolation |
| `audit_results/sc_hz_gate2_memory_pilot_K5000_*/` | 40 per-iid receipts + summary.json |
| Unit test suite | 47/47 pass |
| 1472 freeze | unchanged (memory-only gate doesn't touch headline) |
| `act/` | clean |

## 7. Cumulative S2 path so far

- ✓ Gate 0: full 558 strict audit → 1472 holds
- ✓ Gate 1: streaming-prune infra + 7 soundness tests
- ✓ **Gate 2: 40/40 memory pass, no OOM, 62.8 GB max**
- Next: Gate 3 final-tail k-piece ReLU
- 2-week kill switch still binding
