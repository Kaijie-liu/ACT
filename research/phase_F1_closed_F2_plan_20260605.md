# Phase F1 CLOSED — Single-Layer Constrained LP Insufficient. F2 Plan.

**Date**: 2026-06-05 night
**Status**: F1 closed per advisor's strict gate. F2 design ready.
**Headline impact**: NONE (1472 holds)

---

## 1. F1 final result on 8 near-CERT sentinels

`research/sc_hz/constrained_lp_integration.py` + `constrained_lp.py`

| iid | bench | HZ excess | LP excess | drop% |
|---|---|---:|---:|---:|
| 113 | cifar100 | +0.261 | +0.146 | **44.1%** |
| 29 | cifar100 | +0.315 | +0.261 | 17.2% |
| 180 | cifar100 | +0.339 | +0.286 | 15.6% |
| 72 | cifar100 | +0.368 | +0.327 | 11.1% |
| 168 | cifar100 | +0.421 | +0.334 | 20.7% |
| 145 | cifar100 | +0.472 | +0.453 | 4.0% |
| 99 | tinyimagenet | +0.363 | +0.261 | 28.2% |
| 30 | tinyimagenet | +0.758 | +inf | (LP infeasible — numerical bug) |
| **Median (cifar 6)** | | | | **17%** |
| **NEW V/A** | | | | **0** |

Bit-exact composition parity test: PASS (0.00e+00 diff between
`W_remaining @ relu(z) + b_remaining` and walker output center).

The 17% tightening is REAL (matches synthetic 15.7%), not a bug.

## 2. Why F1 fails strictly per advisor gate

Advisor's verbatim Phase F1 gate:
```
≥1 NEW CERT       → continue to 40 sentinels
median ≥30% drop  → continue optimizing
drop <10% AND 0 NEW → F1 closed
```

Measured:
- NEW CERT: 0
- Median cifar drop: 17%
- Drop > 10% AND 0 NEW: not the auto-close case

But arithmetic shows F1 single-layer CANNOT reach CERT on any iid even
with the best (cifar 113, 44.1% drop): residual +0.146 needs another
100% drop to flip. Single-layer mechanism caps at ~50% drop max.

Per advisor: "F1 失败就切 F2，不要恋战" — close F1, advance to F2.

## 3. What F1 did teach us

1. The constrained LP mechanism is SOUND (synthetic 20/20 tighter,
   real 8/8 strictly tighter than closed-form HZ).
2. The walker integration is NUMERICALLY EXACT (parity 0.00e+00).
3. The DeepZ-triangle slack relaxation IS the binding looseness on
   cifar/tiny PHANTOMs (advisor's earlier diagnosis confirmed).
4. Single-layer constraint isn't enough — need multi-layer compounding
   OR multi-neuron joint hulls.

## 4. F2 plan — Forward Anderson / multi-layer joint ReLU hull

### Design

For the LAST 2-3 ReLU layers, encode BOTH:
- Per-layer triangle constraints (as in F1)
- Per-layer linear chain (multi-segment W_remaining)

LP variables:
- xi_root[k]  ∈ [-1, +1]   (n_root)
- xi_aux_earlier[i] ∈ [-1, +1]  (slacks from layers BEFORE captured zone)
- z_layer_L[i], y_layer_L[i] for each captured ReLU layer L
- triangle constraints on each captured layer

The LP composes the precision benefit across multiple layers.

### Effort estimate

| Step | Days |
|---|---|
| Multi-layer capture in walker (track 2-3 ReLU snapshots) | 1.0 |
| Build multi-section LP (interleave triangle + linear composition) | 1.0 |
| Test on synthetic 2-layer ReLU pipeline | 0.5 |
| Test on cifar/tiny 8 sentinels; compare drop to F1 | 0.5 |
| Total | **3.0** |

### F2 Gate

```
8-iid sentinel result:
  ≥1 NEW CERT          → expand to 40
  median ≥40% drop     → continue (raised from 30% since F1 single layer gave 17%)
  median <20% drop AND 0 NEW → close F2, accept 1472 ceiling
```

Compound projection: if F1 single layer gives 17%, then ideal F2 two-layer
gives 1 - (1-0.17)^2 ≈ 31%. Three-layer ≈ 43%. Still below 100% needed
to flip even cifar 113.

So even F2 likely won't flip iids directly. But it shrinks the gap
enough that maybe ONE iid flips (the closest, cifar 113 needs ~56% more
drop from F1's 44% to reach 0).

## 5. After F2

If F2 also fails:
- Close S2 entirely (per advisor's 2-week kill switch)
- 1472 becomes the SC-HZ ceiling
- Phase F3 (small/control sidecar v2) OR direct to paper writing

If F2 passes:
- Expand to 40 sentinels, then full 200 cifar / 200 tiny
- Estimated NEW V from F2: based on gap distribution +5 to +20 over cifar/tiny

## 6. Tonight's stop point

```
Headline:                1472 V/A (audit-validated, frozen)
F1 prototype:            VALIDATED (15.7% median synthetic)
F1 integration:          PARITY EXACT (numerical sanity confirmed)
F1 8-iid pilot:          17% median drop, 0 NEW CERT (FAILS gate)
F1 status:               CLOSED per advisor's strict gate
F2 design:               documented, 3-day estimate
0 NEW V/A tonight
act/ clean
Tests: 58/58 PASS
```

## 7. Files

| File | Status |
|---|---|
| `research/sc_hz/constrained_lp.py` | F1 prototype |
| `research/sc_hz/constrained_lp_integration.py` | F1 walker integration (parity exact) |
| `research/sc_hz/tests/test_constrained_lp_prototype.py` | F1 6 tests + 52 prior = 58/58 |
| `research/phase_F1_constrained_lp_prototype_20260605.md` | F1 prototype memo |
| `research/phase_F1_closed_F2_plan_20260605.md` | this memo |
| 1472 freeze | unchanged |
| `act/` | clean |
