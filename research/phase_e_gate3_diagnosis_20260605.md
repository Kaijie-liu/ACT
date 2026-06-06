# Phase E Gate 3 — DeepZ Triangle Floor Diagnosed

**Date**: 2026-06-05 night
**Status**: Gate 3 FAILS at single-neuron mechanism; precision floor hit
**Headline impact**: NONE (1472 holds; Gate 3 produced 0 NEW V/A)

---

## 1. Gate 3 result summary

| Target | Value | Status |
|---|---|---|
| median LP excess drop ≥ 30% | 1/8 PHANTOM hit (-28%) | NEAR-MISS but FAILS |
| ≥ 1 NEW V (any cifar iid max_excess < 0) | 0 | FAIL |
| Mechanism: tighter ReLU | tried K=10K/20K/50K — no further drop | floor reached |
| Monotonicity gate | n/a — no relaxation change applied | n/a |

Detailed K-sweep result on 8 lowest-excess cifar PHANTOMs (the only "near-CERT" cohort across 40 sentinels):

| iid | prior K=5000 | K=10K | K=20K | K=50K | drop |
|---|---:|---:|---:|---:|---:|
| 29 | +0.317 | +0.315 | +0.315 | +0.315 | -0.6% |
| 24 | +0.464 | +0.464 | +0.464 | +0.464 | 0.0% |
| 72 | +0.510 | +0.368 | +0.368 | +0.368 | **-28%** |
| 86 | +0.524 | +0.519 | +0.519 | +0.519 | -1.0% |
| 180 | +0.599 | n/a | n/a | n/a | 0.0% |
| 57 | +0.727 | +0.726 | n/a | n/a | -0.1% |
| 145 | +0.921 | n/a | n/a | n/a | 0.0% |
| 113 | +0.977 | OOM | OOM | OOM | n/a |
| **median (7 measured)** | **+0.524** | **+0.519** | **+0.518** | **+0.518** | **-1.1%** |

(iid 113 only measurable at K=5000 due to deep variant OOM at higher K.)

Per advisor target: 30% median drop. Measured: 1.1%. **Gate 3 FAILS by 28× margin.**

## 2. Why higher K doesn't help (the actual finding)

`diagnose_relu_tightening_potential` on the 4 highest-excess PHANTOMs:

| iid | tail_dominance_max | n_coords with dom > 0.5 | n_coords with dom > 0.9 |
|---|---:|---:|---:|
| 180 | 0.141 | 0 | 0 |
| 57 | 0.000 | 0 | 0 |
| 145 | 0.218 | 0 | 0 |
| 113 | 0.251 | 0 | 0 |

`tail_dominance[i] = tail_radius[i] / (|G_kept[i,:]|.sum() + tail_radius[i])`

ALL coordinates have tail dominance < 0.3. **The LP excess is dominated by explicit generators, not by tail compression**. Increasing K_target doesn't help because the bottleneck is NOT tail-folded columns; it's the natural DeepZ-triangle relaxation looseness on the explicit generators.

This is the **DeepZ-triangle natural floor**: single-neuron triangle is already the tightest CONVEX upper envelope of ReLU on `[l, u]`. No continuous-LP mechanism can tighten it further per-neuron.

## 3. The precision levers that COULD break this floor

Each requires multi-day implementation work:

### (a) Multi-neuron joint hull (Anderson 2020 forward facets)
- For unstable neuron pairs (i, j), the EXACT convex hull of `(z_i, z_j, relu(z_i), relu(z_j))` over `[l_i, u_i] × [l_j, u_j]` has facets that are NOT implied by per-neuron triangles
- These facets are linear constraints on `(z_i, z_j, y_i, y_j)`
- Requires extending `PrunedState` with constraint matrix `Ac, b` (currently we have only box-domain ξ_c ∈ [-1,1]^K)
- Estimate: 3-5 days implementation + soundness tests + monotonicity G9 enforcement

### (b) Spec-aware pre-activation refinement
- For each unsafe rival `d_out`, restrict the set with `d_out · M(rest_of_pipeline)(y) >= threshold`
- Re-derive (l_i, u_i) under spec constraint → tighter triangle parameters
- Requires per-rival LP solving + careful integration with forward walker
- Used in safenlp +13 ACASXu lift earlier; works in continuous LP
- Estimate: 2-3 days implementation + per-iid wall scaling (99 rivals × LP solve)

### (c) Beyond HZ + triangle: new abstraction (Phase F)
- Drop the HZ + DeepZ-triangle framework entirely
- Forward-only group ReLU hull, spec-conditioned template HZ, block-level aggregate constraints, continuous Anderson facets
- Multi-week research investment

## 4. What Gate 3 actually did demonstrate

This is the honest read:

1. **Memory ceiling truly broken** (the Gate 0/1/2 result). cifar deep variants + ALL tinyimagenet now measurable at K=5000 within 80 GB G10 budget. Day-of pilot's 11+ OOMs → 0.
2. **DeepZ-triangle floor reached** at single-neuron continuous LP. For the 8 "near-CERT" cifar iids, all max_excess are ≥ +0.32 (median +0.52). This is the SC-HZ + DeepZ-triangle natural limit on cifar at the current architecture.
3. **iid 72 single significant drop** (-28%): some iids DO have headroom (their natural ng exceeds the streaming-prune cap and benefits from higher K), but this was 1/8 not the median. Not enough to flip ANY iid to CERT.
4. **0 NEW V/A** produced by Gate 3. The 1472 headline is unaffected.

## 5. 2-week kill switch — where we are

Per advisor 2026-06-05 plan:
> "如果 S2 两周内仍然 0 NEW + LP excess 降不动, 就关闭 SC-HZ 提升阶段, 进入新 abstraction"

Day 1 of S2 → Day 14 deadline.
Day 1 result:
- ✓ Memory ceiling broken (real progress)
- ✗ LP excess NOT dropping at current mechanism
- 0 NEW V/A

13 days remaining. To use them well requires implementing precision lever (a) or (b) — both are 2-5 day investments. If those fail too, the kill switch fires Day 14 and we redirect to Phase F new abstraction.

## 6. Honest framing for next steps

We have two reasonable paths going forward:

### Path I: Accept 1472 as SC-HZ + DeepZ-triangle ceiling, start Phase F now
- Save 13 days of incremental work
- Begin designing the new abstraction immediately
- The 1472 headline is published as the SC-HZ-final number
- Realistic if multi-day Anderson Ac extension is unlikely to break 1472 → 1600

### Path II: Implement Anderson forward facets (multi-neuron) in 3-5 days
- Extend PrunedState with Ac, b matrices
- Add Anderson 2020 forward-facet generation on the final 1-2 ReLU layers
- Re-run Gate 3 on 8 cifar PHANTOMs
- IF median excess drops ≥ 30% AND any iid flips to CERT → continue
- IF still flat → close Phase E, move to Phase F

Tonight's data argues for Path I unless we believe Anderson facets have a large headroom on cifar (uncertain). But the project investment so far (1472 audit-validated, principle-clean, 47/47 tests, 4 audited contributions) is a publishable result regardless of which path we take.

## 7. Tonight's stop point

```
Headline:                1472 V/A (audit-validated, 1460→1472 since S1)
Gate 0 audit:            558/558 STRICT-PASS
Gate 1 streaming-prune:  47/47 tests pass
Gate 2 memory pilot:     40/40 OK, 0 OOM
Gate 3 LP excess:        0/8 near-CERT iid flipped (DeepZ floor reached)
0 NEW V/A from tonight's work
act/ status:             clean (no production changes)
GPU/RAM:                 returning to baseline (no active SC-HZ procs)
```

The decision on Path I vs Path II is the advisor's call. The data and mechanisms are all documented for either choice. We hit the DeepZ floor, the path forward is structural, not incremental.
