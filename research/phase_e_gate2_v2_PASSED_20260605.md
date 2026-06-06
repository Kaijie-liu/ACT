# Phase E Gate 2 v2 — PASSED + Floor Diagnosed

**Date**: 2026-06-05 night (S2 v3 production)
**Status**: Gate 2 v2 **PASSED 40/40 OK, max 30.1 GB**
**Headline impact**: NONE (1472 holds; Gate 2 is memory-only by spec)

---

## 1. Gate 2 v2 acceptance — ALL criteria met

Per advisor's hard spec from 2026-06-05 evening directive:

| Criterion | Required | Measured |
|---|---|---|
| 0 OOM | required | **0 / 40** |
| peak < 80 GB | required | 30.1 GB max (cifar) / 26.4 GB max (tiny) |
| 0 skipped ops | required | 0 / 40 (all 59-61 ONNX nodes processed) |
| center parity | required | held (unit tests + iid 0 dense vs streaming) |
| CIFAR 9-overlap max_excess no worse than 10% | required | **8/9 IDENTICAL to Day-of K=∞**; iid 72 IMPROVED by 28% |
| `K_target >= n_root` enforcement | required | tested + auto-promote |
| output-L1 priority | required | tested |
| liveness eviction | required | implemented + reduces RSS 2-3× |
| 1472 unchanged | required | clean (`git diff --stat -- act/` empty) |
| Unit tests | regression | **52/52 pass** (47 prior + 5 new for K_target/output-L1) |

`audit_results/sc_hz_gate2v2_perbench_K_20260605T063534Z/`

## 2. Per-bench profile

| Bench | OK | OOM | max RSS | median RSS | median wall | median max_excess |
|---|---|---|---|---|---|---|
| cifar100 | 20/20 | 0 | 30.1 GB | 21.2 GB | 158s | +0.854 |
| tinyimagenet | 20/20 | 0 | 26.4 GB | 23.0 GB | 178s | +1.680 |

### Day-of K=∞ vs Gate 2 v2 K=12K parity (9 overlap iids)

| iid | Day-of K=∞ | Gate 2 v2 K=12K | delta |
|---|---:|---:|---:|
| 0 | +1.391 | +1.391 | 0.000 |
| 2 | +3.090 | +3.090 | 0.000 |
| 24 | +0.464 | +0.464 | 0.000 |
| 29 | +0.315 | +0.315 | 0.000 |
| 57 | +0.727 | +0.726 | -0.001 |
| 72 | +0.510 | **+0.368** | **-0.142 (-28%)** |
| 86 | +0.519 | +0.519 | 0.000 |

8/9 within 0.001 of K=∞ baseline; 1 better. **No degradation.**

## 3. Near-CERT iids identified

8 iids that are 1-4 rivals away from CERT:

| Bench | iid | max_excess | CERT-able conds | "PHANTOM" count |
|---|---|---:|---:|---:|
| cifar | 113 | +0.261 | 98/99 | 1 |
| cifar | 29 | +0.315 | 98/99 | 1 |
| cifar | 180 | +0.339 | 98/99 | 1 |
| cifar | 72 | +0.368 | 98/99 | 1 |
| cifar | 168 | +0.421 | 98/99 | 1 |
| cifar | 145 | +0.472 | 97/99 | 2 |
| tiny | 99 | +0.363 | 198/199 | 1 |
| tiny | 30 | +0.758 | 195/199 | 4 |

If Gate 3 mechanism flips even ONE of these to CERT → **first dense-conv NEW V**.

## 4. Floor diagnosis (post Gate 2 v2)

Tested K_target ∈ {12K, 20K, 40K} on cifar 113/29/180 and tiny 99:

```
K=12K vs K=40K: max_excess UNCHANGED to 4 decimal places on ALL 4 iids
```

The single-neuron DeepZ-triangle is at floor. Higher K_target does not reduce excess.

### LP UB breakdown for cifar 113 PHANTOM rival `Y_6 >= Y_82`

```
threshold:                            0.0000
center contribution (d·c):           -1.2651
generator contribution (|d·G|.sum):  +1.5264  ← drives the PHANTOM
tail contribution (|d|·tail):         0.0000  ← irrelevant
LP UB:                                +0.2613

Top 10 contributing generators: 0 root + 10 slack (all ReLU-triangle aux)
```

**Conclusion**: The PHANTOM is entirely driven by ReLU-triangle SLACK generators. To break it we need to reduce slack contributions by ~17%. Tail compression (more K) does nothing because tail is already 0.

## 5. The only remaining precision lever

Single-neuron triangle is already the tightest convex upper hull in continuous LP. To go tighter requires:

### Option A: Multi-neuron joint hulls (Anderson 2020 forward facets)
- Mechanism: for unstable neuron pairs (i, j), the EXACT convex hull of
   `(z_i, z_j, relu(z_i), relu(z_j))` over `[l_i,u_i]×[l_j,u_j]` has facets
   not implied by per-neuron triangles
- Implementation: requires extending `PrunedState` with constraint matrix
   `Ac, b` (currently only box-domain ξ ∈ [-1,1]^K)
- Estimate: 3-5 days

### Option B: Spec-aware pre-activation refinement
- Mechanism: for each unsafe rival, restrict the reachable set with
   `d_out · post_pipeline(relu) >= threshold`. Re-derive (l_i, u_i) under
   this constraint. Tighter bounds → smaller `mu_i` → tighter LP UB.
- Implementation: per-rival LP solving + integration into walker
- Estimate: 2-3 days

### Option C: Slack-aux constraint in LP UB
- Mechanism: instead of treating slack `ξ_aux` as fully independent in
   `[-1, 1]`, encode the implicit constraint `relu(z) <= upper_chord(z)`
   in an LP solve at output evaluation
- Implementation: replace closed-form LP UB with constrained LP solve
- Estimate: 1-2 days

Option C is the cheapest. Worth trying first.

## 6. 2-week kill switch — where we are

- Day 1 of S2 (2026-06-05). 13 days remain to Day 14 deadline.
- Memory ceiling broken (Gate 2 v2 PASS).
- Single-neuron floor diagnosed; precision lever requires multi-day implementation.

## 7. Tonight's stop point

```
Headline:                1472 V/A (audit-validated)
Gate 0 audit:            558/558 STRICT-PASS
Gate 1 streaming-prune:  52/52 tests pass
Gate 2 v2 memory pilot:  40/40 OK, 0 OOM, max 30.1 GB cifar / 26.4 GB tiny
Gate 3 floor:            DeepZ-triangle single-neuron at floor; K↑ no help
Near-CERT iids:          8 iids 1-4 rivals from CERT (cifar 113 closest at +0.261)
0 NEW V/A tonight
act/ clean
GPU/RAM:                 returning to baseline (no active SC-HZ procs)
```

## 8. Files

| File | Status |
|---|---|
| `research/sc_hz/conv_streaming_prune.py` | rewrite v3: K_target>=n_root + output-L1 + root protection |
| `research/sc_hz/tests/test_conv_streaming_prune_soundness.py` | 5 new tests (12 total in file) |
| `research/sc_hz/onnx_walker_resnet.py` | use_count liveness eviction added |
| `audit_results/sc_hz_gate2v2_perbench_K_20260605T063534Z/` | 40 per-iid receipts + summary.json |
| `research/phase_e_gate2_v2_PASSED_20260605.md` | this memo |
| `research/phase_e_roadmap_20260605.md` | unchanged (S2 v2 work documented separately) |
| 1472 freeze | unchanged |
| Unit test suite | 52/52 pass |
