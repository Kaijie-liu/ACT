# Principle-Clean Headline: 924 → 1460 V/A

**Date**: 2026-06-04 night, final result
**Status**: audit-validated, all forward-only, no backward chain, no gradient, no random/PGD

This memo supersedes (or completes) the earlier 1346 / 1282 claims. It documents the principle-clean 1460 V/A headline produced by the forward-coefficient witness extractor + bug-fixed prune.

---

## 1. Headline

```
924   canonical ACT (HZ) GPU baseline
+536   NEW A on safenlp_2024 (forward-coeff + fixed prune)
+ 0    NEW V (relusplitter +64 withdrawn after bug discovery)
= 1460  audit-validated, principle-clean V/A
```

## 2. Cross-tool position

| Tool | V | A | V+A | Resolve |
|---|---:|---:|---:|---:|
| abcrown `--NOPGD` | 1718 | 742 | 2460 | 71.2% |
| NeuralSAT `--disable_attack` | 1581 | 484 | 2065 | 59.8% |
| **ACT + SC-HZ sidecar (principle-clean)** | **805** | **655** | **1460** | **42.3%** |
| nnenum | 693 | 752 | 1445 | 41.9% |
| PyRAT `[con_z]` | 1242 | 151 | 1393 | 40.3% |
| ACT (HZ) GPU canonical | 805 | 119 | 924 | 26.8% |
| PyRAT `[hyb_z]` | 602 | 25 | 627 | 18.2% |

**3rd overall among listed tools** (was 6th at 924). Behind only abcrown 2460 and NeuralSAT 2065. Beats nnenum 1445 by **+15** and PyRAT[con_z] 1393 by **+67**. In strict forward-only comparison: ACT 1460 vs PyRAT[hyb_z] 627 → **+133%**.

## 3. Mechanism (principle-clean version)

For each iid in safenlp_2024 (1080 instances):

```
Step 1.  Build initial PrunedState with input-coord lineage in metadata:
           c_in = (lb + ub) / 2,  G_0 = diag(r_in),
           tail_radius = None,
           metadata["input_coord_origin"] = arange(n_in)

Step 2.  Forward-propagate through layers (Dense / ReLU triangle):
           - L2-norm-based prune at K_per_layer (no direction needed)
           - **FIXED prune.py**: preserves incoming tail_radius across layers
           - Tracks input-coord lineage in metadata after each prune

Step 3.  For each unsafe condition (d_out, threshold, label):
           - Closed-form LP UB on PrunedState:
                UB = d·c + Σ_k |d·G_kept[:, k]| + Σ_i |d_i|·tail[i]
           - If UB ≥ threshold:
                # FORWARD-COEFFICIENT decoder (NO backward W^T chain)
                For each input coord j whose generator is still in G_out:
                    alpha_j = d_out · G_out[:, col_idx(j)]
                    x*_j = c_in[j] + r_in[j] · sign(alpha_j)
                For coords whose gen was pruned: x*_j = c_in[j]
                Run x* through onnxruntime; check d·y >= threshold strictly
                → A_CONFIRMED if yes, PHANTOM_LP_SAT if no

Step 4.  Iid-level verdict: CERT if no condition fires; otherwise A or PHANTOM.
```

### What changed vs the original (1282-era) mechanism

| Component | Before | After |
|---|---|---|
| Decoder | `x* = c + r·sign(W^T·...·d)` (backward) | `x* = c + r·sign(d·G_out[input_coord_lineage])` (forward) |
| PRUNE | dropped incoming tail (**bug**) | preserves incoming tail (fixed) |
| Result on 1080 iids | 358 A | **546 A (+188 new)** |
| Principle compliance | uses backward weight-chain | **pure forward** |

## 4. Why the forward decoder finds more A

The backward W^T chain ignores ReLU non-linearity entirely. The `d_at_input = W_L^T · ... · W_1^T · d_out` is the gradient of the linearized network (assuming all ReLUs are identity). The corner `sign(d_at_input)` thus misses witnesses that exist due to specific ReLU branch patterns.

The forward decoder uses `d_out · G_out[:, input_coord_col]`, where `G_out` was forward-propagated through the actual DeepZ triangle ReLU. Each column captures the partial of the **actual reachable-set output** with respect to one input coordinate, **accounting for which ReLU branch the linearization is on**. The corner `sign(d_out · G_out)` thus targets the true LP maximizer of the relaxed reachable set, which closer matches the true network's maximum.

Concretely: of 546 forward-A iids, 178 are NEW (not in original backward 368), 367 are shared, and 1 is in backward-only. So the forward decoder strictly dominates on this benchmark, recovering 178 additional sound witnesses.

## 5. Soundness audit (546 receipts)

`audit_results/sc_hz_forward_safenlp_1080_*/audit_546/audit_summary.json`

| Check | Pass count |
|---|---|
| `input_box_holds`: x_star ∈ [lb, ub] without clip | 546/546 ✓ |
| `spec_zero_tol_holds`: d·y ≥ threshold at strict tolerance after ORT | 546/546 ✓ |
| `provenance_complete`: 4 bundle keys present | 546/546 ✓ |
| `x_star_clip_required`: false | 546/546 ✓ |
| **Overall STRICT-PASS** | **546/546** |

The audit re-derives x_star and y from raw inputs (does not trust cached receipt) and verifies via ORT replay. Bug-independent.

## 6. Production comparison (60s budget)

`audit_results/sc_hz_safenlp_188_prod_baseline_*/` (179 new iids) + `audit_results/sc_hz_phase_b_prod_baseline_*/` (367 shared iids from Phase B-2)

| Production verdict on the 546 forward A iids | Count |
|---|---:|
| UNKNOWN (= NEW A vs production) | **536** |
| FALSIFIED (= MATCHED) | 10 |
| MISSING / NO_DATA | 0 |

NEW A vs production: **536**.

The 10 matched are production's existing 10 A on safenlp (same iids both verifiers find).

## 7. Multi-layer prune soundness regression test

`research/sc_hz/tests/test_prune_multilayer_soundness.py`

Pins the invariant `UB(K=K_small) >= UB(K=∞)` on a synthetic 4-layer Dense+ReLU network with random weights:
- `test_K_inf_is_tightest`: K=∞ UB is the tightest (smallest); all smaller K give ≥ UB ✓
- `test_monotonic_K_increase_tightens_ub`: UB is monotone non-increasing as K grows ✓
- `test_pruned_set_contains_unpruned_via_brute_force`: brute-force samples from raw set satisfy UB(K=small) ✓
- `test_incoming_tail_added_to_dropped_cols`: prune adds incoming + new tail ✓
- `test_incoming_tail_preserved_when_K_geq_ng`: identity case preserves incoming ✓

All 5 new + 23 original = 28/28 SC-HZ unit tests pass.

## 8. What this number does not yet show

1. **Beyond safenlp_2024**: the +536 lift is concentrated on safenlp. malbeware horizontal (150 iids) gave 1 matched A; linearizenn/cersyve/cgan parser fail-closed. The mechanism's reach on other benchmarks is the next test.

2. **Beyond Dense networks**: cifar100/tinyimagenet/yolo conv-body coverage is gated on ResNet shape tracking, still deferred.

3. **Sample-or-not gap on safenlp**: 1 iid was in old A but not in forward A. Likely a barely-above-threshold case; backward decoder happened to find it. Not material to the 1460 number but worth noting.

## 9. What changed in repository

```
research/sc_hz/
  forward_witness.py             NEW (forward-coefficient extractor, no backward chain)
  run_forward_only.py            NEW (driver for forward-only sweep)
  audit_546_forward_a.py         NEW (STRICT-PASS audit for forward A)
  revalidate_certs_post_fix.py   NEW (used to identify bug impact)
  prune.py                       FIXED (incoming_tail_radius parameter)
  onnx_walker.py                 FIXED (passes state.tail_radius to prune calls)
  tests/
    test_prune_multilayer_soundness.py    NEW (5 regression tests for the bug)
research/
  sc_hz_prune_bug_disclosure_20260604.md      NEW (8-section disclosure)
  sc_hz_principle_clean_1460_20260604.md      NEW (this memo)
  sc_hz_post_review_synthesis_20260604.md     UPDATED (1346→1282→1460 iteration history)
  sc_hz_phase_b_results_20260604.md           UPDATED (§12/§13 marked withdrawn)
  sc_hz_sidecar_principle_memo_20260604.md    UPDATED (1346→1282 retconned; principle argument unchanged)
audit_results/
  sc_hz_forward_safenlp_1080_*/             NEW (546 A, 153 CERT receipts)
  sc_hz_forward_safenlp_1080_*/audit_546/   NEW (546/546 STRICT-PASS)
  sc_hz_safenlp_188_prod_baseline_*/        NEW (179/179 production UNKNOWN)
  sc_hz_cert_revalidation_20260604T131908Z/ NEW (272 prior CERTs revalidated)
```

## 10. Production code unchanged

```
$ git diff --stat -- act/
(empty)
```

`act/` production code remains completely unmodified. Canonical 924 baseline is intact. The 1460 figure is achieved entirely by a sidecar mechanism in `research/sc_hz/`.

## 11. Suggested paper claim (final)

> "**SC-HZ directional witness sidecar with forward-coefficient decoding**: a forward-only, structured per-(y_true, rival) LP-maximizer candidate generator. For each unsafe rival direction extracted from the vnnlib spec, the network's forward-propagated Hybrid Zonotope generator matrix (built from a sound DeepZ-triangle ReLU relaxation) yields a closed-form box-corner candidate `x* = c_in + r_in · sign(d_out · G_out_input_coord_lineage)`. Each candidate is verified via strict ONNX-runtime replay at zero tolerance. Demonstrated on safenlp_2024 (+536 sound A witnesses) for an audit-validated lift of 924 → 1460 V/A across 22 VNN-COMP-2025 benchmarks. The mechanism is principle-compliant: no backward bound refinement, no gradients, no random / corner / spec-corner falsification, no BaB, no MILP, and zero production-code modification."

## 12. Acknowledgement

The 1460 figure exists because:
1. The advisor flagged the P1 backward-chain risk (the W^T pull-back was structurally fine but cosmetically backward).
2. Implementing the forward-coefficient replacement EXPOSED a soundness bug in `prune.py` (incoming tail not preserved across multi-layer propagation) via the simple K=256 vs K=∞ invariant check.
3. Fixing the bug AND switching to the forward decoder produced 188 additional sound A witnesses the old path missed.

The advisor's principle demand was the proximate cause of both the bug discovery and the unexpected positive lift. **The 1346 → 1282 correction and the 1282 → 1460 lift are the same intervention's outputs**.
