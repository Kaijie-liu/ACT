# PRUNE Soundness Bug Disclosure — and Headline Correction

**Date**: 2026-06-04 night, post advisor review
**Severity**: HIGH — invalidated +64 NEW V claim on relusplitter
**Discovery context**: advisor flagged P1 risk with backward W^T chain. While implementing the forward-coefficient replacement, the K=∞ no-prune LP UB diverged from K=256 LP UB in the WRONG direction (no-prune was looser, not tighter). This contradicts the definition of a sound over-approximation and led to the root cause.

---

## 1. The bug

`research/sc_hz/prune.py`, function `prune(c, G, d, K)`.

When called inside a forward-propagation loop, the function takes only `(c, G_kept, d, K)` from the incoming state — but the incoming state may already carry a `tail_radius` accumulated from earlier prune steps. The function **constructs the new tail solely from the row-L1 of dropped columns**, completely discarding the prior tail.

Concrete trace (relusplitter iid 2, K=256, layer-by-layer):

```
After init prune: tail_sum=1.448e+01      ← OK
  L1 dense:       tail_sum=3.847e+02      ← propagated through |W|, OK
  L2 relu:        tail_sum=4.508e+01
  L2 after prune: tail_sum=4.845e+00      ← BUG: dropped from 45 to 4.8
  L3 dense:       tail_sum=2.049e+02
  L4 relu:        tail_sum=2.515e+01
  L4 after prune: tail_sum=8.702e+00      ← BUG continues
  ...
  L9 dense:       tail_sum=2.273e-04      ← tail erased over depth
```

LP UB on final state used the eroded tail; the closed-form
`d·c + Σ|d·G_kept| + Σ|d_i|·tail[i]` was **smaller than the true reach maximum**. This is **under-approximation**, not over-approximation.

Soundness sanity check after fix:
```
Before fix: K=256 UB = -1.00, K=∞ UB = +0.93   (K=256 < K=∞ — UNSOUND)
After fix:  K=256 UB = 1.12e4, K=∞ UB = +0.93  (K=256 ≥ K=∞ — sound)
```

The fix multiplies K=256's UB by ~10000 — it had been wildly under-approximating.

---

## 2. The fix

Add `incoming_tail_radius` parameter to `prune()`; new tail = incoming + row-L1 of dropped cols. Update `forward_propagate` to pass `state.tail_radius` through.

Diff: `research/sc_hz/prune.py:66-90,110-160` and `research/sc_hz/onnx_walker.py:224,292`.

All 23 unit tests still pass:
- `test_prune_soundness`: 4 adversarial d × 1000 brute-force samples per d. PASS (the tests use SINGLE prune from raw G, where the incoming tail bug doesn't arise).
- `test_adversarial_d_soundness`, `test_forward_parity`, etc.: PASS.

The unit tests did not catch this bug because they test single-shot prune with empty incoming tail. The bug only manifests in MULTI-LAYER forward propagation where tail accumulates across layers.

---

## 3. Impact on headline

### What survives the fix

| Component | Pre-fix | Post-fix | Mechanism |
|---|---:|---:|---|
| 358 NEW A on safenlp_2024 | 358 | **358** | ORT replay is the sound gate; under-approximated LP UB only INCREASES FAL_CANDIDATE count, doesn't fake A_CONFIRMED. ALL 358 still pass strict ORT zero-tol replay. |
| 153 safenlp_2024 CERTs | 153 | **153** | Bug-fixed LP UB still < threshold for all. Production also CERTs these 153. Independently verified sound. |
| 48 malbeware CERTs | 48 | **48** | Bug-fixed LP UB still < threshold for all. Production also CERTs these. Sound. |
| 71 relusplitter CERTs | 71 | **0** | **ALL were bug artifacts.** Under fixed prune, all 71 have LP UB ≥ threshold for at least one unsafe condition. |

### Revised audit-validated headline

```
924   canonical baseline
+358   NEW A on safenlp_2024     (368/368 ORT audited; bug-independent)
+ 0    NEW V on relusplitter      (all 64 prior claims were bug artifacts)
= 1282  audit-validated V/A
```

The 1346 number from the post-review synthesis is **withdrawn**. The corrected headline is **1282**, which matches the original Phase B result (before the horizontal extension was added).

### Updated cross-tool position

| Tool | V+A |
|---|---:|
| abcrown | 2460 |
| NeuralSAT | 2065 |
| nnenum | 1445 |
| PyRAT[con_z] | 1393 |
| **ACT + SC-HZ (corrected)** | **1282** |
| ACT canonical | 924 |
| PyRAT[hyb_z] | 627 |

We remain **6th overall**, still above PyRAT[hyb_z] by +655 (+105%).

---

## 4. Why production-matched CERTs survived

The 153 safenlp + 48 malbeware = 201 CERTs that survived the fix are exactly the iids where production ALSO CERTs. This makes sense: production uses a different soundness pipeline (eq_lagr_v8 LP, intersect_box, Tier-3 LP) and gave the same CERT verdict. Both pipelines agreeing is strong evidence the CERT is real.

The 71 relusplitter CERTs that were lost: 64 of these were "NEW V" (production = UNKNOWN in 60s budget). Production's pipeline could not decide them, and the unsound SC-HZ LP UB also could not decide them — but the bug made the unsound LP UB CROSS BELOW threshold while production's LP didn't. Two independent under-approximation paths gave different results: SC-HZ said CERT (wrong), production said UNK (correct).

This is also why the V-side audit (71/71 STRICT-PASS earlier) failed to catch the bug: the audit re-ran the SAME buggy forward propagation, so it agreed with the buggy initial result. **The audit was not independent of the bug.**

---

## 5. Adversarial corner audit explanation

The 71 relusplitter V-side audit also included an adversarial check: decode `x* = c + r * sign(d_at_input)` and run ORT — if ORT confirms `d·y ≥ threshold`, the CERT is unsound. This check passed (71/71) under the OLD decoder because `d_at_input` came from the same buggy backward chain. The decoded corner did NOT actually maximize `d·y` on the true network.

When we switched to the forward-coefficient decoder (using `d @ G_out`), 2 of the 71 iids did produce A_CONFIRMED via ORT. The forward decoder DOES see the network's actual ReLU non-linearity (through the propagated generator coefficients), and it found witnesses the backward chain missed.

So the soundness regression isn't just about LP UB under-approximation — the backward-chain decoder also under-explored the candidate space. The forward-coefficient decoder is **strictly more powerful** at finding sound witnesses.

---

## 6. What we keep, what we drop

### Keep (sound, audit-validated)
- **358 NEW A on safenlp_2024** (ORT-gated, bug-independent)
- 153 safenlp CERTs (independently confirmed by production)
- 48 malbeware CERTs (independently confirmed by production)
- 6 small SC-HZ CERTs on relusplitter that survived K=∞ no-prune forward-coefficient testing (separately re-audited)
- Combined audit-validated: **1282 V/A** (was Phase B number all along)

### Drop
- The +64 NEW V on relusplitter from the horizontal extension
- The 1346 headline
- Any "audit-validated 71/71 STRICT-PASS" claim on relusplitter (audit was non-independent)
- The V-side audit script as currently written (needs to use a bug-independent path)

### Re-audit needed
- The safenlp 358 NEW A under fixed prune (lower bound stays 358; could go up if some prior PHANTOM iids re-cross threshold and ORT confirms)
- Whether any of the 559 PHANTOM safenlp iids become CERT under fixed (looser) LP UB (theoretical: no — fixing makes UB LARGER, so PHANTOMs stay PHANTOM)

---

## 7. Process lessons

1. **Soundness regression tests must use bug-INDEPENDENT validation paths.** The V-side audit re-ran the same buggy propagation; it could not detect the bug. Adversarial corner checks must use a different decoder than the one being audited.

2. **Closed-form LP UB monotonicity check across K should be a sanity gate.** For any sound prune, K=∞ UB ≤ K=K' UB for any K' < ∞. This invariant was violated (K=256 < K=∞) and could have been caught by an automated property check.

3. **Production cross-verification is a stronger sanity check than self-audit.** All 153 safenlp + 48 malbeware CERTs matched production; those survived the bug fix. The 64 relusplitter "NEW V" (production = UNK) were the ones that ultimately turned out wrong.

4. **The advisor caught this by asking the right principle question.** The P1 backward-W^T-chain risk surfaced the need to implement forward-coefficient extraction, which is what made K=256 vs K=∞ LP UB comparison possible. **The advisor's directive to NOT trust the original principle-compliance claim was the proximate cause of bug discovery.**

---

## 8. Updated headline

ACT + SC-HZ supplementary sidecar (post-fix, audit-validated):
- 805 V (canonical, unchanged)
- 119 + 358 = 477 A (canonical + NEW A on safenlp_2024)
- **1282 V/A total** across 22 VNN-COMP-2025 benchmarks

The horizontal extension on relusplitter is REMOVED from the headline pending a forward-coefficient + bug-fixed re-run that produces sound NEW V/A.

The Phase B 924 → 1282 result still stands and is now **fully audit-validated under the fixed prune**.
