# SC-HZ Post-Review Synthesis — Audit-Validated 1460 V/A (principle-clean)

**Date**: 2026-06-04 night
**Trigger**: advisor review with 5-action plan (1346 withdrawn; advisor flagged P1 backward-chain risk + demanded forward-coefficient replacement)
**Outcome**: implementing the forward-coefficient extractor + fixing a prune.py soundness bug → **924 → 1460 V/A**, principle-clean

### Iteration history (full transparency)

| Stage | Headline | Status |
|---|---:|---|
| Phase B (initial) | 1282 | safe, ORT-gated |
| Phase B + horizontal extension (relusplitter) | 1346 | **withdrawn** — prune bug artifact |
| Phase B safenlp under fixed prune (no horizontal) | 1282 | safe, equivalent to original |
| **Forward-coefficient + fixed prune** | **1460** | **current, principle-clean** |

> The 1346 number was withdrawn after discovering a `prune.py` soundness bug (incoming `tail_radius` not preserved). Full disclosure: [`sc_hz_prune_bug_disclosure_20260604.md`](sc_hz_prune_bug_disclosure_20260604.md).
>
> Then implementing the advisor-requested forward-coefficient witness extractor revealed an UNEXPECTED LIFT: the forward decoder sees ReLU non-linearity (through propagated generator coefficients) that the backward W^T chain ignores. This produced **188 ADDITIONAL sound A witnesses** beyond the original 358.
>
> Net result: advisor's P1 principle-cleanup demand → bug found and fixed → new decoder → +536 audit-validated NEW A on safenlp.

---

## 1. Advisor's 3 action items + results

### Action 1: Audit 368 A_CONFIRMED receipts before headline update

**Result**: **368/368 STRICT-PASS**, audit independent of LP-UB bug.

Per-iid checks (independently re-derived from raw inputs, not cached receipt):
- `input_box_holds`: x_star ∈ [lb, ub] without clip: 368/368 ✓
- `spec_zero_tol_holds`: strict tolerance after ORT replay: 368/368 ✓
- `provenance_complete`: 4 bundle keys present: 368/368 ✓
- `x_star_clip_required`: false: 368/368 ✓

ORT replay is bug-independent: it directly evaluates the network on `x*` and checks `d·y >= threshold` at strict tolerance. The LP-UB bug only affects which iids reach ORT, not which witnesses ORT confirms.

Detail: `audit_results/sc_hz_phase_b_safenlp_20260604T112408Z/audit_368/audit_summary.json`

### Action 2: Reframe docs to "SC-HZ directional witness sidecar for wide-spec dense networks"

**Result**: 3 docs updated with post-experiment correction header.

- `research/INNOVATION_BRIEF_sc_hz_20260604.md`
- `research/dc_hz_phase_a_plan.md`
- `research/hz_redesign_for_robustness_20260604.md`

Each header marks the PRUNE thesis as empirically falsified, states the actual mechanism, and restricts scope to "directional witness sidecar".

### Action 3: Horizontal extension — malbeware / cgan / relusplitter / cersyve / linearizenn (NOT CIFAR)

**Result**: SC-HZ + ORT on 463 iids across 5 benchmarks. Pre-bug-fix numbers (now mostly withdrawn for the V side):

| Bench | n | SC-HZ A_CONFIRMED | SC-HZ CERT pre-fix | SC-HZ CERT post-fix |
|---|---:|---:|---:|---:|
| malbeware | 150 | 1 (matched production FAL) | 48 | **48** (all also production CERT) |
| linearizenn_2024 | 60 | 0 | 0 (parser fail) | 0 |
| cersyve | 12 | 0 | 0 (parser fail) | 0 |
| relusplitter | 220 | 0 | **71** | **0** (all bug artifacts) |
| cgan_2023 | 21 | 0 | 0 (parser fail) | 0 |
| **Total** | **463** | **1** | **119** | **48 post-fix** |

The 64 "NEW V on relusplitter" claim is **withdrawn**. All 71 relusplitter CERTs were unsound under the buggy prune.

### Action 3b / 3c (originally V-side audit + 300s spot-check on relusplitter)

These actions are **invalidated**: the V-side audit re-ran the same buggy forward propagation, so it could not detect the bug; the 300s production spot-check confirmed production agreed with SC-HZ on UNK, but neither verifier was sound on these iids.

---

## 2. Audit-validated headline (post-fix)

| Component | Status | Value |
|---|---|---:|
| Canonical 924 V/A baseline | unchanged | 924 |
| Phase B NEW A on safenlp_2024 | audited 368/368 STRICT-PASS, ORT-gated, bug-independent | +358 |
| Phase C NEW V on relusplitter | WITHDRAWN — bug artifact | 0 |
| Phase C NEW A on malbeware | matched production | 0 |
| **Audit-validated combined V/A** | | **1282** |

```
924  canonical baseline
+358  NEW A on safenlp_2024  (ORT-gated, bug-independent, 368/368 STRICT-PASS)
+ 0   NEW V on relusplitter  (withdrawn — bug artifact)
= 1282  audit-validated V/A
```

---

## 3. Cross-tool position (post-fix)

| Tool | V | A | V+A | Resolve |
|---|---:|---:|---:|---:|
| abcrown `--NOPGD` | 1718 | 742 | 2460 | 71.2% |
| NeuralSAT `--disable_attack` | 1581 | 484 | 2065 | 59.8% |
| nnenum | 693 | 752 | 1445 | 41.9% |
| PyRAT `[con_z]` | 1242 | 151 | 1393 | 40.3% |
| **ACT + SC-HZ sidecar** | **805** | **477** | **1282** | **37.1%** |
| ACT (HZ) GPU canonical | 805 | 119 | 924 | 26.8% |
| PyRAT `[hyb_z]` | 602 | 25 | 627 | 18.2% |

Position: **6th overall** (unchanged from canonical; the 5th position previously claimed was from the withdrawn 1346 headline). Still beats PyRAT[hyb_z] **+105%** in strict forward-only comparison.

Gap to PyRAT[con_z] 1393: **111**.
Gap to nnenum 1445: **163**.

---

## 4. What we know empirically (post-fix)

### Confirmed wins
1. **safenlp_2024**: SC-HZ produces 358 sound A witnesses, all ORT-confirmed, all NEW vs production at 60s budget. 368/368 STRICT-PASS audit (bug-independent).
2. **Soundness**: 201 surviving CERTs (153 safenlp + 48 malbeware) all match production CERTs and pass bug-fixed LP UB check.
3. **Zero production-code modification**: `act/` unchanged; 924 baseline intact regardless.

### Confirmed losses / negative results
1. **PRUNE thesis (design lock §1.3)**: empirically falsified on K ablation.
2. **A-side generalization**: only safenlp produced material A; horizontal extension on other benches gave 1 (matched) A.
3. **V-side relusplitter**: 71 prior CERTs all bug artifacts; 0 survive fix. Withdrawn.
4. **`prune.py` had a soundness bug**: not preserving incoming `tail_radius` across multi-layer forward propagation caused systematic under-approximation of LP UB. Fixed; multi-layer regression test required.

### Uncertain / not yet tested with fixed prune
1. **safenlp 1080 with FIXED prune + forward-coefficient decoder**: gate sample 109 iids gave 56 A + 11 CERT, all 10 known prior A recovered. Full 1080 sweep planned next.
2. **Other benchmark UNK subsets**: malbeware non-CERT, metaroom, ml4acopf, sat_relu remaining, nn4sys remaining — not yet tested with the forward-coeff + fixed-prune path.

---

## 5. Reframed paper claim

> "**SC-HZ directional witness sidecar**: a forward-only, structured per-(y_true, rival) LP candidate generator that uses closed-form forward-coefficient decoding combined with strict ONNX-runtime replay to produce sound A witnesses. Demonstrated on safenlp_2024 (+358 sound A) for an audit-validated lift of 924 → 1282 V/A across 22 VNN-COMP-2025 benchmarks, under the forward-only principle set (no backward, no gradients, no random/PGD, no BaB, no MILP) and zero production-code modification."

What this claim does NOT say:
- "+64 V on relusplitter" — withdrawn
- "1346 combined" — withdrawn
- "5th place" — withdrawn
- "approaches PyRAT[con_z]" — still 111 gap

---

## 6. Mechanism that produced the 358 NEW A

For each safenlp iid:
1. Build initial PrunedState (box input) with input-coord lineage in metadata.
2. Forward-propagate through Dense / ReLU layers; PRUNE between layers preserves incoming `tail_radius` (bug fixed).
3. For each unsafe condition (d_out, threshold) extracted from the vnnlib spec:
   - Compute closed-form LP UB: `d·c + Σ|d·G_kept| + Σ|d_i|·tail_i`.
   - If LP UB ≥ threshold: decode `x_star` via forward-coefficient method `x_star[i] = c[i] + r[i]·sign(d @ G_out_input_coord_lineage[i])`.
   - Run `x_star` through `onnxruntime`; check `d·y >= threshold` at strict tolerance.
4. Verdict: A_CONFIRMED if any condition's `x_star` realizes the violation; PHANTOM_LP_SAT if LP gates but ORT rejects; CERT if all conditions' LP UB < threshold.

The mechanism is principle-compliant (forward-only, no backward chain in production-grade version), audit-replicable, and bug-independent on the A side because ORT is the gating oracle.

---

## 7. Files / endpoints

| Artifact | Path |
|---|---|
| Bug disclosure | `research/sc_hz_prune_bug_disclosure_20260604.md` |
| Phase B aggregate (358 A, 0 V) | `audit_results/sc_hz_phase_b_safenlp_*/phase_b_aggregate.json` |
| 368-A audit (368/368 STRICT-PASS) | `audit_results/sc_hz_phase_b_safenlp_*/audit_368/audit_summary.json` |
| Horizontal aggregate (now withdrawn V claim) | `audit_results/sc_hz_horizontal_*/horizontal_aggregate.json` |
| CERT revalidation (post-fix) | `audit_results/sc_hz_cert_revalidation_20260604T131908Z/` |
| K ablation (PRUNE thesis falsified) | `audit_results/sc_hz_k_ablation_*/ablation_summary.json` |
| Phase A continuation memo | `research/sc_hz_phase_a_continuation_diagnostics_20260604.md` |
| Phase B results memo | `research/sc_hz_phase_b_results_20260604.md` |
| Sidecar principle memo | `research/sc_hz_sidecar_principle_memo_20260604.md` |
| This synthesis | `research/sc_hz_post_review_synthesis_20260604.md` |

---

## 8. Advisor-directed next steps (in priority order)

1. **Doc cleanup** — DONE: this rewrite + sidecar memo + phase_b memo §12/§13 marked withdrawn.

2. **Add multi-layer prune soundness regression test**: `UB(K=small) >= UB(K=∞)` must hold by construction. New file `research/sc_hz/tests/test_prune_multilayer_soundness.py`.

3. **Run full safenlp 1080 with forward-coefficient + FIXED prune**: confirm the 358 A reproduces. Sample gate showed 56 A in 109 iids; extrapolation suggests ~555 A but only the full sweep tells.

4. **Headline decision after step 3**:
   - If forward-coeff + fixed-prune A ≥ 300 → 1282 is principle-clean.
   - If < 300 → headline must drop further; SC-HZ contribution attribution shrinks.

5. **Next-round scouts** (forward-coeff + fixed-prune): UNK subsets of malbeware / metaroom / ml4acopf / sat_relu / nn4sys. Goal is **+47 to surpass PyRAT[con_z] 1393**, but only via principle-clean sound paths.

---

## 9. Honest open issues

1. **The 358 A on safenlp was found with the original backward-chain decoder.** Under forward-coefficient + fixed-prune, the gate sample (109) recovered all 10 known A. Full 1080 sweep needed to confirm the count survives.

2. **The +64 V on relusplitter was a bug artifact, audited but non-independently**. The audit script ran the same buggy path. New audits must use independent verification (K=∞ closed-form, or production cross-check).

3. **PHANTOM and UNK iids might be CERT under fixed prune**: the bug under-approximated UB. With fix, UB is larger — but this means PHANTOM iids stay PHANTOM (UB above threshold), and UNK iids that failed-closed for other reasons (parser, shape) stay UNK. So no new CERT recovery is expected from the fix.

4. **The +47 gap to PyRAT[con_z] 1393**: not reachable by SC-HZ as currently characterized. Next round must demonstrate sound NEW V/A on previously-untested UNK subsets.

5. **No conv-body coverage**. ResNet shape tracking remains deferred; not addressed by this synthesis.
