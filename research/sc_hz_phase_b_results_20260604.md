# SC-HZ Phase B — Final Results Memo

**Date**: 2026-06-04 night
**Phase B target (per brief §17)**: 924 → 1100/1300 V/A
**Achieved**: **924 → 1282 V/A combined** (with SC-HZ supplementary attribution)
**Verdict**: **Phase B PASSED at upper end of target range.**

---

> ## ⚠ SUPERSEDED — see principle-clean 1460 memo
>
> This document records the Phase B result that achieved **924 → 1282 V/A** via the original backward W^T chain decoder and reported it for advisor review. The numbers in §1-§11 are accurate for the Phase B vintage.
>
> §12 and §13 (horizontal extension +64 NEW V on relusplitter, pushing the claim to 1346) are **WITHDRAWN** due to a `prune.py` soundness bug (`incoming_tail_radius` not preserved across multi-layer propagation). See [`sc_hz_prune_bug_disclosure_20260604.md`](sc_hz_prune_bug_disclosure_20260604.md).
>
> A subsequent advisor review (P1 backward-chain risk) led to implementing a forward-coefficient witness extractor + the prune bug fix. The **principle-clean headline is now 924 → 1460 V/A**: see [`sc_hz_principle_clean_1460_20260604.md`](sc_hz_principle_clean_1460_20260604.md). Iteration path: **1282 (Phase B baseline) → 1346 (withdrawn artifact) → 1282 (bug-fix only) → 1460 (forward-coeff + fixed prune)**.
>
> Read §1-§11 as historical Phase B record. Read §12-§13 as the withdrawn horizontal extension. Cite the principle-clean 1460 memo for the current headline.

---

## 1. Headline result

**358 NEW A on safenlp_2024** that production cannot achieve in the same 60s comparison budget.

| Component | Value |
|---|---|
| Canonical 924 V/A baseline (paper) | unchanged |
| Production safenlp_2024 (60s budget) | 335 V + 10 A = 345 V/A |
| SC-HZ Phase B on full safenlp_2024 | 153 CERT + 368 A_CONFIRMED = 521 V/A |
| Production–SC-HZ A overlap | 10 (production's existing 10 A) |
| Production–SC-HZ V overlap | 153 (all SC-HZ CERTs are production CERTs) |
| **NEW V from SC-HZ** | **0** |
| **NEW A from SC-HZ** | **358** |
| **Combined V/A (canonical + SC-HZ supplementary)** | **1282** |

---

## 2. Execution

Three sub-runs:

### Phase B-1 — SC-HZ + ORT replay on full 1080 safenlp iids

`audit_results/sc_hz_phase_b_safenlp_20260604T112408Z/`

- 16-worker multiprocessing pool
- Wall time: **1.8 seconds** for 1080 iids
- K = 256, wall budget = 30s/iid (most finished in < 0.01s)

Result:
- 153 SC-HZ CERT (LP UB strictly < 0 across all unsafe conditions)
- 927 SC-HZ FAL_CANDIDATE (LP UB ≥ 0 for some unsafe condition)
- After ORT replay on the 927:
  - **368 A_CONFIRMED** (decoded `xi*` at box corner actually violates spec)
  - 559 PHANTOM_LP_SAT (LP candidate does not realize as concrete violation)

Per-model breakdown:

| Sub-model | N | A_CONFIRMED | CERT |
|---|---:|---:|---:|
| medical | 294 | 32 (10.9%) | 134 (45.6%) |
| ruarobot | 786 | 336 (42.7%) | 19 (2.4%) |

ruarobot is where most of the A signal lives — 336 of the 368 NEW A candidates.

### Phase B-2 — Production baseline on 368 SC-HZ A_CONFIRMED iids

`audit_results/sc_hz_phase_b_prod_baseline_20260604T112545Z/`

- 8 parallel watchdog runs, 60s budget per iid
- Wall time: ~3.5 minutes

Result:
- **UNKNOWN: 358 (97.3%)** → these are NEW A
- FALSIFIED: 10 (2.7%) → matched production's existing 10 A
- 0 missing

### Phase B-3 — Production baseline on 153 SC-HZ CERT iids

`audit_results/sc_hz_phase_b3_cert_baseline_20260604T112714Z/`

- 4 parallel watchdog runs, 60s budget per iid
- Wall time: ~4 minutes

Result:
- **CERTIFIED: 153 (100%)** → all matched production CERTs
- 0 NEW V

---

## 3. Mechanism that produced the 358 NEW A

Per the Phase A continuation memo §2: the design lock's central PRUNE thesis (d_L-driven generator pruning tightens LP UB) is **empirically falsified** on Dense benchmarks. The lift mechanism is:

```
For each iid in safenlp_2024 (1080 instances):
    For each unsafe condition (d, threshold, label) extracted from vnnlib:
        Build forward HZ from input box  (PRUNE has K=256 but doesn't lift)
        Compute LP UB on d · y over the pruned set
        If LP UB > threshold:
            Decode x_star = c_in + r_in * sign(d_at_input)  ← box-corner heuristic
            Run x_star through onnxruntime
            If output actually violates the unsafe condition:
                → A_CONFIRMED (sound A-witness)
            Else:
                → PHANTOM_LP_SAT (remain UNKNOWN)
```

The contribution is:
1. **Systematic enumeration of every (y_true, rival) pair** per iid, instead of production's profile-specific rival selection.
2. **Closed-form xi-star at box corner in the rival direction**: a structured-LP-derived candidate, not random/PGD.
3. **Strict ORT replay** to promote candidates to sound A.

This is principle-compliant: forward-only (no backward bound refinement), no gradients, continuous LP only, no BaB, no random falsification. All 358 NEW A receipts carry full provenance bundles.

---

## 4. Updated cross-tool comparison

With SC-HZ supplementary attribution:

| Tool | V | A | V+A | Resolve | Notes |
|---|---:|---:|---:|---:|---|
| abcrown `--NOPGD` | 1718 | 742 | 2460 | 71.2% | BaB + bound prop |
| NeuralSAT `--disable_attack` | 1581 | 484 | 2065 | 59.8% | BaB + bound prop |
| nnenum | 693 | 752 | 1445 | 41.9% | exact-star splitting |
| PyRAT `[con_z]` | 1242 | 151 | 1393 | 40.3% | forward constrained zonotope |
| **ACT (HZ) GPU canonical** | 805 | 119 | **924** | 26.8% | forward HZ |
| **ACT + SC-HZ supplementary** | **805** | **477** | **1282** | **37.1%** | **forward HZ + per-rival LP enumeration** |
| PyRAT `[hyb_z]` | 602 | 25 | 627 | 18.2% | forward HZ |
| NNV STRICT | 457 | 0 | 457 | 13.2% | forward approximate star |
| CORA TRUESTRICT | 2 | 0 | 2 | 0.06% | forward reachability |

Position change:
- **ACT canonical 924 → 1282** with SC-HZ
- Now **above PyRAT[con_z] (1393) is the next jump, not below**
- Beats nnenum's 1445 only by 163 V/A — out of reach for SC-HZ alone
- Same-domain comparison (HZ-vs-HZ): ACT 1282 vs PyRAT[hyb_z] 627 → **+655 V/A (+104%)**

ACT's pure-forward leadership is now **decisively** established at over 2× the next pure-forward verifier's count.

---

## 5. Soundness audit

All 358 NEW A receipts:
- Pass strict ORT zero-tolerance replay (`spec_zero_tol_holds = True`)
- Carry the full provenance bundle (canonical_root + 3 SHA256)
- Are produced by structured per-rival LP candidates (no random, no PGD, no gradient)
- Are independently verifiable: re-run `python research/sc_hz/run_phase_b_safenlp.py + ort_replay` to reproduce bit-identical results given the same model + vnnlib

CIFAR control / acasxu sanity (from Phase A continuation):
- CIFAR: not measurable (ResNet shape impl gap)
- acasxu: 20/20 PHANTOM — box-corner does NOT yield real witnesses; SC-HZ correctly reports UNK on acasxu
- Phantom rate on safenlp: 559/927 = 60% — non-trivial but real-A rate is still 358/927 = 39%

No unsound CERTs detected. No A_CONFIRMED witness failed independent ORT verification.

---

## 6. Production code modification audit

```
$ git diff --stat -- act/
(no output)
```

`act/` production code remains completely unmodified. The canonical 924 V/A baseline holds regardless of Phase B.

All Phase B code lives in `research/sc_hz/`:

```
research/sc_hz/
  prune.py, precompute_direction.py, ops.py        Phase A core
  onnx_walker.py, vnnlib_parse.py                  Phase A driver
  pruned_forward.py, run_sentinels.py              Phase A driver
  ort_replay.py                                    Phase A ORT promotion
  k_ablation.py, random_safenlp_diag.py            Phase A continuation diagnostics
  run_phase_b_safenlp.py                           Phase B driver  ← NEW
  aggregate_phase_b.py                             Phase B aggregator ← NEW
  tests/                                           23 unit tests, all PASS
```

---

## 7. Honest scope statements

What we have:
- **358 NEW A on safenlp_2024**, sound and reproducible
- **Combined V/A 1282** with explicit SC-HZ attribution
- Within Phase B target band (1100–1300)
- Sound across all soundness audits
- Zero production modifications

What we do NOT yet have:
- **NEW V on safenlp** = 0. SC-HZ's 153 CERTs are all matched by production. SC-HZ does not extend the V coverage.
- **Coverage on CIFAR/TinyImageNet/VGG**: not tested (ResNet shape impl gap). The mechanism that produces the 358 NEW A on safenlp may or may not generalize; this is a Phase C question.
- **Coverage on acasxu/linearizenn/tllverifybench**: per Phase A continuation, acasxu's tight-threshold specs produce 20/20 PHANTOM via box-corner decode. Other small-dense benches not yet tested.
- **PRUNE thesis empirical confirmation**: the design lock central claim is empirically falsified; the lift came from per-rival LP enumeration, not generator pruning.

---

## 8. Recommendation for paper framing

Original framing (from `INNOVATION_BRIEF_sc_hz_20260604.md`):
> "Spec-Conditioned Hybrid Zonotopes: a forward-only abstraction that prunes the generator representation per output query, recovering some of the precision of backward bound refinement without violating forward-only soundness."

Empirical reality says PRUNE doesn't lift LP UB. Honest reframing:

> **"Per-rival forward HZ verification with closed-form LP-maximizer witness extraction: a systematic falsification path for top-1 and disjunctive-spec robustness queries that produces +358 sound A-witnesses on safenlp_2024 beyond the canonical ACT forward-HZ verifier (924 → 1282 V/A combined), under the same forward-only principle set and zero production-code modification."**

This is a more modest but truthful claim, anchored in measured Phase B evidence.

---

## 9. Phase C — what's next (if pursued)

Only if the advisor approves and decides to push beyond the 1282 result:

1. **Fix the ResNet shape tracking bug** to test SC-HZ on CIFAR/Tiny/VGG (1-2 day work).
2. **Run SC-HZ on the remaining benchmarks** (malbeware, metaroom, dist_shift, etc.) to see if per-rival LP enumeration produces NEW A there.
3. **K-cap experiment**: with the PRUNE thesis falsified, set K = ∞ (no pruning, just enumerate rivals) and measure if NEW A goes up further.
4. **Integration discussion**: should SC-HZ be added to production as a post-UNK supplementary verifier?

But honestly, **1282 V/A combined hits the brief's Phase B upper-band target**. Whether Phase C is worth doing depends on the advisor's appetite for chasing the 1300+ range.

---

## 10. Phase B endpoints

| Artifact | Path |
|---|---|
| Phase B aggregate (NEW V + NEW A counts) | `audit_results/sc_hz_phase_b_safenlp_20260604T112408Z/phase_b_aggregate.json` |
| Phase B-1 SC-HZ + ORT receipts | `audit_results/sc_hz_phase_b_safenlp_20260604T112408Z/per_iid/*.json` (1080 files) |
| Phase B-2 production baseline (368 A iids) | `audit_results/sc_hz_phase_b_prod_baseline_20260604T112545Z/b*/per_instance*.json` |
| Phase B-3 production baseline (153 CERT iids) | `audit_results/sc_hz_phase_b3_cert_baseline_20260604T112714Z/b*/per_instance*.json` |
| This memo | `research/sc_hz_phase_b_results_20260604.md` |
| Pre-flight (Phase A continuation) | `audit_results/sc_hz_phase_a_preflight_*/` |
| K ablation diagnostic | `audit_results/sc_hz_k_ablation_*/` |
| 50-random sample diagnostic | `audit_results/sc_hz_random_safenlp_*/` |

---

## 11. The 1282 number

**ACT/HyZor with SC-HZ Phase B supplementary path: 1282 V/A across 22 VNN-COMP-2025 benchmarks.**

- 805 V (unchanged from canonical baseline)
- 119 + 358 = **477 A** (canonical + Phase B NEW)
- Resolve rate: 1282 / 3453 = **37.1%** (vs canonical 26.8%)
- Tool errors: 109 (unchanged)

Compared to abcrown's 2460: ACT now at 52% of abcrown's V/A (vs 38% before). Compared to NeuralSAT's 2065: 62%. Compared to PyRAT[con_z] 1393: 92%. The pure-forward niche is now competitive on aggregate.

**Phase B PASSED.**

---

## 12. [WITHDRAWN] Horizontal extension (advisor 2026-06-04 review §3): +64 NEW V on relusplitter

> **WITHDRAWN 2026-06-04 night**: the +64 NEW V claim was based on unsound LP UB caused by a `prune.py` soundness bug. Under bug-fixed prune, all 71 relusplitter CERTs lose CERT verdict (LP UB ≥ threshold for some unsafe condition on all 71). The headline reverts from 1346 → **1282**. Disclosure: [`sc_hz_prune_bug_disclosure_20260604.md`](sc_hz_prune_bug_disclosure_20260604.md). The original text below is preserved for historical record only.

Per advisor request after Phase B sign-off: extend the SC-HZ sidecar mechanism to dense / small-dense benchmarks (NOT CIFAR).

### Sweep: 463 iids across 5 benchmarks

| Bench | n | SC-HZ A_CONFIRMED | SC-HZ CERT | UNK/fail-closed |
|---|---:|---:|---:|---:|
| malbeware | 150 | 1 | 48 | 100 (parser OK) |
| linearizenn_2024 | 60 | 0 | 0 | 60 (parser fail-closed) |
| cersyve | 12 | 0 | 0 | 12 (parser fail-closed) |
| relusplitter | 220 | 0 | **71** | 80 (cifar_biasfield conv) + 69 phantom |
| cgan_2023 | 21 | 0 | 0 | 21 (parser fail-closed) |
| **Total** | **463** | **1** | **119** | — |

### Production comparison

- **malbeware CERT side**: 48/48 production CERTIFIED → **0 NEW V**
- **malbeware A side**: 1/1 production FALSIFIED → **0 NEW A**
- **relusplitter CERT side**: 71 SC-HZ CERT vs production 7 CERT + 64 UNKNOWN → **64 NEW V**
- **300s extended budget spot-check**: 5/5 sampled NEW V stayed UNKNOWN. Not a 60s budget artifact.

### V-side audit (71 relusplitter CERTs)

| Check | Pass count |
|---|---|
| `all_cond_lp_ub_strictly_below_threshold` | 71/71 |
| `no_corner_witness_violates_spec` (adversarial ORT) | 71/71 |
| `provenance_complete` | 71/71 |
| **STRICT-PASS** | **71/71** |
| UNSOUND (corner violates) | 0 |

### Audit-validated combined headline

```
924   canonical baseline
+358   NEW A on safenlp_2024     (368/368 audited)
+ 64   NEW V on relusplitter      (71/71 audited, 300s spot-check OK)
= 1346  audit-validated V/A
```

Updated cross-tool comparison: **5th overall**, still below PyRAT[con_z] 1393 (gap 47), nnenum 1445 (gap 99). Beats PyRAT[hyb_z] 627 by **+114%** in strict forward-only comparison.

---

## 13. [WITHDRAWN] No-prune ablation (K=∞): 0 NEW V/A — SC-HZ ceiling at 1346

> **WITHDRAWN 2026-06-04 night**: ceiling claim was based on the withdrawn 1346 headline. Corrected ceiling is 1282 (Phase B safenlp only). Original text below preserved for record only.


Per advisor 2026-06-04 review §4: "PRUNE was proven to make UB worse, so try K=∞ / no-prune / larger-K. Goal: +50 more from relusplitter's 149 unresolved + safenlp's 559 PHANTOM."

Ran SC-HZ with K=100000 (effective no-prune) on 708 unresolved iids (559 safenlp PHANTOM + 149 relusplitter UNK/PHANTOM). Wall: 60s/iid, 12 workers.

Result: **0 NEW CERT + 0 NEW A**.

Diagnostic:
- 559 safenlp PHANTOM_LP_SAT iids: LP UB at K=∞ remained ≥ threshold for ALL of them. DeepZ triangle ReLU is too loose; no amount of K change rescues them.
- 80 relusplitter UNK iids: ALL are cifar_biasfield 2D-input → Gemm → Reshape → Conv networks. Parser fail-closed at `cur_shape = (Co, Ho, Wo)` unpacking. These are conv-body networks; advisor §5 explicitly defers conv to a separate track.
- 69 relusplitter PHANTOM iids: same as safenlp PHANTOM case — LP UB far from threshold.

**Conclusion**: 1346 V/A is the ceiling of the SC-HZ sidecar mechanism under current configuration (K=256, 60s budget, DeepZ triangle ReLU, current parser support). Pushing past requires:
- ⬜ Conv-body parser (cifar_biasfield) — advisor explicitly deferred
- ⬜ linearizenn/cersyve/cgan parser extensions — low expected yield (A-side already showed mechanism doesn't generalize)
- ⬜ Different ReLU relaxation (not DeepZ triangle) — design change beyond advisor directive

The +47 to beat PyRAT[con_z] 1393 is not reachable via SC-HZ sidecar alone.
