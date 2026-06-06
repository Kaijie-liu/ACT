# Overnight Morning Report — 2026-06-06 → 2026-06-07

**Period**: 2026-06-06 evening through 2026-06-07 morning
**Mode**: full autonomous execution per advisor directive
**Headline**: 1472 → **1480 audited+ORT-consistent NEW V** (+8 final)

---

## TL;DR (read this first)

```
+60 candidate (research-confirmed walker outputs)
   ↓ r93 cross-check (advisor mandatory)
+20 r93-audited (collins_rul 39 were double-counted)
   ↓ ORT consistency (caught a SOUNDNESS BUG!)
+8 ORT-confirmed and provenance-bundled
   ↓
1472 → 1480 (the actual paper-grade headline)
```

The audit pipeline caught a **SOUNDNESS BUG** in F1 LP on DAG networks
(cersyve) — 12 supposed "F1 LP CERTs" were FALSE: ORT-sampled inputs
violated the spec at rates of 70-99/100. Without ORT consistency, the
+60 → +20 collins_rul correction alone would have been wrong by an
additional 12 false CERTs.

---

## 1. Per-step results

### 1.1 ORT consistency check on 20 audit-accepted iids

`/tmp/ort_consistency.py` ran 100 random samples per iid through ORT:

| Bench | r93-Accepted | ORT-Consistent | ORT-Violation |
|---|---:|---:|---:|
| cersyve | 12 | **0** | **12** ← F1 LP DAG bug |
| cora_2024 | 3 | 3 | 0 |
| dist_shift_2023 | 5 | 5 | 0 |
| **TOTAL** | **20** | **8** | **12 false CERT** |

Cersyve violations magnitude: 70-99 out of 100 samples violated; max
spec excess +0.44 to +0.90. Not a numerical edge case — a clear bug.

### 1.2 Bug diagnosis and fix

**Root cause**: `forward_resnet_capture` tracks ONE `last_relu_record`
by topological order. For DAG networks with parallel ReLU branches
(cersyve has two-branch architecture joined by final Add), the
"topologically last ReLU" is not the same as "last ReLU on every
input→output path". F1 LP UB is computed with constraints from only
one branch → unsound underestimate.

HZ closed-form `lp_ub_rival_margin` remains sound because it uses
`|d·G|.sum()` over ALL generators including all branches' slacks.

**Fix applied**: pre-pass to detect DAG branchiness; if multiple ReLU
nodes AND multi-consumer values exist, disable F1 capture (force
`last_relu_record = None`). Verified:
- cersyve iid 0: walker.last_relu_record = None (F1 disabled — safe)
- dist_shift iid 3: walker.last_relu_record CAPTURED (sequential — OK)

Code: `research/sc_hz/constrained_lp_integration.py` lines 200-220.

Test suite: 73 tests, OK (expected failures=1) — no regression.

### 1.3 Provenance bundle for 8 truly accepted iids

`audit_results/sprint_truly_accepted_8_20260606.json`:

| Bench | iid | Mechanism | UB excess | ORT violations |
|---|---:|---|---:|---:|
| cora_2024 | 2 | HZ_closed | -0.6477 | 0/100 |
| cora_2024 | 38 | HZ_closed | -2.7925 | 0/100 |
| cora_2024 | 59 | HZ_closed | -2.1275 | 0/100 |
| dist_shift_2023 | 3 | HZ_closed | -6.4674 | 0/100 |
| dist_shift_2023 | 22 | F1_LP | -0.9614 | 0/100 |
| dist_shift_2023 | 38 | HZ_closed | -5.3604 | 0/100 |
| dist_shift_2023 | 53 | F1_LP | -2.9687 | 0/100 |
| dist_shift_2023 | 70 | HZ_closed | -15.5458 | 0/100 |

Each record has model SHA256 + spec SHA256 + UB excess + ORT consistency.

### 1.4 Advisor's "triangle everywhere" exploration

`/tmp/triangle_explore.py`:
Tested compound triangle constraint propagation (LP at every layer
using accumulated triangle constraints from prior layers).

```
2-layer toy (W3=[1,-1], mixed sign):
  exact:    2.0
  HZ:       3.5 (loose by 1.5)
  F1:       3.0 (last-ReLU only triangle, drops 14%)
  FC-HZ:    3.0 (all-layer triangle, no additional drop)
  Compound: 3.0 (per-layer LP bound tightening, no additional drop)

20 random 2-layer instances (n_h=8):
  Compound strictly < FC-HZ: 0/20
  Median compound drop vs FC-HZ: +0%
```

**Finding**: per-layer LP-based bound tightening within triangle
relaxation gives **0% additional** drop over FC-HZ. This is consistent
with F2b/FC-HZ closure findings:
- F1 single-neuron triangle: 17% real cifar (works on small dense)
- F2b pairwise joint hull: 0% additional on real cifar
- FC-HZ multi-layer triangle: 8% additional on synthetic
- Compound LP bound tightening: 0% additional

**Bottom line**: triangle relaxation has a hard ceiling in continuous
LP. The advisor's "triangle everywhere" hint validates the F2b/FC-HZ
diagnosis: **triangle is already the per-neuron tightest convex hull;
multi-layer / per-layer extensions can't break the dense-conv
barrier.** To break it requires either:
- Non-convex (forbidden by P3)
- Backward bound refinement (forbidden by P1)
- New abstraction (Phase H2 multi-month research)

---

## 2. Final numbers

```
True audited + ORT-consistent: +8 NEW V
  cora_2024:           +3 (iids 2, 38, 59)
  dist_shift_2023:     +5 (iids 3, 22, 38, 53, 70)
Headline:               1472 → 1480
Distance to 2000:       520 (was 528 at baseline)
```

---

## 3. Updated FORWARD_PLAN baseline

All Phase H2 scenarios should re-baseline from **1480**:

| Scenario | Phase H2 result | Headline |
|---|---|---:|
| OPTIMISTIC | A passes Z3 + B passes | 1700-1850 |
| LIKELY | One of A/B passes Z2, marginal Z3 | 1580-1700 |
| PESSIMISTIC | All 4 candidates fail | 1480 (current) |
| WORST | Soundness bug | <1472 rollback |

**2000+ unreachable** without principle relaxation or fundamentally
new abstraction.

---

## 4. Honest critique of the sprint method

1. **+60 was wrong in 3 ways**:
   - Collins_rul 39 were r93 CERT already (double count)
   - Cersyve 12 were F1 LP DAG bug (unsound)
   - cora 1 was r93 CERT in another row (correctly caught by r93 audit)
   - True NEW V: 8 (cora 3 + dist_shift 5)

2. **The audit discipline saved us from a paper-stage embarrassment**:
   - Without r93 cross-check: would have claimed +60 = 1532
   - Without ORT consistency: would have claimed +20 = 1492 (with
      12 false CERTs)
   - With both: +8 = 1480 (defensible)

3. **The +8 are genuinely paper-grade**:
   - r93 confirmed UNK in canonical baseline
   - Current build proves CERT (HZ closed or F1 LP, all sequential)
   - ORT spot-check passes 100/100 random inputs
   - Provenance bundle with model + spec SHA256

---

## 5. Recommended next-day actions

### A. Update documents to reflect 1480 (not 1492 or 1532)
- `SPRINT_AUDIT_RESULT_20260606.md`: change 1492 → 1480
- `PHASE_H2_new_abstraction_design_20260606.md`: baseline 1480
- `FORWARD_PLAN_principle_internal_levers_20260606.md`: T1 column
- `PAPER_1472_CHARACTERIZATION_20260606.md`: reflect +8 incremental

### B. Add unit tests for DAG soundness
- Test: cersyve-style DAG model should NOT produce F1 LP CERT
- Test: sequential model should still produce F1 LP CERT
- Add to `research/sc_hz/tests/test_dag_soundness.py`

### C. Phase H2-D SETPH (small exact tail projected hull)
- Per advisor's suggested sequence: D first (1 week kill-switch)
- Z0 toy gate: ≥50% drop over F1
- Z1 cifar 113 gate: ≤+0.05 excess
- Z2 8 sentinel gate: ≥1 NEW CERT or ≥60% median drop
- If D fails: pivot H2-A OPC-FD

### D. Phase H2-A OPC-FD (output-projected constrained forward domain)
- Bigger bet (2-3 weeks)
- The advisor's "triangle everywhere" exploration suggests OPC-FD's
   mechanism (early projection + retained constraints) is the right
   next direction since triangle alone is at ceiling

### E. L4 walker REMAINS PAUSED (advisor principle ruling required)

---

## 6. Files delivered (overnight)

| File | Status |
|---|---|
| `/tmp/sprint_audit.py` | r93 audit script (priority verdict aggregation) |
| `/tmp/sprint_audit_60.json` | per-iid audit records (60 total, 20 accepted) |
| `/tmp/ort_consistency.py` | ORT consistency check |
| `/tmp/ort_consistency.json` | per-iid ORT receipts (8 consistent, 12 false CERT exposed) |
| `/tmp/triangle_explore.py` | advisor's "triangle everywhere" exploration |
| `audit_results/sprint_truly_accepted_8_20260606.json` | provenance bundle for 8 NEW V |
| `research/sc_hz/constrained_lp_integration.py` | DAG safety check added |
| `research/CRITICAL_F1_LP_DAG_SOUNDNESS_BUG_20260606.md` | bug post-mortem |
| `research/SPRINT_AUDIT_RESULT_20260606.md` | needs update 1492→1480 |
| `research/OVERNIGHT_MORNING_REPORT_20260606.md` | this report |

---

## 7. The honest message to advisor (morning meeting)

> "Overnight execution caught a critical soundness bug. F1 LP on DAG
> networks (cersyve) was producing false CERTs; ORT consistency check
> exposed 12 violations out of 12 supposed F1 LP CERTs. After fix
> (DAG safety check disables F1 capture on branchy networks), 73 tests
> still pass and walker is safe.
>
> True audited+ORT-consistent incremental: +8 NEW V (cora 3 + dist_shift 5).
> Audited headline: 1472 → 1480. Provenance bundle written for 8.
>
> Triangle-everywhere exploration: 0% additional drop over FC-HZ on
> synthetic. Confirms F2b/FC-HZ closure — triangle relaxation has a
> hard ceiling. The path to dense-conv NEW V requires Phase H2 new
> abstraction (your A OPC-FD / D SETPH proposals).
>
> Next: H2-D SETPH one-week kill-switch sprint, then H2-A OPC-FD if
> D fails. L4 walker stays paused. Distance to 2000+: 520; remains
> structurally unreachable without Phase H2 breakthrough."

---

## 8. Sleep / wake cycle

Tonight's autonomous execution:
- Audit: caught double-counting (39 collins_rul)
- ORT: caught F1 LP DAG bug (12 cersyve)
- Triangle exploration: confirmed no signal beyond FC-HZ
- Provenance: 8 iids bundled
- All P1-P5 principles respected
- L4 walker: 0 lines code written
- 73 tests OK

The path forward is clear: Phase H2-D SETPH next session.
Good morning.
