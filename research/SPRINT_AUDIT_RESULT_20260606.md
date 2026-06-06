# Sprint Audit Result — Lever 1 +60 candidate → +20 r93-audited → +8 ORT-confirmed NEW V

**Date**: 2026-06-06 morning (REVISED 2026-06-07 morning after overnight ORT consistency)
**Audit script**: `/tmp/sprint_audit.py` (r93 stage) + `/tmp/ort_consistency.py` (ORT stage)
**Audit output**: `/tmp/sprint_audit_60.json` + `/tmp/ort_consistency.json`
**Critical bug memo**: `research/CRITICAL_F1_LP_DAG_SOUNDNESS_BUG_20260606.md`
**Per advisor 2026-06-06**: no 1532 FINAL headline until audit passes;
significant double-counting risk on collins_rul / dist_shift.

**SECOND audit stage** (overnight 2026-06-07) caught a SOUNDNESS BUG:
F1 LP on DAG networks was unsound. The 12 cersyve "F1 LP CERTs" were
FALSE — ORT sample-of-100 showed 70-99/100 spec violations. After this
second stage, the true incremental drops from +20 to **+8**.

Three-stage audit chain:
```
+60 candidate → +20 r93-audited → +8 ORT-confirmed (= 1472 → 1480)
                                    cora_2024:        +3
                                    dist_shift_2023:  +5
                                    cersyve:           0 (F1 LP DAG bug, rejected)
                                    collins_rul:       0 (r93 double-count)
```

---

## TL;DR

```
+60 candidate NEW V (pre-audit, research-confirmed)
After r93 production cross-check (Stage 1 audit):
  cersyve:               +12 (r93 had 0 V — but see Stage 2!)
  cora_2024:             + 3 (r93 had 16 V; my 3 outside)
  collins_rul_cnn_2022:  + 0 (DOUBLE COUNT vs r93 39 V CERTIFIED)
  dist_shift_2023:       + 5 (r93 had 0 V; all 5 outside)
Stage 1 incremental:    +20 NEW V

After ORT consistency check (Stage 2 audit, overnight):
  cersyve:                 0 (12 ORT-VIOLATIONS! F1 LP DAG bug — REJECTED)
  cora_2024:               3 (ORT 0/100 violations — CONFIRMED)
  dist_shift_2023:         5 (ORT 0/100 violations — CONFIRMED)
Stage 2 ORT-CONFIRMED:   +8 NEW V

──────────────────────────────────────
TRUE audited+ORT incremental: +8 NEW V
Adjusted headline: 1472 → 1480 (NOT 1532, NOT 1492)
```

**Advisor's double-count concern was 100% correct on collins_rul.**
The 39 collins_rul "NEW V" were already in r93's 39 V baseline. The
walker simply re-verified them; they were never new.

---

## 1. Audit methodology

For each of the 60 candidate NEW V iids:

### Step A — r93 cross-check (advisor mandatory)
- Load r93 canonical CSV (`audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS/<bench>/per_instance.csv`)
- For multi-row iids (multiple sources: CPU/GPU/smoke/sidecar), aggregate
   verdicts by priority: CERTIFIED > FALSIFIED > VERIFIED > UNKNOWN_TIMEOUT
   > UNKNOWN > ERROR
- An iid is "truly UNK in r93" only if NO row reports CERT/FAL
- If r93 reports CERT/FAL → REJECTED as double-count

### Step B — Walker re-run
- Load model + spec
- Run forward_resnet_capture
- Compute HZ closed-form excess + F1 LP excess
- If mechanism is `HZ_closed`: require HZ excess < 0
- If mechanism is `F1_LP`: require F1 excess < 0
- Otherwise: REJECTED

### Step C — Provenance
- Record the actual HZ / F1 LP excess values as the "certificate"
- Note any provenance issue

---

## 2. Per-bench audit details

### 2.1 cersyve — 12/12 r93-ACCEPTED → 0/12 ORT-CONFIRMED (REJECTED)

`r93 cersyve` baseline: 0 V + 0 A + 12 UNK (all UNK).
Stage 1: confirmed all 12 candidate iids are UNK in r93 AND F1 LP CERT.

**Stage 2 ORT consistency: ALL 12 FAILED** — 100-sample ORT replay
showed 70-99/100 inputs violated the spec, max excess +0.44 to +0.90.

Root cause: cersyve has a BRANCHY DAG topology (two parallel ReLU
branches joined by final Add). My F1 LP integration only tracks ONE
`last_relu_record` by topological order; for DAG networks this MISSES
constraints from the other branch, producing UNSOUND F1 LP UB.

HZ closed-form remained sound for cersyve (uses all generators across
all branches). Only F1 LP was affected.

**FIX APPLIED** (`research/sc_hz/constrained_lp_integration.py`):
DAG safety check disables F1 capture on branchy networks. Verified:
cersyve iid 0 walker now reports `last_relu_record = None`. Tests
73/73 OK with expected failures=1.

**Net NEW V: +0 (after ORT consistency stage 2)**

See `research/CRITICAL_F1_LP_DAG_SOUNDNESS_BUG_20260606.md` for full
post-mortem.

### 2.2 cora_2024 — 3/4 ACCEPTED

`r93 cora_2024` baseline: 16 V + 4 A + 165 UNK.
4 candidate iids: 2, 5, 38, 59.
- iid 2: r93 = UNKNOWN ✓, walker HZ excess -0.64 ✓ → ACCEPTED
- iid 5: r93 = UNKNOWN ✓, walker HZ excess -6.05 ✓ → ACCEPTED
- iid 38: r93 = UNKNOWN ✓, walker HZ excess -2.79 ✓ → ACCEPTED
- iid 59: r93 = CERTIFIED ✗ (already in the 16) → REJECTED

**Net NEW V: +3 (clean)**

### 2.3 collins_rul_cnn_2022 — 0/39 ACCEPTED

`r93 collins_rul_cnn_2022` baseline: 39 V + 11 A + 12 UNK.
The 39 V in r93 baseline ARE EXACTLY the 39 candidate iids I "discovered".
The `_load_unk_iids` function in the sprint code picked them up as
UNK because at least one CSV row (CPU smoke) reported UNK; but the
canonical headline counts them as V via the GPU full run.

ALL 39 are r93 CERTIFIED → ALL 39 REJECTED.

**Net NEW V: +0 (advisor's double-count concern empirically confirmed)**

**Lesson**: the Dropout parser fix did NOT yield NEW V on collins_rul.
What it did do: unblock the walker on those 39 iids (which had been
walker-fail before). The verdicts were already CERT in the production
GPU pipeline; this audit just confirms the walker now agrees.

### 2.4 dist_shift_2023 — 5/5 ACCEPTED

`r93 dist_shift_2023` baseline: 0 V + 0 A + 72 UNK.
The "72/72 production leader" claim in `PAPER_1472_CHARACTERIZATION` was
WRONG: r93 has 0/0/72. My 5 candidate iids are r93 UNK and walker-CERT.

5 candidate iids: 3 HZ-closed CERT (iids 3, 38, 70) + 2 F1-LP CERT
(iids 22, 53). All 5 have r93 = UNKNOWN. All 5 walker excess < 0.

**Net NEW V: +5 (clean)**

---

## 3. Adjusted headline

```
r93 canonical baseline (audited frozen):   1472 V/A
Sprint accepted increment:                  +20 NEW V
                                              +12 cersyve
                                              + 3 cora_2024
                                              + 0 collins_rul (rejected)
                                              + 5 dist_shift_2023
─────────────────────────────────────────
Audited headline (NEW):                    1480 V/A
```

---

## 4. What this teaches about the FORWARD_PLAN

1. **The Step 0 measurement was directionally correct but optimistic.**
   - L1 H0 forecast: 56-185 NEW V/A
   - Audited actual: 20 NEW V (below low end)
   - Reason: H0 sample-of-5 extrapolation didn't account for r93-vs-production multi-row CSV.

2. **Parser fixes ARE valuable, but for different reasons than headline V/A.**
   - Dropout fix on collins_rul: 0 NEW V (all 39 were already counted) but
     it DID promote the walker from "fail" to "CERT" on 39 iids.
   - This is **architecturally important** for paper claims (the verifier
     can now handle Dropout), but it does NOT incrementally count V/A.

3. **Same-iid/timeout/profile attribution discipline was essential.**
   The original sprint memo overcounted by 3×. Without this audit, we
   would have written "1472 → 1480" in the paper and immediately been
   exposed by reviewers cross-checking against r93.

4. **Step 0 sample-of-5 is too small for individual bench accuracy.**
   For high-stakes benches (collins_rul where I was about to claim
   +39), need sample-of-20 or full population.

---

## 5. Phase H2 baseline correction

The Phase H2 design doc projected scenarios from "1532 current Lever 1
result":
```
OPTIMISTIC: 1750-1900 (cifar barrier broken)
LIKELY:     1600-1750
PESSIMISTIC: 1532
WORST:      <1472 rollback
```

These should be re-baselined from **1492**:
```
OPTIMISTIC: 1710-1860
LIKELY:     1560-1710
PESSIMISTIC: 1492 (current)
WORST:      <1472
```

2000+ remains structurally unreachable.

---

## 6. What still needs cross-check

| Item | Status |
|---|---|
| Production V/A direct cross-check | r93 = production baseline ASSUMED in this audit. If production differs from r93 in any of cersyve/cora/dist_shift, the audit is wrong. |
| ORT consistency on 20 accepted | NOT done. Required before official headline change. |
| Provenance bundle per iid | NOT done. Required before paper claim. |

---

## 7. Action items

### A. Audit confirms +20, not +60. Update FORWARD_PLAN / SPRINT_RESULTS memos.
### B. Run ORT consistency on the 20 accepted iids.
### C. Bundle provenance receipts for the 20 (model SHA + spec SHA + walker output + LP certificate).
### D. Do NOT update SAS2026_sound.tex headline yet; wait for B + C.
### E. After B + C pass: **1472 → 1480** becomes paper-grade.

---

## 8. Updated next-step priority (post-audit)

Per advisor's 2026-06-06 sequence:

1. ✓ Audit 60 candidate NEW V → +20 audited (THIS MEMO)
2. (1 day) Run ORT consistency + provenance bundle on 20 accepted
3. (1 week) H2-D SETPH kill-switch scout
   - Z0 toy gate: ≥50% additional drop over F1
   - Z1 cifar 113 gate: ≤+0.05 excess OR CERT
   - Z2 8 sentinel gate: ≥1 NEW CERT OR ≥60% median drop
4. If D fails Z0/Z1: pivot H2-A OPC-FD (2-3 weeks)
5. L4 walker stays PAUSED until advisor principle ruling

---

## 9. The honest message to advisor

> "+60 candidate NEW V was wrong. After your mandatory r93 cross-check
> with multi-row aggregation, the true incremental is +20 NEW V
> (cersyve 12 + cora 3 + dist_shift 5). collins_rul's apparent +39
> was 100% double-counting against r93 CERTIFIED. Adjusted headline:
> 1472 → 1480. Distance to 2000+: 508 (was 468). Phase H2 starts from
> 1492 baseline, not 1532. ORT consistency on the 20 is the next
> required step before paper-grade."

---

## 10. Files

| File | Status |
|---|---|
| `/tmp/sprint_audit.py` | Audit script |
| `/tmp/sprint_audit_60.json` | Per-iid audit records |
| `research/SPRINT_AUDIT_RESULT_20260606.md` | this memo |
| `research/SPRINT_RESULTS_principle_internal_levers_20260606.md` | NEEDS UPDATE: 1532 → 1492 |
| `research/PHASE_H2_new_abstraction_design_20260606.md` | NEEDS UPDATE: baseline 1480 (post-ORT-audit) → 1492 |
