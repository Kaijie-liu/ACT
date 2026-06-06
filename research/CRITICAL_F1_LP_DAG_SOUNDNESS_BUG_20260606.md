# CRITICAL: F1 LP Soundness Bug on Branchy DAG Networks

**Date**: 2026-06-06 night (overnight ORT consistency exposed the bug)
**Severity**: SOUNDNESS BUG — F1 LP reports FALSE CERTIFICATES on DAGs
with parallel ReLU branches.
**Discovery method**: ORT consistency check on 20 audit-accepted iids;
12 cersyve iids showed 70-99/100 sample violations with max excess
+0.44 to +0.90 — clearly NOT CERTIFIABLE.
**Audited headline correction**: 1472 → **1480** (not 1532, not 1492).

---

## 1. The bug in one paragraph

`forward_resnet_capture` in `research/sc_hz/constrained_lp_integration.py`
tracks ONE `last_relu_record` by graph topological order. For networks
with parallel ReLU branches (e.g., cersyve's two-branch architecture
joined by final Add), this single-record assumption is wrong. The
"last ReLU" by topological order may not be the LAST ReLU on EVERY
input-to-output path. F1 LP UB is computed using only this single
record, missing constraints from the other branches' ReLUs. The
resulting LP UB is smaller than the true sound UB, causing FALSE CERT.

HZ closed-form `lp_ub_rival_margin` remains sound (uses all generators
across all branches via `|d·G|.sum()`). Only F1 LP is affected.

---

## 2. Evidence from ORT consistency check

Sampled 100 inputs uniformly from each iid's box; ran ORT; counted
violations:

```
cersyve/iid 0: 70/100 violated, max excess +0.7240
cersyve/iid 1: 96/100 violated, max excess +0.7636
cersyve/iid 2: 62/100 violated, max excess +0.4619
... (all 12 cersyve iids show similar massive violation rates)

cora_2024/iid 2:  0/100 violated, max excess -5.51 ✓
cora_2024/iid 38: 0/100 violated, max excess -5.33 ✓
cora_2024/iid 59: 0/100 violated, max excess -5.18 ✓

dist_shift_2023/iid 3:  0/100 violated, max excess -27.21 ✓
dist_shift_2023/iid 22: 0/100 violated, max excess -38.11 ✓
... (all 5 dist_shift iids consistent)
```

cora and dist_shift are sequential networks (no parallel ReLU
branches in the F1-relevant subgraph), so F1 LP works correctly.
Cersyve has DAG branches, F1 LP fails.

---

## 3. Why HZ closed-form is sound but F1 LP is not

HZ closed-form computes:
```
LP_UB_HZ = d · c + Σ_k |d · G_kept[:, k]|
```
where G_kept has columns from ALL ReLU slacks across ALL branches.
The `|·|.sum()` is a sound upper bound regardless of branch structure.

F1 LP computes:
```
LP_UB_F1 = max d · y_out s.t.
  y_out = W_remaining @ y + b_remaining
  y satisfies triangle constraints w.r.t. z = c_z + G_z @ xi
  xi ∈ [-1, +1]^K
```

The `last_relu_record` provides (c_z, G_z, l, u) for THE LAST ReLU
in topological order. For branchy DAG, this misses ReLU constraints
on the OTHER branch. The LP then treats the other branch's ReLU
outputs as fully free (only bounded by the HZ closed-form generators),
which is LOOSER than the true HZ but TIGHTER than what F1 should report.

Net effect: F1 LP UB is BELOW the true UB → false CERT.

---

## 4. Audit correction

| Bench | Pre-audit | Post-r93-audit | Post-ORT-audit (TRUE) |
|---|---:|---:|---:|
| cersyve | 12 | 12 | **0 (F1 LP UNSOUND)** |
| cora_2024 | 4 | 3 | 3 ✓ |
| collins_rul_cnn_2022 | 39 | 0 | 0 (double-count) |
| dist_shift_2023 | 5 | 5 | 5 ✓ |
| **TOTAL** | **60** | **20** | **+8 TRUE NEW V** |

**Adjusted headline: 1472 → 1480.**

The +60 initial was wrong (collins_rul double-count); +20 was wrong
(cersyve F1 LP unsound); +8 is the true incremental.

---

## 5. Fix design

### Short-term (mandatory before any further F1 LP CERT claim)

Add a "DAG safety check" in `forward_resnet_capture`:
- Pre-pass to detect DAG branches (input has multiple paths to output)
- If DAG branches exist AND walker captures last_relu_record:
   `r.last_relu_record = None` and `r.W_remaining = None` to force
   fall-back to HZ closed-form only
- Document this as a soundness invariant

### Long-term (Phase H2 prerequisite)

The F1 LP integration must handle DAG correctly: capture ReLU records
from ALL parallel branches and compose constraints in the output LP.
This is a non-trivial implementation but soundness-mandatory before
Phase H2 sprints use F1 LP on branchy networks.

---

## 6. Other potentially affected benches

Benches where my walker may have produced FALSE CERTs via F1 LP:
- cersyve: CONFIRMED FALSE on 12 iids
- Any other bench with DAG / residual structure where F1 LP gave CERT:
   need to re-verify

Sequential / chain networks (cifar, tiny, acasxu small dense): F1 LP
likely safe but warrants ORT spot-check too.

---

## 7. Action items (re-prioritized after the bug)

### A. Apply DAG safety check to walker (block further F1 LP CERT on DAG)
- Pre-pass detects branches
- If branchy: F1 LP CERT path disabled, fallback to HZ closed-form only

### B. Re-audit any prior F1 LP CERT claims
- The 12 cersyve are now confirmed FALSE
- collins_rul 39 were already rejected as double-count (separately)
- Other F1 LP CERTs that ORT-pass remain VALID

### C. Final accepted incremental: +8 NEW V
- cora_2024: 3 iids (2, 38, 59) — ORT consistent
- dist_shift_2023: 5 iids (3, 22, 38, 53, 70) — ORT consistent
- All have provenance: hz_excess or f1_excess negative AND ORT-verified

### D. Re-baseline Phase H2 from 1480 (not 1492, not 1532)
- 1472 + 8 = 1480
- Distance to 2000: 520 (not 468, not 508)

---

## 8. The honest message to advisor

> "ORT consistency check on the 20 audit-accepted iids revealed a
> SOUNDNESS BUG in F1 LP on branchy DAG networks (cersyve). All 12
> cersyve "F1 LP CERTs" are FALSE — ORT found massive violations
> (70-99/100 samples violated, max excess +0.44 to +0.90). The
> single-`last_relu_record` design only handles sequential networks;
> for DAG with parallel ReLU branches, F1 LP UB is WRONG.
>
> After this bug fix, the true accepted incremental is +8 NEW V
> (cora 3 + dist_shift 5), not +20, not +60.
>
> Adjusted headline: 1472 → 1480.
>
> This is a sobering finding. The +60 was the maximally-credible
> research-confirmed number; +20 was the r93-cross-checked number;
> +8 is the TRULY SOUND number after ORT confirmation. The audit
> discipline you mandated has caught a serious bug; without it,
> 12 false CERTs would have entered the paper headline."

---

## 9. Files

| File | Status |
|---|---|
| `/tmp/ort_consistency.json` | full ORT consistency receipts |
| `research/CRITICAL_F1_LP_DAG_SOUNDNESS_BUG_20260606.md` | this critical memo |
| `research/SPRINT_AUDIT_RESULT_20260606.md` | NEEDS UPDATE: 1492 → 1480 |
| `research/sc_hz/constrained_lp_integration.py` | NEEDS FIX: DAG safety check |
| 73 unit tests: OK with expected failures=1 | clean (no soundness tests on DAG yet — to be added) |
