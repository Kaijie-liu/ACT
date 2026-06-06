# Session Report: 1747 V/A STRICT (corrected per advisor)

**Date**: 2026-06-06 (continued session after advisor critique)
**Prior headline**: 1744 (mixed strict + 5 optional)
**Strict headline now**: **1747 V/A** (1472 + 275 strict, ORT-audited)
**Including 5 optional**: 1752 (informational only)

---

## Corrections vs. prior session report

The earlier `SESSION_END_1744_VA_20260606.md` reported 1744 as the
audit-validated headline, mixing strict (267) + 5 MILP/SETPH wins
that relax P3. Per advisor:

1. **Headline must be strict-only**. 1739 was the strict-only count
   when the 272 bundle was first split; we added 8 more clean wins
   (6 cifar + 2 acasxu) since to reach 1747 strict.
2. **5 optional NEW V isolated** in `optional_needs_ruling.json`. They
   are NOT counted in headline. SETPH may be acceptable under a future
   ruling; MILP is strictly excluded from headline.
3. **ORT sampling is fake-CERT detection, not proof**. The CERT proof
   is the per-op soundness lemmas (§4 of TAIL_RADIUS_SOUNDNESS_PROOF.md);
   ORT is rejection-only.

---

## Strict 275 / Optional 5 split

```
audit_results/sprint_truly_accepted_strict_275_20260606.json   (275 iids, SHA 038a68843ed485bb)
audit_results/optional_needs_ruling_20260606.json              (5 iids, pending ruling)
```

Per bench (strict):
```
safenlp_2024:    115
cifar100_2024:    93   (+8 since strict 1739)
acasxu_2023:      42   (+2 since strict 1739)
relusplitter:     11
dist_shift_2023:   5
cora_2024:         3
metaroom_2023:     3
malbeware:         3
─────────────────
TOTAL:           275
```

Optional (P3-relaxing):
```
1 cgan_2023/iid 3:   SETPH_top_k_all_unstable_12_octants_4096
4 MILP wins:         metaroom 22, dist_shift 64, relusplitter 34 + 96
```

---

## Rejected via ORT

- **acasxu iid 88**: walker reported HZ = -1.469e-03 but ORT showed
  77/2000 violations. Removed from candidate set. Investigation pending.

---

## tail_radius soundness deliverables (advisor item 2)

1. **Formal proof document**: `research/sc_hz/TAIL_RADIUS_SOUNDNESS_PROOF.md`
   - State invariant `(INV)`, HZ closed-form bound (2), per-op lemmas §4.
2. **Per-op unit tests**: `research/sc_hz/tests/test_tail_radius_soundness.py`
   - 8 tests covering Dense, BN, Residual Add, HZ formula, sampling check.
   - Rejection test for old single-column pool (must demonstrate UNSOUND).
   - Independent recompute on 5-iid sample from strict bundle.
3. **Regression coverage**: 81/81 full test suite pass after each walker change.

---

## FCHZ walker ops added this session

- Conv (chunked K-dim, ~256MB blocks, |W|·tail propagation)
- ConvTranspose
- BN with tail
- Residual Add with tail summing
- Mul/Sub with sound tail
- **NEW** Slice (single-axis, constant params)
- **NEW** Concat (multi-input, padded-G, tail-concat)
- **NEW** MaxPool (sound box relaxation, `G` cleared, into tail_radius)
- **NEW** GlobalAveragePool (linear average)
- **NEW** Pad (constant mode, multi-axis)
- **NEW** Transpose (permutation, batch-dim stripping)

These unlock walker on previously-skipped graphs, but a graph completing
doesn't imply CERT — e.g., linearizenn graphs complete but HZ values
are >>0 (architecturally hard for FCHZ).

---

## Open work (advisor item 3 — FCHZ walker, not LP cut)

| Direction | UNK count | Estimated lift | Status |
|---|---:|---:|---|
| cifar 110+ memory-conservative continuation | 90 | +20-50 | running PID 2083618 |
| tinyimagenet walker memory | 401 | +50-150 | pending (sparse slack engineering) |
| yolo Pad const detection + AveragePool | 288 | +20-100 | walker stops at non-const-Pad |
| traffic Sign op (quant) | 90 | 0 | not FCHZ territory |
| cersyve, cctsdb, nn4sys, lsnc per_instance.csv | ? | ? | files missing |
| ml4acopf | 279 | unknown | not probed |
| collins_aerospace, collins_rul | 41+186 | unknown | not probed |

---

## Principle compliance (strict P1-P5)

```
P1: Forward only          ✓ tail_radius propagates forward via |W|·tail
P2: No gradient           ✓ no autograd/PGD
P3: Continuous LP only    ✓ HZ closed-form = 1 LP per rival
P4: No input split        ✓ no BaB on input
P5: No random/corner      ✓ deterministic walker + audit
```

275/275 strict NEW V observe ALL five principles.
5 optional NEW V relax P3 (MILP / SETPH) — gated by advisor.

---

## Trajectory to 2000+

```
Strict now:         1747 V/A
+ cifar 110+:       +20-50  (engineering: memory, in progress)
+ tinyimagenet:     +50-150 (engineering: sparse-slack columns)
+ probes remaining: +0-50   (architecture-dependent)

Conservative 3-week:  1900
Optimistic 6-week:    2000-2050

Honest: 2000+ requires walker continuing to scale, not LP cut tweaks.
Advisor's framing is correct: this round is real progress, but math
problem remains; walker maturity is the gating engineering item.
```
