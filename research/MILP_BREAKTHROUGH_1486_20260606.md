# MILP Breakthrough: 1484 → 1486 (and counting) via P3 Relaxation

**Date**: 2026-06-06 evening
**P3 RELAXED per user**: bounded-binary MILP allowed (open-source HiGHS only, NO Gurobi)
**Headline**: **1486 V/A** (audit-validated) — sweep ongoing, expecting more

---

## TL;DR

Per user's "可以放宽P3啊 当时是想不使用 gurobi这种商业工具 请你大刀阔斧来吧":

```
P3 RELAXED:
  Was: continuous LP only
  Now: bounded-binary MILP up to K_max binaries per call
       Solver: scipy.optimize.milp (HiGHS, open-source, NO Gurobi)

Mechanism: Tjeng et al. exact ReLU encoding
  For each unstable neuron i in top-K (by |d_eff|):
    Variables: continuous y_i + binary b_i ∈ {0, 1}
    Constraints:
      y_i ≥ 0, y_i ≥ z_i
      y_i ≤ u_i · b_i
      y_i ≤ z_i - l_i · (1 - b_i)
    where b_i = 1 means "ReLU active", b_i = 0 means "inactive"
  For non-top-K unstable: standard triangle relaxation
```

---

## 1. Results so far

| Iid | Bench | F1 LP UB | MILP UB | n_unstable | Audit |
|---|---|---:|---:|---:|---|
| 3 | cgan_2023 | +7.97e-5 | -1.18e-4 | 12 | already accepted via SETPH |
| **22** | **metaroom_2023** | **+5.77e-2** | **-5.91e-3** | **3** | **✓ AUDIT PASS** |
| **64** | **dist_shift_2023** | **+3.43** | **-13.73** | **13** | **✓ AUDIT PASS** |

Both metaroom 22 and dist_shift 64 audited via:
- r93 cross-check (both UNKNOWN baseline — no double count)
- DAG safety (both sequential, n_relus=3)
- ORT consistency (both 0/100 violations)
- Provenance bundle (SHA256 captured)

**Headline: 1484 → 1486 V/A** (with potentially more from running sweeps)

---

## 2. Cumulative state

```
1472 frozen baseline
+ cora_2024 (HZ closed):        +3 (iids 2, 38, 59)
+ dist_shift_2023 (3 HZ + 2 F1): +5 (iids 3, 22, 38, 53, 70)
+ cgan_2023 (SETPH @ top_k=12): +1 (iid 3)
+ metaroom_2023 (F1 LP):        +3 (iids 27, 28, 49)
+ metaroom_2023 (MILP):         +1 (iid 22) ← P3 RELAXED
+ dist_shift_2023 (MILP):       +1 (iid 64) ← P3 RELAXED
══════════════════════════════════════════════════
Total audit-validated NEW V:    +14
Headline:                       1486 V/A
Mechanisms used:                HZ_closed(6) + F1_LP(5) + SETPH(1) + MILP(2)
```

---

## 3. MILP impact per bench (sweep results so far)

| Bench | UNK | Sweep status | New MILP V |
|---|---:|---|---:|
| cgan_2023 | 21 | DONE | 1 (already counted) |
| dist_shift_2023 | 72 | DONE | +1 NEW (iid 64) |
| sat_relu | 50 | DONE | 0 |
| acasxu_2023 | 98 | DONE | 0 (need multi-layer MILP) |
| malbeware | 14 | in progress | TBD |
| relusplitter | 213 | pending | TBD |
| metaroom | 11 | in progress | (1 already found: iid 22) |
| cersyve | 12 | DAG-only | 0 (DAG safety) |
| collins_aero | 6 | walker fail | 0 |
| collins_rul | 12 | pending | TBD |
| tllverify | 29 | pending | TBD |
| safenlp_2024 | 737 | in progress | TBD |
| linearizenn | 47 | DAG-only | 0 (DAG safety) |
| cifar100 | 200 | DAG too slow | 0 (need multi-layer MILP + faster walker) |
| tinyimagenet | 199 | not started | TBD |

---

## 4. Why acasxu 0 NEW V via MILP

acasxu networks have 5-6 sequential ReLU layers. Last-layer MILP
encodes only the LAST ReLU exactly. Earlier layers still use triangle
relaxation. For acasxu, the PHANTOM gap is distributed across all
layers — last-layer-only MILP can't capture earlier-layer slack.

**Solution**: multi-layer MILP via FCHZState walker (engineering pending).

The encoder `research/sc_hz/milp_multilayer.py` is implemented;
needs a walker that produces FCHZState with all per-layer SlackRecords.

Expected gain after multi-layer: +5 to +30 on acasxu.

---

## 5. Why cifar DAG-blocked

cifar100 networks are residual ResNets with parallel branches.
F1 LP capture is unsound on DAG networks (cersyve bug discovered earlier).
Walker DAG safety check disables `last_relu_record` capture for cifar.

Without last_relu_record, MILP encoder can't be applied directly.

**Solution path**:
1. Multi-layer MILP via FCHZState walker would handle this naturally
   (encodes all branches, not just one chain)
2. Or relax DAG safety carefully for MILP (since MILP captures more
   constraints than F1 LP alone)

---

## 6. Honest distance to 2000+

```
1486 V/A current (after MILP +2)
+ in-progress sweeps (estimated):
  - safenlp 737: 0-30 (most are 2-layer MLPs, might flip)
  - relusplitter 213: 0-50 (small dense networks, MILP should help)
  - dist_shift 72: more? maybe
  - metaroom: maybe more
─────────────────────────────────────
Sweep-realistic 24h target: 1500-1550

If multi-layer MILP implemented (1 week):
  - acasxu: +5 to +30
  - cifar/tiny: +30 to +200 (if walker can do DAG MILP)
  - linearizenn: +5 to +20
─────────────────────────────────────
Multi-layer MILP realistic: 1550-1700

If full DAG MILP for cifar (3-4 weeks):
  - cifar100: +50 to +200 (matching benchmarks)
  - tinyimagenet: +50 to +200
─────────────────────────────────────
Aggressive MILP path: 1700-2000+
```

**2000+ now becomes achievable** with multi-layer MILP + cifar DAG path.

---

## 7. Mechanism portfolio

After P3 relaxation, the paper claim becomes:

```
"Forward-only neural network verifier with adaptive mechanism portfolio:
 1. HZ closed-form (1 LP)
 2. F1 LP single-layer triangle
 3. FC-HZ multi-layer triangle
 4. SETPH octant enumeration (for very small unstable)
 5. Bounded-binary MILP last-layer (Tjeng encoding, HiGHS)
 6. Multi-layer MILP (engineering)
 
 All forward-only, no gradient, no input-split.
 Open-source solver only (HiGHS, no Gurobi).
 Mechanism dispatched by per-iid characteristics (unstable count, DAG structure,
 F1 LP margin)."
```

This is a defensible paper contribution beyond mere "another MILP verifier"
because:
- Forward-only (most MILP verifiers do backward)
- HZ-based state representation (most MILP verifiers use star sets)
- HiGHS open-source (most use Gurobi)
- Mechanism portfolio adapts per-iid

---

## 8. Files

| File | Status |
|---|---|
| `research/sc_hz/milp_relu.py` | Tjeng last-layer MILP encoder |
| `research/sc_hz/milp_multilayer.py` | Multi-layer MILP via big-M (awaiting walker) |
| `research/MILP_BREAKTHROUGH_1486_20260606.md` | this memo |
| `audit_results/sprint_truly_accepted_14_20260606.json` | 14-iid provenance bundle |
| Tests | 73 OK (expected failures=1) |

---

## 9. Honest message

> "P3 relaxation executed within minutes of approval:
> - Tjeng exact ReLU MILP encoder implemented + validated
> - cgan iid 3 (already SETPH-CERT) confirmed via MILP: F1=+8e-5 → MILP=-1.2e-4
> - metaroom iid 22 NEW V: F1=+5.77e-2 → MILP=-5.91e-3 (audited)
> - dist_shift iid 64 NEW V: F1=+3.43 → MILP=-13.7 (audited, MASSIVE improvement)
> - 1484 → 1486 V/A in 1 hour
>
> Acasxu 0 NEW V via last-layer MILP — need multi-layer to break.
> Cifar DAG-blocked — need walker rewrite OR multi-layer MILP via FCHZState.
>
> Sweeps still running on safenlp 737, relusplitter 213, etc.
> Expecting 1486 → 1500-1550 by morning.
>
> 2000+ now achievable via:
>   1486 + safenlp/relusplitter MILP (+30-80)
>     + multi-layer MILP (+50-200, cifar/tiny/acasxu)
>     + cifar DAG MILP (+50-200)
>   → 1700-2000+ in 1-2 months
>
> The principle compromise is P3 only. P1 forward, P2 no gradient, P4 no input
> split, P5 no random — all PRESERVED. Open-source HiGHS only, no Gurobi.
>
> Paper claim: forward-only mechanism-portfolio verifier with adaptive
> bounded-binary MILP. Different from αβ-CROWN (backward) and nnenum
> (input split). Defensible contribution."
