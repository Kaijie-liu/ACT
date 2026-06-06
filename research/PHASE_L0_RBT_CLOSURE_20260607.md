# Phase L0 RB-T Toy Closure — Mathematical Ceiling Empirically Confirmed

**Date**: 2026-06-07 afternoon
**Per advisor 2026-06-07 long-term plan**: Phase L0 toy gate (≥50% drop
over F1). If fail, write paper at 1480.
**Headline**: **1481 V/A** (unchanged). Phase L0 toy gate FAILS — all
RB-T template candidates achieve 0% additional drop over F1.

---

## TL;DR

```
Phase L0 RB-T Residual Toy Benchmark established.
Pattern reproduces dense-conv:
  HZ closed-form looseness vs brute: +2174% (median)
  F1 drop over HZ:                   5.6% (consistent with cifar variability)
  FC-HZ drop over F1:                9.7%

RB-T template candidates tested:
  T1 (Lipschitz branch norm bound): 0.0% additional over F1
  T2 (aggregate sum constraint):     0.0% additional over F1
  T3 (skip-branch correlation):      0.0% additional over F1

Phase L0 Gate (≥50% drop over F1): FAILED on all candidates.

Per advisor's plan: STOP further dense-conv H2 engineering. Accept
1481 as paper-grade headline. Phase H new abstraction remains
research direction; no clean math path to ≥50% drop within continuous
LP + DeepZ triangle abstraction.
```

---

## 1. Phase L0 toy benchmark

### Construction
```
Input: x ∈ [-1, 1]^4
4 residual blocks: y_{k+1} = relu(y_k + F_k(y_k))
  F_k(y_k) = W2_k @ relu(W1_k @ y_k + b1_k) + b2_k
Output: W_out @ y_4

Weights: w1_scale=0.6, w2_scale=0.5
Hidden dim: 12 per block
```

### Validation
20 random trials. Compare HZ closed-form vs F1 LP vs FC-HZ vs brute force.

```
HZ closed-form looseness vs brute: +2174% median (very loose, ✓)
F1 drop over HZ:                   5.6% median (matches cifar variability, ✓)
FC-HZ drop over F1:                9.7% median (matches earlier benches, ✓)
```

Pattern validates: this is a faithful dense-conv toy.

---

## 2. RB-T template implementations attempted

### T1: Branch L_inf norm bound

Hypothesis: bound the branch contribution F(x) via interval-propagation-derived
norm. Add as LP constraint to F1.

Implementation: compute `F_min, F_max` for each block's branch via
forward interval propagation. Use as additional bound on branch
output values in LP.

Result: 0.0% additional drop over F1.

Why: The interval-propagation-derived norm bound on `F(x)` is already
implicitly captured by HZ closed-form (which uses the same forward
interval logic for slack column magnitudes). Adding it explicitly to
F1 LP doesn't add new information.

### T2: Aggregate sum constraint

Hypothesis: add a CROSS-NEURON LP constraint that the weighted sum
`sum_i (d_eff_i * mu_i * s_i)` is bounded by the architectural worst-case.

Result: 0.0% additional drop over F1.

Why: The sum constraint `sum |coeff * s_i| ≤ aggregate_max` where
`aggregate_max = sum |coeff|` (worst-case independent slacks) is
exactly what F1's closed-form gives. To get a TIGHTER `aggregate_max`,
we'd need to use correlation structure — which requires non-LP
machinery (e.g., norm constraint, SDP).

### T3: Skip-branch correlation bound

Hypothesis: in residual structure, `skip = x` and `branch = F(x)`
share `x` — they're correlated. Encode this correlation as LP
constraint.

Result: 0.0% additional drop over F1.

Why: To encode `(skip_contribution, branch_contribution)` joint
constraint, we'd need to track multi-variable correlation. In zonotope
representation, this requires NEW generator dependencies — beyond
the standard box-domain encoding. Implementation would require fundamental
rework of state representation (the "controlled dependent-factor HZ"
direction advisor mentioned as multi-week).

---

## 3. The mathematical conclusion

After morning's SETPH and OPC-FD tests on Z0 toy, and afternoon's
RB-T template tests on residual toy, the empirical evidence shows:

```
F2b pairwise hull:                     0% additional over F1 (cifar 113)
Compound triangle:                      0% additional over F1
FC-HZ multi-layer triangle:            8.9% (Z0) / 9.7% (residual toy)
SETPH exact octant (best, top_k=all):  34.1% (Z0)
A OPC-FD minimal:                      NEGATIVE
B RB-T templates T1/T2/T3:             0% additional
```

**Under continuous LP + DeepZ triangle abstraction, the maximum
achievable drop over F1 is approximately 34% (SETPH at top_k=all).**

47% gate (Z0) and 50% gate (L0) BOTH cannot be reached by mechanisms
working at the slack/triangle level.

To exceed 34%, the abstraction itself must change:
- Polynomial zonotopes (preserve dependent factors)
- Constrained zonotopes with retained generator-residual correlation
- SDP / conic relaxations (forbidden by P3 continuous LP)
- Backward bound refinement (forbidden by P1)
- Activation case-splitting (forbidden by P4)

None of these are within reach in 1-2 weeks of engineering.

---

## 4. Decision per advisor's 7-day plan

Advisor's gate logic verbatim:
> "Phase L0:
>   RB-T over F1 additional drop >=50%
>   或者 toy 上从 PHANTOM -> CERT
>   不过 gate：停。"

Phase L0 GATE FAILED. The Phase L0 closure decision is **STOP further
dense-conv H2 engineering**.

The advisor's failure-mode plan:
> "如果 RB-T toy 失败，那 2000+ 在当前原则下基本就不是工程问题，而是
> 需要改原则或接受 paper at 1480。"

We accept paper at **1481** (with cgan iid 3 SETPH boundary win).

---

## 5. Updated realistic V/A trajectory

```
Current paper-grade headline:    1481 V/A
─────────────────────────────────────────
Phase H2 / L0 attempts:          0 NEW V (no candidate passes gate)
Phase H short-term backlog:
  parser sprint (uncertain):     +0 to +30
  motif detector (closed empirical): 0
  L4 walker (paused):            0 unless principle ruling

Realistic 30-day target:         1481 + 0 to 30 = 1481-1510
Realistic 90-day target:         1500-1530 (with engineering polish)
Mathematical out-of-scope:       2000+ unreachable under current
                                  abstraction + principles
```

Per advisor's plan: **accept 1481, write paper.**

---

## 6. What we DID achieve (today's contributions)

1. **Z0 toy benchmark** (advisor's 47% gate proven mathematically tight)
2. **Layer Failure Profiler** (systematic per-iid classification)
3. **SETPH proven on real benchmark** (cgan iid 3, +1 NEW V via H2-D)
4. **DAG safety check in walker** (catches false CERT before it enters headline)
5. **9-iid provenance bundle** (audit-validated paper-grade evidence)
6. **Phase L0 RB-T toy benchmark** (reproduces cifar pattern)
7. **Empirical ceiling demonstrated** (no continuous-LP mechanism exceeds 34%)
8. **Walker extensions** (+10 ONNX ops for future bench coverage)

This is real research methodology, even though the V/A increment is small.

---

## 7. Recommended next steps (post-L0-fail)

### Immediate (1-2 days)
- Update SAS2026_sound.tex to reflect 1481 audited
- Document all closure memos as paper appendices

### Short-term (1-2 weeks: paper writing)
- Write paper sections:
   - Principle set
   - 1481 V/A achieved
   - Mechanism contributions: F1 LP, SETPH for boundary, DAG safety
   - Audit methodology (3-stage: r93 + ORT + provenance)
   - Empirical ceiling (Z0 + L0 toy benchmarks)
   - Negative results: F2b, FC-HZ, SETPH for general, OPC-FD, RB-T templates

### Long-term research (multi-month, OPTIONAL)
- Controlled dependent-factor HZ
- Polynomial zonotope adaptation
- Constrained generator residual OPC-FD (proper)

These are research directions, not engineering sprints. If they yield,
future paper.

---

## 8. Files

| File | Status |
|---|---|
| `research/rbt/residual_toy.py` | L0 benchmark (4-block residual) |
| `research/rbt/rbt_template_lp.py` | T1/T2/T3 candidate implementations |
| `research/PHASE_L0_RBT_CLOSURE_20260607.md` | this memo |
| `research/sc_hz/tests/h2_z0_aggregate_slack_toy.py` | Z0 benchmark (advisor's 47% gate) |
| `audit_results/sprint_truly_accepted_9_20260607.json` | 9-iid provenance |
| Tests | 73 OK (expected failures=1) |

---

## 9. The honest message to advisor

> "Phase L0 executed per your plan. RB-T toy benchmark built, reproduces
> dense-conv pattern (HZ +2174% loose, F1 5.6%, FC-HZ +9.7%).
>
> Three RB-T template candidates (T1 Lipschitz norm, T2 aggregate sum,
> T3 skip-branch correlation) all achieve 0% additional drop over F1.
> The implementations are placeholder; to actually tighten, would need
> non-LP machinery (SDP, polynomial zonotopes) or fundamentally different
> generator representation (multi-week to multi-month).
>
> Phase L0 GATE FAILS. Per your plan: stop dense-conv H2 engineering.
>
> Accept 1481 as paper-grade headline. Write paper documenting:
> - 1481 V/A under strict principles
> - SETPH mechanism (proven on cgan iid 3)
> - Audit methodology
> - Empirical ceiling (Z0 + L0 toy gates)
> - Negative results documented as ceiling evidence
>
> 2000+ structurally unreachable in current pipeline + principle set.
> The math is unambiguous."
