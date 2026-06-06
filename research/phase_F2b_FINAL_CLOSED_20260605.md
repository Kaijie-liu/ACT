# Phase F2b FINAL CLOSED — All 3 advisor gates run, decisive 0% on real cifar

**Date**: 2026-06-05 night
**Status**: F2b soundness-first rewrite complete, all gates run. F2b CLOSED
on definitive real-cifar evidence. Dense-conv short line CLOSED.
**Headline impact**: NONE (1472 holds)

---

## 1. Soundness-first rewrite per advisor 2026-06-05

Advisor 2026-06-05 directive: pause F2b scoring; run 3 gates BEFORE any
cifar verdict-able run.

### Gate A — Toy exact-hull benchmark (advisor's exact suggestion)

```
Toy:   z_1 = ξ_1+ξ_2, z_2 = ξ_1-ξ_2, max (y_1+y_2)
exact:                                    2.0
F1 per-neuron triangle LP:                3.0  (loose by 50%)
F2b pairwise joint cut LP:                2.0  (EXACT)
```

✓ PASS: F2b strictly tighter, achieving exact. Mechanism works when
binding pair correlation matters.

### Gate B — Convex-hull validity (5000 real samples)

For each derived cut, sample 5000 (ξ, ξ_tail) → real (z, relu(z))
points; check `cut.α_i·y_i + cut.α_j·y_j ≤ cut.rhs` holds.

✓ PASS: 0 violations across 5000 samples × derived cuts.

### Gate C — Monotonicity on larger synthetic (K=1000, n_pre=100)

✓ PASS: 0 widening violations in 5 trials.

### Gate D — Decisive real cifar 113 test (advisor's Step 4)

```
cifar 113 worst PHANTOM rival:
  n_unstable = 25, K = 29856, n_pre = 100
  HZ:                          +0.2613
  F1:                          +0.1458
  F2b top_k=4   (6 cuts):      +0.1458   (0%)
  F2b top_k=10  (45 cuts):     +0.1458   (0%)
  F2b top_k=20  (190 cuts):    +0.1458   (0%)
  F2b top_k=25  (300 cuts ALL unstable): +0.1458   (0%)
```

✗ FAIL by advisor's strict gate "≥1 NEW CERT OR median drop ≥40%": 0%.

## 2. Why F2b works on toy but fails on real cifar

**Toy 2-neuron**: only 1 pair to constrain; LP optimum MUST sit on that
pair. F2b cut binds at vertex of LP polytope.

**Real cifar 25-neuron**: LP optimum spreads contribution across many
unstable neurons. F1's per-neuron triangle LP at optimum has:
- α_i·y_i + α_j·y_j ≤ α_i·chord_i(z_i*) + α_j·chord_j(z_j*) = V_F1_pair
- F2b cut: α_i·y_i + α_j·y_j ≤ rhs = max_ξ (α_i·relu + α_j·relu)
- Since chord ≥ relu, V_F1_pair ≥ rhs ALWAYS
- But F1's LP optimum has α_i·y_i + α_j·y_j FAR SMALLER than V_F1_pair
   for the top pairs because LP spreads across all 25 unstable neurons
- All 300 pairwise cuts are SLACK at the F1 optimum

This is a STRUCTURAL property of dense-conv: when many neurons each contribute
modestly to the LP optimum, pairwise constraints don't bind. No matter how
tight per-pair cut is, the LP routes around it.

## 3. The 1472 ceiling reaffirmed

Two independent tests, two independent mechanisms confirm dense-conv 1472:
- F1 single-neuron triangle LP: 17% real, 44% peak; not enough
- F2b same-layer pairwise joint hull: 0% real gain even all-pair coverage

Both are STRUCTURALLY at floor of continuous-LP relaxation under principles.

Under advisor's 5 binding principles:
- P1 forward-only / P2 no gradient / P3 continuous LP / P4 no BaB / P5 no random
- **1472 IS the SC-HZ + DeepZ + continuous-LP ceiling for dense-conv**.

## 4. Advisor's directive — next steps

Per advisor's plan: "如果 F2b 失败 → 停止 dense-conv 短线，转 F3/F4."

### F3: small/control + parser + constrained LP

Targets:
- acasxu_2023, linearizenn_2024, ml4acopf_2024, tllverifybench_2023
- nn4sys, cctsdb_yolo_2023, traffic_signs (need parser work)
- cersyve (Slice/Reshape/Gather)
- metaroom (specific operators)

Estimated:
- Parser fixes: 3-5 days
- LP constrained on small dense: directly applies F1 (proven works)
- Payoff: +30 to +150 V/A (uncertain, parser-bound)

### F4: New HZ-style abstraction

Beyond current scope of 2-week kill switch. Multi-month research:
- forward constrained zonotope (full Ac matrix tracking)
- block-level ReLU group constraints
- spec-conditioned templates
- output projection through forward pass

## 5. Today's complete deliverables

| File | Purpose |
|---|---|
| `research/sc_hz/constrained_lp.py` | F1 LP (kept — 17% real tightening infra) |
| `research/sc_hz/constrained_lp_integration.py` | F1 walker hook (parity 0.00e+00) |
| `research/sc_hz/multi_neuron_hull.py` | F2b pairwise joint cut (kept as diagnostic only) |
| `research/sc_hz/tests/test_constrained_lp_prototype.py` | 6 F1 tests |
| `research/sc_hz/tests/test_multi_neuron_hull.py` | 4 F2b prototype tests |
| `research/sc_hz/tests/test_f2b_toy_validation.py` | 5 F2b soundness-first gates |
| `research/phase_F1_constrained_lp_prototype_20260605.md` | F1 design |
| `research/phase_F1_closed_F2_plan_20260605.md` | F1 closure + F2 plan |
| `research/phase_F2b_closed_dense_conv_ceiling_20260605.md` | F2b first-attempt closure |
| `research/phase_F2b_FINAL_CLOSED_20260605.md` | this memo — definitive close |

## 6. Test suite status

```
67/67 tests PASS  (52 prior + 6 F1 + 4 F2b proto + 5 F2b validation)
act/                clean
0 SC-HZ procs
~100 GB free RAM
```

## 7. Decision required

Pick:
- **(A) F3 parser/small-control** — uncertain +30/+150, 1-2 weeks
- **(B) Paper at 1472** — definite path, 1-2 weeks
- **(C) Both A then B in sequence** — extend kill switch by 1 week

I recommend **(C)**: start F3 parser fixes for 3-5 days (cheap, well-scoped),
re-assess at day 7. If F3 produces ≥30 NEW V/A on small-control, extend.
If not, immediately commit to paper at 1472+epsilon.
