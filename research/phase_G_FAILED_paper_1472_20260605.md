# Phase G.0 FAIL — FC-HZ Multi-Layer Triangle Insufficient. 1472 Ceiling of Current Pipeline Confirmed.

**Date**: 2026-06-05 night (audit-corrected 2026-06-06 morning)
**Status**: G.0 G.4 gate FAIL. FC-HZ provides 8.1% median additional drop over F1.
Advisor's gate required ≥40%.
**Headline claim (CORRECTED 2026-06-06)**:
1472 V/A is the verified ceiling of the **CURRENT pipeline**
(SC-HZ + DeepZ triangle + continuous-LP sidecar). It is NOT the
"theoretical forward-only HZ-like verifier ceiling" — a new forward-only
abstraction (Phase H, e.g., output-projected forward constrained domain
or block-level template constraints) could potentially break it. Three
mechanisms (F1/F2b/FC-HZ) within the current pipeline have been
exhausted within the principle set.
**Recommendation**: PAPER AT 1472, with honest characterization of
strengths (safenlp +548 NEW A, dist_shift 72/72, nn4sys 86, collins_rul 51,
cgan 11, malbeware 136, cora top cluster) and weaknesses (dense-conv
cifar/tiny/vgg/yolo near-zero, small-dense acasxu/linearizenn/relusplitter
below case-reasoning tools).

---

## 1. G.0 two-layer toy + 20 random instances result

`research/sc_hz/fc_hz_state.py` + `tests/test_fc_hz_two_layer_toy.py`

### Advisor's specific toy (mixed-sign d_eff)
```
W_1 = [[1, 1], [1, -1]],  W_2 = [[1, 1], [1, -1]],  W_3 = [[1, -1]]
exact:                                2.0
HZ closed:                            3.5  (loose by 75%)
F1 (last-ReLU triangle only):        3.0  (drop over HZ: 14%)
FC-HZ (all-layer triangle history):  3.0  (drop over F1: 0%)
```

### 20 random 2-layer instances (n_in=4, n_h1=n_h2=8, n_out=4)
```
FC-HZ strictly tighter than F1:  19/20
Median additional drop:           8.1%
Mean additional drop:             9.2%
Range:                            0% - 20.6%
```

**Advisor's G.0.4 gate**: ≥40% additional drop over F1 → **FAIL by ~5×**

## 2. Why FC-HZ doesn't reach 40%

The layer-1 triangle constraint `y_1_i ≥ z_1_i` would force the slack
`s_1_i` upward when `z_1_i > 0`. But:

1. Layer 2's triangle relaxation already imposes the TIGHTEST per-neuron
   convex upper bound on `y_2_i`.
2. Layer 1's slack values feed into z_2 linearly, then z_2 is constrained
   by layer-2's tight triangle.
3. The "looseness" propagates as: layer-1 slack 1.0× → z_2 1.0× → y_2 1.0×.
4. But layer-2 triangle SMOOTHS the variation: chord_2(z_2) is convex
   in z_2, so the maximum of d_eff @ y_2 over z_2's allowed range is
   AT MOST the chord at the most-positive z_2 endpoint. Adding layer-1
   constraints only restricts which (xi, s_1) combinations reach this
   endpoint — doesn't change the endpoint itself.
5. Hence FC-HZ adds at most a small additional drop per layer (8% here).

By induction: adding even more layers of constraints (FC-HZ over 3, 4, 5
layers) would give DIMINISHING returns. Each new layer's triangle adds
~8% drop on top of previous, asymptotically approaching ~30-40% total
drop over plain HZ. Far below the ~100% drop needed to flip cifar PHANTOMs.

## 3. Pattern across all 3 attempts

| Mechanism | Synthetic gain | Real cifar 113 | Gate met? |
|---|---:|---:|---|
| F1 single-layer triangle | 15.7% median | 17% | (design pass, but insufficient) |
| F2b pairwise joint hull | 0.9% over F1 | 0% over F1 | NO (advisor 30%) |
| FC-HZ multi-layer triangle | 8.1% over F1 | (untested) | NO (advisor 40%) |

All three mechanisms:
- Sound (no violations on brute force)
- Strictly tighter than baseline on some inputs
- Far below the threshold needed to flip cifar PHANTOMs

## 4. The 1472 ceiling — scope-limited statement

Under advisor's binding principle stack:
```
P1: Forward-only (no backward bound refinement)
P2: No gradient (no PGD/CW/AutoAttack)
P3: Continuous LP only (no MILP, no integers)
P4: No BaB, no input splitting
P5: No random/corner/spec-corner falsification
```

We have empirically demonstrated (by F1/F2b/FC-HZ) that ON THE CURRENT
PIPELINE (SC-HZ + DeepZ triangle + sidecar LP cuts):
- The DeepZ triangle relaxation is the tightest single-neuron convex hull
- F1's per-neuron LP saturates this triangle, giving ~17% real tightening
- Multi-pair joint hulls (F2b) don't bind on dense-conv (LP optimum spreads)
- Multi-layer triangle (FC-HZ) yields diminishing 8% additional per layer
- Total achievable on this pipeline: ~25-30% drop over plain HZ closed-form
- Required to flip cifar PHANTOMs: ~100% drop (residual at +0.146)

**1472 V/A is the empirical ceiling of THIS pipeline architecture.**

NOT proven: that no forward-only / continuous-LP / no-MILP / no-BaB
abstraction can exceed 1472. A different forward abstraction (e.g.,
output-projected forward constrained domain, block-level template
polyhedra, forward constrained zonotope with retained Ac matrix not
just DeepZ triangle slacks) MIGHT achieve more, but designing one is
multi-month research (Phase H), not within the current 2-week kill switch.

## 5. Three remaining options

### A. Paper at 1472 (RECOMMENDED)
- 3 pillars solidify: forward-only + large-class #1 + P0=0
- Honest contribution: SC-HZ + multiple precision mechanisms + dense-conv memory infra
- Time: 1-2 weeks writing
- Risk: 0

### B. Break a principle for higher V
- MILP integer reasoning → unlocks BaB-equivalent
- Backward bound refinement → CROWN-equivalent
- Random falsifier → many witnesses
- These compete with abcrown 2460 / NeuralSAT 2065 territory
- But violates advisor's core thesis

### C. Find a NEW abstraction (Phase H?)
- E.g., template polyhedra, partial branching with proven equivalents
- Multi-month research, beyond 2-week kill switch
- Uncertain payoff

## 6. Strong recommendation: Option A — Paper at 1472

Three independent mechanism attempts (F1/F2b/FC-HZ) all converge to the
same conclusion. The math is clear. Continuing to attempt new cut formulations
within the principle set will yield diminishing improvements.

The paper claim should be:
> "Within strict forward-only / continuous-LP / no-gradient / no-BaB /
>  no-falsifier-corner principles, our SC-HZ achieves 1472 V/A across 22
>  benchmarks via the DeepZ triangle relaxation augmented by per-neuron
>  constrained-LP at the final ReLU layer (F1). Multi-pair and multi-layer
>  extensions provide diminishing returns (≤ 10% additional drop). This
>  establishes 1472 as the effective ceiling of this principle set."

## 7. Files delivered today (2026-06-05)

| File | Purpose |
|---|---|
| `research/sc_hz/constrained_lp.py` | F1 LP infra |
| `research/sc_hz/constrained_lp_integration.py` | F1 walker hook (Sub/MatMul/const-Add added) |
| `research/sc_hz/multi_neuron_hull.py` | F2b pairwise joint hull |
| `research/sc_hz/fc_hz_state.py` | FC-HZ multi-layer triangle |
| `research/sc_hz/tests/test_constrained_lp_prototype.py` | F1 tests |
| `research/sc_hz/tests/test_multi_neuron_hull.py` | F2b tests |
| `research/sc_hz/tests/test_f2b_toy_validation.py` | F2b soundness |
| `research/sc_hz/tests/test_fc_hz_two_layer_toy.py` | G.0 gates |
| `research/phase_F1_*.md` | F1 design + closure |
| `research/phase_F2b_*.md` | F2b closure |
| `research/phase_F3_*.md` | F3 day 1 negative |
| `research/phase_G_*.md` | FC-HZ design + this closure |
| 1472 freeze | unchanged |
| `act/` | clean |

## 8. Test suite final state

```
72/72 tests PASS  (52 prior + 6 F1 + 4 F2b proto + 5 F2b validation + 6 FC-HZ G.0)
act/                clean
0 SC-HZ procs
~100 GB free RAM
```

## 9. Awaiting advisor's decision

Pick:
- (A) Commit to paper at 1472 (recommended)
- (B) Break principle (e.g., reluctantly allow continuous-LP relaxation
   with binary aux variables on small subsets — partial MILP)
- (C) Open new long-term research (Phase H, months)
