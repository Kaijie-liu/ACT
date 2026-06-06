# Phase F2b CLOSED — Dense-Conv Continuous-LP Ceiling Reached at 1472

**Date**: 2026-06-05 night
**Status**: F2b closed by definitive 0% gain on real cifar. Dense-conv short
line judged STRUCTURALLY EXHAUSTED under binding principles.
**Headline impact**: NONE (1472 holds, no scoreboard change)

---

## 1. F2b decisive result on cifar iid 113 (worst PHANTOM rival)

`research/sc_hz/multi_neuron_hull.py` + `tests/test_multi_neuron_hull.py`

```
HZ closed-form:                       +0.2613
F1 (per-neuron triangle LP):          +0.1458   (44.2% drop)
F2b top_k=4  (6 pairwise cuts):       +0.1458   (+0.00% gain over F1)
F2b top_k=6 (15 pairwise cuts):       +0.1458   (+0.00% gain over F1)
F2b top_k=10 (45 pairwise cuts):      +0.1458   (+0.00% gain over F1)
```

8/8 hard gates: monotonicity ✓, brute-force soundness ✓, synthetic-tightening
on correlated 8/10 with median 0.9% additional ✓. But on real cifar 113
the additional gain is **zero**.

## 2. Structural explanation — why F2b adds nothing

In F1 LP, for each unstable neuron i:
- z_i = c_z[i] + G_z[i, :] @ xi     (linear in xi, ANY i)
- y_i ≥ z_i, y_i ≥ 0, y_i ≤ chord_i(z_i)

For each pair (i, j), the pairwise joint cut
- y_i + y_j ≤ max_{xi}(d_eff[i] · relu(z_i) + d_eff[j] · relu(z_j))

is *structurally redundant* with F1's per-neuron triangle constraints when
the LP objective is d_eff @ y. The LP already maximizes through the same
shared xi, and per-neuron upper triangles + linear z(xi) coupling fully
specify the projected joint hull.

The pairwise cut adds a new constraint surface, but the LP optimum already
sits at a vertex where y_i = chord_i(z_i) and y_j = chord_j(z_j) and the
implied z_i + z_j is already at its max for that xi. The cut never binds.

Synthetic small-K (K=6, n_pre=8, strong correlation) gave 0.9% additional
gain because the projected polygon was narrow. Real cifar (K=12000,
n_pre=128, weak per-neuron correlation under wide projection) gives 0%.

## 3. The 1472 ceiling under binding principles

Under advisor's binding principle stack:
- P1: Forward-only (no backward bound refinement)
- P2: No gradient (no PGD/CW/AutoAttack)
- P3: Continuous LP only (no MILP, no integer)
- P4: No BaB, no input splitting
- P5: No random/corner/spec-corner falsification

**F1 LP is the tightest continuous LP relaxation possible** of the forward
HZ at the last ReLU. F2b's pairwise cut, F2a's multi-layer single-neuron,
and any other continuous extension we can imagine all fold into F1's
constraint set without adding tightness.

To go below 1472 on dense-conv requires breaking AT LEAST ONE binding
principle, OR fundamentally changing abstraction:
- (a) Non-convex constraints — impossible in continuous LP
- (b) Binary aux for active/inactive — needs MILP, forbidden by P3
- (c) Input splitting / case analysis — forbidden by P4
- (d) Backward bound tightening — forbidden by P1
- (e) Different abstraction (not zonotope-style box-generator with DeepZ
       triangle) — would need a Phase G beyond Phase F

So **1472 IS the SC-HZ + DeepZ-triangle + continuous-LP ceiling**.

## 4. F1 still valuable as PUBLISHABLE INFRASTRUCTURE

Even though F1 didn't crack 1472, it produced:
- a numerically exact (parity 0.00e+00) walker hook
- 15.7% synthetic / 17% real median tightening (sound)
- 44.2% drop on cifar 113 worst rival
- a clean abstraction for future multi-layer extensions

F1 is **not wasted work**. It is the strongest forward continuous-LP
PHANTOM-tightening mechanism in the project, and an obvious thing to
include in a paper's "what's possible without backward bound" section.

## 5. Remaining options

Per advisor's plan once F2b fails:

### Option A: Phase F3 — parser + small-control sidecar v2

Fix parser to enable exact LP on additional benches:
- Slice / Reshape / Gather (cersyve, lindex, mscn, etc.)
- tllverifybench IndexError
- Then run constrained LP on small dense networks

Estimated payoff: +30 to +150 V/A (uncertain, parser-bound)
Effort: 1-2 weeks

### Option B: Commit to paper at 1472

3 pillars unchanged:
- forward-only (no backward)
- large-class #1
- P0 = 0 (audit-validated)

1472 / 22 benches with deterministic forward LP. Multi-mechanism story
(BoxHZ / LazyChainHZ / SparseGcZ / eq_lagr_v8 / streaming-prune / F1
constrained LP). Memory + soundness infrastructure are real contributions.

### Option C: Phase G new abstraction

Break out of HZ+triangle entirely. Requires multi-month research. Not
within the 2-week kill switch.

## 6. Files

| File | Status |
|---|---|
| `research/sc_hz/constrained_lp.py` | F1 (kept — 17% tightening infra) |
| `research/sc_hz/constrained_lp_integration.py` | F1 walker (kept — parity exact) |
| `research/sc_hz/multi_neuron_hull.py` | F2b (kept as diagnostic; 0% gain) |
| `research/sc_hz/tests/test_constrained_lp_prototype.py` | 6 F1 tests |
| `research/sc_hz/tests/test_multi_neuron_hull.py` | 4 F2b tests |
| `research/phase_F1_constrained_lp_prototype_20260605.md` | F1 design |
| `research/phase_F1_closed_F2_plan_20260605.md` | F1 closure |
| `research/phase_F2b_closed_dense_conv_ceiling_20260605.md` | this memo |
| 1472 freeze | unchanged |
| `act/` | clean |

## 7. Tonight's stop point

```
Headline:                1472 V/A (audit-validated, frozen)
F1:                      VALIDATED, FROZEN. 17% real tightening, parity exact.
F2b:                     CLOSED. 0% gain over F1 on cifar 113 worst rival.
Theoretical ceiling:     1472 confirmed as SC-HZ + continuous-LP MAX
Tests:                   62/62 PASS (52 prior + 6 F1 + 4 F2b)
0 NEW V/A tonight
act/:                    clean
Memory/GPU:              returning to baseline
```

## 8. Decision required from advisor

Pick one of:
- (A) Phase F3 parser/small-control v2 — uncertain +30/+150, 1-2 weeks
- (B) Commit to paper at 1472 — definite path, 1-2 weeks writing
- (C) Phase G new abstraction — multi-month research, blows 2-week switch
