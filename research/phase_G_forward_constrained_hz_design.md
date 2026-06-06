# Phase G — Forward Constrained HZ (FC-HZ) Design

**Date**: 2026-06-05 night
**Status**: DESIGN. To be executed if F3 day-5 < 30 NEW V/A.
**Goal**: Break dense-conv 1472 ceiling. Long-term: 2000+.

---

## 1. Diagnosis (definitive, post-F2b)

F2b proved: pairwise multi-neuron joint hull at same layer is washed out
by LP optimum spreading across 25+ unstable neurons. The remaining
LOOSENESS is AGGREGATE: many neurons each contribute small slack,
summing to +0.146 PHANTOM.

The fundamental limitation of DeepZ triangle:
- Each ReLU's slack ξ_aux is treated as INDEPENDENT in [-1, 1]
- This is the per-neuron triangle relaxation upper bound
- F1 adds correlation through shared z_i = c_i + G_i @ ξ_root, gaining 17%
- F2b adds pairwise joint cuts, gaining 0% additional on cifar
- ALL future per-neuron and per-pair cuts will fail for the same reason

To break this, the abstraction must encode **constraints AMONG slacks**
(not just within slacks). This is FC-HZ.

## 2. FC-HZ definition

### Current HZ (SC-HZ + DeepZ triangle):
```
State: (c, G_kept, tail, ReLU_aux_metadata)
Set:   { y = c + G_kept · ξ + tail · ξ_tail :
         ξ ∈ [-1, +1]^K, ξ_tail ∈ [-1, +1]^n }
```

### FC-HZ:
```
State: (c, G_kept, tail, A_c, B_s, b)
Set:   { y = c + G_kept · ξ + tail · ξ_tail :
         ξ ∈ [-1, +1]^K, ξ_tail ∈ [-1, +1]^n,
         A_c · ξ + B_s · s ≤ b   (LINEAR CONSTRAINT MATRIX)
       }
```

where:
- A_c: constraint matrix on ξ (root + earlier-layer slacks)
- B_s: constraint matrix on the current layer's slack s
- b: RHS vector

The constraint `A_c · ξ + B_s · s ≤ b` captures the "aggregate slacks
cannot all simultaneously be at worst" property that pairwise cuts
fail to.

## 3. ReLU operator in FC-HZ

For an unstable neuron i at layer L:
- Current DeepZ: y_i = λ_i z_i + μ_i + μ_i ξ_aux, ξ_aux ∈ [-1, 1]
- FC-HZ: y_i = λ_i z_i + μ_i + μ_i s_i, with s_i ∈ [-1, +1] AND:
   - Triangle constraint as linear inequality:
      y_i ≥ 0  →  λ_i z_i + μ_i + μ_i s_i ≥ 0  →  -μ_i s_i ≤ λ_i z_i + μ_i
      y_i ≥ z_i →  λ_i z_i + μ_i + μ_i s_i ≥ z_i →  ...
   - These become rows in the augmented constraint matrix

The KEY insight: each ReLU layer adds a SHARED constraint block, not
independent slacks. As we propagate forward, A_c, B_s, b GROW
column-wise and row-wise. By the output:
- The output value is still a linear function of (ξ, s)
- But ξ and s are constrained by all accumulated rows
- LP UB at output is the CONSTRAINED LP, tighter than independent slacks

## 4. Soundness sketch

The reachable set under DeepZ triangle is a SUBSET of the FC-HZ set
because FC-HZ adds constraints (constraints can only shrink the set).
But FC-HZ MUST still contain the TRUE reachable set:
- True set: y_i = relu(z_i), achievable for some ξ values
- DeepZ triangle: y_i ∈ triangle envelope
- FC-HZ: same envelope + linear correlation
- For any TRUE point, set s_i := 2·(y_i - λ_i z_i - μ_i)/μ_i so y_i is
   recovered, and all triangle inequalities hold (because y_i ≥ 0,
   y_i ≥ z_i, y_i ≤ chord(z_i) hold for true points by definition).

Therefore FC-HZ ⊇ true reachable set ⊆ DeepZ envelope. FC-HZ is a SOUND
TIGHTENING of DeepZ.

## 5. LP UB in FC-HZ

For an unsafe rival d_out:
```
max d_out · y_out
s.t.  y_out = c_out + G_out · ξ + tail_out · ξ_tail
      ξ ∈ [-1, +1]^K
      ξ_tail ∈ [-1, +1]^n
      A_c · ξ + B_s · s ≤ b
```
where s is implicitly part of ξ in the LP formulation (just extend ξ to
include slack variables, with A_c augmented accordingly).

Solver: HiGHS continuous LP. No MILP, no integers, no backward.

## 6. Implementation phases

### Phase G.0 — TWO-LAYER ReLU toy (CRITICAL CORRECTION 2026-06-05)

The original G.0 said "50 same-layer correlated ReLUs". This is WRONG —
F1 already constrains a single ReLU layer's triangle. A single-layer toy
proves nothing FC-HZ-specific.

**Mandatory structure**: two ReLU layers with non-trivial linear mixing:
```
x ∈ [-1, 1]^n_in
z_1 = W_1 · x      → ReLU → y_1
z_2 = W_2 · y_1    → ReLU → y_2
output = W_3 · y_2
```

**Concrete toy** (advisor's pair extended to 2 layers):
```
W_1 = [[1, 1], [1, -1]]    # 2 → 2
W_2 = [[1, 1], [1, -1]]    # 2 → 2
W_3 = [[1, 1]]             # 2 → 1
x ∈ [-1, 1]^2
objective = y_2_1 + y_2_2
```

**Gates**:
- Brute force exact: compute via grid sampling
- HZ closed-form: must be LOOSE (significantly > exact)
- F1 (constrain LAST ReLU only): must be still LOOSE (> exact, < HZ)
- **FC-HZ (constrain BOTH ReLUs)**: must achieve ≥40% additional drop
   relative to F1's improvement over HZ
- Brute-force samples all satisfy FC-HZ UB (soundness)
- FC-HZ UB ≤ F1 UB on all 20 random 2-layer instances (monotonicity)

**Hard rule**: single-layer toy passing is NOT enough — FC-HZ must beat F1
on 2-layer structure to be a real mechanism. If 2-layer toy doesn't show
≥40% additional drop over F1, FC-HZ is functionally equivalent to F1 and
this entire Phase G mechanism is bust.

### Phase G.1 — Implement FC-HZ ops
- BoxHZ initial: same as current
- Affine ops (Dense, Conv): same as current (linear ops just transform A_c)
- BatchNorm: same as Dense
- ReLU triangle with constraint accumulation: NEW
- Reduction operators (PEE, Girard): need extension

### Phase G.2 — cifar 113 gate
- F1 excess +0.1458
- FC-HZ target: ≤ +0.05 OR direct CERT
- If FC-HZ excess > +0.10 → close G

### Phase G.3 — 8 near-CERT sentinel
- median drop ≥60% OR ≥1 NEW CERT
- Else close G

### Phase G.4 — 40 sentinel
- ≥5 NEW V/A OR clear margin shift
- Else close G

### Phase G.5 — Full sweep
- Conditional on G.4 pass

## 7. Effort estimate

| Phase | Days |
|---|---|
| G.0 toy aggregate | 1.0 |
| G.1 implement ops | 4.0 |
| G.2 cifar 113 single test | 0.5 |
| G.3 8 sentinel | 1.0 |
| G.4 40 sentinel | 1.5 |
| G.5 full sweep | 3.0 |
| **Total** | **~11 days** |

Within 2-month timeline if F3 day-5 closes by day 7.

## 8. Hard kill switches

- G.0 toy doesn't tighten → close G, accept 1472 ceiling, paper
- G.2 cifar 113 stays > +0.10 → close G, paper
- G.3 8 sentinel < 1 NEW → close G, paper

## 9. Risks

1. **Constraint matrix size**: each ReLU layer adds 2n constraints (one
   per unstable neuron for upper and lower triangle). After 5 ReLU
   layers and 100 unstable per layer, ~1000 constraints. LP size
   becomes (K + 100×n_unstable) variables and ~1000 constraints.
   Solvable by HiGHS but slow.

2. **Numerical conditioning**: stacked equality/inequality rows from
   triangle constraints may produce ill-conditioned LPs. May need
   preconditioning or regularization.

3. **Reduction operators incompatible**: PEE/Girard assume box-domain
   slacks. Need new reduction respecting A_c constraints.

## 10. Why this is different from F1/F2b

| | F1 | F2b | FC-HZ |
|---|---|---|---|
| Per-neuron triangle | yes | yes | yes |
| Linear coupling via z | yes (shared ξ) | yes | **explicit Ac matrix** |
| Pairwise hull cuts | no | yes (heuristic) | no need (subsumed by Ac) |
| Aggregate slack correlation | NO | NO | **YES via Ac** |
| LP relaxation | per-neuron | + pair cuts | **full polytope** |
| Toy 2-neuron | exact | exact | exact |
| Cifar 25-neuron | 17% | 0% | **tested at G.2** |

FC-HZ is the right next step BECAUSE F2b proved pair cuts don't help,
and the only mechanism that could is full Ac.
