# §2 — Background

## 2.1 Neural network verification

Given a network `N : ℝ^n_in → ℝ^n_out`, an input pre-condition `P ⊆ ℝ^n_in`, and an unsafe post-condition `U ⊆ ℝ^n_out`, the verification problem is to decide
```
∃ x ∈ P : N(x) ∈ U  ?
```
- **CERTIFIED (UNSAT, V)**: prove the answer is NO (`P` is safe under `N` w.r.t. `U`).
- **FALSIFIED (SAT, A)**: produce a witness `x* ∈ P` with `N(x*) ∈ U`.
- **UNKNOWN**: neither.

For piecewise-linear `N` (ReLU networks), the reachable set `N(P)` is a finite union of polyhedra. Verifying `N(P) ∩ U = ∅` is decidable but exponential in network depth/width in the worst case.

Sound over-approximating verifiers compute an abstract reachable set `R^♯ ⊇ N(P)`, then check `R^♯ ∩ U = ∅`. If yes → CERTIFIED. If no → either FAL (witness extracted + replayed) or UNKNOWN (LP-feasible but witness fails replay).

## 2.2 Abstract interpretation primer

Cousot & Cousot 1977 formalize sound static analysis via:

**Concrete domain**: `(C, ⊑_C)` where `C = 2^(ℝⁿ)` and `⊑_C = ⊆`.

**Abstract domain**: `(A, ⊑_A, ⊥_A, ⊤_A, ⊔_A, ⊓_A)` with partial order, bottom/top, join/meet.

**Concretization** `γ : A → C` is monotone and `γ(⊥_A) = ∅`, `γ(⊤_A) = ℝⁿ`.

**Abstraction** `α : C → A` is monotone with `α(γ(a)) ⊑_A a` and `c ⊑_C γ(α(c))`. Together `(α, γ)` form a **Galois connection** with `α ⊣ γ`.

**Sound abstract transformer** for concrete function `f : C → C`: any `f^♯ : A → A` such that `γ(f^♯(a)) ⊇_C f(γ(a))` for all `a ∈ A`.

**Widening** `∇ : A × A → A` for fixpoint computation: sound (`γ(a ∇ b) ⊇ γ(a) ∪ γ(b)`) AND termination-ensuring (any ascending chain `a_0 ⊑ a_0 ∇ a_1 ⊑ ...` stabilizes).

For NN verification, existing abstract domains include:
- **Interval / box** (Goubault 2001): `a = (l, u) ∈ ℝ^n × ℝ^n`, `γ(a) = [l_1, u_1] × ... × [l_n, u_n]`.
- **Zonotope** (Singh et al. 2018, DeepZ): `a = (c, G)`, `γ(a) = c + G·[-1, 1]^p`.
- **Polyhedron** (Cousot & Halbwachs 1978): `a = {x : Ax ≤ b}`.
- **CROWN** (Zhang et al. 2018): bound-propagation domain with per-neuron linear lower/upper bounds.
- **DeepPoly** (Singh et al. 2019): backwards-aware polyhedral abstraction.
- **Star sets** (Tran et al. 2019): bounded linear sets `c + G·ξ` with predicate constraints.

Each of these is formally framed as an abstract domain with Galois connection and sound transformers.

## 2.3 Hybrid zonotopes (Bird 2022)

A **hybrid zonotope** (HZ) is the set
```
Z = { Gc·ξ_c + Gb·ξ_b + c : Ac·ξ_c + Ab·ξ_b ≤ b, ξ_c ∈ [-1,1]^p, ξ_b ∈ {-1,+1}^q }
```
parameterised by the 6-tuple `(Gc, Gb, c, Ac, Ab, b)` plus an `eq_mask` flag specifying which constraint rows are EQUALITIES (`=`) versus inequalities (`≤`).

`Gc` is the **continuous generator** matrix, `Gb` the **binary generator** matrix, `c` the center. `(Ac, Ab, b)` define the **constraint polytope** in mixed-integer factor space.

HZ generalizes:
- `Gb = 0, Ac = 0, Ab = 0, b = 0` ⇒ pure **zonotope**
- `Ac = I, Ab = 0, b = c-coordinate` ⇒ **constrained zonotope**
- `Gc = 0, Gb = I` ⇒ **vertex form** of an integer polytope

The binary generators give HZ the ability to represent **non-convex** sets (specifically, unions of polytopes parameterised by binary patterns) — critical for exact ReLU representation.

### 2.3.1 Exact ReLU encoding (hz1, Ortiz 2023)

For an unstable ReLU `y = max(0, x)` on `x ∈ [l, u]` with `l < 0 < u`, the exact HZ representation uses:
- 4 new continuous generators (slack)
- 1 new binary generator (sign of x)
- 3 new equality constraints

The binary generator `b` encodes `b = +1 ⇒ x ≥ 0 (active)` and `b = -1 ⇒ x ≤ 0 (inactive)`. The associated constraints enforce `y = x` when active and `y = 0` when inactive.

For a network with `N` unstable neurons, this gives `4N` continuous generators, `N` binaries, `3N` constraints — **linear in network size**. The LP relaxation of this encoding (binary `b ∈ [-1, +1]`) gives the **triangle relaxation** of DeepZ per neuron, plus shared-ξ cross-neuron correlation.

### 2.3.2 Closed-form set operations (Bird 2022)

| Operation | HZ formula | Complexity |
|---|---|---|
| Linear map `RZ` | `⟨R·Gc, R·Gb, R·c, Ac, Ab, b⟩` | exact, O(1) |
| Minkowski sum `Z₁ ⊕ Z₂` | block-diagonal `Gc, Gb`, stacked constraints | exact, additive |
| Halfspace ∩ `{x : a·x ≤ d}` | adds 1 constraint row | exact, +1 row |
| Generalized intersection ∩_R | adds n constraint rows | exact, +n rows |
| Cartesian product | block-diagonal everywhere | exact |
| Union ∪ | 1 new binary | exact, +1 binary |

These set operations make HZ a natural fit for forward propagation through linear and ReLU layers.

### 2.3.3 LP-relaxation feasibility

For a target unsafe set `U = {y : C·y ≤ d}`, the question "is `Z ∩ U` non-empty in the LP relaxation?" reduces to:
```
∃ ξ_c ∈ [-1,1]^p, ξ_b ∈ [-1,1]^q (relaxed):
    Ac·ξ_c + Ab·ξ_b ≤ b
    AND
    C·(Gc·ξ_c + Gb·ξ_b + c) ≤ d
```
This is a single LP. If infeasible → `Z ∩ U = ∅` in the relaxation, hence in the original (sound). If feasible → a candidate witness `xi* = (ξ_c*, ξ_b*)`; either ORT-replay confirms it or the verdict is UNKNOWN.

This is the **HZVerifier Phase 2** check (§5).

## 2.4 Why HZ is not yet an abstract domain in the Cousot sense

The existing HZ literature (Bird 2022, Ortiz 2023, Zhang 2022-23) defines:
- The 6-tuple representation ✓
- Sound set operations (linear, intersection, union, ReLU) ✓
- LP-based emptiness checks ✓

But it does NOT define:
- A **formal partial order** `⊑_HZ` over HZ representations
- A **Galois connection** between `2^(ℝⁿ)` and the HZ lattice
- A **bounded join** that avoids the binary-count explosion of naïve union
- A **widening operator** with termination proof — essential for fixpoint analysis (recurrent / iterated computations)
- A **systematic enumeration of sound abstract transformers** for all NN ops

The present paper (§3) closes these gaps. It defines `⊑_HZ`, the join `⊔_HZ`, the widening `∇_HZ`, and proves soundness of new abstract transformers including joint K=2 ReLU (§4). HZ thus becomes the **first mixed-integer abstract domain** with Cousot-style formalism.

## 2.5 Notation summary

| Symbol | Meaning |
|---|---|
| `Z, Z_1, Z_2` | HZ representations or their concretizations |
| `(Gc, Gb, c, Ac, Ab, b)` | HZ 6-tuple |
| `ξ_c ∈ [-1, 1]^p` | continuous factors |
| `ξ_b ∈ {-1, +1}^q` | binary factors (relaxed to `[-1, 1]^q` for LP) |
| `n` | dimension (output of current layer) |
| `n_g, n_b, n_c` | continuous/binary generator counts, constraint row count |
| `γ(Z)` | concretization (set of points) |
| `α` | abstraction function |
| `⊑, ⊔, ⊓` | order, join, meet (abstract) |
| `⊆, ∪, ∩` | concrete subset, union, intersection |
| `∇` | widening |
| `f^♯` | abstract transformer for concrete `f` |
| `eq_mask` | bool array marking equality constraint rows |
