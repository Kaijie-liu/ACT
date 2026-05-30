# Joint K=2 ReLU Encoding for HZ — Mathematical Derivation

## Setup

We have 2 unstable pre-activation neurons `(x_1, x_2)` in a hybrid zonotope:
```
x_i = c_i + Gc[i,:] @ ξ_c + Gb[i,:] @ ξ_b    (i=1,2)
ξ_c ∈ [-1,1]^p,  ξ_b ∈ {-1,+1}^q
Ac @ ξ_c + Ab @ ξ_b ≤ b   (with eq_mask)
```
Pre-activation bounds: `l_i ≤ x_i ≤ u_i` with `l_i < 0 < u_i` (UNSTABLE).

ReLU output: `y_i = max(0, x_i)`.

## Per-neuron baseline (hz1 / eq_lagr_v8)

For each neuron independently:
- Add 4 continuous generators (slack), 1 binary, 3 equality constraints (hz1 Theorem 1)
- Or eq_lagr_v8: tighter via Lagrangian + PEE elimination

Total for the pair: **8 gens + 2 bins + 6 cons** (before PEE).

After PEE: per-neuron eq_lagr_v8 eliminates redundant gens via shared ξ. Equivalent to forward CROWN-slope absorption (memory: `project_eq_elim_hero_20260515`).

## The precision question

**Claim**: Per-neuron + PEE captures cross-neuron correlation **through shared input ξ**. So joint K=2 encoding gives no extra precision unless we explicitly add **joint upper-envelope constraints** that are not implied by per-neuron + shared-ξ.

### Counter-example showing joint adds precision

Let `x_1 = ξ`, `x_2 = -ξ` (perfectly anti-correlated), `ξ ∈ [-1, 1]`.

Per-neuron triangle bounds:
- `y_1 ∈ [0, 1]` (from `x_1 ∈ [-1, 1]`)
- `y_2 ∈ [0, 1]` (from `x_2 ∈ [-1, 1]`)
- Product: `[0,1]² (area = 1)`

**True joint ReLU image** (by exhaustion of `ξ ∈ [-1, 1]`):
- `ξ ≥ 0`: `(y_1, y_2) = (ξ, 0)`, `y_1 ∈ [0, 1]`
- `ξ ≤ 0`: `(y_1, y_2) = (0, -ξ)`, `y_2 ∈ [0, 1]`
- Union = L-shape on the axes

**Convex hull of L-shape**: triangle `{(y_1, y_2) : y_1 + y_2 ≤ 1, y_1 ≥ 0, y_2 ≥ 0}` (area `1/2`).

**Per-neuron with shared ξ**: the LP relaxation over `(ξ_y1, ξ_y2, b_1, b_2)` with `ξ` shared between `x_1` and `x_2` SHOULD also recover the triangle, IF the per-neuron HZ correctly maintains the `x_1 = ξ, x_2 = -ξ` link.

Concretely the per-neuron encoding gives constraints:
- `y_1 ≥ 0, y_1 ≥ x_1 = ξ, y_1 ≤ u_1·(x_1 - l_1)/(u_1 - l_1) = (ξ+1)/2`
- `y_2 ≥ 0, y_2 ≥ x_2 = -ξ, y_2 ≤ (-ξ+1)/2`

Summing the two triangle upper bounds:
- `y_1 + y_2 ≤ (ξ + 1)/2 + (-ξ + 1)/2 = 1` ✓

**So per-neuron triangle ALREADY captures `y_1 + y_2 ≤ 1` through the shared `ξ`.**

### Conclusion of the anti-correlation example

Per-neuron triangle + shared ξ = joint triangle relaxation. **No precision gain** from K=2 joint encoding when the correlation is captured by an explicit shared `ξ`.

## When does joint K=2 add precision?

Joint K=2 adds precision **iff** the cross-neuron coupling is NOT representable through shared ξ. This happens when:

1. **PEE has dropped the shared ξ** (project_eq_elim eliminates generators; if both `x_1` and `x_2` were defined via the eliminated generator, their correlation is LOST after PEE).

2. **Girard reduction folded the shared ξ into the box envelope** (it gets summed into the diagonal slack at line 380 of `_hz_reduce_constraints`).

3. **Triangle ReLU at the previous layer** dropped the binary information: triangle gives a 4-gen relaxation that doesn't preserve which sign pattern was active, losing per-binary correlation.

**Hypothesis**: the GPU 0-verdict precision gap on conv 0-verdict comes from cases 1-3 happening at mid-network. After enough reductions, the per-neuron + shared-ξ correlation is gone.

## Joint K=2 encoding that survives reduction

To survive PEE/Girard, the joint encoding must add **EXPLICIT joint constraints in (y_1, y_2) space** — not just relying on shared ξ that may be eliminated.

For (x_1, x_2) with current bounds `[l_i, u_i]`:

**Sound joint upper envelope** (any direction `(a_1, a_2) ∈ ℝ²`):
```
a_1·y_1 + a_2·y_2 ≤ Φ(a_1, a_2)
```
where `Φ(a_1, a_2) = max{a_1·max(0,x_1) + a_2·max(0,x_2) : (x_1, x_2) ∈ input}`.

For the input being a box `[l_1, u_1] × [l_2, u_2]`:
```
Φ(a_1, a_2) = max(0, a_1)·u_1 · 𝟙[a_1 > 0] + max(0, a_2)·u_2 · 𝟙[a_2 > 0] · (joint correction)
```

Actually for an INDEPENDENT box, `Φ(a_1, a_2) = max(0, a_1·u_1) + max(0, a_2·u_2)` — the per-neuron triangle gives this already.

For a CORRELATED zonotope (not box), `Φ(a_1, a_2)` depends on the actual coupling. This is where joint K=2 can add precision **if** we encode the joint upper envelope directly:

```
y_1 + y_2 ≤ Φ_+(1, 1) = max{max(0,x_1) + max(0,x_2) : (x_1,x_2) ∈ Z_in}
y_1 - y_2 ≤ Φ_+(1, -1) = max{max(0,x_1) - max(0,x_2) : (x_1,x_2) ∈ Z_in}
... etc for 8 octant directions
```

These are SOUND joint cuts, computed once per pair by an inner LP over Z_in.

## Cost analysis

For a pair `(x_1, x_2)`:
- Per-neuron eq_lagr_v8 + PEE: ~10-20 ops total
- Joint K=2 with 8-direction envelope: 8 inner LPs (each <1ms for small Z_in)
- Extra constraints added: 8 (or fewer, after pruning trivial ones)
- Extra generators: 0 (the envelope is a polytope constraint, not a generator)

**Memory cost**: minimal (8 extra constraint rows per pair). 

**Compute cost**: 8 inner LPs per pair. For N unstable neurons paired in N/2 pairs, total 4·N inner LPs. Comparable to PEE's QR cost.

## Pairing heuristic

Which pairs of neurons benefit most? Two criteria:

1. **Strong correlation in input**: if `corr(x_1, x_2) ≈ ±1` based on current Gc rows, joint envelope is much tighter than product.
2. **Both unstable**: pairing a stable with an unstable neuron wastes a pair slot.

Concrete metric: `score(i, j) = |Gc[i] · Gc[j]| / (||Gc[i]|| · ||Gc[j]||)` (cosine similarity of generator coefficients). Pair greedy by descending score.

## Soundness statement

**Theorem (sound joint K=2 ReLU)**: For any pair `(x_1, x_2)` with current bounds and HZ generators `Gc[1:2, :], Gb[1:2, :]`, and any direction `a = (a_1, a_2)`:
```
Φ_+(a_1, a_2) = max{a_1·max(0,x_1) + a_2·max(0,x_2) : (x_1, x_2) ∈ HZ_input}
```
The constraint `a_1·y_1 + a_2·y_2 ≤ Φ_+(a_1, a_2)` is a sound over-approximation of the joint ReLU image.

**Proof**: For any `(x_1, x_2) ∈ HZ_input`, the pair `(y_1, y_2) = (max(0,x_1), max(0,x_2))` satisfies `a_1·y_1 + a_2·y_2 ≤ Φ_+(a_1, a_2)` by definition of Φ_+. □

## Implementation roadmap

1. **K=2 pairing**: greedy by Gc cosine similarity (compute once per ReLU layer)
2. **Joint envelope LP**: 8 directions, each an LP over the input HZ
3. **Add envelope rows to HZ**: 8 new constraint rows of form `[0, …, a_1, a_2, …, 0] · ξ_y ≤ Φ_+(a_1, a_2)` (where ξ_y are the new ReLU output generators)
4. **Wire into**: a new ReLU method `relu_eq_lagr_joint_k2` (extends `eq_lagr_v8` with the joint envelope)
5. **Soundness gate**: 8/8 regression pack
6. **GPU sweep**: 7 GPU 0-verdict benchmarks

## Open question — when to actually pair

If the pre-activation Gc has been REDUCED (Girard / PEE), per-neuron eq_lagr_v8 already captures the surviving correlation. Joint K=2 adds NEW joint constraints from the CURRENT bounds — these are valid additional cuts even if shared-ξ correlations were lost upstream.

**Therefore K=2 is most useful precisely when Girard/PEE has reduced the shared structure** — the LP relaxation of per-neuron + shared-ξ is loose, but the joint upper-envelope LP recovers some of the lost correlation.

This matches the empirical observation (multi-corner sidecar) that HZ output is "phantom" on conv 0-verdict — the reduced HZ's LP relaxation is too loose. Joint K=2 cuts can tighten it.
