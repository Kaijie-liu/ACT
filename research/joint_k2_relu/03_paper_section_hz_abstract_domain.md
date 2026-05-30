# HZ as an Abstract Domain (Paper §3 draft)

## 3.1 Background: abstract interpretation

Following Cousot & Cousot [1977], an abstract domain is a tuple `(A, ⊑, ⊥, ⊤, ⊔, ⊓, γ)` where:
- `A` is a set of abstract elements
- `⊑` is a partial order
- `⊥, ⊤` are bottom/top
- `⊔, ⊓` are join, meet
- `γ : A → 2^(ℝⁿ)` is the concretization function

For NN verification, the concrete domain is `2^(ℝⁿ)` (the powerset of reachable activation values), and abstract operators `f^♯` must be **sound**: `γ(f^♯(a)) ⊇ f(γ(a))` for all activation transformers `f` (Linear, Conv, ReLU, etc.).

CROWN, DeepPoly, ZONOTOPE, OCTAGON, POLYHEDRON are each cast as abstract domains. **HZ is not yet**. This paper closes that gap.

## 3.2 HZ as a set representation (recap)

A hybrid zonotope (Bird 2022) is a set
```
Z = { Gc·ξc + Gb·ξb + c | Ac·ξc + Ab·ξb ≤ b,  ξc ∈ [-1,1]^p, ξb ∈ {-1,+1}^q }
```
parameterised by the 6-tuple `⟨Gc, Gb, c, Ac, Ab, b⟩` plus an `eq_mask` flag for which constraint rows are equalities (`Ac·ξc + Ab·ξb = b`).

## 3.3 HZ as an abstract domain

### 3.3.1 Domain `A_HZ`

`A_HZ` = the set of HZ 6-tuples (modulo set equivalence): `(Gc, Gb, c, Ac, Ab, b, eq_mask)` over a fixed dimension `n` (the activation dimension at a particular layer). Note the dimensions `n_g, n_b, n_c` are variable across elements.

### 3.3.2 Concretization `γ_HZ`

```
γ_HZ(⟨Gc, Gb, c, Ac, Ab, b⟩) = Z (as defined in §3.2)
```

This is a partial function: `γ_HZ` is undefined if the underlying polytope is empty. In practice, `γ_HZ(⊥) = ∅`.

### 3.3.3 Partial order `⊑_HZ`

`Z_1 ⊑_HZ Z_2 ⇔ γ_HZ(Z_1) ⊆ γ_HZ(Z_2)`.

This is decidable via LP: `Z_1 ⊑ Z_2` iff `∀ z ∈ Z_1, ∃ ξ' ∈ feasible region of Z_2 : z = Gc'·ξc' + Gb'·ξb' + c'`. In general checking this requires bilinear programming; for canonical HZ inputs (e.g. one is a sub-representation of the other) it reduces to LP.

We do not need an EXACT decision procedure for verification — a sound `⊑_HZ` test (witness via LP) suffices.

### 3.3.4 Join `⊔_HZ` (sound over-approximation)

**Naïve join (Bird 2022)**: introduce one new binary `b'` linking two HZs. Doubles `n_b` per join.

**Bounded join (this paper, Theorem 3.1)**: for HZs `Z_1, Z_2` of the same dimension, define
```
Z_1 ⊔_HZ Z_2 := ⟨[Gc_1 | Gc_2 | (c_2 - c_1)/2],
                  [Gb_1 | Gb_2],
                  (c_1 + c_2)/2,
                  diag-block(Ac_1, Ac_2 augmented with new col),
                  diag-block(Ab_1, Ab_2),
                  [b_1; b_2]⟩
```
with carefully designed augmentation rows (proof in Appendix A). This costs `O(n_g_1 + n_g_2 + 1)` continuous generators and `O(n_b_1 + n_b_2)` binaries — no new binary.

**Soundness theorem**: `γ_HZ(Z_1 ⊔_HZ Z_2) ⊇ γ_HZ(Z_1) ∪ γ_HZ(Z_2)`.

(Proof: by construction the new generator column links the two centers via a continuous parameter in [-1, 1]; for parameter = -1 we recover Z_1, for +1 we recover Z_2.)

### 3.3.5 Meet `⊓_HZ` (exact for halfspace intersection)

For HZ `Z` and halfspace `H = {x : a·x ≤ d}`:
```
Z ⊓ H = ⟨Gc, Gb, c, [Ac; a·Gc], [Ab; a·Gb], [b; d - a·c]⟩
```
This is **exact** (Bird Theorem 3.4): adds one constraint row, no new generators.

For HZ ⊓ HZ, exactness requires the "generalized intersection" of Bird §3.3.3 with up to `n` new constraint rows.

### 3.3.6 Widening `∇_HZ` (this paper)

For fixpoint computation (RNN / GNN verification), define a widening operator `∇_HZ : A_HZ × A_HZ → A_HZ` satisfying:
1. **Soundness**: `γ_HZ(Z ∇_HZ Z') ⊇ γ_HZ(Z) ∪ γ_HZ(Z')`
2. **Termination**: any ascending chain `Z_0 ⊑ Z_0 ∇ Z_1 ⊑ ...` stabilizes in finite steps.

**Construction (Theorem 3.5)**: given two HZs `Z, Z'`:
1. Compute box envelopes `B(Z), B(Z')` (cheap, via interval bounds).
2. Define `Z ∇_HZ Z' := Z' if B(Z') ⊆ B(Z), else Z' ⊔_HZ box(B(Z) ⊔ B(Z'))`.
3. Termination: the box-component is monotonically refined, and box-domain widening is well-known to terminate.
4. Soundness: by construction the joined element contains both arguments.

This widening enables HZ-based fixpoint verification of recurrent / iterated computations — a NEW capability versus existing HZ literature.

### 3.3.7 Abstract operators

For each network operation, we define a sound abstract transformer:

| Operation | Abstract operator | Soundness |
|---|---|---|
| Linear `f(x) = Wx + b` | `f^♯(Z) = ⟨W·Gc, W·Gb, W·c + b, Ac, Ab, b_c⟩` | exact (Bird Prop 3.1) |
| Conv2D | flatten + Linear (block-Toeplitz) | exact, same proof |
| ReLU (exact, hz1) | adds 4 gens / 1 bin / 3 cons per neuron | exact |
| ReLU (triangle) | DeepZ relaxation: +1 gen, 0 bin, 0 cons | sound over-approx |
| **ReLU (joint K=2)** | **per-neuron + joint 8-direction envelope per pair** | **sound over-approx, this paper** |
| MaxPool | Sigmoid-like K-piece approx | sound over-approx |
| Sigmoid / Tanh | K-piece linearization (Bird hz4 OVERT) | sound over-approx |
| Add (skip) | block-diagonal `Gc` stack | exact |
| Concat | block-diagonal | exact |

### 3.3.8 Galois connection (open question)

A full Galois connection `(α_HZ, γ_HZ)` requires the existence of a tightest HZ over-approximation `α_HZ : 2^(ℝⁿ) → A_HZ`. For arbitrary sets in `ℝⁿ`, this is non-constructive — the set of all HZs is a complete lattice (under set inclusion), but not chain-complete in a computable sense.

We propose a **restricted Galois connection** for the subset `S ⊆ 2^(ℝⁿ)` of polyhedral unions: for any finite union of polyhedra, there is a least HZ-over-approximation, constructible in finite time.

For NN verification, every reachable set IS a finite union of polyhedra (since networks have piecewise-linear activations and inputs are polyhedra). So the restricted Galois connection covers the verification setting.

## 3.4 Soundness of the joint K=2 ReLU operator (new this paper)

**Theorem 3.6 (Joint K=2 sound)**: For any HZ `Z_in` with bounds `(lb, ub)`, any pair of unstable neurons `(i, j)`, and any direction `(a_i, a_j) ∈ ℝ²`:
```
sup{ a_i · max(0, x_i) + a_j · max(0, x_j) : (x_i, x_j, …) ∈ Z_in }
```
is an upper bound on `a_i · y_i + a_j · y_j` for any post-ReLU value `(y_1, …) ∈ relu(Z_in)`.

Therefore the augmented HZ
```
Z_out_aug = Z_out ∩ { y : a_i·y_i + a_j·y_j ≤ Φ_+(a_i, a_j, i, j) }
```
satisfies `γ(Z_out_aug) ⊇ relu(γ(Z_in))`.

**Proof**: by direct LP duality on the convex hull of the joint ReLU image. (See Appendix B.)

**Corollary (precision gain)**: when neurons `i, j` are anti-correlated through shared input generators, `Φ_+(1, 1, i, j) < u_i + u_j` strictly; the augmented HZ excludes the unreachable corner `(u_i, u_j)`.

## 3.5 Open theoretical questions

1. **Best join**: is `⊔_HZ` (§3.3.4) the tightest sound join in `A_HZ`? Conjecture: yes for the family without new binaries.
2. **K-neuron joint envelopes**: for `K ≥ 3` (Singh PRIMA-style), what is the maximum precision gain over `K=2`? Memory: prior experiments [Singh PRIMA k=2 negative on acasxu] suggest diminishing returns on small-dense, but conv networks may differ.
3. **Widening completeness**: does `∇_HZ` (§3.3.6) compute the EXACT least fixpoint for piecewise-linear recurrences? Or only a sound over-approximation?

These are deferred to future work.

## 3.6 Implementation status

- ✅ `joint_k2_envelope.py` implements §3.4 (joint K=2 sound ReLU operator)
- ✅ 3 unit tests + 8/8 regression pack PASS (soundness)
- ⏳ GPU 7-benchmark sweep (running)
- ⏳ Full §3.3.6 widening (deferred — pilot first on RNN benchmarks not in current scope)
- ⏳ Formal proof of Theorem 3.1 (bounded join) — outline only

## References

- Bird, T.J. (2022) "Hybrid Zonotopes: A Mixed-Integer Set Representation for the Analysis of Hybrid Systems", PhD Dissertation, Purdue.
- Cousot, P., Cousot, R. (1977) "Abstract Interpretation: A Unified Lattice Model".
- Ortiz, J., Vellucci, A., Koeln, J., Ruths, J. (2023) "Hybrid Zonotopes Exactly Represent ReLU Neural Networks", arXiv:2304.02755.
- Singh, G. et al. (2019) "PRIMA: General and Precise Neural Network Certification via Multi-Neuron Convex Relaxations".
- Anderson, R., Huchette, J., Tjandraatmadja, C., Vielma, J.P. (2020) "Strong mixed-integer programming formulations for trained neural networks".
- Zhang, Y. et al. (2022, 2023) "Reachability Analysis ... via Hybrid Zonotopes", arXiv:2210.03244 / 2303.10513.
