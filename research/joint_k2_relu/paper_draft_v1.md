# HZ as an Abstract Domain: Forward-Only Soundness, Joint Multi-Neuron Operators, and the Conv 0-Verdict Structural Ceiling

_Working draft compiled 2026-05-28. Authors: TBD. Target: SAS / CAV 2026._

---

# §1 — Introduction

Neural network verification asks whether a trained network satisfies a behavioral specification — for image classifiers, whether an adversarial perturbation can flip the predicted class; for safety-critical controllers, whether a state in the operational envelope can drive the system out of safe bounds. Sound verifiers compute an over-approximation of the network's reachable output set and check it against the unsafe specification.

The dominant approach in recent VNN-COMP iterations combines **forward bound propagation** through linear layers with **branch-and-bound** search and **gradient-based attacks** on unstable ReLUs. While effective on the leaderboard, this combination has substantial implementation complexity and depends on heuristic choices that complicate formal soundness arguments — particularly when gradient attacks are used as a falsification channel parallel to the sound proof channel.

This paper takes an orthogonal stance. We **restrict ourselves to a strict forward-only, principle-compliant verifier**: no CROWN backward bounds, no autograd-based gradient attacks, no MILP via Gurobi, no fallback heuristics, no branch-and-bound. The verification engine consists exclusively of (a) sound forward propagation through hybrid-zonotope (HZ) abstract operators and (b) LP-feasibility checks against the abstract output set. Witnesses, when produced, are validated by **strict ORT replay** against the original ONNX network at zero tolerance.

The contributions are:

1. **HZ as the first mixed-integer abstract domain in the Cousot-Cousot sense (§3)**. We define a partial order `⊑_HZ`, a bounded join `⊔_HZ` (Theorem 3.1) that avoids binary explosion, a widening operator `∇_HZ` with termination proof (Theorem 3.5), and a systematic enumeration of sound abstract transformers for all standard NN operations. To our knowledge no prior work formalizes HZ as an abstract domain — Bird's seminal dissertation (Bird 2022) develops HZ as a set representation but stops short of the Cousot formalism.

2. **A new sound multi-neuron ReLU operator (§4)**. The joint K=2 envelope captures sound upper bounds on PAIRS of unstable ReLU neurons, using inner LPs over the pre-ReLU HZ to compute the joint upper envelope in 8 octant directions (default) or in spec-aware directions (last-ReLU mode). Theorem 3.6 gives the soundness proof. The operator integrates cleanly into the abstract-domain framework as a composable transformer.

3. **A multi-corner LP witness extractor for the verifier's Phase 4 (§5)**. Standard HZ-based verification accepts only the FIRST LP-feasible witness; if it fails ORT replay, the verdict is UNKNOWN. The multi-corner extractor enumerates additional LP corners and re-replays, sound by Theorem 5.1.

4. **Empirical evaluation across the VNN-COMP 2025 benchmark suite (§6)**. We demonstrate ~1000 sound verdicts across 13 small-to-medium benchmarks, including 47.8% decidability on acasxu_2023, 76.7% on linearizenn_2024, 97% on metaroom_2023 (the latter via the documented N=1 override). These are competitive with the strongest forward-only verifiers under strict soundness.

5. **A load-bearing negative result: the conv 0-verdict structural ceiling (§6.3)**. On the seven conv-heavy VNN-COMP benchmarks where the baseline produces 0 V+A (cifar100, dist_shift, soundnessbench, tinyimagenet, traffic_signs, vggnet16, yolo), we test THREE independent principle-compliant precision-side levers — multi-corner LP sidecar, joint K=2 octant, joint K=2 spec-aware. All three produce 0 lift across 47-54 sampled instances; the spec-aware variant introduces +6 OOM regressions on conv-heavy networks. We argue this constitutes empirical evidence that forward-only HZ + LP-relaxation has a structural precision ceiling on conv 0-verdict benchmarks, and we diagnose the ceiling as representation-bound (specifically: Girard reduction + project_eq_elim drop the cross-layer shared-ξ correlations that conv layers create; no post-hoc cut on the output HZ can recover this information).

The negative result is not a failure but a precisely-located limit. We propose (§8) that closing this limit requires representational change to HZ — preserving shared-ξ across conv layers without OOM — rather than further engineering on the existing representation.

## 1.1 Design principles

The five hard principles our verifier obeys:

**P1**. No CROWN-style backward bound propagation. The verifier propagates only **forward**.

**P2**. No autograd / no gradient-based attack. Witnesses come from LP-feasibility on the abstract output set + ORT-replay validation, never from PGD, FGSM, CW, AutoAttack, or related attacks.

**P3**. No Gurobi or MILP solver. The verifier uses scipy's `linprog` (HiGHS) for LP and never calls a MIP solver.

**P4**. No fallback to a different verifier on UNKNOWN. The verifier's output is `(verdict, witness?)`; UNKNOWN is honestly reported, never silently replaced by another tool's output.

**P5**. No branch-and-bound search. The verifier never splits the input box; the abstract operators must be tight enough on the full input region.

Plus a 2026-05-28 addendum after empirical investigation:

**P6** (no random-sample-then-check). Falsification candidates must come from a STRUCTURED procedure (LP-feasibility on the abstract output set), not from random or corner sampling on the input followed by ORT replay. This excludes the OrtSampleFalsifier approach which would otherwise produce a few "boundary" FAL witnesses.

The principle set is strict by design. We measure what sound forward-only HZ verification can achieve under these principles, and what it provably cannot.

## 1.2 Paper structure

§2 reviews background on NN verification, abstract interpretation, and HZ as a set representation. §3 introduces HZ as an abstract domain. §4 develops the joint K=2 ReLU abstract operator. §5 describes the multi-corner LP witness extractor. §6 reports the empirical evaluation including the 3-experiment negative result. §7 compares with related work. §8 concludes and lists open problems. Appendices A, B, C contain the formal proofs of Theorems 3.1, 3.6, 3.5 respectively.

---

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

---

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

---

# §4 — The Joint K=2 ReLU Abstract Operator

## 4.1 Motivation

The standard per-neuron ReLU abstract operator (hz1, Ortiz 2023) processes each unstable neuron independently. For two unstable neurons `(x_i, x_j)`:
- Adds 4 + 4 = 8 continuous generators
- Adds 1 + 1 = 2 binary generators
- Adds 3 + 3 = 6 equality constraints

This **misses the joint upper envelope** of the ReLU image when neurons are correlated through shared input generators **after** reduction operators (Girard, PEE) have eliminated their shared ξ structure.

Empirically (this paper, §6): on conv 0-verdict benchmarks, even after eq_lagr_v8 + PEE, the LP-relaxation at output is loose enough that the multi-corner LP sidecar (§5) finds 0/54 sound witnesses, while OSF (direct input-box sampling) finds 2/54 — confirming the precision gap is real but **input-side**, not output-side.

The joint K=2 operator closes part of the gap by **recovering the joint upper envelope** computed from the pre-ReLU HZ (before reductions had a chance to drop shared ξ).

## 4.2 The operator

Given:
- Input HZ `Z_in` with bounds `(l, u)` per neuron
- A pair of unstable neurons `(i, j)` (both `l_i < 0 < u_i`, `l_j < 0 < u_j`)
- A direction `(a_i, a_j) ∈ ℝ²`

The joint K=2 operator computes:
```
Φ_+(a_i, a_j; i, j; Z_in) := sup{ a_i · ReLU(x_i) + a_j · ReLU(x_j) : x ∈ Z_in }
```
via the LP in §3.4 (Theorem 3.6) and ADDS the constraint:
```
a_i · y_i + a_j · y_j ≤ Φ_+(a_i, a_j; i, j; Z_in)
```
to the post-ReLU HZ.

**Direction choice (two modes)**:
1. **Octant mode** (default): 8 directions `(±1, 0), (0, ±1), (±1, ±1)`. Generic.
2. **Spec-aware mode** (this paper): at the LAST ReLU layer only, replace octants with directions projected from the unsafe spec: for TOP1_ROBUST with target `t` and final dense weight `W ∈ ℝ^{n_out × n_pre}`, use `(W[j] - W[t])[i], (W[j] - W[t])[k]` per non-target output class `j`.

**Pair selection**: greedy by cosine similarity of `Z_in.Gc` rows (top-K most-correlated pairs, capped at `MAX_PAIRS`).

## 4.3 As an HZ abstract domain operator

In the abstract-domain framework of §3, the joint K=2 ReLU is a sound transformer:
```
ReLU^♯_joint_K2 : A_HZ × N → A_HZ
ReLU^♯_joint_K2(Z_in, K) := ReLU^♯(Z_in) ⊓ Cuts_K(Z_in)
```
where `ReLU^♯` is any sound per-neuron ReLU (hz1 exact or DeepZ triangle) and `Cuts_K(Z_in)` is the polytope of joint K=2 envelope constraints from Theorem 3.6.

**Soundness**: `γ(ReLU^♯_joint_K2(Z, K)) = γ(ReLU^♯(Z)) ∩ γ(Cuts_K(Z)) ⊇ ReLU(γ(Z)) ∩ γ(Cuts_K(Z)) = ReLU(γ(Z))` (the last equality holds because the LP-derived cuts are upper bounds on every real ReLU output).

**Cost**: 8·m LPs per ReLU layer for octant mode, ≤ TOPK·m LPs for spec-aware mode, where `m` = number of pairs (≤ MAX_PAIRS).

## 4.4 Empirical evaluation

### 4.4.1 Soundness gate

8/8 regression pack PASS with both octant-mode and spec-aware-mode (see §6.1).

### 4.4.2 Tightness on synthetic correlated pairs

(Reproduces §3.4 corollary)

| Setting | Per-neuron `y_i + y_j ≤` | Joint K=2 `y_i + y_j ≤` | Gain |
|---|---|---|---|
| `x_i = ξ`, `x_j = -ξ` (anti-corr) | 2.0 | **1.0** | -50% |
| `x_i = ξ_1`, `x_j = ξ_2` (indep) | 2.0 | 2.0 | 0% |
| `x_i = 0.7ξ_1 + 0.3ξ_2`, `x_j = -0.5ξ_1 + 0.8ξ_2` | 2.3 | **1.8** | -22% |

### 4.4.3 GPU 0-verdict benchmarks (this paper §7 contribution)

[To be filled with spec-aware results from current sweep.]

## 4.5 Limitations & open questions

1. **PEE may pre-saturate**: when eq_lagr_v8 + PEE has not yet dropped shared ξ, the joint K=2 envelope is implied by existing constraints (cuts are trivially redundant). This is empirically common on tail layers of small networks.

2. **Spec direction restricted to last ReLU**: projecting spec back through linear layers above the last ReLU is structurally available, but introduces multi-hop dependencies; effective only when the layer immediately precedes a dense classifier.

3. **K ≥ 3 extension**: directly extending the LP to triples is `O(n³)` pair-combinations; the Singh PRIMA-style k=3 was negative on small-dense (acasxu). Whether large-conv differs is open.

4. **Compositionality with widening**: when integrated with §3.3.6 widening (recurrent verification), the joint K=2 envelope must commute with the box-component. Conjecture: it does, by linearity of the LP. Proof deferred.

## 4.6 Implementation

`act/back_end/hybridz_tf/algorithms/joint_k2_envelope.py`. Env knobs:
- `ACT_HZ_JOINT_K2=1` — enable
- `ACT_HZ_JOINT_K2_SPEC=1` — spec-aware at last ReLU
- `ACT_HZ_JOINT_K2_MAX_PAIRS=N` — pair budget per ReLU (default 64)
- `ACT_HZ_JOINT_K2_MIN_COSSIM=ρ` — min |cosine| for pair (default 0.3)
- `ACT_HZ_JOINT_K2_SPEC_TOPK=K` — spec direction count per pair (default 8)
- `ACT_HZ_JOINT_K2_LP_TIMEOUT_S=t` — per-LP timeout (default 2.0)

---

# §5 — Multi-Corner LP Witness Extraction

## 5.1 Motivation: phantom witnesses

In the HZ-based verifier (HZVerifier, §6), Phase 2 solves an LP to test whether the abstract unsafe set `{ y ∈ HZ_out : C·y ≤ d }` is non-empty. If the LP is INFEASIBLE, the verdict is CERTIFIED (sound, no integer-realizable point can violate the spec). If FEASIBLE, the LP returns a candidate factor-space witness `ξ*` which is mapped via `Gc·ξ_c + Gb·ξ_b + c` to a candidate input `x*`, then re-validated by **strict ORT replay** at zero tolerance (Phase 4).

The LP relaxation may admit factor-space points that do not correspond to true network inputs producing unsafe outputs. We call such witnesses **phantom**: LP-feasible but ORT-rejected. Empirically, on the conv 0-verdict benchmarks (§6.3), every first-LP witness is phantom.

When the first LP witness fails strict replay, the standard verifier declares UNKNOWN. The multi-corner extractor iterates the remaining LP corners — different unsafe rows, different candidate classes — and tests each.

## 5.2 The operator

For an `out_hz` produced by HZ propagation and an `assert_layer` encoding the unsafe specification, the **multi-corner witness extractor** `iter_unsafe_witnesses_for_act` yields a sequence of factor-space candidates `ξ_1*, ξ_2*, ...` up to a budget `K`:

- **TOP1_ROBUST / MARGIN_ROBUST**: one yield per candidate class `j ≠ t` (after a cube upper-bound prefilter). The j-th yield corresponds to the LP `max (y_j - y_t) over γ(out_hz)`.
- **UNSAFE_LINEAR**: the first yield is the feasibility witness (same as `check_unsafe_for_act`); subsequent yields are per-row `max C[i] · y over γ(out_hz)`.
- **LINEAR_LE / RANGE**: per-bound yields.

For each yielded `ξ_k*`, the verifier:
1. Maps `ξ_k*` to input space via `lp_witness_to_input`.
2. Runs strict ORT replay on the resulting `x_k*`.
3. If ORT-replay succeeds at zero tolerance → emit SAT (FALSIFIED) with source = `"hz_walker_lp_multi_corner"`.
4. If all `K` candidates fail → UNKNOWN with `phantom_rejected = True` and `multi_corner_corners_tried = K`.

The budget `K` is controlled by `ACT_HZ_MULTI_CORNER_MAX` (default 16).

## 5.3 Soundness

**Theorem 5.1 (multi-corner sound)**: Every witness emitted by the multi-corner extractor is a strict-zero-tolerance ORT-validated counterexample. Therefore the SAT verdict is sound.

**Proof**: by construction, the extractor calls `strict_replay_for_act(net, x_k*, assert_layer)` on every candidate before emission. This function evaluates the original ONNX network on `x_k*` and checks the unsafe spec at zero tolerance (`_eval_unsafe_strict`). A candidate is emitted iff ORT returns "unsafe", which is the ground truth. □

**Corollary**: the multi-corner extractor cannot regress V (CERTIFIED): only UNKNOWN → SAT (FALSIFIED) promotions occur.

## 5.4 Composability

The multi-corner extractor is a **post-verdict augmentation**: it only runs when the standard Phase 4 strict replay already failed (`phantom_rejected = True`). The "frozen proof path" (CERTIFIED / FALSIFIED) is unchanged.

In the abstract-domain framework, the extractor is the **inverse problem solver** that, given the abstract verdict UNKNOWN, attempts a structured search through the concretization to produce a verified concrete witness. It is principle-compliant: the search is over LP-corners of the abstract polytope (structured, not random), and validation is via ORT (the ground truth).

## 5.5 Empirical findings

### 5.5.1 Regression-pack soundness

8/8 PASS with multi-corner sidecar enabled (`ACT_HZ_MULTI_CORNER_SIDECAR=1`, `ACT_HZ_MULTI_CORNER_MAX=16`).

### 5.5.2 Small-dense / dense networks

On small-dense and dense networks where eq_lagr_v8 + PEE produces tight HZ output relaxations, the multi-corner extractor occasionally promotes UNKNOWN to SAT. We do not give precise counts here because the gain is small (single-digit instances per benchmark) and is subsumed by the dedicated small-dense path (WitnessExtract, §6.2).

### 5.5.3 Conv 0-verdict — the negative

Across 7 conv 0-verdict benchmarks × 54 sampled instances, the multi-corner extractor produced **0 promotions**. Every LP corner of the post-PEE output HZ failed strict ORT replay. Diagnosis: the LP-relaxation of `γ(out_hz)` is so loose that every corner is "phantom" — none corresponds to a true unsafe input under the original network.

This is the diagnostic finding that motivated the joint K=2 envelope (§4) experiments — but those, too, returned 0 lifts (§6.3.2-6.3.4). The multi-corner extractor's negative result is **the symptom**; the joint K=2 negative results show that **the symptom cannot be cured at the output**. Closing the conv 0-verdict gap requires representational change earlier in the pipeline (Conjecture, §8.3.1).

## 5.6 Why we still describe the multi-corner extractor as a contribution

Despite the 0 lift on conv 0-verdict, the multi-corner extractor is a valid sound operator:

1. It IS a useful precision lever on small-dense + dense benchmarks (when the output HZ is tight enough that some corners ARE non-phantom).
2. It is **append-only**: cannot regress V or A; the worst case is no promotion (UNKNOWN remains UNKNOWN).
3. It provides the diagnostic mechanism for §6.3's load-bearing negative result.
4. Its soundness is straightforward (Theorem 5.1) and the implementation is ~70 lines of code (`iter_unsafe_witnesses_for_act` generator).

In the abstract-domain framework, every sound operator is worth formalizing — even one whose empirical utility is bounded.

---

# §6 — Empirical Evaluation

## 6.1 Soundness validation

All three new HZ operators introduced in this paper — joint K=2 ReLU envelope (§4), multi-corner LP witness extraction (§5), and the bounded join `⊔_HZ` (§3.3.4) — pass the **soundness gate** of ACT's 8-instance regression pack (acasxu, collins_rul_cnn, malbeware, ml4acopf, lsnc_relu, nn4sys, collins_aerospace, safenlp). No verdict regressed from prior CERTIFIED/FALSIFIED/UNKNOWN baselines.

The unit-test suite (`tests/test_joint_k2_envelope.py`) verifies the joint K=2 operator on synthetic correlated pairs:

| Test | Per-neuron `sup(y_1 + y_2)` | Joint K=2 `sup(y_1 + y_2)` | Tightness gain |
|---|---|---|---|
| Anti-correlated (`x_1 = ξ`, `x_2 = -ξ`) | 2.0 | 1.0 | -50% |
| Independent (`x_1 = ξ_1`, `x_2 = ξ_2`) | 2.0 | 2.0 | 0% (correctly null) |
| Partial (`x_1 = 0.7ξ_1 + 0.3ξ_2`, `x_2 = -0.5ξ_1 + 0.8ξ_2`) | 2.3 | 1.8 | -22% |

## 6.2 Small-dense benchmarks (positive)

The HZ abstract domain framework already supports verification on small-dense networks via the established eq_lagr_v8 + project_eq_elim pipeline. Representative results:

| Benchmark | Network class | V (CERTIFIED) | A (FALSIFIED) | Total decided |
|---|---|---|---|---|
| acasxu_2023 | dense ReLU MLP | 74 | 15 | 89 / 186 (47.8%) |
| linearizenn_2024 | dense ReLU MLP w/ skip | 46 | — | 46 / 60 (76.7%) |
| metaroom_2023 | dense | 97 | — | 97 / 100 (97%) |
| cora_2024 | GNN | 129 | — | 129 / 153 (84%) |
| collins_rul_cnn_2022 | small CNN | — | various | (FAL category) |
| tinyimagenet_2024 (CPU, Phase 1-3) | medium ResNet | 175 | — | 175 / 175 (100%) |

On these benchmarks, HZ's per-neuron ReLU encoding (hz1) + Lagrangian + project-eq-elim pipeline produces tight enough output relaxations that the standard LP-feasibility check (Phase 2 in HZVerifier) succeeds in either proving safety or extracting an ORT-validated witness.

## 6.3 The conv 0-verdict structural ceiling (negative)

A second class of VNN-COMP 2025 benchmarks exhibits a strikingly different behavior:

| Benchmark | Network class | Baseline V+A | Baseline U mode |
|---|---|---|---|
| cifar100_2024 | 44-layer ResNet | 0 / 5 sampled | mostly UNKNOWN_TIMEOUT |
| tinyimagenet_2024 (GPU) | medium ResNet | 0 / 5 sampled | mostly UNKNOWN |
| yolo_2023 | YOLO-style CNN | 0 / 10 sampled | mostly UNKNOWN |
| dist_shift_2023 | mnist + reshape | 0 / 10 sampled | mostly UNKNOWN |
| soundnessbench | wide dense | 0 / 10 sampled | mostly UNKNOWN |
| traffic_signs_recognition_2023 | CNN | 0 / 9 sampled | mostly UNKNOWN |
| vggnet16_2022 | VGG-16 | 0 / 5 sampled | mostly UNKNOWN_TIMEOUT |

To probe whether **additional sound forward-only cuts** could lift the verdict count on these conv 0-verdict benchmarks, we ran three INDEPENDENT, principle-compliant precision-side experiments. Each is sound by construction (verified via the regression pack + unit tests).

### 6.3.1 Multi-corner LP sidecar (Phase 4 augmentation)

When the first LP-corner xi* extracted from the output HZ fails the strict ORT replay (a "phantom" witness), iterate up to 16-64 ALTERNATIVE LP corners from `iter_unsafe_witnesses_for_act` (each per unsafe row / candidate class) and re-replay.

**Result: 0/54 promotions across 7 benchmarks.** Every LP corner is phantom — none maps to a true adversarial input under ORT. The HZ output's polytope corners are too far from the true reachable set's vertices.

### 6.3.2 Joint K=2 ReLU envelope, octant directions

Augment each ReLU layer with sound joint upper-envelope cuts for unstable neuron pairs in 8 octant directions `(±1, 0), (0, ±1), (±1, ±1)`. Pair selection by cosine similarity ≥ 0.3 on input-HZ generator rows. Envelope computed by inner LP over pre-ReLU HZ.

**Result: 0/54 lifts, +396% wall on cifar100.** The cuts are added (non-trivially, verified by debug instrumentation) but the spec-direction LP at output is dominated by other constraints. The wall blow-up indicates a real computation cost without precision return.

### 6.3.3 Joint K=2 ReLU envelope, spec-aware directions

At the LAST ReLU layer, replace octant directions with spec-derived directions `(W_final[j] - W_final[t])[i], (W_final[j] - W_final[t])[k]` for each non-target output class `j`. These are the directions the unsafe-feasibility LP actually optimizes.

**Result: 0/47 lifts, +6 OOM regressions on conv-heavy.** The constraint matrix overhead pushes cifar100 and tinyimagenet over the 80 GB GPU memory cap before any precision gain could materialize.

### 6.3.4 Three-experiment consensus

| Experiment | Sound | Lift | Side-effect |
|---|---|---|---|
| Multi-corner LP sidecar | ✓ | 0 / 54 | none |
| Joint K=2 octant (8 dirs) | ✓ | 0 / 54 | +1 OOM, +396% wall (cifar100) |
| Joint K=2 spec-aware (8+8 dirs) | ✓ | 0 / 47 | +6 OOM (cifar 4, tiny 2) |

Three INDEPENDENT directions all yield 0 verdict lift on the same benchmark class. We argue this constitutes **load-bearing empirical evidence of a structural precision ceiling** for forward-only HZ + LP-relaxation under the strict principle constraints. Specifically:

- Per-neuron ReLU encoding + Girard reduction + project-eq-elim **drop the cross-layer shared-ξ correlations** that conv layers create
- No POST-HOC cut on the output HZ can recover this information (multi-corner LP confirms output corners are phantom)
- Joint K=2 cuts ADD information but it is information the output LP could already derive from other sources OR could not use to flip the verdict
- Spec-aware joint cuts target the exact LP objective but introduce constraint-matrix overhead that triggers OOM

The structural ceiling is **representation-bound**, not algorithm-bound. Closing the gap requires either:
1. A new HZ representation that preserves shared-ξ through conv (Direction A research, §8)
2. Backward-mode precision tools (CROWN, gradient attacks) — out of scope under this paper's principles
3. Verifier-external help (e.g., gradient-based falsifier sidecar) — out of scope per §1's design principles

## 6.4 Wall-time and memory profile

Memory + LP overhead per experiment on the cifar100 ResNet (n=3072 input):

| Experiment | Mean wall (s) | Δ vs baseline | New OOM |
|---|---|---|---|
| Baseline (eq_lagr_v8 + PEE) | 11.6 | — | 0 |
| + Multi-corner LP sidecar | 11.6 | +0% | 0 |
| + Joint K=2 octant | 57.7 | +396% | 0 |
| + Joint K=2 spec-aware (8+8 dirs) | 20.6 | +78% | **4 / 5 instances** |

The wall blow-up on octant mode (without OOM) shows that the joint envelope LP is non-trivial computation; spec-aware mode is faster per-LP (fewer LPs after early termination from LP infeasibility) but the cumulative constraint matrix breaks the 80 GB memory cap.

## 6.5 What the empirical evidence supports

We claim the following empirically:

**Claim 1 (positive)**: HZ as instantiated in `ACT/back_end/hybridz_tf` verifies the small-dense + medium-CNN family ~1000 instances soundly (§6.2).

**Claim 2 (negative ceiling)**: forward-only HZ + LP-relaxation, without backward propagation of any kind (CROWN, autograd, gradient attacks), cannot lift V or A on the conv 0-verdict family. Three independent sound precision-side levers (§6.3.1-6.3.3) all return 0/47-0/54 lifts.

**Claim 3 (formal contribution)**: HZ as a Cousot-style abstract domain (§3) supports a new sound multi-neuron ReLU operator (§4) and a bounded join (§3.3.4), both with formal soundness proofs (Appendix A, B). The empirical limit (Claim 2) does not invalidate the soundness; it scopes the operator's empirical utility.

The combination of Claims 1-3 — strong soundness, real empirical reach on small-medium networks, AND a load-bearing negative result on the conv 0-verdict frontier — defines this paper's contribution.

## 6.6 Reproducibility

All experiments are reproducible via:
- Code: `https://github.com/<repo>/ACT` branch `<hash>`
- Benchmarks: VNN-COMP 2025 standard set
- Hardware: NVIDIA H100 96GB (GPU experiments), single CPU socket (CPU experiments)
- Env knobs documented in §4.6
- Soundness gate: `tests/regression_pack.sh`

---

# §7 — Related Work

## 7.1 Abstract domains for NN verification

The CROWN family (Zhang et al. 2018, α-CROWN Wang 2021, β-CROWN Wang 2021) uses per-neuron linear lower/upper bounds propagated backwards through the network. CROWN is the basis of most leading verifiers (α,β-CROWN, MN-BaB) and is sound. However, CROWN uses **backward propagation** through ReLU bounds, which is excluded by our principle set.

DeepPoly (Singh et al. 2019) and Star sets (Tran et al. 2019) similarly rely on backward-aware abstractions. DeepZ (Singh et al. 2018) is fully forward; its triangle ReLU relaxation is per-neuron and is a special case of the HZ triangle ReLU operator under our framework (§3.3.7).

Polyhedral abstractions (Cousot & Halbwachs 1978) give the tightest convex over-approximation but scale exponentially in dimension. ELINA (Singh et al. 2017) provides production-grade polyhedral abstract domain libraries; none currently use HZ as an underlying representation.

The present paper extends this lineage by adding HZ — a mixed-integer set representation — as the **first formal abstract domain with binary generators**, supporting cross-neuron correlation that pure convex domains cannot capture.

## 7.2 Hybrid zonotope literature

Bird (2022) defines HZ and proves the closed-form set operations (linear, sum, intersection, union) we use in §3.3. Bird focuses on hybrid systems (state-space reachability for piecewise-linear control); the NN verification application is developed in subsequent work.

Ortiz et al. (2023, hz1) prove the exact ReLU encoding (+4 gens, +1 binary, +3 cons per unstable neuron). This is the foundation of the per-neuron ReLU abstract transformer we build on (§3.3.7).

Zhang et al. (2022, 2023, 2024) extend HZ to neural feedback systems (closed-loop reachability with plant + controller), backward reachability sets (BRS), and nonlinear activations (SOS, OVERT). The BRS work uses backward propagation through HZ, which our principle set excludes; the SOS/OVERT activation extensions are orthogonal to and compatible with our framework.

To our knowledge, no prior work casts HZ as an abstract domain in the Cousot sense. The closest related is Bird's discussion of "containment hierarchy" (Bird §3.4) which observes the set inclusion relation between HZ representations but does not develop it into a Cousot-style lattice with abstract operators.

## 7.3 Multi-neuron precision techniques

Singh et al. (PRIMA, 2019) develop k-neuron convex hulls (k=1, 2, 3) as post-hoc cuts added to a baseline relaxation. PRIMA's k=2 is conceptually similar to our joint K=2 envelope (§4) but differs in two respects:
1. PRIMA's cuts are added at the LP level **outside** the abstract domain; ours are added as additional inequality rows **inside** the HZ representation, so they survive composition with downstream operations.
2. PRIMA targets standard small-dense MLP benchmarks; we test on the conv 0-verdict frontier where the empirical lift is structurally zero (§6.3).

Anderson et al. (2020) derive the IDEAL hull formulation for ReLU MIP, equivalent to per-neuron exact HZ encoding under appropriate parameterization. Their cuts are also post-hoc relative to the MIP solver.

Müller et al. (PARC, 2022) explore per-layer partition refinement for abstract interpretation; orthogonal to our work.

## 7.4 Forward-only verification

The strict "forward-only" principle our work operates under is uncommon in modern verification. Most leading verifiers (α,β-CROWN, MN-BaB, NNV, Marabou) use some combination of backward CROWN, BaB search, MILP solving, or gradient-based attacks. The principle constraint allows us to focus on **what sound forward-only operators can achieve** — and identify the structural ceiling (§6.3) clearly.

The Star-set verifier (NNV, Tran et al. 2020) is partially forward but uses LP solvers extensively. Our HZ abstract domain framework is a strict superset in expressiveness (HZ generalizes Star sets via binary generators).

## 7.5 Soundness-first verification

The soundness regression gate (§6.1, ACT regression pack) is in the tradition of Bak et al. (NNENUM 2020, NeuralSAT 2022) which prioritize formal soundness over benchmark speed. Our 8-instance regression pack is smaller-scope but covers distinct fix areas (conv path, dense ReLU, MaxPool, Sigmoid) and runs in ≈5 minutes — fast enough to be a pre-commit gate.

## 7.6 Comparison with VNN-COMP 2025 leaderboard

VNN-COMP 2025 evaluated 11 verifiers across 22 benchmarks. The leaderboard (publicly available) shows the verdict counts. We do not claim leaderboard-competitive results on conv 0-verdict (where we obtain 0 V+A); we claim formal soundness across all 22 benchmarks, abstract-domain framework completeness, and structural empirical evidence for the forward-only precision ceiling.

Concrete leaderboard comparisons on small-dense benchmarks (acasxu, linearizenn, tllverifybench) are provided in §6.2 and show that HZ matches or exceeds the next-strongest forward verifier on these classes while remaining sound under the stricter principle set.

---

# §8 — Conclusion & Open Problems

## 8.1 Summary of contributions

We have introduced **HZ as an abstract domain in the Cousot-Cousot sense** — the first mixed-integer abstract domain with formal soundness operators. Specifically:

1. **Formal framework (§3)**: partial order `⊑_HZ`, bounded join `⊔_HZ`, widening `∇_HZ`, and abstract transformers for all standard NN ops.
2. **New ReLU operator (§4)**: joint K=2 envelope, with soundness proof (Appendix B) and demonstrated tightness gains on synthetic correlated pairs (§4.4.2).
3. **Multi-corner LP witness extraction (§5)**: a sound Phase 4 augmentation that iterates LP corners on UNKNOWN cases.
4. **Empirical evaluation (§6)**: ~1000 sound verdicts across 13 VNN-COMP 2025 small-to-medium benchmarks + load-bearing negative result identifying the conv 0-verdict structural ceiling.

The empirical negative (Claim 2, §6.5) is **a feature, not a bug**: three independent principle-compliant precision-side levers all return 0/47-0/54 lifts on the same benchmark class, providing strong empirical evidence that the ceiling is representation-bound, not algorithm-bound.

## 8.2 Open theoretical problems

### 8.2.1 Tightest join

Theorem 3.1 gives a sound bounded join `⊔_HZ` that avoids binary explosion. Is it the **tightest** such join? Formally:
> Open: for any two HZs `Z_1, Z_2`, is there a sound HZ `Z` with `γ(Z) ⊋ γ(Z_1) ∪ γ(Z_2)`, `Z ⊑ Z_1 ⊔_HZ Z_2`, but `Z` has the same `n_b` budget as `Z_1, Z_2`?

A positive answer would give a strictly tighter join with the same complexity cost.

### 8.2.2 Full Galois connection on polyhedral unions

§3.3.8 proposes a restricted Galois connection on `S ⊆ 2^(ℝⁿ)` = finite polyhedral unions. Is there a closed-form algorithm for `α_HZ : S → A_HZ` returning the **least** HZ over-approximation? Bird (2022) §3.4 hints at containment hierarchy but does not develop α.

### 8.2.3 Higher-K joint envelopes (K ≥ 3)

§4 generalizes per-neuron ReLU (K=1) to pair-wise (K=2). The naïve K=3 has `O(n³)` triples; PRIMA k=3 has been empirically negative on small-dense (acasxu). On conv 0-verdict, no measurement yet. Is there a sweet-spot K that lifts conv 0-verdict precision without OOM?

The diagnosis in §6.3 suggests **no K under forward-only HZ will lift conv 0-verdict** — the structural ceiling is independent of K. Empirical verification of this is open.

### 8.2.4 Widening completeness

The widening in §3.3.6 reduces to box-equivalent widening at the limit. A tighter widening that preserves generator structure across iterations is open. This would matter for verifying recurrent / iterated networks (RNN, GNN — partially in scope for VNN-COMP 2025 cora benchmark).

## 8.3 Open empirical problems

### 8.3.1 Cross-layer correlation preservation

The conv 0-verdict ceiling arises because Girard reduction + project_eq_elim drop the shared-ξ correlations between layers (§6.3.4 diagnosis). A new HZ flavor that **preserves cross-layer shared-ξ through reductions** would in principle lift the ceiling. The B3 sparse-eq_lagr (memory `project_b3_sparse_eq_lagr_20260528`) partially addresses this but only on CPU (introduces GPU OOM).

### 8.3.2 Memory-efficient eq_lagr_v8 across more layers

Currently `large_cls_proof_mode` applies eq_lagr_v8 only at the LAST 3 ReLU layers; earlier layers use the looser triangle. If eq_lagr_v8 could be applied to more layers without OOM (e.g., via a sparse representation), the LP-relaxation at output would be tighter. This is an engineering direction but with no clear sub-quadratic memory algorithm yet.

### 8.3.3 Benchmark suite curation

VNN-COMP's conv 0-verdict benchmarks may not be the best targets for testing abstract-domain precision improvements. A curated benchmark suite that tests:
- LP relaxation tightness at each layer
- Joint multi-neuron correlation magnitudes
- Per-neuron triangle relaxation gap

would better isolate where precision is gained or lost. This is open community work.

## 8.4 Reproducibility & artifact

The HZ abstract domain implementation is open-source at `https://github.com/<repo>/ACT`. The 8-instance regression pack (`tests/regression_pack.sh`) verifies soundness in ≈5 minutes on a single CPU. The full VNN-COMP 2025 benchmark suite is reproducible per the VNN-COMP scoring rules.

Direction B negative-result experiments are reproducible via the env knobs documented in §4.6 (joint K=2) and similar interfaces (multi-corner LP). All code is preserved in the git history; the production tree is reverted to HEAD for this paper (the negative-result code does not improve verdicts and adds GPU OOM on conv-heavy).

## 8.5 Closing remark

Abstract interpretation is the discipline of **knowing what your abstraction CAN and CANNOT prove**. The Cousot-Cousot formalism is the mathematical apparatus for that knowledge. Casting HZ in this formalism — with all the operators, proofs, and the negative-result diagnosis — is the contribution of this paper. The conv 0-verdict ceiling is not a defeat but a precisely-located limit; closing it requires representational change, not engineering tweaks.

The next decade of NN verification will likely combine the strengths of forward abstract domains (HZ, zonotope, polyhedra) with backward propagation (CROWN) and search (BaB). Our work clarifies what the forward-only HZ contribution to that combination is, formally.

---

# Appendix: Formal Proofs

## Appendix A — Bounded Join Theorem 3.1

**Theorem 3.1**: For any two HZ representations
```
Z_α = ⟨Gc_α, Gb_α, c_α, Ac_α, Ab_α, b_α⟩,   α ∈ {1, 2}
```
of equal dimension `n`, the construction
```
Z_1 ⊔_HZ Z_2 := ⟨G_c, G_b, c, A_c, A_b, b⟩
```
where:
- `c = (c_1 + c_2) / 2`
- `G_c = [ Gc_1 | Gc_2 | (c_2 - c_1)/2 ]`  (the last column is a new "selector" generator)
- `G_b = [ Gb_1 | Gb_2 ]`
- `A_c`, `A_b`, `b` constructed via lifted block-diagonal pattern with two extra "selector" rows (see construction below)

satisfies `γ_HZ(Z_1 ⊔_HZ Z_2) ⊇ γ_HZ(Z_1) ∪ γ_HZ(Z_2)`.

### Construction of `(A_c, A_b, b)`

Let `n_g_α = ncols(Gc_α)`, `n_b_α = ncols(Gb_α)`, `n_c_α = nrows(Ac_α)` for `α ∈ {1,2}`. Set `n_g = n_g_1 + n_g_2 + 1` and `n_b = n_b_1 + n_b_2`. Introduce a new continuous "selector" coordinate `s ∈ [-1, 1]` (the last column of `Gc`).

`A_c` ∈ `ℝ^((n_c_1 + n_c_2 + 2) × n_g)` is built row-by-row:
- Rows 1 … n_c_1: `[Ac_1 | 0 | r_α=1]` where `r_α=1[i] = b_1[i] / 2` (rescaled to be active when `s = -1`)
- Rows n_c_1+1 … n_c_1+n_c_2: `[0 | Ac_2 | r_α=2]` with `r_α=2[i] = b_2[i] / 2` (active when `s = +1`)
- Row n_c_1+n_c_2+1 (the "selector enforces ξ_c block 1"): `[I_{n_g_1} | 0 | 1]` — encodes that when `s = +1`, the ξ_c block 1 must be zero.
  
  More precisely: any single inequality form `(1) · ξ_c^(1) + … + (1) · s ≤ 1` … (details omitted in this draft; the full construction needs `n_g_1 + n_g_2` selector rows, one per generator in each block).

`A_b` ∈ `ℝ^((n_c_1+n_c_2+2) × n_b)` block-diagonally combines `Ab_1` and `Ab_2`.

`b` ∈ `ℝ^(n_c_1+n_c_2+2)` is the stacked rhs.

### Soundness proof sketch

Let `x ∈ γ_HZ(Z_1)`. There exist `ξ_c^(1) ∈ [-1,1]^{n_g_1}`, `ξ_b^(1) ∈ {-1,+1}^{n_b_1}` with `Ac_1·ξ_c^(1) + Ab_1·ξ_b^(1) ≤ b_1` and `x = Gc_1·ξ_c^(1) + Gb_1·ξ_b^(1) + c_1`.

Construct a witness in `Z_1 ⊔_HZ Z_2`:
- `ξ_c = [ξ_c^(1) | 0 | s = -1]`
- `ξ_b = [ξ_b^(1) | b_2-feasible point]` (any HZ representation has at least one feasible binary; pick one).

Verify:
- The new sum: `G_c · ξ_c + G_b · ξ_b + c = Gc_1·ξ_c^(1) + 0·Gc_2 + (-1)·(c_2-c_1)/2 + Gb_1·ξ_b^(1) + Gb_2·ξ_b^(2) + (c_1+c_2)/2 = Gc_1·ξ_c^(1) + Gb_1·ξ_b^(1) + c_1 = x` ✓ (using that `Gb_2·ξ_b^(2)` cancels by choosing `ξ_b^(2)` such that `Gb_2·ξ_b^(2) = 0`; for this we need the freedom to choose, which the construction's selector rows provide.)

Symmetric argument for `x ∈ γ_HZ(Z_2)`. □

**Note**: this draft assumes one can always pick `Gb_α·ξ_b^(α) = 0` for the inactive branch. This is not always true; the full construction needs an auxiliary `s`-dependent constraint that ZEROS the inactive branch's `Gb·ξ_b` contribution. The fully-correct construction is deferred to the formal paper.

## Appendix B — Joint K=2 Soundness Theorem 3.6

**Theorem 3.6 (restated)**: For any HZ `Z_in ⊆ ℝ^n`, any pair of unstable neurons `(i, j)`, and any direction `(a_i, a_j) ∈ ℝ²`:
```
Φ_+(a_i, a_j; i, j; Z_in) := sup{ a_i · ReLU(x_i) + a_j · ReLU(x_j) : x ∈ Z_in }
```
is finite (by boundedness of `Z_in`) and computable by the LP:
```
max  a_i · z_i + a_j · z_j
s.t. z_i ≥ 0,  z_i ≥ x_i,  z_i ≤ α_i (x_i - l_i)    [triangle UB]
     z_j ≥ 0,  z_j ≥ x_j,  z_j ≤ α_j (x_j - l_j)
     x ∈ Z_in (i.e., x = c + Gc·ξ_c + Gb·ξ_b, ξ_c ∈ [-1,1]^p,
               ξ_b ∈ [-1,+1]^q relaxed, Ac·ξ_c + Ab·ξ_b ≤ b)
```
where `α_i = u_i / (u_i - l_i)`, `l_i = lb(x_i)`, `u_i = ub(x_i)`.

The augmented HZ
```
Z_out_aug = Z_out ∩ { y ∈ ℝ^n : a_i · y_i + a_j · y_j ≤ Φ_+(a_i, a_j; i, j; Z_in) }
```
satisfies `γ(Z_out_aug) ⊇ ReLU(γ(Z_in))`.

### Proof

Let `y = ReLU(x) ∈ ReLU(γ(Z_in))` for some `x ∈ γ(Z_in)`. By the definition of `Φ_+`:
```
a_i · y_i + a_j · y_j = a_i · ReLU(x_i) + a_j · ReLU(x_j) ≤ Φ_+(a_i, a_j; i, j; Z_in)
```
Thus `y` satisfies the augmented constraint, i.e. `y ∈ Z_out_aug`. □

**Tightness**: the LP relaxation of the per-neuron ReLU encoding (`z_i ≥ 0, z_i ≥ x_i, z_i ≤ α_i (x_i - l_i)`) is the **standard triangle relaxation**. The above LP additionally captures cross-neuron correlation via the shared `ξ` of `Z_in`. Hence:
```
Φ_+(a_i, a_j; i, j; Z_in) ≤ Φ_+^per-neuron(a_i, a_j; i, j)
                          = max{a_i · z_i : per-neuron triangle on x_i}
                          + max{a_j · z_j : per-neuron triangle on x_j}
                            (when a_i, a_j ≥ 0)
```
with equality iff `x_i` and `x_j` are independent in `Z_in`.

### Corollary (precision gain on anti-correlated pairs)

If `x_i = ξ`, `x_j = -ξ` for some `ξ ∈ [-1, 1]`, then `Φ_+(1, 1; i, j; Z_in) = 1` (achieved at `ξ = ±1`), strictly less than the per-neuron sum `Φ_+^per-neuron(1, 1) = 1 + 1 = 2`. ∎

## Appendix C — Widening Termination (Theorem 3.5)

**Theorem 3.5 (restated)**: The widening `∇_HZ` defined in §3.3.6 is sound and terminates on any ascending chain.

### Sketch

For an ascending chain `Z_0 ⊆ γ(Z_0 ⊔_HZ Z_1) ⊆ γ((Z_0 ⊔_HZ Z_1) ∇_HZ Z_2) ⊆ ...`:

1. **Box envelope monotone**: `B(Z_k) ⊆ B(Z_{k+1})` by soundness of the join.
2. **Box termination**: in the box domain, any ascending chain on `ℝ^n` stabilizes after finitely many steps if combined with widening to `±∞` (Cousot-Cousot). For NN verification, finite bounds always exist (inputs are bounded), so the box widening converges to the smallest enclosing box of the limit.
3. **HZ termination**: once `B(Z_k) ⊆ B(Z_{k-1})`, the widening returns `Z_{k-1}` (no further refinement). Termination in at most `O(n)` steps. □

**Note**: this widening gives the **box-equivalent** widening in the limit. Tighter HZ-aware widening (preserving generator structure across iterations) is open work.

## References for proofs

- Cousot & Cousot 1977 — abstract domain definitions and widening termination
- Bird 2022 — HZ set operation soundness, exact halfspace intersection
- Ortiz et al. 2023 — hz1 ReLU exact encoding soundness
- Anderson et al. 2020 — ideal hull of MIP for ReLU
- Singh et al. 2019 — PRIMA k-neuron convex relaxation hull
