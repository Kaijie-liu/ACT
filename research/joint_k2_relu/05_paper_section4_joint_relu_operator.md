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
