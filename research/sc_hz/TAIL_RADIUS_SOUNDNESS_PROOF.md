# Tail-Radius FCHZ State: Soundness Invariant + Per-Op Lemmas

**Audit purpose**: a formal proof obligation for the `tail_radius` mechanism
introduced 2026-06-06 after the unsound single-column tail pool was retired.
The previous unsound version is described in §6 (rejection regression).

---

## 1. State definition

A reachable state is a triple

```
s = (c ∈ ℝⁿ, G ∈ ℝⁿˣᴷ, tail_radius ∈ ℝⁿ_≥0)
```

where `tail_radius` may be ⊥ (meaning identically zero).

The set represented by `s` is

```
R(s) = { c + G·ξ + δ  :  ξ ∈ [-1,1]ᴷ,  δ_i ∈ [-tail_radius_i, +tail_radius_i] }    (1)
```

Note: `ξ` is shared across rows, but `δ_i` is **independent per row** —
this is what makes per-row `tail_radius` strictly tighter (and sound)
versus the single-column pool which forced one shared `δ`.

---

## 2. Soundness invariant

**(INV)** For every state `s` reachable by the walker from initial state
`s₀` (representing the verification input box) by a sequence of supported
operators, the actual reachable set `R*(s)` (under the original neural-net
semantics from the input box) satisfies `R*(s) ⊆ R(s)`.

---

## 3. HZ closed-form upper bound

For any direction `d ∈ ℝⁿ`,

```
max_{y ∈ R(s)} d · y
   = d·c + max_ξ d·G·ξ + max_δ d·δ
   = d·c + Σ_k |d·G_k| + Σ_i |d_i| · tail_radius_i     (2)
```

This is the bound returned by `hz_closed_form_ub(s, d)`.

Independent `δ_i` make `Σ_i |d_i| · tail_radius_i` the exact maximum;
under a shared single `δ`, the max would collapse to `|Σ_i d_i tail_i|`
which is strictly smaller — see §6.

The certified condition is

```
hz_closed_form_ub(s_out, d) − t  <  0     (CERT, sound by INV)    (3)
```

for every unsafe halfspace `(d, t)`.

---

## 4. Per-op soundness lemmas

### 4.1 Initial state (input box `[lb, ub]`)

```
c = (lb + ub)/2,  G = diag(r) where r = (ub - lb)/2,  tail_radius = 0.
```

Sound: every `x ∈ [lb, ub]` is `c + G·ξ` for `ξ_i = (x_i - c_i)/r_i ∈ [-1,1]`.

### 4.2 Dense (Linear) layer  `y = W·x + b`

Lift from `s = (c, G, τ)` to `s' = (W·c + b, W·G, |W|·τ)`.

Proof. For any `x = c + G·ξ + δ ∈ R(s)`:

```
W·x + b = W·c + b + W·G·ξ + W·δ.
```

Bound on `W·δ`: row `i`, `|(W·δ)_i| = |Σ_j W_{ij} δ_j| ≤ Σ_j |W_{ij}| · |δ_j| ≤ (|W|·τ)_i`.

Treat `(W·δ)_i` as `δ'_i ∈ [-(|W|·τ)_i, +(|W|·τ)_i]`. (Note: the new `δ'` is
NOT generally independent across rows — but the per-row bound on its
*magnitude* is sound for any covariance, and `hz_closed_form_ub` only
uses the magnitude.) ∎

### 4.3 Conv2D `y = Conv(x; W, b)`

Conv is a structured linear operator. Same as 4.2 with `|W| @ τ` replaced
by `conv2d(reshape(τ, input image shape), |W|, no_bias)`.

In code: `_propagate_tail()` reshapes `τ` to `(C_in, H_in, W_in)`, applies
`F.conv2d` with `|W|` filters and zero bias, flattens. ∎

### 4.4 ConvTranspose

Same argument: ConvTranspose is linear; tail propagates via `conv_transpose2d(τ, |W|)`. ∎

### 4.5 Batch normalization  `y = a·x + β`

With `s = (c, G, τ)`, `s' = (a·c + β, a·G, |a|·τ)`. ∎

### 4.6 Residual Add  `y = x_a + x_b`

If `s_a = (c_a, G_a, τ_a)` and `s_b = (c_b, G_b, τ_b)` share the same `ξ`,
then `s' = (c_a + c_b, G_a + G_b, τ_a + τ_b)`.

Proof. Any element of `R*(s')` is `(x_a) + (x_b)` for some `x_a ∈ R*(s_a)` and
`x_b ∈ R*(s_b)`; both representable in their respective forms with the
**same shared** `ξ` (from input). The row-wise `δ_a + δ_b` lies in
`[-(τ_a+τ_b), +(τ_a+τ_b)]` componentwise. ∎

### 4.7 Add with constant bias  `y = x + b`

`s' = (c + b, G, τ)`. ∎

### 4.8 Mul by constant  `y = α·x` (`α` scalar or broadcast)

`s' = (α·c, α·G, |α|·τ)` (componentwise `|α|` on tail). ∎

### 4.9 Sub from constant  `y = b - x`

`s' = (b - c, -G, τ)`. (Tail unchanged because `|-1| = 1`.) ∎

### 4.10 ReLU (hz_only mode)

For each ReLU output `y_i = max(0, x_i)`:

- **Inactive** (`u_i ≤ 0`): `y_i = 0`. Set `c'_i = 0`, `G'_i = 0`, `τ'_i` unchanged irrelevant — set to 0.
- **Active** (`l_i ≥ 0`): `y_i = x_i`. Propagate as identity.
- **Unstable** (`l_i < 0 < u_i`): use the **DeepZ slope** relaxation `y_i ≈ λ_i x_i + μ_i + ε_i`
  where `λ_i = u_i/(u_i - l_i)`, `μ_i = -λ_i · l_i / 2`, and
  `ε_i ∈ [-μ_i, +μ_i]`. This **box error** is added to `tail_radius`.

  - `c'_i = λ_i · c_i + μ_i`
  - `G'_i = λ_i · G_i`  (lambda-scaled affine core)
  - `τ'_i = λ_i · τ_i + μ_i`

The DeepZ triangle relaxation is **sound** for `y_i = ReLU(x_i)`: the line
`y = λ x + μ` is the upper edge of the triangle `{(x, y) : y ≥ 0, y ≥ x,
y ≤ λ x + μ}` that overapproximates `(x, max(0, x))` on `[l_i, u_i]`.

The `+ μ` lift handles the bias to the center of the relaxation; the
`± μ` box error covers both the lower envelope (`y = 0` for `x < 0`) and
upper envelope (`y = x` for `x > 0`).

For each unstable row, soundness is `R*([l, u]) → R*(ReLU output) ⊆`
the new `(c', G', τ')` representation. ∎

### 4.11 Slice, Concat, Pad, Transpose, GlobalAveragePool

All are pure linear / index-permutation ops. Tail is permuted /
zero-padded / averaged along with `c` and `G`. ∎

### 4.12 MaxPool

MaxPool is **nonlinear**. We relax to the per-output box bound
`[max over window of l_i, max over window of u_i]`. This is sound but
loose: the actual output is in this box (max preserves bounds), but `G`
linkage is destroyed (set `G_out = 0`, push everything into `tail_radius`).

The CERT obtained from a MaxPool-using network is therefore conservative
— may PHANTOM more, never UNSOUND. ∎

---

## 5. Final classification

```
hz_closed_form_ub(s_final, d) − t  <  0
```

This is sound by INV (§2) + the per-op lemmas (§4).

---

## 6. Rejected regression: single-column tail pool (UNSOUND)

The previous formulation packed all unstable-neuron `μ_i` values into a
**single new generator column** `tail_col` (so each row's slack came from
the same `ξ_{new}`). The HZ bound used

```
UB_old = d·c + Σ_k |d·G_k| + |Σ_i d_i · μ_i|         (UNSOUND)
```

Counterexample: `μ = (+1, -1)`, `d = (1, 1)`:

```
UB_old = ... + |1·(+1) + 1·(-1)| = 0
UB_sound = ... + |1|·1 + |1|·1 = 2
```

For independent ReLU box errors, `UB_sound` is correct; `UB_old`
undercounts by canceling per-row errors. Detected via 3 ORT violations
on relusplitter iids 161, 192, 193.

`test_single_column_pool_is_unsound` in
`research/sc_hz/tests/test_tail_radius_soundness.py` **must always pass**
— it asserts that any future regression to the single-column pool gives
a strictly-smaller bound than the per-row tail, which would be unsound.

---

## 7. Independent recompute audit

The 275-iid strict bundle is checked by:

1. **Per-op unit tests**: §4 lemmas tested individually (`test_tail_radius_soundness.py`).
2. **Sampling soundness check**: 5000 random `(ξ, δ)` samples per
   constructed state — confirm `UB ≥ max_sampled`.
3. **ORT replay**: 500-2000 input samples per accepted iid via
   `onnxruntime` — confirm 0 violations of any unsafe halfspace.
4. **Spot recompute**: 5 random iids per audit re-run via fresh walker —
   confirm `hz_max < 0` reproduces.

iids that violate any of (1-4) are **rejected**, not adjusted to fit.
Example: acasxu iid 88 reported HZ ≈ -1.469e-03 but ORT showed 77/2000
violations → rejected from the strict bundle.

---

## 8. Open soundness items (for advisor review)

- (a) Per-row `δ` independence: §3 assumes per-row independent `δ`. When
  `δ` is propagated through Dense/Conv, the resulting `(W·δ)` becomes
  cross-row correlated. Bound (2) still holds because it uses only row
  magnitudes. **Verified by sampling test §1 of unit tests** but a
  formal coupling argument should be written for paper §4.
- (b) For multi-step ReLU chains, the `tail_radius` grows by `λ_i τ_i + μ_i`
  per layer. After many layers `tail_radius` dominates `G`; HZ becomes
  effectively a box. This is the cifar/tinyimagenet looseness ceiling.
  Solution direction: sparse-slack-per-layer columns retain per-layer
  identity in `G` longer (deferred engineering).
- (c) MaxPool destroys `G` linkage — sound but very loose. Could be
  improved by per-channel slack columns instead of full `G` zeroing.
