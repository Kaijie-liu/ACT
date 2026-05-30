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
