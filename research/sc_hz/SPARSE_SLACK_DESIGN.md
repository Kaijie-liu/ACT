# Sparse-Slack-Columns FCHZ — Design Doc

**Purpose**: unlock tinyimagenet (400 UNK), cifar 150+ deep, vggnet16 in a
memory-efficient way while preserving HZ closed-form bound tightness.

**Constraint**: P1-P5 strict, sound by construction, no Gurobi, must pass
existing 81 tests + soundness invariant.

---

## 1. The problem

Two operating modes in current walker:

- **regular**: keep full `G ∈ ℝⁿˣᴷ` (every layer's ReLU slack adds a column).
  Bound: tight. Memory: O(n·K). cifar 100-deep iids OOM at iid 110+.
- **hz_only**: replace `G` columns past first ReLU with per-row `tail_radius`.
  Bound: lossy after 2-3 ReLU layers. Memory: O(n).

**The gap**: a vanilla MLP of 100 ReLU layers spreads `G` to ~100 columns,
but most columns have only a few non-zeros (the unstable neurons in that
layer). The K=100 is sparse — but stored dense.

---

## 2. Insight

At ReLU layer `ℓ`, unstable neurons gain a *new* generator column with
non-zero entries only at the unstable rows of layer ℓ. After a forward Dense
op, the new column gets dense: `(W · G_ℓ)_i ≠ 0` for any `i` with
`W_{i,j} ≠ 0` (which is usually all rows).

**Once a layer's slack column has been multiplied by a dense Dense/Conv, it
goes dense.** So the sparsity benefit is only across that single boundary —
in a feed-forward net, sparsity collapses immediately.

The exception: in **ResNet-like** networks, the residual `add` after ReLU
preserves sparsity in the skip path. Some Conv channels are unaffected
by an earlier layer's slack column → sparsity survives across skip blocks.

This makes the engineering work *per architecture pattern*, not generic.

---

## 3. Design A: post-Conv sparse → dense compression

Currently after Conv, every new G column is fully dense (n_out × K_in).
But most rows have small magnitudes — they correspond to channels barely
affected by the layer's slack.

**Compress**: after each Conv, for each column `k`, if `|G[:,k]|.max() < ε·tail_radius`,
absorb it into tail (per-row): `tail_radius += |G[:,k]|`. Drop the column.

This is **sound** (Σ_i |d_i| |G[i,k]| ≥ |d · G[:,k]|) and **lossy** (we
trade column tightness for memory).

### Threshold ε
- Too small → keep too many cols → memory issue persists
- Too large → lose precision rapidly
- Adaptive: keep K_target ≈ 256 columns max; drop the smallest by L∞ norm.

### Memory complexity
- O(n · K_target) for G — bounded.
- O(n) for tail_radius.
- Total: O(n · 256) — predictable.

---

## 4. Design B: sparse CSR representation

Use `scipy.sparse.csr_matrix` for G. Conv `Y[i,k] = Σ_j W[i,j] G[j,k]`
becomes a sparse·dense matmul. Memory: O(nnz(G)).

### When sparsity holds
- After ReLU: column for layer ℓ has only `n_unstable_ℓ` non-zeros.
  Typical n_unstable for late layers is <10% → 10× sparsity.
- After Dense/Conv: fill-in. Sparsity drops to ~50%+ in 1-2 ops.

### Engineering cost
- All Conv/Dense paths must dispatch on sparse type.
- HZ bound `Σ_k |d · G[:,k]|` is straightforward in CSR.
- tail_radius unchanged.

### Estimated lift
Likely small benefit over Design A for CNN nets (fill-in kills sparsity
after 1 conv). Better suited for transformer-like sparse activations.

---

## 5. Recommended: Design A (compression) first

- Simpler: ~50 lines walker change
- Bounded memory: O(n · K_max)
- Sound by construction (compression goes to tail_radius)
- Configurable K_max budget per layer

### Implementation plan

1. Add `G_max_cols` parameter (default 512) to `forward_fchz`.
2. After every Conv/Dense (where G fills), call `_compress_G_to_tail(state, G_max_cols)`:
   ```python
   def _compress_G_to_tail(state, K_max):
       if state.G.shape[1] <= K_max: return state
       col_inf = np.abs(state.G).max(axis=0)  # L∞ per column
       keep_idx = np.argsort(col_inf)[::-1][:K_max]
       drop_idx = np.argsort(col_inf)[::-1][K_max:]
       extra_tail = np.abs(state.G[:, drop_idx]).sum(axis=1)
       new_G = state.G[:, keep_idx]
       new_tail = (state.tail_radius if state.tail_radius is not None
                          else np.zeros(state.n)) + extra_tail
       return FCHZState(c=state.c, G=new_G, n_root=state.n_root,
                              slack_records=state.slack_records,
                              tail_radius=new_tail)
   ```
3. Unit test: invariant still holds (UB after compression ≥ UB before).
4. Memory test: cifar 110-deep + tinyimagenet should pass.

---

## 6. Soundness proof

Claim: compressing `G` columns into per-row tail is sound.

For any state `s = (c, G, τ)` and target compression set `D` (drop indices),
the new state `s' = (c, G[:,keep], τ + Σ_{k∈D} |G[:,k]|)` satisfies:

```
R(s) = { c + G·ξ + δ : ξ ∈ [-1,1]^K, δ_i ∈ [-τ_i, +τ_i] }
R(s') = { c + G_keep·ξ_keep + δ' :
          ξ_keep ∈ [-1,1]^|keep|, δ'_i ∈ [-τ'_i, +τ'_i] }
```

For any `y ∈ R(s)`, decompose:
```
y = c + Σ_{k∈keep} G[:,k]·ξ_k + Σ_{k∈D} G[:,k]·ξ_k + δ
```

Set `ξ_keep := ξ|_{keep}`. Then `y - c - G_keep·ξ_keep = Σ_{k∈D} G[:,k]·ξ_k + δ`.

For each row `i`: `|(Σ_{k∈D} G[i,k]·ξ_k) + δ_i| ≤ Σ_{k∈D} |G[i,k]| + τ_i = τ'_i`. ∎

So `R(s) ⊆ R(s')`. The HZ closed-form on `s'` is therefore a sound upper bound.

---

## 7. Expected lift

| Bench | Current state | After sparse-slack |
|---|---|---|
| cifar 110-199 | partial (some still OOM) | full sweep (+10-30 V) |
| tinyimagenet 401 UNK | OOM at Conv_3 | reaches output (+50-100 V) |
| vggnet16 41 UNK | 169 GiB OOM | walker fits (+0-15 V, bounds may be loose) |

**Conservative**: +60 V from cifar + tinyimagenet alone → 2153 + 60 = 2213. ✅ 2200+

---

## 8. Implementation order

1. Add `_compress_G_to_tail` helper to walker (50 lines)
2. Call it in Conv, Dense, ConvTranspose, Add paths
3. Add 3 unit tests (compression sound + tighter bound preserved + cifar 110)
4. Run cifar 150-199 + tinyimagenet small batch
5. ORT validate, append to bundle

Total: ~2 hours implementation + sweep.
