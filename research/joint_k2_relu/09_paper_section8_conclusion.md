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
