# Spec-Conditioned HZ (SC-HZ) — Phase A Design Lock

**Date**: 2026-06-04 night
**Authorization**: research-only proposal under the roadmap future-work track.
No production default or paper headline changes are authorized by this file.
**Scope**: SC-HZ generator budgeting ONLY in Phase A; selective exact-HZ ReLU,
batched per-rival GPU, stable fastpaths, and MaxPool work are subsequent
tracks gated on Phase A success.

Filename note: this file keeps the historical `dc_hz` filename so existing
references do not break. The mechanism name is now **SC-HZ**, because the
important feature is conditioning on the verification spec/rival, not generic
"direction conditioning".

This document is the design lock **before any code begins**. It encodes the
exact mechanism, scope, principle invariants, soundness proofs, and hard
stop gate. Per the lesson from the §6b denominator burn AND the §7 final-tail
hull `cifar_finaltail_hull_plan.md` success template, the prototype cannot
deviate from this spec; if a deviation is required, this file is edited
first and the design lock is re-issued.

---

## 1. Mechanism — explicit forward HZ with per-rival generator pruning

### 1.1 Pre-computation: the rival direction `d_L`

Given a model with hidden layers L = 1, ..., N and weight matrices
`W_1, ..., W_{N+1}` (where `W_{N+1}` is the output classifier), and given
a top-1 robustness query with true class `y_t` and rival set `R`:

```
For each rival r ∈ R:
    d_N^r = W_{N+1}[r, :] - W_{N+1}[y_t, :]            # output rival direction
    for L = N - 1, ..., 0:                              # backward through weights only
        d_L^r = W_{L+1}^T · d_{L+1}^r                   # pure linear-algebra
```

**Concrete shapes**: each `d_L^r` has the shape of layer L's output (the
hidden activation), pre-ReLU. For a Conv2D layer producing `(C, H, W)` output,
`d_L^r` is `(C, H, W)`. The "backward through W_{L+1}^T" operation is the
adjoint of the linear operator: `Conv2DTranspose` for Conv, matrix transpose
for Dense, etc.

**What `d_L^r` does NOT depend on**:
- The input box `[lb, ub]` of THIS iid.
- Any bound at any layer.
- The forward HZ state.
- Autograd / gradients.

**What `d_L^r` DOES depend on**:
- The model weights only.

This is purely a model-architectural quantity, fixed before any forward
propagation begins. It is computed ONCE per (model, rival) pair and can be
cached across iids that share the model.

### 1.2 Forward HZ propagation with per-rival K-cap

For each rival r, we propagate a separate HZ that has been pruned per layer
to keep only the top-K generators by relevance to `d_L^r`:

```
Initialize  h_0^r  =  input HZ  (same for all r)
For L = 1, ..., N:
    h_L^r  =  STANDARD_LAYER_OP(h_{L-1}^r, W_L)         # forward as usual
    h_L^r  =  PRUNE(h_L^r, d_L^r, K)                    # NEW: per-rival prune
At output:
    LP UB on (d_N^r)^T y    →    rival_lp_min^r
If max_r (rival_lp_min^r) < 0:    CERT
Else:                              decode xi_star → ORT replay → FAL / UNK
```

The `STANDARD_LAYER_OP` is the existing HZ Conv2D / Dense / ReLU triangle /
MaxPool. **Phase A reuses these without modification**; the only addition is
`PRUNE`.

### 1.3 PRUNE: top-K with sound tail-box merge

```
PRUNE(h_HZ, d_L, K):
    c, G = h_HZ.c, h_HZ.G                         # G: (hidden_dim, ng)
    if G.shape[1] <= K:
        return h_HZ                                # nothing to prune
    relevance = abs( d_L.reshape(-1) @ G )         # (ng,) per-column scalar
    order = argsort(-relevance)                     # descending
    keep = order[:K]                                # top-K indices
    drop = order[K:]
    G_kept = G[:, keep]                             # (hidden_dim, K)
    r_tail = abs(G[:, drop]).sum(axis=1)            # (hidden_dim,) row-wise L1
    G_new  = concatenate([G_kept, r_tail[:, None]], axis=1)  # (hidden_dim, K+1)
    new_factor_ids = keep_ids + [fresh_tail_aux_id]
    return HZ(c=c, G=G_new, factor_ids=new_factor_ids,
              binary_generators=h_HZ.binary_generators)        # binary side untouched
```

The binary-generator side of the HZ (Gb, Ab, b) is left untouched by PRUNE;
only the continuous-generator side Gc is pruned. This preserves the existing
eq_lagr_v8 / large_cls_proof_mode pipeline at the tail.

### 1.4 Soundness of PRUNE

Claim: the pruned HZ over-approximates the original HZ.

Proof: any point `c + G ξ` with `ξ ∈ [-1, 1]^ng` can be expressed as:
```
c + G ξ  =  c + G[:, keep] · ξ_keep  +  G[:, drop] · ξ_drop
```
Define `ξ_tail = sign(G[:, drop] · ξ_drop) ∈ [-1, +1]` per row (after a
common rescaling argument). The contribution of the dropped generators is
bounded coordinate-wise by:

```
| G[:, drop] · ξ_drop |_i  <=  Σ_{j ∈ drop} |G[i, j]|  =  r_tail_i  =  |r_tail[:, None] · ξ_tail|_i
```

Therefore `(c + G ξ) ∈ (c + G_kept · ξ_keep + r_tail · ξ_tail)` with
`ξ_keep ∈ [-1, +1]^K, ξ_tail ∈ [-1, +1]`. The pruned HZ contains the
original HZ. **Sound.**

The bound is tight when the dropped generators are axis-aligned in one
coordinate; loose when they are mutually-correlated in directions
non-aligned with the kept generators. This is the standard Girard
reduction's soundness property; we are simply CHOOSING what to keep by a
different criterion (per-rival relevance) instead of column-norm.

### 1.5 The output query and verdict

After all per-rival forwards complete, the output of rival r's propagated
HZ is:

```
y^r  =  W_{N+1} · h_N^r  +  b_{N+1}
```

The rival margin `(y^r[r] - y^r[y_t])` lives in 1-D and is bounded by an LP
over the K+1 generators of `h_N^r`. This LP is much smaller than the
original full-HZ LP, **AND** has been per-rival tightened by pruning
irrelevant generators.

Verdict:
- CERT: every rival r has `LP_UB(rival_margin^r) < 0` strictly.
- FAL candidate: some rival r has `LP_UB(rival_margin^r) >= 0`; the LP
  produces a `ξ*` realizing that maximum. Decode `ξ*` to input by the
  existing receipt_factor_aware_endcap_lp decoder; run strict ORT replay.
  Pass ⇒ FAL; fail ⇒ UNK.
- UNK: between CERT and FAL.

The receipt format mirrors the existing CIFAR endcap receipt format so the
audit harness can ingest it without changes.

---

## 2. Why Phase A is SC-HZ generator budgeting ONLY

The full proposal in `hz_redesign_for_robustness_20260604.md` §3 lists 5
mechanisms. Phase A implements **only the SC-HZ generator-budgeting
mechanism**.

Reasons:
1. SC-HZ is the only current hypothesis with enough structural reach to move
   multiple weak benchmark families at once.
2. Its first job is not to prove a big score; it is to prove a signal:
   new V/A or meaningful LP-UB reduction on a representative sentinel set.
3. It can be implemented standalone in `research/sc_hz/` without modifying
   production code (zero risk to the 924 V/A baseline).
4. Batched GPU forward only matters for full-sweep scale. Phase A can use
   serial or small-batch per-rival forward if that keeps the implementation
   auditable.

Selective exact-HZ ReLU is deferred until SC-HZ has signal. Stable fastpaths
and exact MaxPool remain independent engineering tracks; they should not be
mixed into Phase A because they would make attribution ambiguous.

---

## 3. Invariants

| ID | Invariant |
|---|---|
| I1 | Forward-only per the adopted definition (§9 of redesign): no bound at L′ > L refines the bound at L. d_L^r is computed from weights only and used only as a generator-relevance score. |
| I2 | No gradients. d_L^r is a weight-matrix transpose product, NOT an autograd-derived quantity. |
| I3 | Continuous LP only. The LP at the end is over the K+1 generators with `ξ ∈ [-1, +1]`. No binary integer reasoning in Phase A. |
| I4 | No BaB, no input splitting. |
| I5 | No random / corner / PGD candidates. FAL candidates come from the per-rival LP `ξ*` decoded back to input via the existing receipt decoder. |
| I6 | Soundness. PRUNE is sound by §1.4. The forward HZ ops are existing sound implementations. |
| I7 | Fail-closed. Shape mismatches, K-budget violations, or model-graph drift all raise and the iid stays UNK. |
| I8 | No production modification. All code lives in `research/sc_hz/`. The existing 924 V/A is unaffected regardless of Phase A outcome. |
| I9 | Provenance bundle on every receipt: `canonical_root + instances_csv_sha256 + onnx_sha256 + vnnlib_sha256`. |
| I10 | Soundness is independent of `d_L^r`. `d_L^r` may improve the reduction order, but PRUNE must over-approximate correctly for any arbitrary or adversarial ordering. |

---

## 4. Scope (Phase A)

| Component | Action |
|---|---|
| `d_L^r` pre-computation | NEW — `research/sc_hz/precompute_direction.py` |
| Per-rival HZ forward | NEW — `research/sc_hz/pruned_forward.py` |
| PRUNE operator | NEW — `research/sc_hz/prune.py` |
| LP solver | REUSE — existing highspy / `_solve_endcap_lp_with_solution` |
| Snapshot consumer | REUSE — existing FLATTEN snapshots OR fresh per-iid capture |
| ORT replay | REUSE — `receipt_factor_aware_endcap_lp._extract_xi_root_witness` |
| Receipt format | REUSE — mirrors the CIFAR endcap receipt |
| Driver | NEW — `research/sc_hz/run_sentinels.py` for the Phase A sentinel set |

What is NOT in scope (Phase A):

- Multi-rival GPU batching (Phase C / R3).
- Selective binary activation (Phase D / R2).
- MaxPool / Stable-fastpath improvements (R4 / R5).
- Any change to `act/pipeline/cli.py`, `act/back_end/hybridz_tf/*`, or
  `verify_once_hz`.
- Modification to the existing endcap LP path or the 924 V/A baseline.

---

## 5. Hard stop gate (Phase A → Phase B)

20 iids per benchmark × 4 benchmarks = 80 sentinels total.

Benchmarks:
- **cifar100_2024** — 20 iids from atlas v3 near-boundary set (lowest positive
  `final_lp_margin` UNK iids). Production endcap LP is at the per-neuron
  triangle math ceiling per §7; SC-HZ tests whether query-local generator
  budgeting can recover information earlier than the final per-neuron hull.
- **tinyimagenet_2024** — 20 iids from the current UNKNOWN/OOM pool, stratified
  by model family and wall time. Dense-conv behavior must be tested outside
  CIFAR.
- **safenlp_2024** — 20 iids drawn from the UNK pool (currently large enough
  UNK). The wide-spec disjunctive nature is where SC-HZ's per-rival forward
  pays off most.
- **acasxu_2023** — 20 iids drawn from the UNK pool (currently 98
  UNK). Small-dense net; per-rival pruning has high directional signal.

### Gate criteria

**PASS** iff:
- All 80 iids complete fail-closed (crashes become explicit ERROR rows, not
  silent drops).
- Cumulative new V/A across the 80 sentinels ≥ **5**, OR median LP UB reduction
  across completed UNKNOWN rows ≥ **25%** relative to production baseline.

**FAIL** iff:
- Any iid crashes or violates soundness (caught by a parallel ORT replay
  check on every claimed CERT — Phase A includes this audit).
- Cumulative new V/A = **0** AND median LP UB reduction < **10%**.

**INCONCLUSIVE** in between → advisor decides whether to expand K-cap or
close. Default action on INCONCLUSIVE is widen K from 256 → 512 and re-run
the 10 iids that showed the smallest LP UB reduction.

If PASS, proceed to Phase B (expanded 6-8 benchmark pilot). If FAIL, close
the SC-HZ direction; the project continues with the 924 V/A paper and only
independent engineering cleanup.

---

## 6. Implementation layout

```text
research/sc_hz/
  __init__.py
  precompute_direction.py        # d_L^r pre-computation
  prune.py                       # PRUNE operator + soundness test
  pruned_forward.py              # per-rival HZ forward with PRUNE
  run_sentinels.py               # 20-iid × 4-bench Phase A driver
  metrics.py                     # gate evaluation
  tests/
    __init__.py
    test_prune_soundness.py      # brute-force containment under random xi
    test_direction_chain.py      # d_L correctness on a synthetic 4-layer net
    test_forward_parity.py       # at K=ng (no prune), produces identical LP UB to baseline
audit_results/
  sc_hz_phase_a_<STAMP>/
    per_iid/
      iid<NNN>.json              # per-iid metrics + LP UB before/after
    gate.json                    # final gate evaluation
```

No file outside `research/sc_hz/` is modified in Phase A.

---

## 7. Unit-test entry matrix (Phase A entry condition)

Before any sentinel run:

| Test | Coverage |
|---|---|
| `test_prune_soundness` | 1000 deterministic quasi-grid / seeded samples `ξ` in `[-1, +1]^ng` on a toy ng=10 case; verify `(c + G ξ)` is contained in the pruned over-approx. This is a unit test, not a falsification method. |
| `test_direction_chain` | Synthetic 4-layer linear-only net; verify `d_0 = W_1^T W_2^T W_3^T d_3` matches a brute-force per-coord chain. |
| `test_forward_parity` | When K is set to ng (no pruning), the pruned forward HZ at output is bit-identical to the baseline forward HZ (LP UB matches within 1e-9). |
| `test_pruned_lp_ub_at_random_K` | At various K ∈ {64, 128, 256, 512}, the pruned LP UB is ≥ the unpruned LP UB (pruning is over-approximating, so LP UB can only loosen or stay same). |

These tests run with pytest and are independent of any production module.

---

## 8. Open design questions

The advisor should review these before Phase A code begins:

### 8.1 K-cap policy

Proposal: **K = 256 per layer, fixed**. Memory budget: per-rival HZ at each
layer has at most (hidden_dim × 257) generator matrix = 257 × hidden_dim ×
8 B. For VGG L33 FLATTEN (hidden_dim ≈ 25088), this is 51 MB per rival per
layer; for cifar100 (hidden_dim ≈ 100 at tail), 0.2 MB.

Alternative: per-layer adaptive K based on the rival direction's
"informativeness" (e.g. higher K where d_L has high variance). Defer to
Phase B based on what the sentinel run shows.

### 8.2 Per-rival forward parallelization

Phase A may run serial-per-rival or small batched groups. For 100-class
cifar100, serial 99-rival forward is expensive but acceptable on 80 sentinels
if the GPU job is bounded and checkpointed. If this is too slow, batch rivals
in groups of 5-10 before changing the mathematics.

If Phase A PASSES, Phase B introduces GPU batching for production scale.

### 8.3 Receipt format

Proposal: extend the existing CIFAR endcap receipt JSON with three new
fields:

```json
{
  ...existing fields...,
  "sc_hz_per_rival_lp_ub":  {rival_int: lp_ub_float},
  "sc_hz_lp_ub_reduction_pct":  median_across_rivals_pct,
  "sc_hz_K_per_layer":  K_int
}
```

No new files; the receipt schema is open for one revision after Phase A
shows what fields are needed.

### 8.4 Snapshot reuse vs fresh capture

Two paths:
- (A) Reuse the existing CIFAR / Tiny FLATTEN snapshots from prior sweeps
  (cheaper, but needs to verify per-rival forward starts from the same `c, G`).
- (B) Capture fresh snapshots in the sentinel driver (clean, but +30s overhead per iid).

Proposal: (B) for safety in Phase A. (A) optimization in Phase B if Phase A
PASSES.

### 8.5 Should `d_L^r` be computed offline (once per model) or per-iid?

Per (1.1) it depends only on weights, so it's offline per model. Cache as
`audit_results/sc_hz_cache/<model_sha256>/d_per_rival.pkl`. Reused across
iids that share the model.

---

## 9. Forbidden during Phase A (restated)

- No touch on production CIFAR / Tiny / nn4sys / malbeware / yolo paths.
- No touch on `act/back_end/hybridz_tf/*`.
- No modification to `verify_once_hz`.
- No CIFAR-ImageHZ revival.
- No `MILP` / `BaB` / `random` / `PGD`.
- No silent fallback when shapes don't match; fail-closed to UNK.

---

## 10. Phase A → next-phase trigger conditions

| Gate outcome | Next action |
|---|---|
| PASS | Phase B: 6-8 benchmark pilot with no-lost audit. Same K=256 + adaptive widening. |
| FAIL (V/A = 0 AND median LP UB reduction < 10%) | Close SC-HZ. Paper remains at 924 V/A. Independent engineering cleanup can proceed separately. |
| INCONCLUSIVE | Widen K=256→512 on the 10 worst sentinels; re-run those 10. If still inconclusive, advisor decides. |

---

## 11. Audit trail

- Research context: `research/hz_redesign_for_robustness_20260604.md` §9.
- Sentinel selection: `research/sc_hz/run_sentinels.py` writes
  `audit_results/sc_hz_phase_a_sentinels_<STAMP>.json` with the picked iids
  and selection criterion.
- Phase A receipts: `audit_results/sc_hz_phase_a_<STAMP>/`.
- Cross-reference: `research/cifar_finaltail_hull_plan.md` (the template) and
  `research/imagehz_vgg_prototype_plan.md` (the design-lock pattern).

---

## 12. What this plan does NOT change

- The 924 V/A canonical baseline.
- The paper skeleton's positive section (Section 3 of
  `paper_skeleton_20260604.md`).
- The three closure analyses (CIFAR-ImageHZ, VGG-ImageHZ, CIFAR final-tail
  hull).
- The provenance contract.
- The receipt format for any existing profile.

The full SC-HZ proposal becomes a *new contribution section* in
the paper only if Phase A through later productionization gates pass; otherwise it's a
closed-negative analysis that delineates the per-neuron-tightness ceiling
even more sharply.

This plan supersedes any prior DC-HZ / early SC-HZ discussion in the redesign analysis;
the redesign analysis remains the strategic context, this file is the
implementation-binding contract for Phase A.
