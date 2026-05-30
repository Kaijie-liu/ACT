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
