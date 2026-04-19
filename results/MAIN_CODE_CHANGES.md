# Main-Code Changes Applied During UCU → ACT Migration

This file records every change made to ACT main code (i.e. files outside `experiments/`) while porting the AIware 2026 paper's RQ1–RQ6 benchmarks from `UCU_Aiware` into this repository. Each entry identifies the file, the root cause, and the observed behavioural change.

Companion results: [COMPARISON.md](COMPARISON.md).

All fixes have been validated against `act/back_end/serialization/test_serialization.py`; the three pre-existing failures (unrelated format-version-1.0 issue in bundled JSONs) remain unchanged and no new regressions were introduced.

---

## Fix 1 — `act2torch` fail-loud on multi-predecessor topologies

**File:** [act/pipeline/verification/act2torch.py](../act/pipeline/verification/act2torch.py)

### Root cause
`ACTToTorch.run()` previously built an `nn.Sequential`-style `VerifiableModel` and *silently skipped* layers whose schema declared `requires_graph_restoration` (notably `ADD` for residual skip connections, plus `CONCAT` / `MAX` / `MIN`). The returned `nn.Sequential` therefore ran a different function than the one `analyze()` computed bounds for, and any `concrete ∈ bound?` check downstream of a dropped merge was meaningless — silently producing unsound verdicts on any residual architecture.

### Fix
`ACTToTorch.run()` now calls `_assert_chain_structure()` up front. If any ACT layer has more than one predecessor, it raises `NotImplementedError` with a descriptive message. Chain networks continue to convert to `VerifiableModel` exactly as before. This is a fail-loud contract that prevents silent unsoundness on multi-input ops until full DAG conversion is implemented.

### Impact
- No silent unsoundness on residual / CONCAT / MAX / MIN architectures.
- Chain behaviour unchanged.

---

## Fix 2 — `NetFactory` deterministic TF-capability filtering

**File:** [act/back_end/net_factory.py](../act/back_end/net_factory.py)

### Root cause
`NetFactory.sample_family()` previously called `rng.choice()` three times per instance (for `activation`, `pool_kind`, `downsample`) even when the YAML-sampled value was already allowed by the active TF set. Each extra RNG consumption shifted all downstream sampling, so `base_seed = 1015796661` produced networks wholly different from UCU's manifest, blocking reproducibility of the published experiments.

### Fix
The override is now deterministic: if the YAML-sampled value is in the allowed set, it is kept verbatim (no RNG consumed); only when the value is not permitted is the first entry of the allowed list used as a fallback. Applied symmetrically to `activation`, `pool_kind`, and `downsample`.

### Impact
- ACT now reproduces the first 4 of UCU's first 5 sampled networks byte-for-byte under `master_seed = 42`, `base_seed = 1015796661`.
- No change to sampling semantics when the YAML value is already TF-compatible.

---

## Fix 3 — `hz_apply_leaky_relu` reuses ReLU's 4+1+3 encoding

**File:** [act/back_end/hybridz_tf/tf_mlp.py](../act/back_end/hybridz_tf/tf_mlp.py)

### Root cause
`hz_apply_leaky_relu` previously introduced 6 continuous generators + 1 binary generator + 5 equality rows per unstable neuron (vs ReLU's 4+1+3). The extra slack compounded through depth: on a 26-layer mlp_block HybridZ's output bound was 38× wider than interval; on a 31-layer mlp_block, 196× wider — enough to make LeakyReLU networks fail bound-sanity checks that ReLU networks passed.

### Fix
The decomposition `y = max(s·x, x) = s·x + (1−s)·ReLU(x)` lets LeakyReLU reuse the ReLU 4+1+3 template exactly. Graph equalities and linking equality are identical; only the output formula gains two extra linear terms (`out_Gc[unstable, col_xi1] = s·α/2`, `out_Gb[unstable, col_z] = s·α/2`). When `s = 0` these terms vanish and LRELU degenerates to ReLU.

### Impact
- Post-fix HybridZ/interval ratios on deep LRELU blocks drop from 38× / 196× to **1.14× / 1.73×** — exponential blow-up eliminated.
- Same `(Gc, Gb, Ac, Ab, b)` shapes as ReLU, so downstream propagators need no changes.

---

## Fix 4 — `dual_tf` forward pass reads predecessor IDs from the graph

**File:** [act/back_end/dual_tf/tf_forward.py](../act/back_end/dual_tf/tf_forward.py)

### Root cause
`compute_forward_bounds` previously read `layer.params["x_src"]` and `layer.params["y_src"]` when handling ADD layers. But `NetFactory.create_network` writes ADD operands into `params["x_vars"]` / `params["y_vars"]` (variable IDs) and the *predecessor layer IDs* into `net.preds[layer.id]` — never populating `x_src` / `y_src`. The missing-key branch fell through to a fallback that produced `bounds_dict[ADD] == bounds_dict[main_pred]` (ignoring the skip-connection contribution) and yielded *unsound* bounds on residual networks: 2/30 residual nets in dual mode produced bounds tighter than the concrete reachable set, triggering a 6.7% detection rate on the `loosen_bounds` negative control.

### Fix
Predecessor layer IDs are now read from `net.preds.get(lid, [])`:
```python
pred_ids = list(net.preds.get(lid, []) or [])
if len(pred_ids) >= 2 and pred_ids[0] in bounds_dict and pred_ids[1] in bounds_dict:
    x_src, y_src = pred_ids[0], pred_ids[1]
```

### Impact
- `dual/loosen_bounds` detection rate: 6.7% → **0.0%** across all networks (negative control recovered).
- No change to non-ADD layer handling.

---

## Design principle: fail-loud over silent-drop

Fixes 1 and 4 share a common theme — ACT's previous code silently dropped information (merge layers in Fix 1, skip connections in Fix 4) rather than refusing to handle topologies it didn't support. Both paths produced plausible-looking but unsound bounds.

The chosen remedy is: **refuse to run rather than run with unsound bounds**. Fix 1 raises `NotImplementedError` on multi-predecessor networks; Fix 4 explicitly reads from the graph structure rather than from absent `params` keys. Neither fix expands ACT's supported topology set — they make the existing boundary explicit and observable.

---

## Not a main-code change, listed for completeness

**[experiments/validation_core.py](../experiments/validation_core.py) — `TARGET_CANDIDATE_WINDOW = 5`**

UCU's RQ1 runs capped `target_layer_id` at 6 across all 450 runs because its older interval analyzer produced Inf bounds on deeper layers, which `get_clean_bounds` filtered out. ACT's analyzer is numerically stable and returns finite bounds for every layer, so the candidate list now spans the full network and pushes target selection into middle-to-deep layers where mutations are harder to detect. Capping the candidate window to 5 mirrors UCU's effective behaviour without re-introducing the overflow. This lives in `experiments/` and is not a main-code change.
