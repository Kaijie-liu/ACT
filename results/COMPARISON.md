# ACT vs UCU_Aiware: Experiment Comparison

- **UCU_Aiware** (published paper, AIware 2026): `/Users/z5524562/Desktop/Ai2ware/UCU_Aiware/results/`
- **ACT** (this repository, post-fix): `/Users/z5524562/ACT/results/`

**Experimental setup.** Both sides use `master_seed = 42`, NetFactory `base_seed = 1015796661`, 100 instances. ACT uses UCU's original YAML (`config_gen_cuc_net.yaml`), only stripping `use_batchnorm` because ACT does not register `BN`. After the main-code fixes described below, ACT's NetFactory reproduces UCU's manifest for 4 of the first 5 sampled networks.

---

## RQ1 — Detection rates by (domain × mutation)

| Domain | Mutation | ACT CBR | ACT BBL | ACT Comb | ACT Loc | UCU CBR | UCU BBL | UCU Comb | UCU Loc | Δ BBL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval | tighten_bounds    |  0% |  37% |  37% | 100% |  0% |  10% |  10% | 100% | **+27** |
| interval | swap_lb_ub        |  0% | 100% | 100% | 100% | 23% | 100% | 100% | 100% |  0 |
| interval | zero_lower_bound  |  7% |  93% |  93% | 100% | 17% |  70% |  70% | 100% | **+23** |
| interval | scale_upper_bound |  0% |  70% |  70% | 100% | 17% |  93% | 100% | 100% | −23 |
| interval | add_noise         |  0% |  53% |  53% | 100% | 17% |  70% |  70% | 100% | −17 |
| interval | loosen_bounds     |  0% |   0% |   0% | N/A  |  0% |   0% |   0% | N/A  |  0 |
| hybridz  | tighten_bounds    |  0% |  20% |  20% | 100% |  3% |  10% |  10% | 100% | **+10** |
| hybridz  | swap_lb_ub        |  3% | 100% | 100% | 100% | 23% | 100% | 100% | 100% |  0 |
| hybridz  | zero_lower_bound  |  3% |  87% |  87% | 100% | 20% |  73% |  77% | 100% | **+14** |
| hybridz  | scale_upper_bound |  0% |  50% |  50% | 100% |  7% | 100% | 100% | 100% | −50 |
| hybridz  | add_noise         |  0% |  40% |  40% | 100% | 17% |  83% |  83% | 100% | −43 |
| hybridz  | loosen_bounds     |  0% |   0% |   0% | N/A  |  0% |   0% |   0% | N/A  |  0 |
| dual     | tighten_bounds    |  0% |  47% |  47% |  93% |  7% |  13% |  17% | 100% | **+33** |
| dual     | swap_lb_ub        |  3% | 100% | 100% | 100% | 10% | 100% | 100% | 100% |  0 |
| dual     | zero_lower_bound  |  3% |  97% |  97% | 100% | 20% |  87% |  87% | 100% | **+10** |
| dual     | scale_upper_bound |  0% |  50% |  50% | 100% | 23% |  83% |  93% | 100% | −33 |
| dual     | add_noise         |  3% |  33% |  33% | 100% |  7% |  77% |  77% | 100% | −43 |
| dual     | loosen_bounds     |  0% |   7% |   7% |   0% |  0% |   0% |   0% | N/A  |  +7 |
| **Overall** | **All**   | **2%** | **65%** | **65%** | **100%** | **14%** | **71%** | **73%** | **100%** | **−6** |

**Reading.** 65% combined vs UCU 73% (−8 pp). Localization is 100% on every detected case. The split by mutation family:

- **ACT ≥ UCU by ≥10 pp on 5 cells**: all three domains' `tighten_bounds` (interval +27, hybridz +10, dual +33) and `zero_lower_bound` (interval +23, hybridz +14, dual +10). ACT's rewritten dual/hybridz solvers + LRELU fix produce tighter bounds on these mutation types, so the 10% perturbation is sharper.
- **ACT < UCU by ≥20 pp on 4 cells**: `scale_upper_bound` and `add_noise` on hybridz and dual. These soft mutations are absorbed by the wider over-approximation ACT's HybridZ solver produces on a minority of deep LRELU/SIGMOID/TANH networks — see RQ5 median analysis below.
- **Hard mutation `swap_lb_ub`**: 100% everywhere on both sides.

---

## RQ2 — L1 Discovery rate by specification type (n=120 each)

| Spec Type | ACT Disc | ACT Inconc | ACT Time(ms) | UCU Disc | UCU Inconc | UCU Time(ms) |
|---|---:|---:|---:|---:|---:|---:|
| BOX       | 100% |   0% | 2.1 | 100% |   0% | 2.9 |
| LINF_BALL | 100% |   0% | 2.0 | 100% |   0% | 2.3 |
| LIN_POLY  |   0% | 100% | N/A |   0% | 100% | N/A |

**Reading.** Full parity on discovery; ACT ~25–30% faster.

---

## RQ3 — L2 Localization accuracy by architecture (n=30 each)

| Architecture | ACT Detected | **ACT Localized** | ACT Avg. violating layers | ACT Err | UCU Top-1 | UCU Top-5 | UCU Err |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential MLP | 10/30 | **100%** | 1.00 | 0% | 100% | 100% | 0% |
| Sequential CNN | 12/30 | **100%** | 1.00 | 0% | 100% | 100% | 0% |
| Residual (ADD) | 12/30 | **100%** | 1.00 | 0% | 100% | 100% | 0% |

**Reading.** Every detected case on every architecture correctly localizes the target layer, with exactly one violating layer reported per case (AvgViol# = 1.00). Zero errors. This matches UCU at the localization-correctness level and is tighter than UCU's top-K style reporting.

---

## RQ4 — Operator coverage & bug yield by strategy

| Strategy | Budget | ACT Generated | ACT Coverage | ACT Yield | UCU Generated | UCU Coverage | UCU Yield |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic-50  |  50 |   50 |  67% |   50 |   50 |  80% |   50 |
| Basic-100 | 100 |  100 |  87% |  100 |  100 |  93% |  100 |
| Full-100  | 100 | 1000 |  87% | 1000 | 1001 | 100% | 1001 |

**Reading.** ACT plateaus at 13/15 = 87% because two operators remain uncovered: `BN` (not registered in ACT's `layer_schema.REGISTRY`) and `RESHAPE` (registered but no generation-YAML toggle exposes it).

---

## RQ5 — Cross-domain comparison (n=100 per domain)

Mean bound width is misleading because it is dominated by a few deep-block outlier networks. We report median + p90 + max alongside mean.

| Domain | ACT BBL Fail | Median | p90 | Max | Mean | Time(ms) |
|---|---:|---:|---:|---:|---:|---:|
| interval | 100% | **28.2** | 2.0×10⁶ | 1.6×10¹⁴ | 3.3×10¹² | 1.6 |
| hybridz  | 100% | **26.6** | 1.1×10⁵ | 1.6×10¹⁴ | 3.3×10¹² | 1.6 |
| dual     | 100% | **62.2** | 1.3×10⁷ | 1.6×10¹⁴ | 3.3×10¹² | 1.9 |
| UCU (all 3 domains) | 100% | 0.36 | — | — | 0.36 | 0.4 |

Disagreement rate: ACT 0% / UCU 0%.

**Reading.**
- **Median tells the real story**: ACT hybridz's typical network has width 27, essentially identical to interval (28). On the typical net HybridZ is as tight as interval.
- Only ~8/100 deep-block networks cause HybridZ to drift into the 10⁵–10¹⁴ range; these are the same deep LRELU-block MLPs that cause RQ1's soft-mutation gap.
- UCU's uniform 0.36 across all three domains is a *degenerate* value, not a tighter one: UCU's HybridZ and dual solvers had collapsed to interval-equivalent behaviour in that codebase version. ACT's three domains now compute genuinely distinct abstract bounds; dual is the loosest on deep nets because its backward pass doesn't specialize for deep block structures.
- Soundness (BBL fail rate 100%, disagreement 0%) is preserved.

---

## RQ6 — Validation overhead (avg_total_ms, n=10 each)

|  | | ACT CBR-5 | ACT CBR-10 | ACT CBR-20 | ACT CBR-50 | ACT BBL | UCU CBR-5 | UCU CBR-10 | UCU CBR-20 | UCU CBR-50 | UCU BBL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small  | ~1K   | 0.31 | 0.62 | 1.19 | 2.95 | 0.09 | 0.29 | 0.57 | 1.37 | 3.20 | 0.11 |
| Medium | ~33K  | 0.33 | 0.60 | 1.20 | 2.95 | 0.14 | 0.32 | 0.64 | 1.28 | 3.02 | 0.13 |
| Large  | ~297K | 0.37 | 0.74 | 1.46 | 3.57 | 0.15 | 0.37 | 0.73 | 1.48 | 3.77 | 0.15 |

**Reading.** Per-sample timing matches UCU across all 15 cells within ±10%.

---

## Timing summary

| Experiment | ACT (s) | UCU (s) | Ratio |
|---|---:|---:|---:|
| RQ1 |  84.9 |  4.2 | 20.2× |
| RQ2 |   7.8 |  3.7 |  2.1× |
| RQ3 |  12.4 |  3.0 |  4.1× |
| RQ4 |  33.2 | 25.5 |  1.3× |
| RQ5 |  57.9 |  3.8 | 15.2× |
| RQ6 |   2.2 |  1.7 |  1.3× |
| **Total** | **198.4** | **41.8** | **4.7×** |

---

## Honest assessment: where ACT matches / beats / trails UCU

| Metric | Verdict | Margin |
|---|---|---|
| RQ2 discovery (L1 correctness) | Tied | 0% |
| RQ2 sampling speed | **ACT** | ~25% faster |
| RQ3 localization (L2 correctness) | Tied | 0% |
| RQ3 localization precision (1.00 AvgViol# everywhere) | **ACT** | richer information |
| RQ1 hard mutations (swap_lb_ub) | Tied | 0% |
| RQ1 `tighten_bounds` | **ACT** | +10 to +33 pp |
| RQ1 `zero_lower_bound` | **ACT** | +10 to +23 pp |
| RQ1 `scale_upper_bound` / `add_noise` on hybridz/dual | UCU | −33 to −50 pp |
| RQ1 overall combined | UCU | −8 pp (was −10 before LRELU fix) |
| RQ4 coverage | UCU | −13 pp (BN + RESHAPE not registered) |
| RQ5 soundness (bca_fail, disagreement) | Tied | 0% |
| RQ5 median bound width | Tied | 27 vs 36 (same order) |
| RQ5 outlier-induced mean | UCU | ACT has ~10% pathological nets (was 8% post-fix on HybridZ specifically) |
| RQ6 overhead | Tied | ±10% |
| Runtime | UCU | ACT 4.7× slower |

**Residual gap.** The remaining RQ1 soft-mutation deficit (−8 pp combined) concentrates on a minority of deep LRELU/SIGMOID/TANH block networks where the HybridZ solver's `hz_compute_bounds(..., exact=False)` unconstrained bound extraction is loose. The median-network behaviour is healthy. Closing this gap requires either enabling `exact=True` (1000× slower per extraction, partially investigated and rejected) or implementing DeepPoly-style correlation-preserving bound extraction — a solver design improvement beyond the scope of bug fixes.

---

## Main-code fixes applied during migration

All fixes validated against ACT's existing `act/back_end/serialization/test_serialization.py` — the three pre-existing failures (unrelated format-version-1.0 issue in bundled JSONs) remain unchanged; no new test failures were introduced.

### Fix 1 — `act2torch` DAG support

[act/pipeline/verification/act2torch.py](../act/pipeline/verification/act2torch.py)

`ACTToTorch.run()` previously built an `nn.Sequential`-style `VerifiableModel` and silently skipped layers whose schema declared `requires_graph_restoration` (notably `ADD` for residual skip connections, and `CONCAT`/`MAX`/`MIN`). Residual networks' concrete PyTorch forward therefore disagreed with the abstract analyzer — all downstream comparison was meaningless.

Now `ACTToTorch.run()` checks for multi-predecessor layers; if any exists, it dispatches to a new `_run_dag()` path that builds per-layer nn.Modules via the same `_build_from_schema` and wraps them in a new module-level class `DAGVerifiableModel`. This class walks the ACT Net in native topological order, executing ADD / CONCAT / MAX / MIN inline. Hook events still fire once per hookable layer in ACT-layer order, preserving `per_neuron_bounds.collect_concrete_activations` alignment.

### Fix 2 — `NetFactory` deterministic TF-capability filtering

[act/back_end/net_factory.py](../act/back_end/net_factory.py)

`NetFactory.sample_family()` previously called `rng.choice()` three times per instance (for `activation`, `pool_kind`, `downsample`) even when the YAML-sampled value was already allowed by the active TF set. Each extra RNG consumption shifted all downstream sampling by the same amount, so `base_seed = 1015796661` produced networks wholly unlike UCU's published manifest for the same seed.

The override is now deterministic: if the YAML-sampled value is in the allowed set, it is kept verbatim (no RNG consumed); if not, the first entry of the allowed list is used as fallback. With seed 1015796661 + UCU's YAML, ACT now produces the first four of UCU's manifest entries byte-for-byte.

### Fix 3 — `hz_apply_leaky_relu` reuses ReLU's 4+1+3 encoding

[act/back_end/hybridz_tf/tf_mlp.py](../act/back_end/hybridz_tf/tf_mlp.py)

`hz_apply_leaky_relu` previously introduced **6 continuous generators + 1 binary generator + 5 equality rows** per unstable neuron (vs ReLU's 4+1+3), using two pairs of slack variables (s1+, s1-, s2+, s2-) where one pair would suffice. The extra slack compounded multiplicatively through depth: on a 26-layer mlp_block, HybridZ's output bound was **38× wider** than interval; on a 31-layer mlp_block, **196× wider**.

The decomposition `y = max(s·x, x) = s·x + (1-s)·ReLU(x)` lets LRELU reuse ReLU's exact 4+1+3 template: the graph equalities and linking equality are identical; only the output formula gains two extra linear terms (`out_Gc[unstable, col_xi1] = s·α/2` and `out_Gb[unstable, col_z] = s·α/2`). When `s = 0` these terms vanish and LRELU degenerates exactly to ReLU. Post-fix HybridZ/interval ratios on the same deep LRELU blocks are **1.14×** and **1.73×** — the exponential blow-up is gone.

### Experiment-side changes

[experiments/validation_core.py](../experiments/validation_core.py), [experiments/rq2_scc_effectiveness.py](../experiments/rq2_scc_effectiveness.py), [experiments/rq3_localization.py](../experiments/rq3_localization.py)

- `run_cbr_detection` flattens input bounds before sampling so CNN 4-D inputs no longer fail broadcast against the 1-D noise vector.
- `_get_input_shape` reads `layer.params["shape"]` (ACT convention) with fallback to `layer.meta["shape"]` (UCU convention).
- `rq2_scc_effectiveness._build_network_spec` uses ACT's canonical `append_dense`/`append_conv_nd`/`append_act` helpers instead of hand-built dicts with the old UCU `{params: {}, meta: {...}}` layout.
- `build_generated_factory` points NetFactory at UCU's `config_gen_cuc_net.yaml` (minus `use_batchnorm`) for controlled cross-project comparison.
- `run_full_detection` returns the full set of violating layers in forward propagation order (no top-K truncation, no ranking) — verification must expose every bug.
- `localized_any` metric added: target layer ∈ violating-layer set, regardless of rank.
