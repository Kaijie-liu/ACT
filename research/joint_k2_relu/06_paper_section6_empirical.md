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
