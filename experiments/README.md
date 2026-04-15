# Experiment Scripts - Data Collection Guide

## Data Collection Workflow

The data in the paper's tables is obtained through the following workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Data Collection Workflow                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. NetFactory generates networks                                    │
│     ↓                                                                │
│  2. the framework verifier runs abstract interpretation              │
│     (interval/hybridz/dual)                                          │
│     ↓                                                                │
│  3. Collect layer bounds (layer_bounds)                               │
│     ↓                                                                │
│  4. TFMutator injects mutations (M1-M6)                              │
│     ↓                                                                │
│  5. CBR: sample and check output consistency                         │
│     BBL: check bound containment invariants                          │
│     ↓                                                                │
│  6. Collect detection rate, localization accuracy, time overhead      │
│     ↓                                                                │
│  7. Generate LaTeX tables                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Running Experiments

```bash
# Run all experiments with default seed
python experiments/run_all.py --seed 42

# Run individual RQ experiments
python experiments/rq1_detection.py --seed 42
python experiments/rq2_scc_effectiveness.py --seed 42
python experiments/rq3_localization.py --seed 42
python experiments/rq4_coverage.py --seed 42
python experiments/rq5_cross_domain.py --seed 42
python experiments/rq6_overhead.py --seed 42
```

## Data Source Description

### RQ1: Detection Rate Data (Table 1)

**Setup:** For RQ1, we generate 30 networks per (domain, mutation) combination, yielding 30 × 3 domains (interval, hybridz, dual) × 6 mutations (M1\_TIGHTEN, M2\_LOOSEN, M3\_SWAP, M4\_ZERO\_LB, M5\_SCALE\_UB, M6\_NOISE) = 540 validation instances. Each instance loads a network, runs abstract interpretation under the specified domain, applies the mutation to a randomly selected intermediate layer, and checks detection via both CBR (sampling-based) and BBL (bound-containment-based).

| Data Item | Source |
|-----------|--------|
| CBR Only | `scc_result.status == SCCStatus.FAIL` |
| BBL Only | `bca_result.status == BCAStatus.FAIL` |
| Combined | `soundness_violated` (detected by either SCC or BCA) |
| Localized | `violation_localized` (detected by BCA with violation records) |

**Run:**
```bash
python experiments/rq1_detection.py --seed 42 -v
```

**Output:** `results/rq1/results.json`, `results/rq1/table_rq1.tex`

### RQ2: CBR Effectiveness Data (Table 2)

**Setup:** For RQ2, we generate 30 networks per (spec\_type, dimension) combination, yielding 30 × 3 spec types (BOX, LINF\_BALL, LIN\_POLY) × 4 input dimensions (4, 16, 64, 256) = 360 validation instances. Each instance builds a network with the specified input dimension (MLP for d=4,16; CNN2D for d=64,256), runs interval analysis, applies M1\_TIGHTEN mutation to the output layer (factor=0.5), and uses CBR with a sampling budget of 20 to check whether counterexamples can be discovered.

| Data Item | Source |
|-----------|--------|
| Discovery Rate | Proportion of samples that found counterexamples |
| Inconclusive | `scc_result.status == SCCStatus.INCONCLUSIVE` |
| Avg Time | CBR runtime (ms) |

**Note:** LIN\_POLY specifications cannot be directly sampled, so Discovery Rate = 0%

### RQ3: Localization Accuracy Data (Table 3)

**Setup:** For RQ3, we generate 30 networks per architecture type, yielding 30 × 3 architectures (sequential\_mlp, sequential\_cnn, residual) = 90 validation instances. Each instance loads a network of the specified architecture, runs interval-domain analysis, applies M1\_TIGHTEN mutation to a randomly selected intermediate layer, and performs BBL detection. Localization accuracy is measured by checking whether the mutated (buggy) layer appears in the top-1 or top-5 layers ranked by bound violation severity.

| Data Item | Source |
|-----------|--------|
| Top-1 Hit | `violations[0].layer_id == target_layer_id` |
| Top-5 Hit | `target_layer_id in [v.layer_id for v in violations[:5]]` |
| Error Rate | `bca_result.status == BCAStatus.ERROR` |

### RQ4: Coverage Data (Table 4)

**Setup:** For RQ4, we evaluate three generation strategies (Basic-50, Basic-100, Full-100) using the NetFactory. Basic strategies generate networks via random sampling with the specified budget (50 or 100). Full-100 uses the same random budget of 100 but supplements with coverage-directed minimal templates to achieve 100% operator coverage. Coverage is measured over 15 trackable operators (the intersection of operators supported by all three abstract domains). All strategies use the same base seed so Basic-100's first 50 networks are identical to Basic-50's.

| Data Item | Source |
|-----------|--------|
| Op Coverage | `covered_layers / total_trackable_layers` |
| Bug Yield | Number of detected bugs |

### RQ5: Cross-Domain Comparison Data (Table 5)

**Setup:** For RQ5, we generate 100 networks per domain, yielding 100 × 3 domains (interval, hybridz, dual) = 300 validation instances. Each instance applies M1\_TIGHTEN mutation and runs BBL detection under the specified abstract domain. We compare BBL detection failure rates and average bound widths across domains to assess how domain precision affects validation effectiveness.

| Data Item | Source |
|-----------|--------|
| BBL Fail Rate | BBL detection failure rate |
| Bound Width | `avg(ub - lb)` average bound width |
| Disagreement | Proportion of inconsistent results across different domains |

### RQ6: Overhead Data (Table 6)

**Setup:** For RQ6, we measure validation overhead across 3 model sizes: small (16→[32,32]→4, ~1K params), medium (64→[128,128,64]→10, ~33K params), and large (256→[512,256,128]→10, ~297K params). For each size, we measure CBR overhead at 4 sampling budgets (5, 10, 20, 50) and BBL overhead (budget-independent). Each timing measurement uses 3 warmup runs followed by 10 timed runs, reporting the average. The "Complementary" column shows the combined CBR+BBL overhead.

| Data Item | Source |
|-----------|--------|
| Params | Number of model parameters |
| CBR (ms) | CBR runtime |
| BBL (ms) | BBL runtime |
| Overhead | `(scc + bca) / analysis_time` |

## Output File Structure

```
results/
├── rq1/
│   ├── metadata.json      # Experiment metadata (seed, configuration)
│   ├── results.json       # Complete experiment results
│   └── table_rq1.tex      # LaTeX table
├── rq2/
│   ├── results.json
│   └── ...
└── experiment_summary.json # Summary statistics
```

## Core Code Paths

- **Data Collector:** `experiments/data_collector.py`
- **CBR:** `act/back_end/validation/scc.py`
- **BBL:** `act/back_end/validation/bca.py`
- **Mutation Operations:** `act/back_end/validation/mutations.py`
- **Abstract Analysis:** `act/back_end/analyze.py`
- **Transfer Functions:** `act/back_end/interval_tf/`, `hybridz_tf/`, `dual_tf/`

## Reproducibility Verification

```bash
# Generate baseline results
python experiments/verify_reproducibility.py --seed 42 --generate-baseline

# Verify reproducibility
python experiments/verify_reproducibility.py --seed 42 --verify
```

## Full Experiment Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments
python experiments/run_all.py --seed 42

# 3. View results
cat results/experiment_summary.json

# 4. Verify reproducibility
python experiments/verify_reproducibility.py --seed 42 --verify
```

## Frequently Asked Questions

### Q: How long do the experiments take?
A: It depends on the number and size of networks.

### Q: How do I run only specific RQs?
A: Use the `--experiments` parameter:
```bash
python experiments/run_all.py --experiments rq1 rq3
```
