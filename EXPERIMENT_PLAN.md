# Experiment Plan: Soundness Validation for Neural Network Verifiers

**Paper**: *[Title Redacted for Anonymous Submission]*

**Problem Statement**: Neural network verifiers may contain bugs (e.g., unsound abstract transformers, numerical errors) that lead to incorrect certification results. This work proposes a two-level validation framework to detect and localize such defects.

---

## 1. Two-Level Validation Architecture

The framework performs two complementary levels of checking, from output-level semantics to internal bound invariants:

```
+---------------------------------------------------------------+
|                       Verifier Under Test                      |
|                                                                |
|   BBL (Bounds-based Localization)                       |
|     Per-layer check: concrete activation v in [lb, ub]         |
|     Layer 1 -> Layer 2 -> ... -> Layer N                       |
|                                                                |
|   CBR (Counterexample-based Refutation)                          |
|     Input x -> [Network] -> Output y                           |
|     input_satisfied(x) AND NOT output_satisfied(y) => witness  |
+---------------------------------------------------------------+
```

### 1.1 CBR (Counterexample-based Refutation)

**Scope**: Output-level -- validates the verifier's final verdict.

**Objective**:
- Detect false CERTIFIED results (verifier claims safe, but concrete counterexamples exist)
- Validate FALSIFIED results (check whether reported counterexamples are genuine)

**Method**:
1. Extract the sampable region from the input specification (BOX or LINF_BALL)
2. Sample concrete input points within the region
3. Execute the network to obtain concrete outputs
4. Check: `input_satisfied(x) AND NOT output_satisfied(y)` indicates a real counterexample

**Verdict logic**:

| Condition | Verifier says CERTIFIED | Verifier says FALSIFIED | Verifier says UNKNOWN |
|-----------|------------------------|-------------------------|-----------------------|
| Counterexample found | **FAIL** (unsoundness) | PASS (witness consistent) | ACCEPTABLE |
| No counterexample found | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |

**Limitations**: Depends on random sampling (low coverage in high dimensions); LIN_POLY specifications cannot be directly sampled; only detects output-level errors.

### 1.2 BBL (Bounds-based Localization)

**Scope**: Internal-level -- validates each abstract transformer step.

**Objective**:
- Check the fundamental soundness invariant of abstract interpretation: concrete values must lie within abstract bounds
- Precisely localize faults to specific layers and neurons

**Method**:
1. Collect concrete activations at each layer via forward hooks
2. Obtain abstract bounds [lb, ub] from the verifier at each layer
3. Align activations with bounds by execution order
4. Check containment invariant: `lb - tau <= v <= ub + tau` (tau = numerical tolerance)

**Violation metric**: `gap = max(lb - tau - v, v - ub - tau, 0)`. A positive gap indicates a soundness violation.

**Verdict logic**:
- `gap > 0` at any neuron: **FAIL**, report violation location and severity
- Alignment failure (non-sequential architecture, repeated calls): **ERROR**, conservatively abstain
- Otherwise: **PASS**

**Advantages**: Deterministic invariant checking (no sampling dependence); precise layer/neuron-level localization; complements CBR when sampling is inconclusive.

**Limitations**: Alignment issues with non-sequential architectures (residual, multi-branch); stateful layers (e.g., BatchNorm) require special handling.

### 1.3 Complementary Relationship

| Scenario | CBR | BBL | Interpretation |
|----------|---------------|---------------|----------------|
| 1 | FAIL | FAIL | Confirmed unsoundness, localizable |
| 2 | FAIL | PASS | Output-level error; bounds correct but constraint/solver issue |
| 3 | FAIL | ERROR | Output-level error; internal state inconclusive |
| 4 | INCONCLUSIVE | FAIL | Sampling missed it, but internal bounds violated |
| 5 | INCONCLUSIVE | PASS | No issues found (limited coverage) |
| 6 | PASS | -- | Only when FALSIFIED verdict is correctly witnessed |

CBR provides black-box input-output checking; BBL provides white-box internal bound checking. Together they cover both output-level and internal-level defects.

---

## 2. Research Questions

| RQ | Question | Target |
|----|----------|--------|
| RQ1 | Detection of injected unsoundness violations | Combined CBR + BBL detection capability |
| RQ2 | Effectiveness boundary of CBR | Impact of specification type and input dimension on sampling |
| RQ3 | Localization accuracy of BBL | Localization capability and alignment reliability across architectures |
| RQ4 | Does TF-aware generation improve coverage? | Operator coverage vs. bug discovery correlation |
| RQ5 | Behavioral differences across abstract domains | Interval vs. HybridZonotope vs. DeepPoly |
| RQ6 | Validation overhead | Runtime cost of CBR and BBL |

---

## 3. Mutation Operators (Bug Injection)

To evaluate detection capability, the following mutation operators inject known defects into abstract transformers:

| ID | Mutation | Effect | Expected Detection |
|----|----------|--------|--------------------|
| M1 | Tighten bounds | Shrink [lb, ub] by factor -- soundness violation | BBL should detect |
| M2 | Loosen bounds | Expand [lb, ub] by factor -- negative control | Neither level should flag |
| M3 | Swap lb/ub | Exchange lower and upper bounds -- severe violation | BBL should detect immediately |
| M4 | Zero lower bound | Set lb = 0 -- soundness violation for negative activations | BBL should detect |
| M5 | Scale upper bound | Multiply ub by factor < 1 -- tighter upper bound | BBL should detect |
| M6 | Add noise | Random perturbation to bounds -- stochastic violation | BBL may detect |

M2 (Loosen) serves as a negative control: since it only widens bounds, it preserves soundness and should not trigger any detection.

---

## 4. Module Reference

### Core Modules

| Module | Location | Responsibility |
|--------|----------|----------------|
| CBR | `cuc/pipeline/verification/validate_verifier.py` | Output-level counterexample search |
| BBL | `cuc/pipeline/verification/per_neuron_bounds.py` | Per-neuron bound containment audit |
| Mutation injection | `cuc/back_end/validation/mutations.py` | TFMutator, MutationType, MutationConfig |
| Reproducibility | `cuc/back_end/validation/reproducibility.py` | Seed management, ExperimentMetadata |
| Device management | `cuc/util/device_manager.py` | CPU/GPU device selection |

### Key APIs

**CBR**:
```python
from cuc.pipeline.verification.validate_verifier import VerificationValidator

validator = VerificationValidator(device="cuda", dtype=torch.float64)
results = validator.validate_counterexamples(networks=[...], solvers=['torchlp'])
# Returns: validation_status, concrete_counterexample, samples_tried
```

**BBL**:
```python
from cuc.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig, run_per_neuron_bounds_check
)

config = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10)
result = run_per_neuron_bounds_check(net, model, input_tensor, tf_mode="interval", config=config)
# Returns: status, violations, worst_gap, layers_checked, neurons_checked
```

**Comprehensive validation (CBR + BBL)**:
```python
validator.validate_comprehensive(
    networks=[...], tf_modes=["interval"], solvers=['torchlp'],
    per_neuron_config=config
)
```

---

## 5. Experiment Configuration

```yaml
# experiments/config.yaml

reproducibility:
  master_seed: 42
  record_seeds: true
  deterministic_mode: true
  verify_on_load: true

experiment:
  num_trials: 30
  output_base_dir: "results"
  save_networks: true
  save_intermediate: true

level1:
  sampling_budget: 20
  strategies: ["uniform", "boundary", "center"]

level2:
  tolerance: 1e-5
  max_violations_per_layer: 10

# --- Per-RQ Configuration ---

rq1:
  description: "Two-level detection capability"
  seed_offset: 1000
  num_networks: 100
  mutations: [M1, M3, M4, M5, M6]
  control: [M2]
  domains: [interval, hybridz, dual]
  mutation_factor: 0.1

rq2:
  description: "CBR effectiveness boundary"
  seed_offset: 2000
  spec_types: [BOX, LINF_BALL, LIN_POLY]
  dimensions: [4, 16, 64, 256]
  networks_per_config: 30

rq3:
  description: "BBL localization accuracy"
  seed_offset: 3000
  architectures: [sequential_mlp, sequential_cnn, residual]
  networks_per_arch: 30
  mutation: M1
  topk: [1, 5]

rq4:
  description: "TF-aware generation coverage"
  seed_offset: 4000
  configs:
    basic_50:  {mode: random, n: 50, seed_offset: 0}
    basic_100: {mode: random, n: 100, seed_offset: 100}
    full_100:  {mode: coverage, target: 0.95, max_attempts: 100, seed_offset: 200}

rq5:
  description: "Cross-domain comparison"
  seed_offset: 5000
  num_networks: 100
  domains: [interval, hybridz, dual]

rq6:
  description: "Validation overhead"
  seed_offset: 6000
  sampling_budgets: [5, 10, 20, 50]
  model_sizes:
    small:  {params: ~2K}
    medium: {params: ~34K}
    large:  {params: ~297K}
```

### Seed Derivation

All randomness is derived deterministically from a single master seed (default: 42):

| Component | Derivation | Example |
|-----------|-----------|---------|
| RQ experiment | `rq_seed = master_seed + seed_offset` | RQ1: 42 + 1000 = 1042 |
| Network generation | `net_seed = SHA256(rq_seed, net_idx, instance_id)` | SHA256(1042, 0, "mlp_...") |
| Weight initialization | `weight_seed = SHA256(net_seed, "weights")` | SHA256(12345, "weights") |
| CBR sampling | `sample_seed = SHA256(net_seed, "scc", sample_idx)` | SHA256(12345, "scc", 0) |
| Mutation injection | `mutation_seed = SHA256(net_seed, "mutation", layer_id)` | SHA256(12345, "mutation", 3) |

---

## 6. RQ6 Results Summary

Configuration: 4 budgets x 3 model sizes = 12 configurations, mean of 10 runs each.

**CBR overhead (scales linearly with budget, in ms):**

| Model Size | Params | Budget=5 | Budget=10 | Budget=20 | Budget=50 |
|-----------|--------|----------|-----------|-----------|-----------|
| Small     | ~2K    | 0.50     | 0.85      | 1.34      | 2.81      |
| Medium    | ~34K   | 0.29     | 0.58      | 1.16      | 2.84      |
| Large     | ~297K  | 0.36     | 0.73      | 1.44      | 3.59      |

**BBL overhead (constant across budgets, in ms):**

| Model Size | BBL |
|-----------|-----|
| Small     | 0.09 |
| Medium    | 0.13 |
| Large     | 0.13 |

**Combined overhead (CBR + BBL, in ms):**

| Model Size | Budget=5 | Budget=10 | Budget=20 | Budget=50 |
|-----------|----------|-----------|-----------|-----------|
| Small     | 0.59     | 0.93      | 1.42      | 2.90      |
| Medium    | 0.42     | 0.70      | 1.28      | 2.97      |
| Large     | 0.49     | 0.86      | 1.57      | 3.72      |

**Key findings:**
1. CBR overhead scales linearly with sampling budget (5x--10x increase from budget 5 to 50)
2. BBL overhead is independent of budget (constant 0.09--0.13 ms)
3. Maximum total overhead remains below 4 ms even at budget = 50
4. At default budget = 20, combined overhead is 1.28--1.57 ms

Results stored in `results/rq6/` (results.json, table_rq6.tex, fig_rq6_overhead.pdf/png/csv, metadata.json).

---

## 7. Directory Structure

```
cuc/
  back_end/
    validation/
      __init__.py              # Unified exports
      mutations.py             # Mutation operators (bug injection)
      reproducibility.py       # Seed management, ExperimentMetadata
  pipeline/
    verification/
      validate_verifier.py     # CBR + BBL validation coordinator
      per_neuron_bounds.py     # BBL: bound containment audit core
      model_factory.py         # Model factory
  util/
    device_manager.py          # Device management (CPU/GPU)

experiments/
  config.yaml                  # Experiment configuration (master_seed = 42)
  rq1_detection.py             # RQ1: Detection capability
  rq2_scc_effectiveness.py     # RQ2: CBR effectiveness
  rq3_localization.py          # RQ3: BBL localization
  rq4_coverage.py              # RQ4: TF-aware coverage
  rq5_cross_domain.py          # RQ5: Cross-domain comparison
  rq6_overhead.py              # RQ6: Overhead
  run_all.py                   # Run all experiments
  verify_reproducibility.py    # Reproducibility verification

results/
  rq1/ ... rq6/                # Per-RQ results (metadata.json, results.json, tables, figures)
```

---

## 8. Reproducibility Checklist

- All experiments use a fixed master_seed (default: 42)
- Per-network derived seeds recorded in manifest.json
- Full environment info recorded (Python, PyTorch, NumPy versions)
- Configuration file hash recorded
- Git commit hash recorded
- Deterministic mode enabled for PyTorch (cudnn.deterministic = True, cudnn.benchmark = False)

### Artifact Structure

```
artifcuc/
  README.md                     # Reproduction guide
  requirements.txt              # Pinned dependency versions
  experiments/
    config.yaml                 # Experiment configuration (master_seed = 42)
    run_all.py                  # Single-command runner
  results/
    rq1/metadata.json           # Contains all seeds
    rq1/results.json            # Expected results (for verification)
    ...
  verify.sh                    # Verification script
```

### Verification

```bash
pip install -r requirements.txt
python experiments/run_all.py --seed 42 --output-dir results_verify
python experiments/verify_reproducibility.py --expected results/ --actual results_verify/
```

---

## 9. Summary

| Level | Name | Scope | Checks | Strength | Limitation |
|-------|------|-------|--------|----------|------------|
| CBR | CBR | Output-level | Verifier verdict vs. concrete execution | Intuitive, fast | Sampling-dependent |
| BBL | BBL | Internal-level | Bound containment invariant | Precise localization | Alignment issues |

The two levels are complementary: CBR catches output-level errors through black-box sampling, while BBL catches internal-level errors through white-box invariant checking. Together they provide comprehensive soundness validation for neural network verifiers.
