#!/usr/bin/env python3
"""
RQ6: Validation Runtime Overhead Measurement

Measures the runtime overhead of CBR and BBL
validation under different configurations.

Reproducible Run:
    python experiments/rq6_overhead.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cuc.back_end.validation import (
    set_all_seeds,
    derive_seed,
    ExperimentMetadata,
)
from cuc.back_end.core import Bounds

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_mock_model(input_dim: int, hidden_sizes: List[int], output_dim: int) -> torch.nn.Module:
    """Create a mock PyTorch model."""
    layers = []
    prev_size = input_dim

    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(prev_size, hidden_size))
        layers.append(torch.nn.ReLU())
        prev_size = hidden_size

    layers.append(torch.nn.Linear(prev_size, output_dim))
    return torch.nn.Sequential(*layers)

def create_layer_bounds(num_layers: int, layer_size: int, seed: int) -> Dict[int, Bounds]:
    """Create layer bounds for testing."""
    set_all_seeds(seed)

    layer_bounds = {}
    for layer_id in range(num_layers):
        center = torch.randn(layer_size) * 0.5
        width = torch.abs(torch.randn(layer_size)) * 0.5 + 0.1
        layer_bounds[layer_id] = Bounds(
            lb=center - width,
            ub=center + width
        )

    return layer_bounds

def simulate_scc_overhead(
    input_dim: int,
    sampling_budget: int,
    num_strategies: int = 3,
    seed: int = 42
) -> Dict[str, float]:
    """
    Simulate CBR overhead.

    Main cost factors:
    - Sample generation
    - Forward pass (model execution)
    - Output checking
    """
    set_all_seeds(seed)

    # Create mock model
    model = create_mock_model(input_dim, [64, 32], 10)
    model.eval()

    # Create input bounds
    input_bounds = Bounds(
        lb=torch.zeros(input_dim),
        ub=torch.ones(input_dim)
    )

    times = {
        "sample_generation": 0.0,
        "forward_pass": 0.0,
        "output_check": 0.0,
        "total": 0.0,
    }

    start_total = time.perf_counter()

    for strategy_idx in range(num_strategies):
        for sample_idx in range(sampling_budget):
            # Sample generation
            start = time.perf_counter()
            t = torch.rand(input_dim)
            x = input_bounds.lb + t * (input_bounds.ub - input_bounds.lb)
            times["sample_generation"] += time.perf_counter() - start

            # Forward pass
            start = time.perf_counter()
            with torch.no_grad():
                y = model(x.unsqueeze(0))
            times["forward_pass"] += time.perf_counter() - start

            # Output check
            start = time.perf_counter()
            _ = y.argmax().item()  # Simple output check
            times["output_check"] += time.perf_counter() - start

    times["total"] = time.perf_counter() - start_total

    return times

def simulate_bca_overhead(
    model: torch.nn.Module,
    layer_bounds: Dict[int, Bounds],
    input_dim: int,
    seed: int = 42
) -> Dict[str, float]:
    """
    Simulate BBL overhead.

    Main cost factors:
    - Hook registration
    - Forward pass with hooks
    - Bound containment checking
    """
    set_all_seeds(seed)

    times = {
        "hook_registration": 0.0,
        "forward_pass": 0.0,
        "containment_check": 0.0,
        "total": 0.0,
    }

    # Create input
    x = torch.randn(1, input_dim)

    start_total = time.perf_counter()

    # Hook registration
    start = time.perf_counter()
    hooks = []
    activations = {}

    def make_hook(idx):
        def hook(module, input, output):
            activations[idx] = output.detach()
        return hook

    for idx, module in enumerate(model.modules()):
        if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
            handle = module.register_forward_hook(make_hook(idx))
            hooks.append(handle)

    times["hook_registration"] = time.perf_counter() - start

    # Forward pass
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    times["forward_pass"] = time.perf_counter() - start

    # Containment check
    start = time.perf_counter()
    for layer_id, bounds in layer_bounds.items():
        # Simulate containment check
        lb = bounds.lb
        ub = bounds.ub
        # Mock activation values
        v = lb + torch.rand_like(lb) * (ub - lb)
        gap = torch.max((lb - v).clamp(min=0), (v - ub).clamp(min=0))
        _ = (gap > 0).any()  # Check for violations

    times["containment_check"] = time.perf_counter() - start

    # Clean up hooks
    for handle in hooks:
        handle.remove()

    times["total"] = time.perf_counter() - start_total

    return times

def run_rq6_experiment(
    master_seed: int,
    config_path: str,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    RQ6: Validation Runtime Overhead Measurement

    Args:
        master_seed: Global master seed
        config_path: Path to config.yaml
        output_dir: Output directory for results
        verbose: Print verbose output

    Returns:
        Experiment results dictionary
    """
    # ===== 1. Load configuration =====
    config = load_config(config_path)
    rq6_config = config["rq6"]

    # ===== 2. Initialize metadata =====
    metadata = ExperimentMetadata(
        experiment_id="rq6_overhead",
        description=rq6_config["description"],
        master_seed=master_seed,
        seed_offset=rq6_config["seed_offset"],
        config_path=config_path,
    )

    experiment_seed = metadata.get_experiment_seed()
    set_all_seeds(experiment_seed)

    print(f"RQ6: Validation Runtime Overhead Measurement")
    print(f"=" * 60)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print()

    # ===== 3. Get experiment parameters =====
    num_networks = rq6_config.get("num_networks", 30)
    sampling_budgets = rq6_config.get("sampling_budgets", [5, 10, 20, 50])
    model_sizes = rq6_config.get("model_sizes", {
        "small": {"input_dim": 16, "hidden": [32, 32], "output_dim": 4},
        "medium": {"input_dim": 64, "hidden": [128, 128, 64], "output_dim": 10},
        "large": {"input_dim": 256, "hidden": [512, 256, 128], "output_dim": 10},
    })
    warmup_runs = rq6_config.get("warmup_runs", 3)
    timing_runs = rq6_config.get("timing_runs", 10)

    print(f"Number of networks: {num_networks}")
    print(f"Sampling budgets: {sampling_budgets}")
    print(f"Model sizes: {list(model_sizes.keys())}")
    print(f"Warmup runs: {warmup_runs}, Timing runs: {timing_runs}")
    print()

    # ===== 4. Initialize results =====
    results = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "sampling_budgets": sampling_budgets,
            "model_sizes": model_sizes,
            "warmup_runs": warmup_runs,
            "timing_runs": timing_runs,
        },
        "scc_overhead": defaultdict(list),
        "bca_overhead": defaultdict(list),
        "summary": {},
    }

    # ===== 5. Run overhead experiments =====
    for size_name, size_config in model_sizes.items():
        print(f"\nModel size: {size_name}")

        input_dim = size_config["input_dim"]
        hidden = size_config["hidden"]
        output_dim = size_config["output_dim"]

        # Create model
        model = create_mock_model(input_dim, hidden, output_dim)
        model.eval()

        # Create layer bounds
        num_layers = len(hidden) * 2 + 1  # Linear + ReLU for each hidden, plus output
        bounds_seed = derive_seed(experiment_seed, size_name, "bounds")
        layer_bounds = create_layer_bounds(num_layers, hidden[0], bounds_seed)

        # Test CBR overhead
        for budget in sampling_budgets:
            config_key = f"{size_name}/budget{budget}"

            if verbose:
                print(f"  CBR: {config_key}")

            # Warmup
            for _ in range(warmup_runs):
                simulate_scc_overhead(input_dim, budget, seed=experiment_seed)

            # Timing runs
            for run_idx in range(timing_runs):
                run_seed = derive_seed(experiment_seed, size_name, budget, run_idx)
                times = simulate_scc_overhead(input_dim, budget, seed=run_seed)

                results["scc_overhead"][config_key].append({
                    "run_idx": run_idx,
                    "size": size_name,
                    "budget": budget,
                    **times,
                })

        # Test BBL overhead
        if verbose:
            print(f"  BBL: {size_name}")

        # Warmup
        for _ in range(warmup_runs):
            simulate_bca_overhead(model, layer_bounds, input_dim, seed=experiment_seed)

        # Timing runs
        for run_idx in range(timing_runs):
            run_seed = derive_seed(experiment_seed, size_name, "bca", run_idx)
            times = simulate_bca_overhead(model, layer_bounds, input_dim, seed=run_seed)

            results["bca_overhead"][size_name].append({
                "run_idx": run_idx,
                "size": size_name,
                **times,
            })

    # ===== 6. Calculate summary statistics =====
    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print(f"{'=' * 60}")

    # CBR overhead summary
    print(f"\nCBR Overhead:")
    print(f"{'Config':<25} {'Total (ms)':<15} {'Sample Gen':<15} {'Forward':<15}")
    print("-" * 70)

    for config_key, records in results["scc_overhead"].items():
        n = len(records)
        avg_total = sum(r["total"] for r in records) / n * 1000  # Convert to ms
        avg_sample = sum(r["sample_generation"] for r in records) / n * 1000
        avg_forward = sum(r["forward_pass"] for r in records) / n * 1000

        print(f"{config_key:<25} {avg_total:<14.2f}  {avg_sample:<14.2f}  {avg_forward:<14.2f}")

        results["summary"][f"scc_{config_key}"] = {
            "n": n,
            "avg_total_ms": avg_total,
            "avg_sample_gen_ms": avg_sample,
            "avg_forward_ms": avg_forward,
        }

    # BBL overhead summary
    print(f"\nBBL Overhead:")
    print(f"{'Size':<15} {'Total (ms)':<15} {'Hook Reg':<15} {'Forward':<15} {'Check':<15}")
    print("-" * 75)

    for size_name, records in results["bca_overhead"].items():
        n = len(records)
        avg_total = sum(r["total"] for r in records) / n * 1000
        avg_hook = sum(r["hook_registration"] for r in records) / n * 1000
        avg_forward = sum(r["forward_pass"] for r in records) / n * 1000
        avg_check = sum(r["containment_check"] for r in records) / n * 1000

        print(f"{size_name:<15} {avg_total:<14.2f}  {avg_hook:<14.2f}  {avg_forward:<14.2f}  {avg_check:<14.2f}")

        results["summary"][f"bca_{size_name}"] = {
            "n": n,
            "avg_total_ms": avg_total,
            "avg_hook_reg_ms": avg_hook,
            "avg_forward_ms": avg_forward,
            "avg_check_ms": avg_check,
        }

    # Combined overhead analysis
    print(f"\nCombined Overhead (CBR + BBL):")
    for size_name in model_sizes.keys():
        bca_total = results["summary"][f"bca_{size_name}"]["avg_total_ms"]

        for budget in sampling_budgets:
            scc_key = f"scc_{size_name}/budget{budget}"
            if scc_key in results["summary"]:
                scc_total = results["summary"][scc_key]["avg_total_ms"]
                combined = scc_total + bca_total
                print(f"  {size_name}/budget{budget}: {combined:.2f} ms (CBR: {scc_total:.2f}, BBL: {bca_total:.2f})")

    # ===== 7. Save results =====
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata.finalize()
    metadata.save(str(output_dir / "metadata.json"))

    results["scc_overhead"] = dict(results["scc_overhead"])
    results["bca_overhead"] = dict(results["bca_overhead"])

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 60}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="RQ6: Validation Runtime Overhead Measurement"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed (default: 42)"
    )
    parser.add_argument(
        "--config", type=str, default="experiments/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq6",
        help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    run_rq6_experiment(
        master_seed=args.seed,
        config_path=args.config,
        output_dir=output_dir,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
