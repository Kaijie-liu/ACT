#!/usr/bin/env python3
"""
RQ2: Counterexample-based Refutation (CBR) Effectiveness

Evaluates when CBR can find concrete counterexamples based on:
1. Specification type: BOX, LINF_BALL, LIN_POLY
2. Input dimensionality: 4, 16, 64, 256

Output:
    - Table (tab:rq2-spec): Discovery rate by specification type
    - Figure (fig:rq2-dim): Discovery rate vs input dimension

Reproducible Run:
    python experiments/rq2_scc_effectiveness.py --seed 42

Output: results/rq2/
    - results.json: Full experimental data
    - table_rq2.tex: LaTeX table for spec type results
    - fig_rq2_dim.csv: Data for dimension figure
    - fig_rq2_dim.pdf: Generated figure (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from act.back_end.validation import set_all_seeds, derive_seed

# ============================================================================
# Configuration
# ============================================================================

SPEC_TYPES = ["BOX", "LINF_BALL", "LIN_POLY"]
DIMENSIONS = [4, 16, 64, 256]
DEFAULT_SAMPLING_BUDGET = 20

# Mapping: flattened input dim -> (input_shape, family)
DIM_TO_CONFIG = {
    4:   {"shape": [1, 4],        "family": "mlp"},
    16:  {"shape": [1, 16],       "family": "mlp"},
    64:  {"shape": [1, 1, 8, 8],  "family": "cnn2d"},
    256: {"shape": [1, 1, 16, 16], "family": "cnn2d"},
}

@dataclass
class SCCResult:
    """Result of a single CBR experiment."""
    network_idx: int
    network_seed: int
    spec_type: str
    dimension: int
    discovered: bool      # Found a counterexample
    inconclusive: bool    # Could not sample (e.g., LIN_POLY)
    num_samples: int
    time_ms: float

# ============================================================================
# Network Generation
# ============================================================================

_net_factory = None

def _get_net_factory():
    """Lazy-init NetFactory for on-the-fly network generation."""
    global _net_factory
    if _net_factory is None:
        from act.back_end.net_factory import NetFactory
        _net_factory = NetFactory(tf_targets=["interval"])
    return _net_factory


def _build_network_spec(dimension: int, spec_type: str, seed: int) -> Dict[str, Any]:
    """Build a network spec following ACT's current NetFactory contract.

    ACT stores layer fields in ``params`` (shape, dtype, kind, in_features,
    etc.) and does not use ``meta``. We reuse ACT's own ``append_*``
    helpers so the spec exactly matches what NetFactory generates
    internally.
    """
    from act.back_end.net_factory import (
        append_dense, append_act, append_flatten, append_conv_nd,
    )

    cfg = DIM_TO_CONFIG[dimension]
    input_shape = cfg["shape"]
    family = cfg["family"]
    rng = random.Random(seed)

    dtype = "torch.float64"
    num_classes = 10
    layers: List[Dict[str, Any]] = []

    # INPUT: shape + dtype go in params (ACT convention)
    layers.append({
        "kind": "INPUT",
        "params": {
            "shape": list(input_shape),
            "dtype": dtype,
            "num_classes": num_classes,
            "value_range": [0.0, 1.0],
        },
    })

    # INPUT_SPEC: kind + bounds go in params
    if spec_type == "BOX":
        layers.append({
            "kind": "INPUT_SPEC",
            "params": {"kind": "BOX", "lb_val": 0.0, "ub_val": 1.0},
        })
    elif spec_type == "LINF_BALL":
        layers.append({
            "kind": "INPUT_SPEC",
            "params": {"kind": "LINF_BALL", "center_val": 0.5, "eps": 0.1},
        })

    # Architecture (use ACT's canonical appenders)
    act_kind = rng.choice(["RELU", "TANH", "SIGMOID"])

    if family == "mlp":
        flat_dim = dimension
        hidden = rng.choice([32, 64, 128])
        append_dense(layers, in_features=flat_dim, out_features=hidden, use_bias=True)
        append_act(layers, act_kind)
        append_dense(layers, in_features=hidden, out_features=num_classes, use_bias=True)
    else:  # cnn2d
        _, C, H, W = input_shape
        out_ch = rng.choice([8, 16, 32])
        append_conv_nd(
            layers, ndim=2, in_ch=C, out_ch=out_ch,
            spatial_dims=(H, W), kernel=3, stride=1, padding=1,
        )
        append_act(layers, act_kind)
        append_flatten(layers)
        flat_dim = out_ch * H * W
        append_dense(layers, in_features=flat_dim, out_features=num_classes, use_bias=True)

    # ASSERT: kind + properties in params
    layers.append({
        "kind": "ASSERT",
        "params": {"kind": "TOP1_ROBUST", "y_true": 0},
    })

    return {"layers": layers}


def _create_network_and_model(dimension: int, spec_type: str, seed: int):
    """Generate a network with the given input dimension and convert to PyTorch."""
    from act.pipeline.verification.act2torch import ACTToTorch

    factory = _get_net_factory()
    spec = _build_network_spec(dimension, spec_type, seed)
    name = f"rq2_d{dimension}_{spec_type}_{seed}"
    act_net = factory.create_network(name, spec)
    model = ACTToTorch(act_net).run()
    model = model.to(dtype=torch.float64)
    model.eval()
    return act_net, model

# ============================================================================
# CBR Validation
# ============================================================================

def run_scc(
    spec_type: str,
    dimension: int,
    net_seed: int,
    sampling_budget: int = DEFAULT_SAMPLING_BUDGET,
) -> SCCResult:
    """
    Run CBR with a network generated for the specified input dimension.

    Flow: generate network with input dim -> analyze -> mutate output bounds ->
          sample inputs within spec -> check violations.
    LIN_POLY is always inconclusive (no seedable box).
    """
    from experiments.validation_core import (
        get_clean_bounds, run_cbr_detection,
        MutationType, MutationConfig, TFMutator,
        _get_last_computation_layer_id, _get_input_shape,
        Bounds, gather_input_spec_layers, seed_from_input_specs,
    )
    from act.back_end.validation import set_all_seeds, derive_seed

    set_all_seeds(net_seed)

    # LIN_POLY is always inconclusive without explicit seed box
    if spec_type == "LIN_POLY":
        return SCCResult(
            network_idx=0,
            network_seed=net_seed,
            spec_type=spec_type,
            dimension=dimension,
            discovered=False,
            inconclusive=True,
            num_samples=0,
            time_ms=0.0,
        )

    try:
        # Generate network with the specified input dimension
        act_net, model = _create_network_and_model(
            dimension, spec_type, derive_seed(net_seed, "netgen"),
        )

        bounds_by_layer, entry_fact = get_clean_bounds(act_net, "interval")

        if not bounds_by_layer:
            return SCCResult(
                network_idx=0, network_seed=net_seed, spec_type=spec_type,
                dimension=dimension, discovered=False, inconclusive=True,
                num_samples=0, time_ms=0.0,
            )

        # Mutate the OUTPUT layer's bounds so CBR can detect violations.
        output_layer_id = _get_last_computation_layer_id(act_net)

        if output_layer_id is None or output_layer_id not in bounds_by_layer:
            return SCCResult(
                network_idx=0, network_seed=net_seed, spec_type=spec_type,
                dimension=dimension, discovered=False, inconclusive=True,
                num_samples=0, time_ms=0.0,
            )

        mutation_config = MutationConfig(
            mutation_type=MutationType.M1_TIGHTEN,
            target_layer_id=output_layer_id,
            seed=derive_seed(net_seed, "mutation"),
            factor=0.5,
        )
        mutated_bounds = TFMutator.mutate_layer_bounds(bounds_by_layer, mutation_config)
        output_bounds = mutated_bounds[output_layer_id]

        if output_bounds is None:
            return SCCResult(
                network_idx=0, network_seed=net_seed, spec_type=spec_type,
                dimension=dimension, discovered=False, inconclusive=True,
                num_samples=0, time_ms=0.0,
            )

        # Get input bounds for sampling
        spec_layers = gather_input_spec_layers(act_net)
        input_bounds = seed_from_input_specs(spec_layers)
        input_bounds = Bounds(
            lb=input_bounds.lb.to(dtype=torch.float64),
            ub=input_bounds.ub.to(dtype=torch.float64),
        )

        input_shape = _get_input_shape(act_net)

        # Run CBR with specified sampling budget
        start = time.perf_counter()
        cbr_result = run_cbr_detection(
            model=model,
            input_bounds=input_bounds,
            output_bounds=output_bounds,
            num_samples=sampling_budget,
            seed=derive_seed(net_seed, "cbr"),
            input_shape=input_shape,
        )
        time_ms = (time.perf_counter() - start) * 1000

        return SCCResult(
            network_idx=0,
            network_seed=net_seed,
            spec_type=spec_type,
            dimension=dimension,
            discovered=cbr_result["discovered"],
            inconclusive=False,
            num_samples=sampling_budget,
            time_ms=time_ms,
        )

    except Exception as e:
        import traceback
        print(f"  Warning: CBR failed for dim={dimension} ({e})")
        traceback.print_exc()
        return SCCResult(
            network_idx=0, network_seed=net_seed, spec_type=spec_type,
            dimension=dimension, discovered=False, inconclusive=True,
            num_samples=0, time_ms=0.0,
        )

# ============================================================================
# Main Experiment
# ============================================================================

def run_rq2_experiment(
    master_seed: int,
    num_networks: int = 30,
    sampling_budget: int = DEFAULT_SAMPLING_BUDGET,
    output_dir: Path = Path("results/rq2"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RQ2: CBR Effectiveness Evaluation.

    Args:
        master_seed: Global master seed
        num_networks: Networks per spec_type x dimension combination
        sampling_budget: Number of samples for CBR
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Results dictionary with table and figure data
    """
    set_all_seeds(master_seed)
    experiment_seed = derive_seed(master_seed, "rq2", 2000)

    print(f"RQ2: CBR Effectiveness Evaluation")
    print(f"=" * 70)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print(f"Networks per config: {num_networks}")
    print(f"Sampling budget: {sampling_budget}")
    print(f"Spec types: {SPEC_TYPES}")
    print(f"Dimensions: {DIMENSIONS}")
    print()

    # Storage for results
    results_by_spec: Dict[str, List[SCCResult]] = defaultdict(list)
    results_by_dim: Dict[int, List[SCCResult]] = defaultdict(list)
    results_by_spec_dim: Dict[tuple, List[SCCResult]] = defaultdict(list)

    total_configs = len(SPEC_TYPES) * len(DIMENSIONS) * num_networks
    completed = 0

    for spec_type in SPEC_TYPES:
        for dim in DIMENSIONS:
            if verbose:
                print(f"\n[{spec_type}] dim={dim}:")

            for net_idx in range(num_networks):
                net_seed = derive_seed(experiment_seed, spec_type, dim, net_idx)

                result = run_scc(
                    spec_type=spec_type,
                    dimension=dim,
                    net_seed=net_seed,
                    sampling_budget=sampling_budget,
                )

                result.network_idx = net_idx
                results_by_spec[spec_type].append(result)
                results_by_dim[dim].append(result)
                results_by_spec_dim[(spec_type, dim)].append(result)

                completed += 1
                if verbose and completed % 50 == 0:
                    print(f"  Progress: {completed}/{total_configs}")

    # =========================================================================
    # Compute statistics
    # =========================================================================

    table_data = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "sampling_budget": sampling_budget,
            "spec_types": SPEC_TYPES,
            "dimensions": DIMENSIONS,
        },
        "raw_results": {},
        "table_rq2_spec": {},
        "figure_rq2_dim": {},
        "summary": {},
    }

    # Convert raw results to serializable format
    for spec_type, results in results_by_spec.items():
        table_data["raw_results"][spec_type] = [asdict(r) for r in results]

    # Table: by spec type (aggregated across dimensions)
    for spec_type in SPEC_TYPES:
        results = results_by_spec[spec_type]
        n = len(results)

        if n == 0:
            continue

        n_discovered = sum(1 for r in results if r.discovered)
        n_inconclusive = sum(1 for r in results if r.inconclusive)
        n_conclusive = n - n_inconclusive

        # Discovery rate among conclusive results
        discovery_rate = n_discovered / n_conclusive if n_conclusive > 0 else 0.0
        inconclusive_rate = n_inconclusive / n

        # Average time (excluding inconclusive)
        conclusive_results = [r for r in results if not r.inconclusive]
        avg_time = sum(r.time_ms for r in conclusive_results) / len(conclusive_results) if conclusive_results else 0.0

        table_data["table_rq2_spec"][spec_type] = {
            "n": n,
            "n_discovered": n_discovered,
            "n_inconclusive": n_inconclusive,
            "discovery_rate": discovery_rate,
            "inconclusive_rate": inconclusive_rate,
            "avg_time_ms": avg_time,
        }

    # Figure: by dimension (for seedable specs only: BOX, LINF_BALL)
    for dim in DIMENSIONS:
        # Filter to only seedable spec types
        seedable_results = [r for r in results_by_dim[dim] if r.spec_type in ["BOX", "LINF_BALL"]]
        n = len(seedable_results)

        if n == 0:
            continue

        n_discovered = sum(1 for r in seedable_results if r.discovered)
        discovery_rate = n_discovered / n

        table_data["figure_rq2_dim"][dim] = {
            "n": n,
            "n_discovered": n_discovered,
            "discovery_rate": discovery_rate,
        }

    # Detailed breakdown by spec_type x dimension
    table_data["breakdown"] = {}
    for (spec_type, dim), results in results_by_spec_dim.items():
        n = len(results)
        if n == 0:
            continue

        n_discovered = sum(1 for r in results if r.discovered)
        n_inconclusive = sum(1 for r in results if r.inconclusive)

        table_data["breakdown"][f"{spec_type}/dim{dim}"] = {
            "n": n,
            "n_discovered": n_discovered,
            "n_inconclusive": n_inconclusive,
            "discovery_rate": n_discovered / n if n > 0 else 0.0,
        }

    # =========================================================================
    # Print results
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("Table: CBR Discovery Rate by Specification Type")
    print(f"{'=' * 70}")
    print(f"{'Spec Type':<15} {'Discovery Rate':>15} {'Inconclusive':>15} {'Avg Time (ms)':>15}")
    print("-" * 70)

    for spec_type in SPEC_TYPES:
        stats = table_data["table_rq2_spec"].get(spec_type, {})
        if not stats:
            continue

        disc_str = f"{stats['discovery_rate']*100:.1f}%"
        inc_str = f"{stats['inconclusive_rate']*100:.1f}%"
        time_str = f"{stats['avg_time_ms']:.2f}" if stats['avg_time_ms'] > 0 else "N/A"

        print(f"{spec_type:<15} {disc_str:>15} {inc_str:>15} {time_str:>15}")

    print(f"\n{'=' * 70}")
    print("Figure: CBR Discovery Rate by Input Dimension (seedable specs)")
    print(f"{'=' * 70}")
    print(f"{'Dimension':>10} {'Discovery Rate':>15} {'Networks':>10}")
    print("-" * 40)

    for dim in DIMENSIONS:
        stats = table_data["figure_rq2_dim"].get(dim, {})
        if not stats:
            continue

        disc_str = f"{stats['discovery_rate']*100:.1f}%"
        print(f"{dim:>10} {disc_str:>15} {stats['n']:>10}")

    # =========================================================================
    # Save results
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(table_data, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_latex_table_rq2(table_data)
    with open(output_dir / "table_rq2.tex", "w") as f:
        f.write(latex_table)

    # Generate figure data CSV
    generate_figure_csv(table_data, output_dir)

    # Generate figure if matplotlib available
    try:
        generate_figure_pdf(table_data, output_dir)
        print(f"  - fig_rq2_dim.pdf")
    except ImportError:
        print("  - fig_rq2_dim.pdf (skipped, matplotlib not available)")

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"  - results.json")
    print(f"  - table_rq2.tex")
    print(f"  - fig_rq2_dim.csv")
    print(f"{'=' * 70}")

    return table_data

def generate_latex_table_rq2(data: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ2 spec type results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ2: L1 discovery rate by specification type}",
        r"\label{tab:rq2-spec}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Spec Type} & \textbf{Discovery Rate} & \textbf{Inconclusive} & \textbf{Avg Time (ms)} \\",
        r"\midrule",
    ]

    for spec_type in SPEC_TYPES:
        stats = data["table_rq2_spec"].get(spec_type, {})

        if stats:
            disc = f"{stats['discovery_rate']*100:.0f}\\%"
            inc = f"{stats['inconclusive_rate']*100:.0f}\\%"
            time_val = f"{stats['avg_time_ms']:.1f}" if stats['avg_time_ms'] > 0 else "N/A"
        else:
            disc = inc = "--\\%"
            time_val = "--"

        # Format spec type name
        escaped_spec = spec_type.replace('_', '\\_')
        spec_label = f"\\textsc{{{escaped_spec}}}"
        lines.append(f"{spec_label} & {disc} & {inc} & {time_val} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_figure_csv(data: Dict[str, Any], output_dir: Path):
    """Generate CSV data for the dimension figure."""
    csv_lines = ["dimension,discovery_rate,n_networks,n_discovered"]

    for dim in DIMENSIONS:
        stats = data["figure_rq2_dim"].get(dim, {})
        if stats:
            csv_lines.append(f"{dim},{stats['discovery_rate']:.4f},{stats['n']},{stats['n_discovered']}")

    with open(output_dir / "fig_rq2_dim.csv", "w") as f:
        f.write("\n".join(csv_lines))

def generate_figure_pdf(data: Dict[str, Any], output_dir: Path):
    """Generate PDF figure for dimension analysis."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    dims = []
    rates = []

    for dim in DIMENSIONS:
        stats = data["figure_rq2_dim"].get(dim, {})
        if stats:
            dims.append(dim)
            rates.append(stats['discovery_rate'] * 100)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(dims, rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(dims, rates, alpha=0.2, color='#2E86AB')

    ax.set_xlabel('Input Dimension', fontsize=12)
    ax.set_ylabel('Discovery Rate (%)', fontsize=12)
    ax.set_title('CBR Discovery Rate vs Input Dimension', fontsize=14)

    ax.set_xscale('log', base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_rq2_dim.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="RQ2: CBR Effectiveness Evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed (default: 42)"
    )
    parser.add_argument(
        "--num-networks", type=int, default=30,
        help="Networks per spec x dimension config (default: 30)"
    )
    parser.add_argument(
        "--sampling-budget", type=int, default=20,
        help="CBR sampling budget (default: 20)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq2",
        help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    run_rq2_experiment(
        master_seed=args.seed,
        num_networks=args.num_networks,
        sampling_budget=args.sampling_budget,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
