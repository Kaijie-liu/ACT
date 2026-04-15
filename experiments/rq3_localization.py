#!/usr/bin/env python3
"""
RQ3: Bounds-based Localization (BBL) Localization Accuracy

Evaluates BBL's ability to correctly localize injected faults by architecture:
- Sequential MLP
- Sequential CNN
- Residual (ADD)

Output Table Format (tab:rq3-loc):
    Architecture -> Top-1 Hit | Top-5 Hit | Error Rate

Reproducible Run:
    python experiments/rq3_localization.py --seed 42

Output: results/rq3/
    - results.json: Full experimental data
    - table_rq3.tex: LaTeX table matching paper format
"""

from __future__ import annotations

import argparse
import json
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

ARCHITECTURES = ["sequential_mlp", "sequential_cnn", "residual"]

@dataclass
class LocalizationResult:
    """Result of a single localization experiment."""
    network_idx: int
    network_seed: int
    architecture: str
    target_layer_id: int
    detected: bool
    top1_hit: bool
    top5_hit: bool
    localized_any: bool   # target appears anywhere in violating-layer list
    error: bool      # Alignment or other error
    num_violations: int
    top_violation_layer_ids: List[int]
    time_ms: float

# ============================================================================
# BBL Localization
# ============================================================================

_factory = None

def _get_factory():
    global _factory
    if _factory is None:
        from experiments.validation_core import build_generated_factory
        _factory = build_generated_factory()
    return _factory

def run_localization(
    architecture: str,
    net_seed: int,
    target_layer_id: int = 3,
) -> LocalizationResult:
    """
    Run BBL localization using the ACT infrastructure.

    Flow: load network -> analyze -> mutate (M1_TIGHTEN) -> BBL -> check localization.
    """
    from experiments.validation_core import (
        run_full_detection, get_networks_for_architecture, MutationType,
    )

    factory = _get_factory()
    matching = get_networks_for_architecture(factory, architecture)

    if not matching:
        # Fall back to all available networks
        matching = factory.list_networks()

    if not matching:
        return LocalizationResult(
            network_idx=0, network_seed=net_seed, architecture=architecture,
            target_layer_id=target_layer_id, detected=False,
            top1_hit=False, top5_hit=False, localized_any=False, error=True,
            num_violations=0, top_violation_layer_ids=[], time_ms=0.0,
        )

    net_idx = net_seed % len(matching)
    network_name = matching[net_idx]

    result = run_full_detection(
        factory=factory,
        network_name=network_name,
        domain="interval",
        mutation_type=MutationType.M1_TIGHTEN,
        net_seed=net_seed,
        target_layer_index=target_layer_id,
        mutation_factor=0.1,
        num_cbr_samples=20,
    )

    if result.error:
        return LocalizationResult(
            network_idx=net_idx, network_seed=net_seed, architecture=architecture,
            target_layer_id=result.target_layer_id, detected=False,
            top1_hit=False, top5_hit=False, localized_any=False, error=True,
            num_violations=0, top_violation_layer_ids=[],
            time_ms=result.bca_time_ms,
        )

    return LocalizationResult(
        network_idx=net_idx,
        network_seed=net_seed,
        architecture=architecture,
        target_layer_id=result.target_layer_id,
        detected=result.bca_detected,
        top1_hit=result.localized_top1,
        top5_hit=result.localized_top5,
        localized_any=result.localized_any,
        error=False,
        num_violations=result.bca_violations_total,
        top_violation_layer_ids=result.bca_top_violation_layers[:10],
        time_ms=result.bca_time_ms,
    )

# ============================================================================
# Main Experiment
# ============================================================================

def run_rq3_experiment(
    master_seed: int,
    num_networks: int = 30,
    output_dir: Path = Path("results/rq3"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RQ3: BBL Localization Accuracy Evaluation.
    """
    set_all_seeds(master_seed)
    experiment_seed = derive_seed(master_seed, "rq3", 3000)

    print(f"RQ3: BBL Localization Accuracy Evaluation")
    print(f"=" * 70)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print(f"Networks per architecture: {num_networks}")
    print(f"Architectures: {ARCHITECTURES}")
    print()

    # Storage for results
    results_by_arch: Dict[str, List[LocalizationResult]] = defaultdict(list)

    total_configs = len(ARCHITECTURES) * num_networks
    completed = 0

    for arch in ARCHITECTURES:
        if verbose:
            print(f"\n[{arch}]:")

        for net_idx in range(num_networks):
            net_seed = derive_seed(experiment_seed, arch, net_idx)
            target_layer = 3 + (net_idx % 5)

            result = run_localization(
                architecture=arch,
                net_seed=net_seed,
                target_layer_id=target_layer,
            )

            result.network_idx = net_idx
            results_by_arch[arch].append(result)

            completed += 1
            if verbose and completed % 30 == 0:
                print(f"  Progress: {completed}/{total_configs}")

    # =========================================================================
    # Compute statistics
    # =========================================================================

    table_data = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "architectures": ARCHITECTURES,
        },
        "raw_results": {},
        "table_rq3": {},
        "summary": {},
    }

    # Convert raw results
    for arch, results in results_by_arch.items():
        table_data["raw_results"][arch] = [asdict(r) for r in results]

    # Table: by architecture
    for arch in ARCHITECTURES:
        results = results_by_arch[arch]
        n = len(results)

        if n == 0:
            continue

        n_detected = sum(1 for r in results if r.detected)
        n_error = sum(1 for r in results if r.error)
        n_valid = n - n_error

        # Among detected (non-error) results
        detected_results = [r for r in results if r.detected and not r.error]
        n_d = len(detected_results)

        localized_rate = sum(1 for r in detected_results if r.localized_any) / n_d if n_d > 0 else 0.0
        avg_violating = (
            sum(len(r.top_violation_layer_ids) for r in detected_results) / n_d
            if n_d > 0 else 0.0
        )
        error_rate = n_error / n

        table_data["table_rq3"][arch] = {
            "n": n,
            "n_detected": n_detected,
            "n_error": n_error,
            "localized_rate": localized_rate,
            "avg_violating_layers": avg_violating,
            "error_rate": error_rate,
        }

    # =========================================================================
    # Print results
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("Table: BBL Localization Accuracy by Architecture")
    print(f"{'=' * 70}")
    print(f"{'Architecture':<20} {'n_det':>8} {'Localized':>12} {'AvgViol#':>12} {'Error Rate':>12}")
    print("-" * 70)

    for arch in ARCHITECTURES:
        stats = table_data["table_rq3"].get(arch, {})
        if not stats:
            continue

        n_det_str = str(stats['n_detected'])
        loc_str = f"{stats['localized_rate']*100:.1f}%"
        avg_str = f"{stats['avg_violating_layers']:.2f}"
        error_str = f"{stats['error_rate']*100:.1f}%"

        print(f"{arch:<20} {n_det_str:>8} {loc_str:>12} {avg_str:>12} {error_str:>12}")

    # =========================================================================
    # Save results
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(table_data, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_latex_table_rq3(table_data)
    with open(output_dir / "table_rq3.tex", "w") as f:
        f.write(latex_table)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"  - results.json")
    print(f"  - table_rq3.tex")
    print(f"{'=' * 70}")

    return table_data

def generate_latex_table_rq3(data: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ3 results."""
    arch_labels = {
        "sequential_mlp": "Sequential MLP",
        "sequential_cnn": "Sequential CNN",
        "residual": "Residual (ADD)",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ3: L2 localization accuracy by architecture}",
        r"\label{tab:rq3-loc}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Detected} & \textbf{Localized} & \textbf{Avg.\ violating layers} & \textbf{Error Rate} \\",
        r"\midrule",
    ]

    for arch in ARCHITECTURES:
        stats = data["table_rq3"].get(arch, {})
        label = arch_labels.get(arch, arch)

        if stats:
            n_det = str(stats["n_detected"])
            loc = f"{stats['localized_rate']*100:.0f}\\%"
            avg = f"{stats['avg_violating_layers']:.2f}"
            error = f"{stats['error_rate']*100:.0f}\\%"
        else:
            n_det = loc = avg = error = "--"

        lines.append(f"{label} & {n_det} & {loc} & {avg} & {error} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="RQ3: BBL Localization Accuracy Evaluation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-networks", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results/rq3")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_rq3_experiment(
        master_seed=args.seed,
        num_networks=args.num_networks,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
