#!/usr/bin/env python3
"""
RQ5: Cross-Domain Behavior Comparison

Compares the behavior of different abstract domains:
- interval: Box bounds (fastest, least precise)
- hybridz:  Hybrid zonotopes (moderate)
- dual:     Dual bounds (slowest, most precise)

Output Table Format (tab:rq5-domains):
    Domain -> BBL Fail Rate | Bound Width | Time (ms)
    + Disagreement rate

Reproducible Run:
    python experiments/rq5_cross_domain.py --seed 42

Output: results/rq5/
    - results.json: Full experimental data
    - table_rq5.tex: LaTeX table matching paper format
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from act.back_end.validation import set_all_seeds, derive_seed

# ============================================================================
# Configuration
# ============================================================================

DOMAINS = ["interval", "hybridz", "dual"]

@dataclass
class DomainResult:
    """Result for a single network under one domain."""
    network_idx: int
    network_seed: int
    domain: str
    bca_failed: bool
    bound_width: float
    time_ms: float

# ============================================================================
# Domain Comparison
# ============================================================================

_factory = None

def _get_factory():
    global _factory
    if _factory is None:
        from experiments.validation_core import build_generated_factory
        _factory = build_generated_factory()
    return _factory

def run_domain_check(
    domain: str,
    net_seed: int,
) -> DomainResult:
    """
    Run domain comparison using the ACT infrastructure.

    Flow: load network -> analyze with specified domain -> mutate (M3_SWAP) -> BBL.
    """
    from experiments.validation_core import run_full_detection, MutationType

    factory = _get_factory()
    available = factory.list_networks()

    if not available:
        return DomainResult(
            network_idx=0, network_seed=net_seed, domain=domain,
            bca_failed=False, bound_width=0.0, time_ms=0.0,
        )

    net_idx = net_seed % len(available)
    network_name = available[net_idx]

    result = run_full_detection(
        factory=factory,
        network_name=network_name,
        domain=domain,
        mutation_type=MutationType.M3_SWAP,
        net_seed=net_seed,
        target_layer_index=0,
        mutation_factor=0.1,
        num_cbr_samples=20,
    )

    return DomainResult(
        network_idx=net_idx,
        network_seed=net_seed,
        domain=domain,
        bca_failed=result.bca_detected,
        bound_width=result.bound_width_avg,
        time_ms=result.bca_time_ms,
    )

# ============================================================================
# Main Experiment
# ============================================================================

def run_rq5_experiment(
    master_seed: int,
    num_networks: int = 100,
    output_dir: Path = Path("results/rq5"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RQ5: Cross-Domain Behavior Comparison.
    """
    set_all_seeds(master_seed)
    experiment_seed = derive_seed(master_seed, "rq5", 5000)

    print(f"RQ5: Cross-Domain Behavior Comparison")
    print(f"=" * 70)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print(f"Networks: {num_networks}")
    print(f"Domains: {DOMAINS}")
    print()

    # Storage for results
    results_by_domain: Dict[str, List[DomainResult]] = defaultdict(list)
    results_by_network: Dict[int, Dict[str, DomainResult]] = defaultdict(dict)

    total_configs = len(DOMAINS) * num_networks
    completed = 0

    for net_idx in range(num_networks):
        net_seed = derive_seed(experiment_seed, net_idx)

        for domain in DOMAINS:
            result = run_domain_check(domain, net_seed)
            result.network_idx = net_idx
            results_by_domain[domain].append(result)
            results_by_network[net_idx][domain] = result

            completed += 1
            if verbose and completed % 100 == 0:
                print(f"  Progress: {completed}/{total_configs}")

    # =========================================================================
    # Compute statistics
    # =========================================================================

    table_data = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "domains": DOMAINS,
        },
        "raw_results": {},
        "table_rq5": {},
        "disagreement": {},
    }

    # Convert raw results
    for domain, results in results_by_domain.items():
        table_data["raw_results"][domain] = [asdict(r) for r in results]

    # Domain statistics
    for domain in DOMAINS:
        results = results_by_domain[domain]
        n = len(results)

        fail_rate = sum(1 for r in results if r.bca_failed) / n
        avg_width = sum(r.bound_width for r in results) / n
        avg_time = sum(r.time_ms for r in results) / n

        table_data["table_rq5"][domain] = {
            "n": n,
            "bca_fail_rate": fail_rate,
            "avg_bound_width": avg_width,
            "avg_time_ms": avg_time,
        }

    # Disagreement rate: networks where domains give different results
    n_disagreements = 0
    for net_idx in range(num_networks):
        domain_results = results_by_network[net_idx]
        outcomes = [domain_results[d].bca_failed for d in DOMAINS]
        if not all(o == outcomes[0] for o in outcomes):
            n_disagreements += 1

    disagreement_rate = n_disagreements / num_networks
    table_data["disagreement"]["rate"] = disagreement_rate
    table_data["disagreement"]["count"] = n_disagreements

    # Pairwise agreement
    for i, d1 in enumerate(DOMAINS):
        for d2 in DOMAINS[i+1:]:
            agree = sum(
                1 for net_idx in range(num_networks)
                if results_by_network[net_idx][d1].bca_failed == results_by_network[net_idx][d2].bca_failed
            )
            table_data["disagreement"][f"{d1}_vs_{d2}"] = agree / num_networks

    # =========================================================================
    # Print results
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("Table: Cross-Domain Comparison")
    print(f"{'=' * 70}")
    print(f"{'Domain':<12} {'BBL Fail Rate':>15} {'Bound Width':>12} {'Time (ms)':>12}")
    print("-" * 55)

    for domain in DOMAINS:
        stats = table_data["table_rq5"][domain]
        fail_str = f"{stats['bca_fail_rate']*100:.0f}%"
        width_str = f"{stats['avg_bound_width']:.2f}"
        time_str = f"{stats['avg_time_ms']:.1f}"
        print(f"{domain:<12} {fail_str:>15} {width_str:>12} {time_str:>12}")

    print("-" * 55)
    print(f"Disagreement rate: {disagreement_rate*100:.1f}%")

    # =========================================================================
    # Save results
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(table_data, f, indent=2)

    latex_table = generate_latex_table_rq5(table_data)
    with open(output_dir / "table_rq5.tex", "w") as f:
        f.write(latex_table)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"  - results.json")
    print(f"  - table_rq5.tex")
    print(f"{'=' * 70}")

    return table_data

def generate_latex_table_rq5(data: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ5 results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ5: Cross-domain comparison}",
        r"\label{tab:rq5-domains}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{BBL Fail Rate} & \textbf{Bound Width} & \textbf{Time (ms)} \\",
        r"\midrule",
    ]

    for domain in DOMAINS:
        stats = data["table_rq5"].get(domain, {})

        if stats:
            fail = f"{stats['bca_fail_rate']*100:.0f}\\%"
            width = f"{stats['avg_bound_width']:.2f}"
            time_val = f"{stats['avg_time_ms']:.1f}"
        else:
            fail = "--\\%"
            width = "--"
            time_val = "--"

        lines.append(f"\\textit{{{domain}}} & {fail} & {width} & {time_val} \\\\")

    # Disagreement rate
    disagree_rate = data.get("disagreement", {}).get("rate", 0)
    lines.extend([
        r"\midrule",
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Disagreement rate: {disagree_rate*100:.0f}\\%}}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Cross-Domain Behavior Comparison"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-networks", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/rq5")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_rq5_experiment(
        master_seed=args.seed,
        num_networks=args.num_networks,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
