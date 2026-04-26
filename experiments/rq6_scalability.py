#!/usr/bin/env python3
"""
RQ6: Validation Overhead Analysis

Measures the runtime overhead of CBR and BBL validation across:
- CBR sampling budgets: {5, 10, 20, 50}
- Model sizes: Small (~4K), Medium (~65K), Large (~260K)

Output Table Format (tab:rq6-overhead):
    Size -> Params | CBR(5,10,20,50) | BBL(5,10,20,50) | Complementary(5,10,20,50)

Reproducible Run:
    python experiments/rq6_scalability.py --seed 42

Output: results/rq6/
    - results.json: Full experimental data
    - table_rq6.tex: LaTeX table matching paper format
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._validation import set_all_seeds, derive_seed

# ============================================================================
# Configuration
# ============================================================================

MODEL_SIZES = {
    "Small":  {"params": 4000,    "layers": 3,  "width": 32},
    "Medium": {"params": 65000,   "layers": 5,  "width": 128},
    "Large":  {"params": 260000,  "layers": 7,  "width": 256},
}

SIZE_ORDER = ["Small", "Medium", "Large"]
BUDGETS = [5, 10, 20, 50]

@dataclass
class OverheadResult:
    """Result of overhead measurement for one network at one budget."""
    network_idx: int
    network_seed: int
    size_class: str
    budget: int
    n_params: int
    scc_time_ms: float
    bca_time_ms: float
    verification_time_ms: float
    complementary_ms: float  # CBR + BBL

# ============================================================================
# Overhead Measurement
# ============================================================================

_factory = None

def _get_factory():
    global _factory
    if _factory is None:
        from experiments.validation_core import build_generated_factory
        _factory = build_generated_factory()
    return _factory

def run_overhead(
    size_class: str,
    net_seed: int,
    scc_budget: int = 20,
) -> OverheadResult:
    """
    Run overhead measurement using the ACT infrastructure.

    Measures analysis, CBR, and BBL timing using pre-built networks.
    """
    from experiments.validation_core import run_overhead_measurement

    factory = _get_factory()
    available = factory.list_networks()

    if not available:
        raise RuntimeError("No networks available in ModelFactory")

    net_idx = net_seed % len(available)
    network_name = available[net_idx]

    result = run_overhead_measurement(
        factory=factory,
        network_name=network_name,
        net_seed=net_seed,
        scc_budget=scc_budget,
    )

    verification_time_ms = result["analyze_time_ms"]
    scc_time_ms = result["scc_time_ms"]
    bca_time_ms = result["bca_time_ms"]
    n_params = result["n_params"]

    return OverheadResult(
        network_idx=net_idx,
        network_seed=net_seed,
        size_class=size_class,
        budget=scc_budget,
        n_params=n_params,
        scc_time_ms=scc_time_ms,
        bca_time_ms=bca_time_ms,
        verification_time_ms=verification_time_ms,
        complementary_ms=scc_time_ms + bca_time_ms,
    )

# ============================================================================
# Main Experiment
# ============================================================================

def run_rq6_experiment(
    master_seed: int,
    num_networks: int = 30,
    output_dir: Path = Path("results/rq6"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RQ6: Validation Overhead Analysis.

    For each (size, budget, network) triple:
      - CBR is measured with the given sampling budget
      - BBL is measured once per network (budget-independent)
      - Complementary = CBR + BBL
    """
    set_all_seeds(master_seed)
    experiment_seed = derive_seed(master_seed, "rq6", 6000)

    print(f"RQ6: Validation Overhead Analysis")
    print(f"=" * 70)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print(f"Networks per size: {num_networks}")
    print(f"CBR budgets: {BUDGETS}")
    print(f"Size classes: {SIZE_ORDER}")
    print()

    # Storage: results_by_key[(size, budget)] = [OverheadResult, ...]
    results_by_key: Dict[tuple, List[OverheadResult]] = defaultdict(list)

    total_configs = len(SIZE_ORDER) * len(BUDGETS) * num_networks
    completed = 0

    for size_class in SIZE_ORDER:
        if verbose:
            print(f"\n[{size_class}]:")

        for net_idx in range(num_networks):
            net_seed = derive_seed(experiment_seed, size_class, net_idx)

            for budget in BUDGETS:
                result = run_overhead(
                    size_class=size_class,
                    net_seed=net_seed,
                    scc_budget=budget,
                )
                result.network_idx = net_idx
                results_by_key[(size_class, budget)].append(result)

                completed += 1

            if verbose and (net_idx + 1) % 10 == 0:
                print(f"  Progress: {completed}/{total_configs}")

    # =========================================================================
    # Compute statistics
    # =========================================================================

    table_data = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "budgets": BUDGETS,
            "size_classes": SIZE_ORDER,
            "model_configs": MODEL_SIZES,
        },
        "raw_results": {},
        "table_rq6": {},
    }

    # Convert raw results
    for (size_class, budget), results in results_by_key.items():
        key = f"{size_class}/budget{budget}"
        table_data["raw_results"][key] = [asdict(r) for r in results]

    # Per (size, budget) statistics
    for size_class in SIZE_ORDER:
        table_data["table_rq6"][size_class] = {}

        for budget in BUDGETS:
            results = results_by_key[(size_class, budget)]
            n = len(results)

            avg_params = sum(r.n_params for r in results) / n
            avg_scc = sum(r.scc_time_ms for r in results) / n
            avg_bca = sum(r.bca_time_ms for r in results) / n
            avg_comp = sum(r.complementary_ms for r in results) / n

            table_data["table_rq6"][size_class][str(budget)] = {
                "n": n,
                "avg_params": avg_params,
                "avg_scc_ms": avg_scc,
                "avg_bca_ms": avg_bca,
                "avg_complementary_ms": avg_comp,
            }

    # =========================================================================
    # Print results
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("Table: Validation Overhead by Model Size and CBR Budget")
    print(f"{'=' * 70}")

    # Header
    budget_cols = "".join(f"{b:>8}" for b in BUDGETS)
    print(f"{'Size':<10} {'Params':>8}  CBR(ms):{budget_cols}  BBL(ms):{budget_cols}  Comp(ms):{budget_cols}")
    print("-" * 110)

    for size_class in SIZE_ORDER:
        stats = table_data["table_rq6"][size_class]
        first_budget_stats = stats[str(BUDGETS[0])]
        params_str = f"~{int(first_budget_stats['avg_params']/1000)}K"

        cbr_vals = "".join(f"{stats[str(b)]['avg_scc_ms']:>8.2f}" for b in BUDGETS)
        bbl_vals = "".join(f"{stats[str(b)]['avg_bca_ms']:>8.2f}" for b in BUDGETS)
        comp_vals = "".join(f"{stats[str(b)]['avg_complementary_ms']:>8.2f}" for b in BUDGETS)

        print(f"{size_class:<10} {params_str:>8}  {cbr_vals}  {bbl_vals}  {comp_vals}")

    # =========================================================================
    # Save results
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(table_data, f, indent=2)

    latex_table = generate_latex_table_rq6(table_data)
    with open(output_dir / "table_rq6.tex", "w") as f:
        f.write(latex_table)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"  - results.json")
    print(f"  - table_rq6.tex")
    print(f"{'=' * 70}")

    return table_data

def generate_latex_table_rq6(data: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ6 results matching paper format."""
    budgets = data["metadata"]["budgets"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ6: Validation overhead by model size}",
        r"\vspace{-4mm}",
        r"\label{tab:rq6-overhead}",
        r"\small",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\begin{tabular}{@{}l r rrrr rrrr rrrr@{}}",
        r"\toprule",
        r"\textbf{Size} & \textbf{Params} &",
        r"\multicolumn{4}{c}{\textbf{CBR}} &",
        r"\multicolumn{4}{c}{\textbf{BBL}} &",
        r"\multicolumn{4}{c}{\textbf{Complementary}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
    ]

    # Budget sub-headers
    budget_hdrs = " & ".join(f"\\textbf{{{b}}}" for b in budgets)
    lines.append(f"& & {budget_hdrs}")
    lines.append(f"  & {budget_hdrs}")
    lines.append(f"  & {budget_hdrs} \\\\")
    lines.append(r"\midrule")

    for size_class in SIZE_ORDER:
        stats = data["table_rq6"].get(size_class, {})
        if not stats:
            continue

        first_stats = stats[str(budgets[0])]
        params = f"$\\sim${int(first_stats['avg_params']/1000)}K"

        cbr_vals = " & ".join(f"{stats[str(b)]['avg_scc_ms']:.2f}" for b in budgets)
        bbl_vals = " & ".join(f"{stats[str(b)]['avg_bca_ms']:.2f}" for b in budgets)
        comp_vals = " & ".join(f"{stats[str(b)]['avg_complementary_ms']:.2f}" for b in budgets)

        lines.append(f"{size_class} & {params} & {cbr_vals} & {bbl_vals} & {comp_vals} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-4mm}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="RQ6: Validation Overhead Analysis"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-networks", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results/rq6")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_rq6_experiment(
        master_seed=args.seed,
        num_networks=args.num_networks,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
