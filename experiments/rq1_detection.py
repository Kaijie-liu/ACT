#!/usr/bin/env python3
"""
RQ1: Two-Level Detection Capability Evaluation

Evaluates the detection capability of CBR and BBL
against injected soundness violations (mutations M1-M6).

Output Table Format (tab:rq1-detection):
    Domain x Mutation -> CBR Only | BBL Only | Combined | Localized

Reproducible Run:
    python experiments/rq1_detection.py --seed 42

Output: results/rq1/
    - results.json: Full experimental data
    - table_rq1.tex: LaTeX table matching paper format
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from act.back_end.validation import (
    set_all_seeds,
    derive_seed,
    MutationType,
    MutationConfig,
    TFMutator,
)

# ============================================================================
# Configuration
# ============================================================================

DOMAINS = ["interval", "hybridz", "dual"]
MUTATIONS = {
    "soundness": [
        MutationType.M1_TIGHTEN,
        MutationType.M3_SWAP,
        MutationType.M4_ZERO_LB,
        MutationType.M5_SCALE_UB,
        MutationType.M6_NOISE,
    ],
    "control": [
        MutationType.M2_LOOSEN,
    ],
}

MUTATION_LABELS = {
    "M1_TIGHTEN": "M1 (Tighten)",
    "M2_LOOSEN": "M2 (Loosen)",
    "M3_SWAP": "M3 (Sign)",
    "M4_ZERO_LB": "M4 (Shape)",
    "M5_SCALE_UB": "M5 (Drift)",
    "M6_NOISE": "M6 (Missing)",
}

@dataclass
class DetectionResult:
    """Result of a single detection experiment."""
    network_idx: int
    network_seed: int
    domain: str
    mutation: str
    scc_detected: bool  # CBR: Counterexample-based Refutation
    bca_detected: bool  # BBL: Bounds-based Localization
    localized: bool     # BBL found the injected layer in top-k
    target_layer_id: Optional[int] = None
    top_violation_layer_id: Optional[int] = None
    scc_time_ms: float = 0.0
    bca_time_ms: float = 0.0

# ============================================================================
# Validation
# ============================================================================

_factory = None

def _get_factory():
    global _factory
    if _factory is None:
        from experiments.validation_core import build_generated_factory
        # Match the UCU_Aiware paper's generation config (seed + 100 nets)
        # so ACT results are comparable to the published baseline.
        _factory = build_generated_factory()
    return _factory

def run_validation(
    domain: str,
    mutation: MutationType,
    net_seed: int,
    target_layer_id: int = 3,
    num_samples: int = 20,
) -> DetectionResult:
    """
    Run validation using the ACT infrastructure.

    Flow: load network -> analyze -> mutate bounds -> detect via BBL/CBR -> localize.
    """
    from experiments.validation_core import run_full_detection

    factory = _get_factory()
    available = factory.list_networks()
    if not available:
        raise RuntimeError("No networks available in ModelFactory")

    net_idx = net_seed % len(available)
    network_name = available[net_idx]

    result = run_full_detection(
        factory=factory,
        network_name=network_name,
        domain=domain,
        mutation_type=mutation,
        net_seed=net_seed,
        target_layer_index=target_layer_id,
        mutation_factor=0.1,
        num_cbr_samples=num_samples,
    )

    if result.error:
        print(f"  Warning: Validation error ({result.error})")

    top_layer = result.bca_top_violation_layers[0] if result.bca_top_violation_layers else None

    return DetectionResult(
        network_idx=net_idx,
        network_seed=net_seed,
        domain=domain,
        mutation=mutation.value,
        scc_detected=result.scc_detected,
        bca_detected=result.bca_detected,
        localized=result.localized_top5,
        target_layer_id=result.target_layer_id,
        top_violation_layer_id=top_layer,
        scc_time_ms=result.scc_time_ms,
        bca_time_ms=result.bca_time_ms,
    )

# ============================================================================
# Main Experiment
# ============================================================================

def run_rq1_experiment(
    master_seed: int,
    num_networks: int = 30,
    output_dir: Path = Path("results/rq1"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RQ1: Two-Level Detection Capability Evaluation.

    Args:
        master_seed: Global master seed
        num_networks: Networks per mutation x domain combination
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Results dictionary with structure matching table format
    """
    set_all_seeds(master_seed)
    experiment_seed = derive_seed(master_seed, "rq1", 1000)

    print(f"RQ1: Two-Level Detection Capability")
    print(f"=" * 70)
    print(f"Master seed: {master_seed}")
    print(f"Experiment seed: {experiment_seed}")
    print(f"Networks per config: {num_networks}")
    print(f"Domains: {DOMAINS}")
    print(f"Mutations: {[m.name for m in MUTATIONS['soundness'] + MUTATIONS['control']]}")
    print()

    # Storage for results
    all_mutations = MUTATIONS["soundness"] + MUTATIONS["control"]
    results_by_domain_mutation: Dict[Tuple[str, str], List[DetectionResult]] = defaultdict(list)

    total_configs = len(DOMAINS) * len(all_mutations) * num_networks
    completed = 0

    for domain in DOMAINS:
        for mutation in all_mutations:
            if verbose:
                print(f"\n[{domain}] {mutation.name}:")

            for net_idx in range(num_networks):
                # Derive deterministic seed for this specific experiment
                net_seed = derive_seed(experiment_seed, domain, mutation.value, net_idx)
                target_layer = 3 + (net_idx % 5)  # Vary target layer

                result = run_validation(
                    domain=domain,
                    mutation=mutation,
                    net_seed=net_seed,
                    target_layer_id=target_layer,
                )

                result.network_idx = net_idx
                results_by_domain_mutation[(domain, mutation.value)].append(result)

                completed += 1
                if verbose and completed % 50 == 0:
                    print(f"  Progress: {completed}/{total_configs}")

    # =========================================================================
    # Compute statistics for table
    # =========================================================================

    table_data = {
        "metadata": {
            "master_seed": master_seed,
            "experiment_seed": experiment_seed,
            "num_networks": num_networks,
            "domains": DOMAINS,
            "mutations": [m.name for m in all_mutations],
        },
        "raw_results": {},
        "table_rq1": {},
        "summary": {},
    }

    # Convert raw results to serializable format
    for (domain, mutation), results in results_by_domain_mutation.items():
        key = f"{domain}/{mutation}"
        table_data["raw_results"][key] = [asdict(r) for r in results]

    # Compute table statistics
    overall_scc = []
    overall_bca = []
    overall_combined = []
    overall_localized = []

    for domain in DOMAINS:
        table_data["table_rq1"][domain] = {}

        for mutation in all_mutations:
            key = (domain, mutation.value)
            results = results_by_domain_mutation[key]
            n = len(results)

            if n == 0:
                continue

            scc_rate = sum(1 for r in results if r.scc_detected) / n
            bca_rate = sum(1 for r in results if r.bca_detected) / n
            combined_rate = sum(1 for r in results if r.scc_detected or r.bca_detected) / n

            # Localized: among BBL detections, how many localized correctly
            bca_detected = [r for r in results if r.bca_detected]
            if bca_detected:
                localized_rate = sum(1 for r in bca_detected if r.localized) / len(bca_detected)
            else:
                localized_rate = None  # N/A

            table_data["table_rq1"][domain][mutation.value] = {
                "n": n,
                "scc_only": scc_rate,
                "bca_only": bca_rate,
                "combined": combined_rate,
                "localized": localized_rate,
            }

            # Aggregate (exclude control mutations)
            if mutation in MUTATIONS["soundness"]:
                overall_scc.extend([r.scc_detected for r in results])
                overall_bca.extend([r.bca_detected for r in results])
                overall_combined.extend([r.scc_detected or r.bca_detected for r in results])
                overall_localized.extend([r.localized for r in bca_detected])

    # Overall summary
    if overall_scc:
        n_total = len(overall_scc)
        table_data["summary"]["overall"] = {
            "n": n_total,
            "scc_only": sum(overall_scc) / n_total,
            "bca_only": sum(overall_bca) / n_total,
            "combined": sum(overall_combined) / n_total,
            "localized": sum(overall_localized) / len(overall_localized) if overall_localized else None,
        }

    # =========================================================================
    # Print results
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("Detection Rates by Domain and Mutation")
    print(f"{'=' * 70}")
    print(f"{'Domain':<12} {'Mutation':<15} {'CBR':>8} {'BBL':>8} {'Combined':>10} {'Localized':>10}")
    print("-" * 70)

    for domain in DOMAINS:
        for mutation in all_mutations:
            stats = table_data["table_rq1"].get(domain, {}).get(mutation.value, {})
            if not stats:
                continue

            scc_str = f"{stats['scc_only']*100:.1f}%"
            bca_str = f"{stats['bca_only']*100:.1f}%"
            combined_str = f"{stats['combined']*100:.1f}%"
            localized_str = f"{stats['localized']*100:.1f}%" if stats['localized'] is not None else "N/A"

            print(f"{domain:<12} {mutation.value:<15} {scc_str:>8} {bca_str:>8} {combined_str:>10} {localized_str:>10}")

    print("-" * 70)
    if "overall" in table_data["summary"]:
        o = table_data["summary"]["overall"]
        loc_str = f"{o['localized']*100:.1f}%" if o['localized'] is not None else "N/A"
        print(f"{'OVERALL':<12} {'All Soundness':<15} {o['scc_only']*100:.1f}%   {o['bca_only']*100:.1f}%   {o['combined']*100:.1f}%      {loc_str}")

    # =========================================================================
    # Save results
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(table_data, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_latex_table(table_data)
    with open(output_dir / "table_rq1.tex", "w") as f:
        f.write(latex_table)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"  - results.json")
    print(f"  - table_rq1.tex")
    print(f"{'=' * 70}")

    return table_data

def generate_latex_table(data: Dict[str, Any]) -> str:
    """Generate LaTeX table matching paper format."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ1: Detection rates by mutation, domain, and validation level}",
        r"\label{tab:rq1-detection}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Mutation} & \textbf{CBR Only} & \textbf{BBL Only} & \textbf{Combined} & \textbf{Localized} \\",
        r"\midrule",
    ]

    all_mutations = MUTATIONS["soundness"] + MUTATIONS["control"]

    for i, domain in enumerate(DOMAINS):
        # Add multirow for domain
        n_mutations = len(all_mutations)

        for j, mutation in enumerate(all_mutations):
            stats = data["table_rq1"].get(domain, {}).get(mutation.value, {})

            # Format values
            if stats:
                scc = f"{stats['scc_only']*100:.0f}\\%"
                bca = f"{stats['bca_only']*100:.0f}\\%"
                combined = f"{stats['combined']*100:.0f}\\%"
                localized = f"{stats['localized']*100:.0f}\\%" if stats['localized'] is not None else "N/A"
            else:
                scc = bca = combined = localized = "--\\%"

            # Get mutation label
            label = MUTATION_LABELS.get(mutation.value, mutation.value)

            if j == 0:
                domain_col = f"\\multirow{{{n_mutations}}}{{*}}{{{domain}}}"
            else:
                domain_col = ""

            lines.append(f"{domain_col}")
            lines.append(f"& {label} & {scc} & {bca} & {combined} & {localized} \\\\")

        if i < len(DOMAINS) - 1:
            lines.append(r"\midrule")

    # Overall row
    lines.append(r"\midrule")
    if "overall" in data["summary"]:
        o = data["summary"]["overall"]
        scc = f"{o['scc_only']*100:.0f}\\%"
        bca = f"{o['bca_only']*100:.0f}\\%"
        combined = f"\\textbf{{{o['combined']*100:.0f}\\%}}"
        localized = f"{o['localized']*100:.0f}\\%" if o['localized'] is not None else "N/A"
        lines.append(f"\\textbf{{Overall}} & \\textbf{{All}} & {scc} & {bca} & {combined} & {localized} \\\\")
    else:
        lines.append(r"\textbf{Overall} & \textbf{All} & --\% & --\% & \textbf{--\%} & --\% \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="RQ1: Two-Level Detection Capability Evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed (default: 42)"
    )
    parser.add_argument(
        "--num-networks", type=int, default=30,
        help="Networks per mutation x domain config (default: 30)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq1",
        help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    run_rq1_experiment(
        master_seed=args.seed,
        num_networks=args.num_networks,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
