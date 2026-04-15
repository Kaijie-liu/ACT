#!/usr/bin/env python3
"""
Real Data Collection Module

Integrates with the framework's existing validation infrastructure to collect
experimental data. Reuses:
- VerificationValidator from act.pipeline.verification.validate_verifier
- run_per_neuron_bounds_check from act.pipeline.verification.per_neuron_bounds
- ModelFactory from act.pipeline.verification.model_factory
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing validation infrastructure
from act.pipeline.verification.validate_verifier import VerificationValidator
from act.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig,
    run_per_neuron_bounds_check,
    compute_abstract_bounds,
)
from act.pipeline.verification.model_factory import ModelFactory
from act.back_end.core import Net, Bounds, Fact
from act.back_end.verifier import find_entry_layer_id
from act.back_end.transfer_functions import set_transfer_function_mode

# Device and seed management
from act.back_end.validation import (
    set_all_seeds,
    derive_seed,
    initialize_device,
    get_default_device,
)

@dataclass
class ValidationData:
    """Complete validation data for a single network."""
    # Network info
    network_name: str
    network_seed: int

    # Domain info
    tf_mode: str

    # CBR results
    level1_status: str  # PASSED, FAILED, ACCEPTABLE, INCONCLUSIVE, ERROR
    level1_counterexample_found: bool
    level1_time_ms: float

    # BBL results
    level2_status: str  # PASS, FAIL, ERROR
    level2_violations_count: int
    level2_layers_checked: int
    level2_neurons_checked: int
    level2_worst_gap: float
    level2_time_ms: float
    level2_violation_layers: List[int]

    # Combined
    soundness_violated: bool
    violation_localized: bool
    total_time_ms: float

def collect_validation_data_for_network(
    network_name: str,
    tf_mode: str = "interval",
    num_samples: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
    per_neuron_config: Optional[PerNeuronCheckConfig] = None,
) -> ValidationData:
    """
    Collect complete validation data for a single network using existing infrastructure.

    Args:
        network_name: Name of the network in ModelFactory
        tf_mode: Transfer function mode ("interval", "hybridz", "dual")
        num_samples: Number of samples for BBL validation
        device: Compute device
        dtype: Data type
        seed: Random seed
        per_neuron_config: Config for per-neuron bounds check

    Returns:
        ValidationData with all measurements
    """
    set_all_seeds(seed)
    total_start = time.perf_counter()

    if per_neuron_config is None:
        per_neuron_config = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10)

    # Create validator (reuses existing infrastructure)
    validator = VerificationValidator(device=device, dtype=dtype)

    # CBR: Counterexample validation
    level1_start = time.perf_counter()
    try:
        level1_summary = validator.validate_counterexamples(
            networks=[network_name],
            solvers=['torchlp']  # Use torchlp for speed
        )
        level1_results = level1_summary.get('results', [])
        if level1_results:
            r = level1_results[0]
            level1_status = r.get('validation_status', 'INCONCLUSIVE')
            level1_ce_found = r.get('concrete_counterexample', False)
        else:
            level1_status = 'INCONCLUSIVE'
            level1_ce_found = False
    except Exception as e:
        level1_status = 'ERROR'
        level1_ce_found = False
    level1_time_ms = (time.perf_counter() - level1_start) * 1000

    # BBL: Bounds validation
    level2_start = time.perf_counter()
    try:
        level2_summary = validator.validate_bounds(
            networks=[network_name],
            tf_modes=[tf_mode],
            num_samples=num_samples,
            per_neuron_config=per_neuron_config,
        )
        level2_results = level2_summary.get('results', [])
        if level2_results:
            r = level2_results[0]
            level2_status = r.get('validation_status', 'ERROR')
            level2_violations = r.get('violations', [])
            level2_violations_count = len(level2_violations)
            level2_total_checks = r.get('total_checks', 0)
            level2_worst_gap = 0.0
            level2_violation_layers = []
            for v in level2_violations:
                if v.get('worst_gap', 0) > level2_worst_gap:
                    level2_worst_gap = v.get('worst_gap', 0)
                for vl in v.get('violated_layers', []):
                    lid = vl.get('layer_id')
                    if lid is not None and lid not in level2_violation_layers:
                        level2_violation_layers.append(lid)
        else:
            level2_status = 'ERROR'
            level2_violations_count = 0
            level2_total_checks = 0
            level2_worst_gap = 0.0
            level2_violation_layers = []
    except Exception as e:
        level2_status = 'ERROR'
        level2_violations_count = 0
        level2_total_checks = 0
        level2_worst_gap = 0.0
        level2_violation_layers = []
    level2_time_ms = (time.perf_counter() - level2_start) * 1000

    total_time_ms = (time.perf_counter() - total_start) * 1000

    # Combined verdict
    soundness_violated = (
        level1_status == 'FAILED' or
        level2_status == 'FAIL'
    )
    violation_localized = (
        level2_status == 'FAIL' and
        len(level2_violation_layers) > 0
    )

    return ValidationData(
        network_name=network_name,
        network_seed=seed,
        tf_mode=tf_mode,
        level1_status=level1_status,
        level1_counterexample_found=level1_ce_found,
        level1_time_ms=level1_time_ms,
        level2_status=level2_status,
        level2_violations_count=level2_violations_count,
        level2_layers_checked=0,  # TODO: extract from results
        level2_neurons_checked=level2_total_checks,
        level2_worst_gap=level2_worst_gap,
        level2_time_ms=level2_time_ms,
        level2_violation_layers=level2_violation_layers,
        soundness_violated=soundness_violated,
        violation_localized=violation_localized,
        total_time_ms=total_time_ms,
    )

def collect_rq1_data(
    networks: Optional[List[str]] = None,
    tf_modes: List[str] = ["interval"],
    master_seed: int = 42,
    num_samples: int = 10,
    verbose: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Collect data for RQ1: Two-level detection capability.

    Uses existing VerificationValidator infrastructure.

    Args:
        networks: List of network names (None = all from ModelFactory)
        tf_modes: Transfer function modes to test
        master_seed: Random seed for reproducibility
        num_samples: Samples for BBL validation
        verbose: Print progress
        device: Compute device

    Returns:
        Dictionary with detection results
    """
    from experiments.validation_core import build_generated_factory
    factory = build_generated_factory(seed=master_seed, num_networks=30)

    if networks is None:
        networks = factory.list_networks()

    results = {
        "metadata": {
            "num_networks": len(networks),
            "tf_modes": tf_modes,
            "master_seed": master_seed,
            "device": device,
        },
        "detection_results": {},
        "summary": {},
    }

    if verbose:
        print(f"Using device: {device}")
        print(f"Testing {len(networks)} networks with {len(tf_modes)} TF modes")

    set_all_seeds(master_seed)

    for net_idx, network_name in enumerate(networks):
        net_seed = derive_seed(master_seed, "rq1", net_idx)

        for tf_mode in tf_modes:
            key = f"{tf_mode}/{network_name}"
            if key not in results["detection_results"]:
                results["detection_results"][key] = []

            try:
                data = collect_validation_data_for_network(
                    network_name=network_name,
                    tf_mode=tf_mode,
                    num_samples=num_samples,
                    device=device,
                    seed=derive_seed(net_seed, tf_mode),
                )

                results["detection_results"][key].append({
                    "network_name": network_name,
                    "network_seed": net_seed,
                    "level1_status": data.level1_status,
                    "level1_detected": data.level1_status == "FAILED",
                    "level2_status": data.level2_status,
                    "level2_detected": data.level2_status == "FAIL",
                    "either_detected": data.soundness_violated,
                    "localized": data.violation_localized,
                    "level1_time_ms": data.level1_time_ms,
                    "level2_time_ms": data.level2_time_ms,
                })

                if verbose:
                    status = "✓" if not data.soundness_violated else "✗"
                    print(f"  {status} {network_name}/{tf_mode}: L1={data.level1_status}, L2={data.level2_status}")

            except Exception as e:
                if verbose:
                    print(f"  ⚠ {network_name}/{tf_mode}: Error - {e}")
                results["detection_results"][key].append({
                    "network_name": network_name,
                    "network_seed": net_seed,
                    "error": str(e),
                })

    # Calculate summary statistics
    for key, records in results["detection_results"].items():
        valid_records = [r for r in records if "error" not in r]
        n = len(valid_records)
        if n == 0:
            continue

        results["summary"][key] = {
            "n": n,
            "level1_detection_rate": sum(r["level1_detected"] for r in valid_records) / n,
            "level2_detection_rate": sum(r["level2_detected"] for r in valid_records) / n,
            "combined_detection_rate": sum(r["either_detected"] for r in valid_records) / n,
            "localization_rate": sum(r["localized"] for r in valid_records) / n,
            "avg_level1_time_ms": sum(r["level1_time_ms"] for r in valid_records) / n,
            "avg_level2_time_ms": sum(r["level2_time_ms"] for r in valid_records) / n,
        }

    return results

def format_rq1_table(results: Dict[str, Any]) -> str:
    """Format RQ1 results as LaTeX table."""
    lines = []
    lines.append("% RQ1 Detection Rates")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{TF Mode / Network} & \\textbf{CBR} & \\textbf{BBL} & \\textbf{Combined} & \\textbf{Localized} \\\\")
    lines.append("\\midrule")

    summary = results.get("summary", {})
    for key, s in summary.items():
        l1 = f"{s['level1_detection_rate']*100:.0f}\\%"
        l2 = f"{s['level2_detection_rate']*100:.0f}\\%"
        combined = f"{s['combined_detection_rate']*100:.0f}\\%"
        localized = f"{s['localization_rate']*100:.0f}\\%" if s['localization_rate'] > 0 else "N/A"
        lines.append(f"{key} & {l1} & {l2} & {combined} & {localized} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)

if __name__ == "__main__":
    # Example: collect real data for RQ1
    print("Collecting RQ1 data using existing VerificationValidator...")

    from experiments.validation_core import build_generated_factory
    factory = build_generated_factory(seed=42, num_networks=30)
    available = factory.list_networks()
    print(f"Available networks: {available[:5]}...")

    rq1_data = collect_rq1_data(
        networks=available[:3] if available else None,  # Test with first 3
        tf_modes=["interval"],
        master_seed=42,
        num_samples=5,
        verbose=True,
    )

    print("\nSummary:")
    for key, summary in rq1_data["summary"].items():
        print(f"  {key}:")
        print(f"    CBR: {summary['level1_detection_rate']*100:.1f}%")
        print(f"    BBL: {summary['level2_detection_rate']*100:.1f}%")
        print(f"    Combined: {summary['combined_detection_rate']*100:.1f}%")

    print("\nLaTeX Table:")
    print(format_rq1_table(rq1_data))
