#!/usr/bin/env python3
"""
Run All RQ Experiments

Orchestrates the execution of all research question experiments with
reproducible seeds and generates a comprehensive summary.

Usage:
    # Run all experiments with default seed
    python experiments/run_all.py

    # Run with specific seed
    python experiments/run_all.py --seed 42

    # Run specific experiments only
    python experiments/run_all.py --experiments rq1 rq3

    # Verify reproducibility
    python experiments/run_all.py --seed 42 --verify
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Experiment configuration
EXPERIMENTS = {
    "rq1": {
        "script": "experiments/rq1_detection.py",
        "description": "Two-Level Detection Capability",
        "output_dir": "results/rq1",
    },
    "rq2": {
        "script": "experiments/rq2_scc_effectiveness.py",
        "description": "CBR Effectiveness Boundary",
        "output_dir": "results/rq2",
    },
    "rq3": {
        "script": "experiments/rq3_localization.py",
        "description": "BBL Localization Accuracy",
        "output_dir": "results/rq3",
    },
    "rq4": {
        "script": "experiments/rq4_coverage.py",
        "description": "TF-Aware Generation Coverage",
        "output_dir": "results/rq4",
    },
    "rq5": {
        "script": "experiments/rq5_cross_domain.py",
        "description": "Cross-Domain Comparison",
        "output_dir": "results/rq5",
    },
    "rq6": {
        "script": "experiments/rq6_overhead.py",
        "description": "Validation Runtime Overhead",
        "output_dir": "results/rq6",
    },
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(
    name: str,
    exp_config: Dict[str, str],
    seed: int,
    config_path: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        name: Experiment name (e.g., "rq1")
        exp_config: Experiment configuration
        seed: Master seed
        config_path: Path to config.yaml
        verbose: Enable verbose output

    Returns:
        Result dictionary with status and timing
    """
    script = exp_config["script"]
    output_dir = exp_config["output_dir"]

    print(f"\n{'=' * 60}")
    print(f"Running {name.upper()}: {exp_config['description']}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Build command
    cmd = [
        sys.executable,
        script,
        "--seed", str(seed),
        "--config", config_path,
        "--output-dir", output_dir,
    ]

    if verbose:
        cmd.append("-v")

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            status = "success"
            print(f"\n{name.upper()} completed successfully in {elapsed:.1f}s")
        else:
            status = "failed"
            print(f"\n{name.upper()} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")

        return {
            "name": name,
            "status": status,
            "elapsed_seconds": elapsed,
            "return_code": result.returncode,
            "output_dir": output_dir,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{name.upper()} failed with exception: {e}")
        return {
            "name": name,
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(e),
            "output_dir": output_dir,
        }

def generate_summary(
    results: List[Dict[str, Any]],
    seed: int,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate comprehensive summary of all experiments."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "master_seed": seed,
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] in ["failed", "error"]),
        "total_time_seconds": sum(r["elapsed_seconds"] for r in results),
        "experiments": results,
    }

    # Load individual experiment results for aggregate statistics
    aggregate = {}
    for result in results:
        if result["status"] != "success":
            continue

        rq_name = result["name"]
        results_file = Path(result["output_dir"]) / "results.json"

        if results_file.exists():
            try:
                with open(results_file) as f:
                    exp_results = json.load(f)
                aggregate[rq_name] = exp_results.get("summary", {})
            except Exception as e:
                print(f"Warning: Could not load {rq_name} results: {e}")

    summary["aggregate_results"] = aggregate

    return summary

def run_all_experiments(
    seed: int,
    config_path: str,
    output_dir: Path,
    experiments: Optional[List[str]] = None,
    verbose: bool = False,
    verify: bool = False,
) -> Dict[str, Any]:
    """
    Run all experiments.

    Args:
        seed: Master seed
        config_path: Path to config.yaml
        output_dir: Base output directory
        experiments: List of specific experiments to run (None = all)
        verbose: Enable verbose output
        verify: Run in verification mode

    Returns:
        Summary of all experiment results
    """
    print(f"{'#' * 60}")
    print(f"# Validation Experiment Suite")
    print(f"{'#' * 60}")
    print(f"Master seed: {seed}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().isoformat()}")

    # Determine experiments to run
    if experiments:
        to_run = {k: v for k, v in EXPERIMENTS.items() if k in experiments}
    else:
        to_run = EXPERIMENTS

    print(f"Experiments to run: {list(to_run.keys())}")

    # Run experiments
    results = []
    for name, exp_config in to_run.items():
        result = run_experiment(
            name=name,
            exp_config=exp_config,
            seed=seed,
            config_path=config_path,
            verbose=verbose,
        )
        results.append(result)

    # Generate summary
    print(f"\n{'#' * 60}")
    print(f"# Summary")
    print(f"{'#' * 60}")

    summary = generate_summary(results, seed, output_dir)

    print(f"\nTotal experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_time_seconds']:.1f}s")

    # Print individual results
    print(f"\nExperiment Results:")
    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} {result['name'].upper()}: {result['status']} ({result['elapsed_seconds']:.1f}s)")

    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Verification mode
    if verify:
        print(f"\n{'=' * 60}")
        print("Verification Mode")
        print(f"{'=' * 60}")

        # Run verification script
        verify_cmd = [
            sys.executable,
            "experiments/verify_reproducibility.py",
            "--seed", str(seed),
            "--results-dir", str(output_dir),
        ]

        try:
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            if verify_result.returncode == 0:
                print("Verification PASSED")
            else:
                print("Verification FAILED")
                print(verify_result.stdout)
                print(verify_result.stderr)
        except Exception as e:
            print(f"Verification error: {e}")

    print(f"\n{'#' * 60}")
    print(f"# Experiment suite completed")
    print(f"# End time: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Run all RQ experiments"
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
        "--output-dir", type=str, default="results",
        help="Base output directory"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="Specific experiments to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification after experiments"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    summary = run_all_experiments(
        seed=args.seed,
        config_path=args.config,
        output_dir=output_dir,
        experiments=args.experiments,
        verbose=args.verbose,
        verify=args.verify,
    )

    # Exit with appropriate code
    if summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
