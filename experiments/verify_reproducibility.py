#!/usr/bin/env python3
"""
Reproducibility Verification Script

Verifies that experiment results can be exactly reproduced with the same seed.
This is critical for paper submission requirements.

Usage:
    # Verify results against expected baseline
    python experiments/verify_reproducibility.py --results-dir results/ --expected-dir results_expected/

    # Re-run and verify (creates new results and compares)
    python experiments/verify_reproducibility.py --seed 42 --rerun

    # Generate expected baseline
    python experiments/verify_reproducibility.py --seed 42 --generate-baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-10

def hash_json(data: Any) -> str:
    """Compute hash of JSON-serializable data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

def compare_values(expected: Any, actual: Any, path: str = "") -> List[str]:
    """
    Compare two values recursively.

    Args:
        expected: Expected value
        actual: Actual value
        path: Path for error reporting

    Returns:
        List of mismatch descriptions
    """
    mismatches = []

    if type(expected) != type(actual):
        mismatches.append(f"{path}: type mismatch ({type(expected).__name__} vs {type(actual).__name__})")
        return mismatches

    if isinstance(expected, dict):
        # Compare dictionaries
        all_keys = set(expected.keys()) | set(actual.keys())
        for key in all_keys:
            key_path = f"{path}.{key}" if path else key
            if key not in expected:
                mismatches.append(f"{key_path}: unexpected key in actual")
            elif key not in actual:
                mismatches.append(f"{key_path}: missing key in actual")
            else:
                mismatches.extend(compare_values(expected[key], actual[key], key_path))

    elif isinstance(expected, list):
        # Compare lists
        if len(expected) != len(actual):
            mismatches.append(f"{path}: length mismatch ({len(expected)} vs {len(actual)})")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                mismatches.extend(compare_values(e, a, f"{path}[{i}]"))

    elif isinstance(expected, float):
        # Compare floats with tolerance
        if abs(expected - actual) > FLOAT_TOLERANCE:
            mismatches.append(f"{path}: {expected} != {actual} (diff={abs(expected-actual)})")

    else:
        # Direct comparison
        if expected != actual:
            mismatches.append(f"{path}: {expected} != {actual}")

    return mismatches

def verify_experiment_results(
    expected_dir: Path,
    actual_dir: Path,
    experiment_name: str
) -> Tuple[bool, List[str]]:
    """
    Verify results for a single experiment.

    Args:
        expected_dir: Directory with expected results
        actual_dir: Directory with actual results
        experiment_name: Name of the experiment

    Returns:
        Tuple of (success, list of mismatches)
    """
    mismatches = []

    # Check metadata
    expected_meta_path = expected_dir / "metadata.json"
    actual_meta_path = actual_dir / "metadata.json"

    if not expected_meta_path.exists():
        return False, [f"Expected metadata not found: {expected_meta_path}"]
    if not actual_meta_path.exists():
        return False, [f"Actual metadata not found: {actual_meta_path}"]

    with open(expected_meta_path) as f:
        expected_meta = json.load(f)
    with open(actual_meta_path) as f:
        actual_meta = json.load(f)

    # Check critical metadata fields
    if expected_meta.get("master_seed") != actual_meta.get("master_seed"):
        mismatches.append(f"master_seed: {expected_meta.get('master_seed')} != {actual_meta.get('master_seed')}")

    if expected_meta.get("seed_offset") != actual_meta.get("seed_offset"):
        mismatches.append(f"seed_offset: {expected_meta.get('seed_offset')} != {actual_meta.get('seed_offset')}")

    # Check results
    expected_results_path = expected_dir / "results.json"
    actual_results_path = actual_dir / "results.json"

    if not expected_results_path.exists():
        return False, [f"Expected results not found: {expected_results_path}"]
    if not actual_results_path.exists():
        return False, [f"Actual results not found: {actual_results_path}"]

    with open(expected_results_path) as f:
        expected_results = json.load(f)
    with open(actual_results_path) as f:
        actual_results = json.load(f)

    # Compare metadata section
    meta_mismatches = compare_values(
        expected_results.get("metadata", {}),
        actual_results.get("metadata", {}),
        "metadata"
    )
    mismatches.extend(meta_mismatches)

    # Compare network seeds if present
    if "networks" in expected_results and "networks" in actual_results:
        for i, (exp_net, act_net) in enumerate(zip(
            expected_results["networks"],
            actual_results["networks"]
        )):
            if exp_net.get("seed") != act_net.get("seed"):
                mismatches.append(f"networks[{i}].seed: {exp_net.get('seed')} != {act_net.get('seed')}")
            if exp_net.get("name") != act_net.get("name"):
                mismatches.append(f"networks[{i}].name: {exp_net.get('name')} != {act_net.get('name')}")

    # Compare detection/effectiveness results (sample to avoid overwhelming output)
    for key in ["detection_results", "effectiveness_results", "localization_results"]:
        if key in expected_results and key in actual_results:
            exp_data = expected_results[key]
            act_data = actual_results[key]

            for sub_key in exp_data:
                if sub_key not in act_data:
                    mismatches.append(f"{key}.{sub_key}: missing in actual")
                    continue

                exp_list = exp_data[sub_key]
                act_list = act_data[sub_key]

                if len(exp_list) != len(act_list):
                    mismatches.append(f"{key}.{sub_key}: length {len(exp_list)} != {len(act_list)}")
                    continue

                # Sample check first few items
                for i in range(min(5, len(exp_list))):
                    if "seed" in exp_list[i] and exp_list[i]["seed"] != act_list[i].get("seed"):
                        mismatches.append(f"{key}.{sub_key}[{i}].seed mismatch")

    # Compare summary statistics
    if "summary" in expected_results and "summary" in actual_results:
        summary_mismatches = compare_values(
            expected_results["summary"],
            actual_results["summary"],
            "summary"
        )
        # Filter to only significant differences
        significant = [m for m in summary_mismatches if "detection_rate" in m or "accuracy" in m]
        mismatches.extend(significant[:10])  # Limit to 10

    return len(mismatches) == 0, mismatches

def verify_all_experiments(
    expected_base: Path,
    actual_base: Path,
    experiments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Verify all experiment results.

    Args:
        expected_base: Base directory for expected results
        actual_base: Base directory for actual results
        experiments: List of experiments to verify (None = all found)

    Returns:
        Verification report
    """
    if experiments is None:
        # Auto-detect experiments
        experiments = []
        for d in expected_base.iterdir():
            if d.is_dir() and d.name.startswith("rq"):
                experiments.append(d.name)

    report = {
        "experiments": {},
        "overall_success": True,
        "total_checked": 0,
        "total_passed": 0,
        "total_failed": 0,
    }

    for exp_name in sorted(experiments):
        expected_dir = expected_base / exp_name
        actual_dir = actual_base / exp_name

        if not expected_dir.exists():
            report["experiments"][exp_name] = {
                "status": "skipped",
                "reason": f"Expected directory not found: {expected_dir}",
            }
            continue

        if not actual_dir.exists():
            report["experiments"][exp_name] = {
                "status": "skipped",
                "reason": f"Actual directory not found: {actual_dir}",
            }
            continue

        success, mismatches = verify_experiment_results(expected_dir, actual_dir, exp_name)

        report["total_checked"] += 1
        if success:
            report["total_passed"] += 1
            report["experiments"][exp_name] = {
                "status": "passed",
                "mismatches": [],
            }
        else:
            report["total_failed"] += 1
            report["overall_success"] = False
            report["experiments"][exp_name] = {
                "status": "failed",
                "mismatches": mismatches[:20],  # Limit output
            }

    return report

def generate_baseline(
    seed: int,
    output_dir: Path,
    config_path: str
) -> bool:
    """
    Generate baseline expected results.

    Args:
        seed: Master seed
        output_dir: Output directory for baseline
        config_path: Configuration file path

    Returns:
        True if successful
    """
    print(f"Generating baseline results with seed={seed}")
    print(f"Output directory: {output_dir}")

    # Run all experiments
    cmd = [
        sys.executable,
        "experiments/run_all.py",
        "--seed", str(seed),
        "--config", config_path,
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment reproducibility"
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
        "--results-dir", type=str, default="results",
        help="Directory with actual results"
    )
    parser.add_argument(
        "--expected-dir", type=str, default=None,
        help="Directory with expected results (default: {results-dir}_expected)"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        help="Specific experiments to verify"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run experiments before verification"
    )
    parser.add_argument(
        "--generate-baseline", action="store_true",
        help="Generate baseline expected results"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    expected_dir = Path(args.expected_dir) if args.expected_dir else Path(f"{args.results_dir}_expected")

    # Generate baseline mode
    if args.generate_baseline:
        success = generate_baseline(args.seed, expected_dir, args.config)
        if success:
            print(f"\nBaseline generated successfully at: {expected_dir}")
            print("You can now run experiments and verify against this baseline.")
        else:
            print("\nBaseline generation failed!")
        sys.exit(0 if success else 1)

    # Re-run mode
    if args.rerun:
        print("Re-running experiments...")
        cmd = [
            sys.executable,
            "experiments/run_all.py",
            "--seed", str(args.seed),
            "--config", args.config,
            "--output-dir", str(results_dir),
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # Verify results
    print(f"\n{'=' * 60}")
    print("Reproducibility Verification")
    print(f"{'=' * 60}")
    print(f"Expected: {expected_dir}")
    print(f"Actual: {results_dir}")
    print()

    report = verify_all_experiments(
        expected_base=expected_dir,
        actual_base=results_dir,
        experiments=args.experiments,
    )

    # Print report
    print(f"Results:")
    print(f"  Total checked: {report['total_checked']}")
    print(f"  Passed: {report['total_passed']}")
    print(f"  Failed: {report['total_failed']}")
    print()

    for exp_name, exp_report in report["experiments"].items():
        status = exp_report["status"]
        if status == "passed":
            print(f"  ✓ {exp_name}: PASSED")
        elif status == "failed":
            print(f"  ✗ {exp_name}: FAILED")
            if args.verbose and exp_report.get("mismatches"):
                for mismatch in exp_report["mismatches"][:5]:
                    print(f"      - {mismatch}")
                if len(exp_report["mismatches"]) > 5:
                    print(f"      ... and {len(exp_report['mismatches']) - 5} more")
        else:
            print(f"  - {exp_name}: {status} ({exp_report.get('reason', '')})")

    print()
    if report["overall_success"]:
        print("=" * 60)
        print("✓ All results match exactly! Reproducibility verified.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ Reproducibility check FAILED!")
        print("Some results do not match the expected baseline.")
        print("=" * 60)

    # Save report
    report_path = results_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    sys.exit(0 if report["overall_success"] else 1)

if __name__ == "__main__":
    main()
