"""
Experiment Scripts for Two-Level Validation Framework

This module contains reproducible experiment scripts for evaluating
the two-level validation framework (CBR, BBL).

Research Questions:
- RQ1: Two-level detection capability evaluation
- RQ2: CBR effectiveness boundary analysis
- RQ3: BBL localization accuracy
- RQ4: TF-aware generation coverage analysis
- RQ5: Cross-domain behavior comparison
- RQ6: Validation runtime overhead measurement

Usage:
    # Run all experiments
    python experiments/run_all.py --seed 42

    # Run specific experiment
    python experiments/rq1_detection.py --seed 42

    # Verify reproducibility
    python experiments/verify_reproducibility.py --seed 42 --verify
"""

__all__ = [
    "rq1_detection",
    "rq2_scc_effectiveness",
    "rq3_localization",
    "rq4_coverage",
    "rq5_cross_domain",
    "rq6_overhead",
    "run_all",
    "verify_reproducibility",
]
