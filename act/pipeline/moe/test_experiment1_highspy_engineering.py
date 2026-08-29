import unittest

from act.pipeline.moe.experiment1_highspy_engineering import (
    _validate_engineering_config,
    summarize_incremental_telemetry,
)


def _telemetry(**updates):
    value = {
        "model_builds": 1,
        "model_build_failures": 0,
        "objective_update_calls": 3,
        "objective_coefficients_changed": 5,
        "row_pool_additions": 1,
        "row_update_calls": 4,
        "row_coefficients_changed": 2,
        "row_bound_updates": 2,
        "integrality_update_calls": 1,
        "run_time_limit_warnings_accepted": 1,
        "budget_extension_calls": 1,
        "budget_extension_seconds": 7.0,
        "solves": 3,
        "cold_start_solves": 2,
        "basis_submission_attempts": 1,
        "basis_submissions_accepted": 1,
        "basis_valid_after_solve": 1,
        "simplex_iterations": 2,
        "ipm_iterations": 0,
        "mip_nodes": 4,
        "model_build_seconds": 0.1,
        "objective_update_seconds": 0.2,
        "row_update_seconds": 0.3,
        "solve_seconds": 1.0,
        "total_seconds": 1.6,
        "status_counts": {"optimal": 2, "time_limit": 1},
        "build_error": None,
    }
    value.update(updates)
    return value


class IncrementalEngineeringRunnerTests(unittest.TestCase):
    def test_terminal_attempt_only_is_aggregated(self):
        first = _telemetry(solves=1, budget_extension_calls=0)
        terminal = _telemetry(solves=3, budget_extension_calls=1)
        row = {
            "sample_rank": 1,
            "status": "SAFE",
            "total_seconds": 10.0,
            "gate": {
                "branches": [{
                    "support": {"layers": [{
                        "lp_incremental_telemetry": _telemetry(solves=2),
                        "milp_incremental_telemetry": None,
                    }]},
                    "attempts": [
                        {"metadata": {"incremental_hz": first}},
                        {"metadata": {"incremental_hz": terminal}},
                    ],
                }]
            },
            "f0": None,
        }
        baseline = [{
            "sample_rank": 1,
            "status": "UNKNOWN",
            "total_seconds": 20.0,
        }]
        summary = summarize_incremental_telemetry([row], baseline)
        self.assertEqual(summary["incremental_sessions"], 2)
        self.assertEqual(summary["telemetry_totals"]["solves"], 5)
        self.assertEqual(
            summary["session_category_counts"],
            {"guarded_support_lp": 1, "expert_property": 1},
        )
        self.assertEqual(summary["paired_status_transitions"], {"UNKNOWN->SAFE": 1})
        self.assertFalse(summary["f0_cross_augmented_hz_reuse"])

    def test_config_requires_explicit_opt_in_and_frozen_scope(self):
        config = {
            "scope": "engineering_performance_rerun_not_confirmatory_overwrite",
            "engineering_allow_support_solver_drift": True,
            "expected_rows": 20,
            "instance_timeout_seconds": 900.0,
            "support": {"solver_backend": "highspy_incremental"},
            "solver": {
                "backend": "highspy_incremental",
                "f0_backend": "highspy_incremental",
            },
        }
        _validate_engineering_config(config)
        with self.assertRaises(ValueError):
            _validate_engineering_config({**config, "expected_rows": 19})
        with self.assertRaises(ValueError):
            _validate_engineering_config({
                **config,
                "support": {"solver_backend": "scipy"},
            })


if __name__ == "__main__":
    unittest.main()
