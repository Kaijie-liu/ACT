# ===- act/pipeline/moe/test_experiment1_confirmatory.py - Tests -----====#

import json
import tempfile
import time
import unittest
from pathlib import Path

from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.audit_experiment1_confirmatory import _cluster_bootstrap
from act.pipeline.moe.experiment1_confirmatory import (
    DEFAULT_CONFIG,
    _boundary_summary,
    _census_summary,
    _run_boundary_with_deadline,
)


def _slow_boundary_row(*_args, **_kwargs):
    time.sleep(0.25)
    return {"status": "SAFE", "reason": "must_not_return"}


def _fast_boundary_row(model, dataset, selection, *_args, **_kwargs):
    if model.spec.top_k != 2 or len(dataset) != 10000:
        raise RuntimeError("spawned child did not load the frozen model/dataset")
    return {
        "sample_rank": selection["sample_rank"],
        "dataset_index": selection["dataset_index"],
        "status": "UNKNOWN",
        "reason": "ENGINEERING_SPAWN_PROBE",
        "total_seconds": 0.0,
    }


class ConfirmatoryProtocolTests(unittest.TestCase):
    def test_tracked_numerical_policy_matches_implementation(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(
            config["numerical_safety"], hz_numerical_policy_manifest()
        )

    def test_multiseed_configs_share_frozen_selection_and_policy(self):
        config_root = DEFAULT_CONFIG.parent
        configs = []
        for seed in (1, 2):
            path = config_root / f"experiment1_multiseed_seed{seed}_r1.json"
            configs.append(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(configs[0]["sample_count"], 40)
        self.assertEqual(configs[1]["sample_count"], 40)
        self.assertEqual(
            configs[0]["selection_manifest"], configs[1]["selection_manifest"]
        )
        self.assertEqual(configs[0]["numerical_safety"], hz_numerical_policy_manifest())
        self.assertEqual(configs[1]["numerical_safety"], hz_numerical_policy_manifest())
        self.assertEqual(configs[0]["instance_timeout_seconds"], 300.0)
        self.assertEqual(configs[1]["instance_timeout_seconds"], 300.0)

    def test_census_summary_closes_guard_accounting(self):
        row = {
            "sample_rank": 100,
            "status": "COMPLETE",
            "route_set_unstable": True,
            "exact_candidate_count": 3,
            "ibp_candidate_count": 5,
            "zonotope_candidate_count": 4,
            "route_conditioned_max_width": 6,
            "candidate_pruned_monolithic_width": 12,
            "total_seconds": 1.0,
            "branches": [
                {
                    "guard_accounting": {
                        "binaries_before": 10,
                        "binaries_after": 5,
                        "binary_eliminated": 5,
                        "lp_support_eliminated": 2,
                        "milp_support_eliminated": 1,
                        "structural_or_propagation_eliminated": 2,
                    },
                    "support": {"seconds": 0.5},
                }
            ],
        }
        summary = _census_summary([row])
        self.assertEqual(summary["exact_reduces_ibp_rows"], 1)
        self.assertEqual(summary["exact_reduces_zonotope_rows"], 1)
        self.assertEqual(
            summary["width_ratio_candidate_gt_topk"]["median"], 0.5
        )
        self.assertEqual(
            summary["guard_accounting"][
                "structural_or_propagation_eliminated"
            ],
            2,
        )

    def test_boundary_summary_reports_f0_increment(self):
        rows = [
            {
                "sample_rank": 100,
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "unique_safe": True,
                "gate_reason": "UNKNOWN_GATE_SUFFICIENCY",
                "f0_invoked": True,
                "f0_seconds": 2.0,
                "total_seconds": 3.0,
                "gate": {"branches": []},
            },
            {
                "sample_rank": 101,
                "status": "UNKNOWN",
                "reason": "UNKNOWN_SOLVER_LIMIT",
                "unique_safe": False,
                "gate_reason": "UNKNOWN_SOLVER_LIMIT",
                "f0_invoked": False,
                "f0_seconds": 0.0,
                "total_seconds": 1.0,
                "gate": {"branches": []},
            },
        ]
        summary = _boundary_summary(rows)
        self.assertEqual(summary["base_semantic_incompleteness"], 1)
        self.assertEqual(summary["f0_invoked"], 1)
        self.assertEqual(summary["f0_resolved"], 1)
        self.assertEqual(summary["f0_added_safe"], 1)
        self.assertEqual(
            summary["f0_paired_runtime_overhead"]["median"], 2.0
        )
        self.assertEqual(summary["f0_observed_time_rows"], 1)
        self.assertEqual(summary["f0_right_censored_time_rows"], 0)

    def test_boundary_summary_does_not_impute_censored_f0_time_as_zero(self):
        row = {
            "sample_rank": 100,
            "status": "TIMEOUT",
            "reason": "INSTANCE_HARD_DEADLINE",
            "unique_safe": False,
            "gate_reason": "UNKNOWN_GATE_SUFFICIENCY",
            "f0_invoked": True,
            "f0_seconds": None,
            "f0_time_observation": {
                "kind": "RIGHT_CENSORED_AT_INSTANCE_DEADLINE",
                "seconds": None,
                "lower_bound_seconds": 17.0,
            },
            "total_seconds": 300.0,
            "gate": {"branches": []},
        }
        summary = _boundary_summary([row])
        self.assertEqual(summary["f0_seconds"], 0)
        self.assertEqual(summary["f0_observed_time_rows"], 0)
        self.assertEqual(summary["f0_right_censored_time_rows"], 1)
        self.assertEqual(
            summary["f0_right_censored_known_lower_bound_seconds"], 17.0
        )
        self.assertIsNone(summary["f0_paired_runtime_overhead"]["median"])

    def test_cluster_bootstrap_uses_sample_clusters(self):
        rows = [
            {"sample_rank": 100, "hit": True},
            {"sample_rank": 100, "hit": True},
            {"sample_rank": 101, "hit": False},
        ]
        interval = _cluster_bootstrap(
            rows,
            lambda row: row["hit"],
            replicates=200,
            seed=3,
        )
        self.assertLessEqual(interval[0], 2.0 / 3.0)
        self.assertGreaterEqual(interval[1], 2.0 / 3.0)

    def test_boundary_hard_deadline_terminates_child(self):
        result_root = Path("/data1/Kane/MOE/ACT/data/moe/results")
        with tempfile.TemporaryDirectory(dir=result_root) as temporary:
            stage_dir = Path(temporary)
            row = _run_boundary_with_deadline(
                model=object(),
                dataset=object(),
                selection={"sample_rank": 100, "dataset_index": 196},
                stage_dir=stage_dir,
                runtime={},
                config={"instance_timeout_seconds": 0.05},
                row_runner=_slow_boundary_row,
            )
        self.assertEqual(row["status"], "TIMEOUT")
        self.assertEqual(row["reason"], "INSTANCE_HARD_DEADLINE")
        self.assertTrue(row["deadline_enforced"])
        self.assertLess(row["total_seconds"], 0.5)

    @unittest.skipUnless(
        Path(
            "/data1/Kane/MOE/ACT/data/moe/checkpoints/"
            "cifar10_top2_e8_seed0_bal010.pt"
        ).exists(),
        "frozen confirmatory checkpoint is not available",
    )
    def test_spawned_child_loads_frozen_inputs(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = json.load(handle)
        config["instance_timeout_seconds"] = 10.0
        result_root = Path("/data1/Kane/MOE/ACT/data/moe/results")
        with tempfile.TemporaryDirectory(dir=result_root) as temporary:
            row = _run_boundary_with_deadline(
                model=None,
                dataset=None,
                selection={"sample_rank": 100, "dataset_index": 196},
                stage_dir=Path(temporary),
                runtime={},
                config=config,
                row_runner=_fast_boundary_row,
            )
        self.assertEqual(row["reason"], "ENGINEERING_SPAWN_PROBE")
        self.assertTrue(row["deadline_enforced"])


if __name__ == "__main__":
    unittest.main()
