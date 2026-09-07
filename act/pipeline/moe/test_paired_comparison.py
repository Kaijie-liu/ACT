import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.pipeline.moe.test_staged_verifier import _config, _constant_model
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package, _audit_safe_structure
from act.pipeline.moe.paired_followup import schedule, method_config, summarize, METHODS


class ComparisonTests(unittest.TestCase):
    def test_schedule_balances_positions_and_covers_all_jobs(self):
        selection = {"models": {"seed0": {}, "seed1": {}, "seed2": {}},
                     "samples": [{"dataset_index": 100 + i} for i in range(100)]}
        jobs = schedule(selection, range(100))
        self.assertEqual(len(jobs), 1200)
        self.assertEqual(len({j["job_id"] for j in jobs}), 1200)
        for model in selection["models"]:
            for method in METHODS:
                for position in range(4):
                    self.assertEqual(sum(j["model"] == model and j["method"] == method
                                         and j["position"] == position for j in jobs), 25)

    def test_method_budget_keeps_numerical_policy(self):
        base = _config()
        cfg = method_config(base, "tier1_only", 300.)
        self.assertEqual(cfg["tier1"]["solver"]["escalation_budget_per_branch"], 300.)
        self.assertEqual(cfg["numerical_safety"], base["numerical_safety"])
        self.assertEqual(base["tier1"]["solver"]["escalation_budget_per_branch"], 25.)

    def test_paired_safe_and_solved_are_different_endpoints(self):
        rows = [{"rank": 0, "model": "seed0", "method": method,
                 "wall_seconds": 1., "status": status}
                for method, status in zip(METHODS, ["SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"])]
        result = summarize(rows, 4)["models"]["seed0"]["paired"]["route_invariance"]
        self.assertEqual(result["safe"]["net_gain"], 1)
        self.assertEqual(result["solved"]["net_gain"], 0)

    def verify(self, model, method):
        config = _config()
        config["comparison_method"] = method
        return verify_staged_linf(model, torch.full((1, 2), .5, dtype=torch.float64), .1, config)

    def test_stable_weighted_route_still_requires_f0(self):
        model = _constant_model(((0., 1.), (3., 0.)))
        result = self.verify(model, "route_invariance")
        self.assertEqual(result.status, "SAFE")
        self.assertTrue(result.evidence["tier2"]["invoked"])
        limited = self.verify(model, "tier1_only")
        self.assertEqual(limited.status, "UNKNOWN")
        self.assertFalse(limited.evidence["tier2"]["invoked"])

    def test_legal_ties_reject_invariance_before_expert_solves(self):
        model = build_output_moe(OutputMoEFactoryConfig(
            input_shape=(2,), num_classes=2, num_experts=3, top_k=2,
            gate=GateKind.SELECTED_SOFTMAX,
            router_hidden=(), expert_hidden=(), seed=7,
        )).cpu().double().eval()
        with torch.no_grad():
            model.router[1].weight.zero_()
            model.router[1].bias.zero_()
            for expert in model.experts:
                expert[1].weight.zero_()
                expert[1].bias.copy_(torch.tensor([2., 0.]))
        with patch("act.pipeline.moe.experiment1c._solve_staged", side_effect=AssertionError("expert ran")):
            result = self.verify(model, "route_invariance")
        self.assertEqual(result.reason, "UNKNOWN_ROUTE_INVARIANCE")
        self.assertEqual(len(result.evidence["route_coverage"]["feasible_route_sets"]), 3)
        self.assertEqual(self.verify(model, "staged").status, "SAFE")

    def test_monolithic_matches_f0_control_and_audits(self):
        model = _constant_model(((0., 1.), (3., 0.)))
        result = self.verify(model, "monolithic_f0")
        self.assertEqual(result.status, "SAFE")
        self.assertEqual(result.evidence["tier1"]["branches"], [])
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as tmp:
            directory = Path(tmp) / "package"
            write_evidence_package(result, directory)
            audit = audit_evidence_package(directory)
            self.assertEqual(audit["status"], "PASS", audit)
        altered = copy.deepcopy(result.evidence)
        altered["tier2"]["property_rows"] = []
        issues = []
        _audit_safe_structure(altered, issues)
        self.assertTrue(issues)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(_constant_model(((2., 0.), (3., 0.))), "invalid")


if __name__ == "__main__":
    unittest.main()
