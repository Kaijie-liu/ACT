"""Complete-verifier controls for an explicitly labeled relation relaxation."""
import copy
import json
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from act.pipeline.moe.test_route_complexity_schedule import config, model
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package


class RelationAblationTests(unittest.TestCase):
    def test_frozen_runner_and_counterbalanced_schedule(self):
        from act.pipeline.moe.relation_ablation import DEFAULT, artifacts, jobs
        frozen = json.loads(DEFAULT.read_text())
        selection, configs = artifacts(frozen)
        self.assertEqual(len(jobs(selection, False)), 60)
        self.assertEqual(len(jobs(selection, True)), 6)
        for name in selection["models"]:
            for arm in configs:
                self.assertEqual(sum(j["model"] == name and j["method"] == arm and j["position"] == 0
                                     for j in jobs(selection, False)), 5)
        for field, value in (("sample_count", 11), ("budget_seconds", 301), ("selection_sha256", "bad")):
            changed = copy.deepcopy(frozen); changed[field] = value
            with self.assertRaises(ValueError): artifacts(changed)
        changed = copy.deepcopy(frozen); changed["methods"]["independent"] = changed["methods"]["shared"]
        with self.assertRaises(ValueError): artifacts(changed)

    def test_shared_safe_independent_unknown_in_both_f0_paths(self):
        net = model(((0., 0.), (0., 0.)))
        with torch.no_grad():
            net.experts[0][1].weight[0, 0] = 1.
            net.experts[0][1].bias[0] = -.3
            net.experts[1][1].weight[0, 0] = -1.
            net.experts[1][1].bias[0] = .7
        for scheduled in (False, True):
            for relation in ("shared_input", "independent_inputs"):
                cfg = config()
                if not scheduled:
                    cfg.pop("route_complexity_schedule")
                cfg["f0"]["expert_relation"] = relation
                report = verify_staged_linf(net, torch.full((1, 2), .5, dtype=torch.float64), .5, cfg)
                self.assertEqual(report.status, "SAFE" if relation == "shared_input" else "UNKNOWN")
                self.assertIsNone(report.witness)
                with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as root:
                    p = Path(root)/"package"
                    write_evidence_package(report, p)
                    self.assertEqual(audit_evidence_package(p)["issues"], [])
                    wrong = copy.deepcopy(report.evidence)
                    wrong["algorithm"]["f0_expert_relation"] = "other"
                    q = Path(root)/"tampered"
                    write_evidence_package(replace(report, evidence=wrong), q)
                    self.assertTrue(audit_evidence_package(q)["issues"])

    def test_all_tie_legal_pairs_and_reuse_unchanged(self):
        net = model(((-.2, 0.), (1., 0.), (2., 0.)))
        for method in ("staged", "monolithic_f0"):
            cfg = config(method); cfg["f0"]["expert_relation"] = "independent_inputs"
            report = verify_staged_linf(net, torch.full((1, 2), .5, dtype=torch.float64), .1, cfg)
            self.assertEqual(report.status, "SAFE")
            self.assertEqual(report.evidence["route_coverage"]["feasible_route_sets"], [[0,1], [0,2], [1,2]])
            with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as root:
                p = Path(root)/"package"
                write_evidence_package(report, p)
                self.assertEqual(audit_evidence_package(p)["issues"], [])

    def test_unknown_relation_rejected_before_solving(self):
        cfg = config(); cfg["f0"]["expert_relation"] = "typo"
        with self.assertRaisesRegex(ValueError, "relation"):
            verify_staged_linf(model(((1.,0.), (1.,0.))), torch.zeros(1,2), .1, cfg)


if __name__ == "__main__": unittest.main()
