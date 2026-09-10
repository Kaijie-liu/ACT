import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.pipeline.moe.test_staged_verifier import _config, _constant_model
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package, build_weighted_top2_f0
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package, _audit_safe_structure
from act.pipeline.moe.scoped_f0_proofs import facts_from_branches, make_scope, reuse_property
from act.pipeline.moe.experiment1c import diagnose_radius


def model():
    value = build_output_moe(OutputMoEFactoryConfig(
        input_shape=(2,), num_classes=3, num_experts=2, top_k=2,
        gate=GateKind.SELECTED_SOFTMAX, router_hidden=(), expert_hidden=(), seed=7,
    )).cpu().double().eval()
    with torch.no_grad():
        value.router[1].weight.zero_()
        value.router[1].bias.zero_()
        for expert, bias in zip(value.experts, ((0., 1., -2.), (3., 0., -2.))):
            expert[1].weight.zero_()
            expert[1].bias.copy_(torch.tensor(bias))
    return value


class ScopedProofTests(unittest.TestCase):
    def test_all_properties_reused_skips_pair_propagation(self):
        # Force the fallback path on an otherwise Tier-1-safe analytic control
        # to test this branch without inventing solver proof facts.
        def force_fallback(*args, **kwargs):
            result = diagnose_radius(*args, **kwargs)
            result.update(status="UNKNOWN", reason="UNKNOWN_GATE_SUFFICIENCY")
            return result
        config = _config()
        config["scoped_proof_reuse"] = True
        with patch("act.pipeline.moe.staged_verifier.diagnose_radius", side_effect=force_fallback), patch(
            "act.pipeline.moe.staged_verifier.shared_input_pair_propagation",
            side_effect=AssertionError("unnecessary pair propagation")):
            report = verify_staged_linf(_constant_model(((2., 0.), (3., 0.))),
                                       torch.full((1, 2), .5, dtype=torch.float64), .1, config)
        self.assertEqual(report.status, "SAFE")
        self.assertTrue(report.evidence["tier2"]["pairs"][0]["all_properties_reused"])
        issues = []
        _audit_safe_structure(report.evidence, issues)
        self.assertEqual(issues, [])

    def verify(self, enabled):
        config = _config()
        config["scoped_proof_reuse"] = enabled
        with patch("act.pipeline.moe.staged_verifier.build_weighted_top2_f0",
                   wraps=build_weighted_top2_f0) as build:
            report = verify_staged_linf(model(), torch.full((1, 2), .5, dtype=torch.float64), .1, config)
        return report, build.call_count

    def test_partial_property_reuse_reduces_calls_without_changing_verdict(self):
        before, before_calls = self.verify(False)
        after, after_calls = self.verify(True)
        self.assertEqual((before.status, after.status), ("SAFE", "SAFE"))
        self.assertEqual((before_calls, after_calls), (2, 1))
        self.assertEqual(after.evidence["tier2"]["reused_property_count"], 1)
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as tmp:
            package = Path(tmp) / "package"
            write_evidence_package(after, package)
            result = audit_evidence_package(package)
            self.assertEqual(result["issues"], [], result)

    def test_scope_and_property_mutations_rejected(self):
        report, _ = self.verify(True)
        evidence = report.evidence
        scope = make_scope(evidence["request_id"], evidence["identity"],
                           evidence["proof_reuse"]["frame_id"], evidence["numerical_safety"])
        facts = facts_from_branches(evidence["tier1"]["branches"], scope)
        self.assertIsNone(reuse_property(facts, scope, (0, 1), 0))
        self.assertIsNotNone(reuse_property(facts, scope, (0, 1), 1))
        for key in ("frame_id", "request_id", "model_state", "lower", "property", "numerical_policy"):
            changed = copy.deepcopy(scope)
            changed[key] = "mutated"
            with self.assertRaises(ValueError):
                reuse_property(facts, changed, (0, 1), 1)
        del facts[(1, 1)]
        self.assertIsNone(reuse_property(facts, scope, (0, 1), 1))

    def test_audit_rejects_wrong_source_and_bound(self):
        report, _ = self.verify(True)
        for mutation in ("bound", "expert", "scope"):
            evidence = copy.deepcopy(report.evidence)
            row = evidence["tier2"]["pairs"][0]["property_rows"][1]
            if mutation == "bound":
                row["accepted_minimum"] += 1
            elif mutation == "expert":
                row["proof_sources"][0]["expert"] = 1
            else:
                row["proof_sources"][0]["scope"]["frame_id"] = "other"
            issues = []
            _audit_safe_structure(evidence, issues)
            self.assertTrue(issues, mutation)


if __name__ == "__main__":
    unittest.main()
