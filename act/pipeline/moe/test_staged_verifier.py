import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from act.back_end.moe import (
    GateKind,
    OutputMoEFactoryConfig,
    build_output_moe,
)
from act.pipeline.moe.staged_verifier import (
    DEFAULT_CONFIG,
    _tensor_identity,
    verify_staged_linf,
    write_evidence_package,
)
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package


def _config():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _constant_model(expert_biases):
    model = build_output_moe(
        OutputMoEFactoryConfig(
            input_shape=(2,),
            num_classes=2,
            num_experts=2,
            top_k=2,
            gate=GateKind.SELECTED_SOFTMAX,
            router_hidden=(),
            expert_hidden=(),
            seed=7,
        )
    ).to(device=torch.device("cpu"), dtype=torch.float64)
    with torch.no_grad():
        router = model.router[1]
        router.weight.zero_()
        router.bias.zero_()
        for expert, bias in zip(model.experts, expert_biases):
            expert[1].weight.zero_()
            expert[1].bias.copy_(torch.tensor(bias, dtype=torch.float64))
    return model.eval()


class StagedVerifierTests(unittest.TestCase):
    def test_tier1_safe_skips_experiment_controls_and_f0(self):
        model = _constant_model(((2.0, 0.0), (3.0, 0.0)))
        report = verify_staged_linf(
            model, torch.zeros(1, 2, dtype=torch.float64), 0.1, _config()
        )
        self.assertEqual(report.status, "SAFE")
        self.assertEqual(report.reason, "SAFE_GATE_ELIMINATION")
        self.assertFalse(report.evidence["tier2"]["invoked"])
        self.assertFalse(
            report.evidence["algorithm"]["matched_no_support_ablation_executed"]
        )
        self.assertFalse(
            report.evidence["algorithm"][
                "unguarded_accounting_propagation_executed"
            ]
        )
        for branch in report.evidence["tier1"]["branches"]:
            self.assertIsNone(branch["matched_no_support_status"])
            self.assertIsNone(branch["guard_accounting"])

    def test_f0_is_required_and_proves_weighted_safe(self):
        model = _constant_model(((0.0, 1.0), (3.0, 0.0)))
        report = verify_staged_linf(
            model, torch.zeros(1, 2, dtype=torch.float64), 0.1, _config()
        )
        self.assertEqual(report.status, "SAFE")
        self.assertEqual(report.reason, "SAFE_WEIGHTED_RANGE")
        self.assertTrue(report.evidence["tier2"]["invoked"])
        self.assertEqual(
            report.evidence["verdict"]["decision_tier"], "TIER2_F0"
        )
        self.assertEqual(
            [row["stage"] for row in report.evidence["transitions"]],
            ["REQUEST_ACCEPTED", "TIER1_COMPLETE", "TIER2_F0_COMPLETE"],
        )
        self.assertTrue(report.evidence["route_coverage"]["coverage_complete"])
        self.assertTrue(report.evidence["route_coverage"]["route_sets_exact"])

    def test_evidence_identity_and_refuse_overwrite(self):
        model = _constant_model(((2.0, 0.0), (3.0, 0.0)))
        center = torch.zeros(1, 2, dtype=torch.float64)
        report = verify_staged_linf(model, center, 0.1, _config())
        result_root = Path("/data1/Kane/MOE/ACT/data/moe/results")
        with tempfile.TemporaryDirectory(dir=result_root) as temporary:
            output = Path(temporary) / "evidence"
            manifest = write_evidence_package(report, output)
            self.assertTrue((output / "evidence.json").is_file())
            self.assertTrue((output / "request.pt").is_file())
            self.assertEqual(manifest["request_id"], report.evidence["request_id"])
            audit = audit_evidence_package(output)
            self.assertEqual(audit["status"], "PASS", audit["issues"])
            with self.assertRaises(RuntimeError):
                write_evidence_package(report, output)

    def test_auditor_accepts_f0_and_rejects_tampered_evidence(self):
        model = _constant_model(((0.0, 1.0), (3.0, 0.0)))
        report = verify_staged_linf(
            model, torch.zeros(1, 2, dtype=torch.float64), 0.1, _config()
        )
        result_root = Path("/data1/Kane/MOE/ACT/data/moe/results")
        with tempfile.TemporaryDirectory(dir=result_root) as temporary:
            output = Path(temporary) / "f0-evidence"
            write_evidence_package(report, output)
            audit = audit_evidence_package(output)
            self.assertEqual(audit["status"], "PASS", audit["issues"])

            evidence_path = output / "evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["verdict"]["certificate_complete"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evidence_sha256"] = hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit = audit_evidence_package(output)
            self.assertEqual(audit["status"], "FAIL")
            self.assertIn("SAFE is not marked complete", audit["issues"])

    def test_tensor_identity_changes_with_value(self):
        left = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        self.assertNotEqual(_tensor_identity(left)["sha256"], _tensor_identity(right)["sha256"])

    def test_numerical_policy_mismatch_is_rejected(self):
        config = _config()
        config["numerical_safety"]["safe_positive_margin"] = 0.0
        model = _constant_model(((2.0, 0.0), (3.0, 0.0)))
        with self.assertRaises(ValueError):
            verify_staged_linf(
                model, torch.zeros(1, 2, dtype=torch.float64), 0.1, config
            )

    def test_f0_acceptance_tolerance_cannot_drift(self):
        config = _config()
        config["f0"]["solver"]["safety_tolerance"] = 1e-8
        model = _constant_model(((2.0, 0.0), (3.0, 0.0)))
        with self.assertRaises(ValueError):
            verify_staged_linf(
                model, torch.zeros(1, 2, dtype=torch.float64), 0.1, config
            )

    def test_model_must_use_eval_cpu_float64_contract(self):
        center = torch.zeros(1, 2, dtype=torch.float64)
        model = _constant_model(((2.0, 0.0), (3.0, 0.0))).train()
        with self.assertRaises(ValueError):
            verify_staged_linf(model, center, 0.1, _config())

        model.eval().float()
        with self.assertRaises(ValueError):
            verify_staged_linf(model, center, 0.1, _config())


if __name__ == "__main__":
    unittest.main()
