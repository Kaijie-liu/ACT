"""Tests for the Route A verifier-front-end specialization closure."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
from torch import nn

from act.pipeline.moe.verifier_specialization_probe import (
    audit_onnx_data_dispatch,
    specialize_official_expert,
)


class _ToyOfficial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Flatten(), nn.Sequential(nn.Flatten())])


class SpecializeOfficialExpertTest(unittest.TestCase):
    def test_specialized_branch_matches_explicit_normalized_expert(self) -> None:
        official = _ToyOfficial().eval()
        specialized = specialize_official_expert(official, 1)
        pixels = torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]])
        normalized = (pixels * 255.0 - specialized.mean) / specialized.std
        torch.testing.assert_close(specialized(pixels), official.experts[1](normalized))

    def test_rejects_invalid_layout_or_index(self) -> None:
        with self.assertRaises(TypeError):
            specialize_official_expert(nn.Linear(2, 2), 0)
        with self.assertRaises(IndexError):
            specialize_official_expert(_ToyOfficial(), 2)


class FrozenSpecializationProbeConfigTest(unittest.TestCase):
    def test_config_closes_dynamic_reject_to_specialized_acceptance(self) -> None:
        path = Path(
            "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/verifier_specialization_probe_r2.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
        self.assertEqual(config["checkpoint"]["experts"], [0, 1, 2, 3])
        self.assertEqual(
            config["required_outcome"]["dynamic_model"],
            "EXISTING_VERIFIER_CANNOT_CONSUME",
        )
        self.assertEqual(
            config["required_outcome"]["all_specialized_experts"],
            "EXISTING_VERIFIER_FRONTEND_ACCEPTS",
        )
        self.assertIn("GatherElements", config["required_outcome"]["forbidden_data_dispatch_operators"])
        self.assertIn("Shape", config["required_outcome"]["allowed_shape_bookkeeping"])
        self.assertIn("not a robustness certificate", config["claim_scope"])


class OnnxDispatchAuditTest(unittest.TestCase):
    @staticmethod
    def _save_graph(path: Path, nodes: list, outputs: list) -> None:
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            nodes,
            "dispatch-audit",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
            outputs,
        )
        onnx.save(helper.make_model(graph), str(path))

    def test_shape_gather_is_not_data_dispatch(self) -> None:
        import tempfile
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE/ACT/data/moe") as root:
            path = Path(root) / "shape.onnx"
            nodes = [
                helper.make_node("Shape", ["input"], ["shape"]),
                helper.make_node("Constant", [], ["index"], value=helper.make_tensor("i", TensorProto.INT64, [], [0])),
                helper.make_node("Gather", ["shape", "index"], ["dimension"], axis=0),
            ]
            self._save_graph(path, nodes, [helper.make_tensor_value_info("dimension", TensorProto.INT64, [])])
            audit = audit_onnx_data_dispatch(path)
        self.assertEqual(audit["data_dispatch_count"], 0)
        self.assertEqual(audit["shape_bookkeeping_count"], 1)

    def test_feature_gather_is_data_dispatch(self) -> None:
        import tempfile
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE/ACT/data/moe") as root:
            path = Path(root) / "feature.onnx"
            nodes = [
                helper.make_node("Constant", [], ["index"], value=helper.make_tensor("i", TensorProto.INT64, [], [0])),
                helper.make_node("Gather", ["input", "index"], ["selected"], axis=1),
            ]
            self._save_graph(path, nodes, [helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1])])
            audit = audit_onnx_data_dispatch(path)
        self.assertEqual(audit["data_dispatch_count"], 1)
        self.assertEqual(audit["shape_bookkeeping_count"], 0)


if __name__ == "__main__":
    unittest.main()
