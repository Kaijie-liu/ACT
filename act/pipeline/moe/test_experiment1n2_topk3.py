# ===- act/pipeline/moe/test_experiment1n2_topk3.py - N2 Runner Tests ===#

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from act.back_end.moe import (
    GateKind,
    OutputLevelMoE,
    OutputLevelMoESpec,
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    UNSAFE_FULL_FORWARD_FALLBACK,
    build_act_moe_program,
    linear_safety_rows,
)
from act.back_end.solver.solver_hz import SparseHZono
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import PROJECT_ROOT
from act.pipeline.moe.experiment1 import _propagate_component
from act.pipeline.moe.experiment1n2_topk3 import (
    DEFAULT_CONFIG,
    aggregate_property_rows,
    aggregate_route_sets,
    independently_audit,
    select_clean_correct_indices,
    validate_checkpoint_contract,
    verify_route_set,
)


def _config():
    with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


def _matching_payload(config):
    model = config["model"]
    training = config["training"]
    return {
        "dataset": model["dataset"],
        "factory_config": {
            "input_shape": model["input_shape"],
            "num_classes": model["num_classes"],
            "num_experts": model["num_experts"],
            "top_k": model["top_k"],
            "gate": model["gate"],
            "router_hidden": model["router_hidden"],
            "expert_hidden": model["expert_hidden"],
            "seed": training["seed"],
        },
        "training_config": {
            "balance_loss": training["balance_loss"],
            "balance_coefficient": training["balance_coefficient"],
            "validation_fraction": training["validation_fraction"],
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "seed": training["seed"],
        },
    }


class _IndexClassifier(torch.nn.Module):
    def forward(self, images):
        indices = images[:, 0, 0, 0].long()
        output = torch.zeros(images.shape[0], 3, device=images.device)
        output.scatter_(1, (indices % 3).unsqueeze(1), 1.0)
        return output


def _constant_linear(in_features, values):
    layer = torch.nn.Linear(in_features, len(values))
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor(values, dtype=layer.bias.dtype))
    return torch.nn.Sequential(torch.nn.Flatten(start_dim=1), layer)


class Experiment1N2TopK3Tests(unittest.TestCase):
    def test_frozen_config_is_exact_normalized_sigmoid_top3_target(self):
        config = _config()
        self.assertEqual(config["model"]["dataset"], "CIFAR10")
        self.assertEqual(config["model"]["num_experts"], 8)
        self.assertEqual(config["model"]["top_k"], 3)
        self.assertEqual(config["model"]["gate"], "normalized_sigmoid")
        self.assertEqual(config["model"]["router_hidden"], [128])
        self.assertEqual(config["model"]["expert_hidden"], [256, 128])
        self.assertEqual(config["training"]["balance_loss"], "switch")
        self.assertEqual(config["training"]["balance_coefficient"], 0.1)
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertEqual(config["training"]["batch_size"], 256)
        self.assertEqual(config["training"]["learning_rate"], 1e-3)
        self.assertEqual(config["training"]["weight_decay"], 1e-4)
        self.assertEqual(config["training"]["seed"], 0)
        self.assertEqual(config["expected"]["products_per_property"], 2)
        self.assertEqual(len(config["checkpoint_sha256"]), 64)
        for field in ("checkpoint", "dataset_root", "output_dir"):
            self.assertTrue(
                Path(config[field]).resolve().is_relative_to(
                    Path("/data1/Kane/MOE").resolve()
                )
            )

    def test_checkpoint_contract_accepts_complete_frozen_provenance(self):
        config = _config()
        report = validate_checkpoint_contract(_matching_payload(config), config)
        self.assertTrue(report["matched"])

    def test_checkpoint_contract_rejects_legacy_payload_missing_epochs(self):
        config = _config()
        payload = _matching_payload(config)
        del payload["training_config"]["epochs"]
        with self.assertRaisesRegex(RuntimeError, "training_config.epochs"):
            validate_checkpoint_contract(payload, config)

    def test_checkpoint_contract_rejects_top2_or_selected_softmax(self):
        config = _config()
        payload = _matching_payload(config)
        payload["factory_config"]["top_k"] = 2
        payload["factory_config"]["gate"] = "selected_softmax"
        with self.assertRaisesRegex(RuntimeError, "factory_config.top_k"):
            validate_checkpoint_contract(payload, config)

    def test_clean_correct_cohort_uses_deterministic_global_ranks(self):
        images = torch.zeros(8, 1, 1, 1)
        images[:, 0, 0, 0] = torch.arange(8)
        labels = torch.tensor([0, 0, 2, 0, 1, 2, 0, 1])
        dataset = TensorDataset(images, labels)
        selected = select_clean_correct_indices(
            _IndexClassifier(),
            dataset,
            (0, 2, 4),
            device=torch.device("cpu"),
            batch_size=3,
        )
        self.assertEqual(
            selected,
            [
                {"sample_rank": 0, "dataset_index": 0},
                {"sample_rank": 2, "dataset_index": 3},
                {"sample_rank": 4, "dataset_index": 5},
            ],
        )

    def test_property_aggregation_requires_every_row_safe(self):
        status, reason = aggregate_property_rows(
            [
                {"status": "SAFE", "reason": SAFE_WEIGHTED_RANGE},
                {"status": "SAFE", "reason": SAFE_WEIGHTED_RANGE},
            ]
        )
        self.assertEqual((status, reason), ("SAFE", SAFE_WEIGHTED_RANGE))
        status, reason = aggregate_property_rows(
            [
                {"status": "SAFE", "reason": SAFE_WEIGHTED_RANGE},
                {"status": "UNKNOWN", "reason": UNKNOWN_WEIGHTED_RELAXATION},
            ]
        )
        self.assertEqual(
            (status, reason), ("UNKNOWN", UNKNOWN_WEIGHTED_RELAXATION)
        )

    def test_relaxation_negative_or_unvalidated_unsafe_is_never_unsafe(self):
        status, reason = aggregate_property_rows(
            [
                {
                    "status": "UNSAFE",
                    "reason": UNKNOWN_WEIGHTED_RELAXATION,
                    "full_model_witness_valid": False,
                }
            ]
        )
        self.assertEqual((status, reason), ("UNKNOWN", UNKNOWN_WEIGHTED_RELAXATION))
        status, reason = aggregate_route_sets(
            [
                {
                    "status": "UNSAFE",
                    "reason": UNKNOWN_WEIGHTED_RELAXATION,
                    "full_model_witness_valid": False,
                }
            ],
            enumeration_exact=True,
        )
        self.assertEqual((status, reason), ("UNKNOWN", UNKNOWN_WEIGHTED_RELAXATION))

    def test_full_forward_replay_is_the_only_unsafe_transition(self):
        validated = {
            "status": "UNSAFE",
            "reason": UNSAFE_FULL_FORWARD_FALLBACK,
            "full_model_witness_valid": True,
        }
        self.assertEqual(
            aggregate_property_rows([validated]),
            ("UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK),
        )
        self.assertEqual(
            aggregate_route_sets([validated], enumeration_exact=True),
            ("UNSAFE", UNSAFE_FULL_FORWARD_FALLBACK),
        )

    def test_incomplete_tie_inclusive_route_enumeration_blocks_safe(self):
        status, reason = aggregate_route_sets(
            [{"status": "SAFE", "reason": SAFE_WEIGHTED_RANGE}],
            enumeration_exact=False,
        )
        self.assertEqual((status, reason), ("UNKNOWN", UNKNOWN_WEIGHTED_SOLVER_LIMIT))

    def test_empty_route_set_result_is_unknown_not_vacuous_safe(self):
        self.assertEqual(
            aggregate_route_sets([], enumeration_exact=True),
            ("UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL),
        )

    def test_toy_top3_route_uses_generic_two_product_data_path(self):
        model = OutputLevelMoE(
            _constant_linear(1, [0.5, 0.0, -0.5]),
            (
                _constant_linear(1, [2.0, 0.0]),
                _constant_linear(1, [0.0, 0.2]),
                _constant_linear(1, [2.0, 0.0]),
            ),
            OutputLevelMoESpec(
                num_experts=3,
                top_k=3,
                gate=GateKind.NORMALIZED_SIGMOID,
                normalized=True,
            ),
        ).double().eval()
        clean = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        lower = torch.full_like(clean, -0.1)
        upper = torch.full_like(clean, 0.1)
        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[0])
        program = build_act_moe_program(
            model,
            center=clean,
            lower=lower,
            upper=upper,
            output_spec=output_spec,
        )
        router = _propagate_component(program.router)
        self.assertIsInstance(router.output_hz, SparseHZono)
        properties = linear_safety_rows(output_spec, 2)
        config = {
            "support": {
                "enabled": False,
                "max_input_dim": 1024,
                "lp_neurons": 0,
                "milp_neurons": 0,
                "lp_time_limit": 0.0,
                "milp_time_limit": 0.0,
            },
            "solver": {
                "gate_elimination_seconds": 2.0,
                "gate_support_seconds": 2.0,
                "difference_support_seconds": 2.0,
                "property_seconds": 2.0,
                "safety_tolerance": 1e-7,
            },
        }
        with tempfile.TemporaryDirectory(
            dir=str(PROJECT_ROOT / "data/moe")
        ) as temporary:
            row = verify_route_set(
                model=model,
                program=program,
                router=router,
                route_set=(0, 1, 2),
                properties=properties,
                clean=clean,
                lower=lower,
                upper=upper,
                clean_prediction=0,
                sample_rank=0,
                output_dir=Path(temporary),
                config=config,
            )
        self.assertEqual(row["status"], "SAFE")
        self.assertEqual(row["gate_family"], "normalized_sigmoid")
        self.assertEqual(row["expected_property_products"], 2)
        self.assertTrue(row["fallback_invoked"])
        self.assertEqual(row["resolved_stage"], "weighted_topk_range")
        self.assertTrue(row["property_rows"])
        self.assertTrue(
            all(item["product_count"] == 2 for item in row["property_rows"])
        )
        self.assertFalse(row["full_model_witness_valid"])

    def test_independent_audit_rejects_unreplayed_unsafe(self):
        config = _config()
        config["cohort"]["clean_correct_ranks"] = [0]
        row = {
            "sample_rank": 0,
            "route_semantics": "ANY_LEGAL_TOPK_UNORDERED_TIE_INCLUSIVE",
            "gate_family": "normalized_sigmoid",
            "products_per_property": 2,
            "status": "UNSAFE",
            "full_model_witness_valid": False,
            "route_set_enumeration_exact": True,
            "exact_feasible_unordered_top3_set_count": 1,
            "route_sets": [],
            "reason": UNSAFE_FULL_FORWARD_FALLBACK,
        }
        with tempfile.TemporaryDirectory(
            dir=str(PROJECT_ROOT / "data/moe")
        ) as temporary:
            root = Path(temporary)
            (root / "results.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            (root / "sample_indices.json").write_text(
                json.dumps({"rows": [{"sample_rank": 0, "dataset_index": 1}]}),
                encoding="utf-8",
            )
            audit = independently_audit(root, config)
        self.assertFalse(audit["passed"])
        self.assertIn("unreplayed_unsafe:0", audit["issues"])


if __name__ == "__main__":
    unittest.main()
