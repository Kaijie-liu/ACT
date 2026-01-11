#!/usr/bin/env python3
#===- tests/test_verification.py - Verification Consolidated Tests ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch
import torch.nn as nn

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InKind, OutKind, InputSpec, OutputSpec
from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer
from act.pipeline.verification.activations import ActivationEvent, collect_concrete_activations
from act.pipeline.verification.bounds_compare import compare_bounds_per_neuron
from act.pipeline.verification.bounds_core import LayerBounds
from act.pipeline.verification.layer_alignment import align_activations_to_act_layers
from act.pipeline.verification.torch2act import TorchToACT, _filter_layer_meta
from act.pipeline.verification.validate_verifier import VerificationValidator


class TestBoundsCompare:
    def test_compare_bounds_per_neuron_pass(self) -> None:
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        bounds = {
            0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
        }
        concrete = {0: torch.tensor([0.25, 0.75])}
        result = compare_bounds_per_neuron(
            bounds_by_layer=bounds,
            concrete_by_layer=concrete,
            atol=1e-6,
            rtol=0.0,
            topk=5,
        )
        assert result["status"] == "PASS"
        assert result["violations_total"] == 0

    def test_compare_bounds_per_neuron_fail(self) -> None:
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        bounds = {
            0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
        }
        concrete = {0: torch.tensor([1.5, -0.5])}
        result = compare_bounds_per_neuron(
            bounds_by_layer=bounds,
            concrete_by_layer=concrete,
            atol=1e-6,
            rtol=0.0,
            topk=5,
        )
        assert result["status"] == "FAIL"
        assert result["violations_total"] == 2
        assert result["violations_topk"]
        assert result["violations_topk"][0]["gap"] > 0.0

    def test_compare_bounds_per_neuron_error_on_shape(self) -> None:
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        bounds = {
            0: LayerBounds(layer_id=0, kind="DENSE", lb=lb, ub=ub, shape=(2,)),
        }
        concrete = {0: torch.tensor([0.5])}
        result = compare_bounds_per_neuron(
            bounds_by_layer=bounds,
            concrete_by_layer=concrete,
            atol=1e-6,
            rtol=0.0,
            topk=5,
        )
        assert result["status"] == "ERROR"
        assert result["violations_total"] == 0


class TestActivations:
    class _ReuseReLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.relu(x))

    def test_collect_concrete_activations_strict_multiple_calls(self) -> None:
        model = self._ReuseReLU()
        x = torch.tensor([[-1.0, 1.0]])
        events, errors, _warnings = collect_concrete_activations(
            model,
            x,
            strict_single_call_per_module=True,
        )
        assert len(events) == 2
        assert errors


class TestLayerAlignment:
    class _Layer:
        def __init__(self, layer_id: int, kind: str, output_shape):
            self.id = layer_id
            self.kind = kind
            self.meta = {"output_shape": output_shape}

    class _Net:
        def __init__(self, layers):
            self.layers = layers

    def test_alignment_ok(self) -> None:
        net = self._Net([
            self._Layer(0, "DENSE", (1, 2)),
            self._Layer(1, "RELU", (1, 2)),
        ])
        events = [
            ActivationEvent("l0", "Linear", 0, (1, 2), torch.zeros(1, 2)),
            ActivationEvent("l1", "ReLU", 0, (1, 2), torch.zeros(1, 2)),
        ]
        result = align_activations_to_act_layers(net, events)
        assert result.ok is True
        assert result.mapping[0].module_type == "Linear"

    def test_alignment_count_mismatch(self) -> None:
        net = self._Net([
            self._Layer(0, "DENSE", (1, 2)),
            self._Layer(1, "RELU", (1, 2)),
        ])
        events = [
            ActivationEvent("l0", "Linear", 0, (1, 2), torch.zeros(1, 2)),
        ]
        result = align_activations_to_act_layers(net, events)
        assert result.ok is False
        assert result.errors

    def test_alignment_kind_mismatch(self) -> None:
        net = self._Net([self._Layer(0, "DENSE", (1, 2))])
        events = [
            ActivationEvent("l0", "Conv2d", 0, (1, 2), torch.zeros(1, 2)),
        ]
        result = align_activations_to_act_layers(net, events)
        assert result.ok is False
        assert result.errors

    def test_alignment_shape_mismatch(self) -> None:
        net = self._Net([self._Layer(0, "DENSE", (1, 2))])
        events = [
            ActivationEvent("l0", "Linear", 0, (1, 3), torch.zeros(1, 3)),
        ]
        result = align_activations_to_act_layers(net, events)
        assert result.ok is False
        assert result.errors


class TestTorch2ActMeta:
    def _make_wrapped_conv(self) -> torch.nn.Sequential:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        labeled = LabeledInputTensor(x, label=0)
        input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
        input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
        input_spec_layer = InputSpecLayer(spec=input_spec)

        conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        relu = torch.nn.ReLU()
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(4, 2)

        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
        output_spec_layer = OutputSpecLayer(spec=output_spec)

        return torch.nn.Sequential(
            input_layer,
            input_spec_layer,
            conv,
            relu,
            flatten,
            linear,
            output_spec_layer,
        )

    def test_torch2act_filters_unknown_meta_keys(self) -> None:
        wrapped = self._make_wrapped_conv()
        TorchToACT(wrapped).run()

    def test_filter_layer_meta_keeps_required_keys(self) -> None:
        meta = {
            "input_shape": (1, 1, 4, 4),
            "output_shape": (1, 1, 2, 2),
            "torch_path": "0",
            "torch_type": "Conv2d",
            "flat_in_dim": 16,
            "flat_out_dim": 4,
            "logical_shape": (1, 1, 4, 4),
            "converter": "torch2act",
        }
        filtered = _filter_layer_meta("CONV2D", meta)
        assert filtered["input_shape"] == meta["input_shape"]
        assert filtered["output_shape"] == meta["output_shape"]
        assert "torch_path" not in filtered
        assert "converter" not in filtered

    def test_filter_layer_meta_unknown_kind_raises(self) -> None:
        try:
            _filter_layer_meta("UNKNOWN_KIND", {"input_shape": (1,), "output_shape": (1,)})
        except KeyError:
            return
        raise AssertionError("Expected KeyError for unknown kind")

    def test_conv2d_padding_same_rejected(self) -> None:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        labeled = LabeledInputTensor(x, label=0)
        input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
        input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
        input_spec_layer = InputSpecLayer(spec=input_spec)

        conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding="same", bias=True)
        relu = torch.nn.ReLU()
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(16, 2)

        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
        output_spec_layer = OutputSpecLayer(spec=output_spec)

        wrapped = torch.nn.Sequential(
            input_layer,
            input_spec_layer,
            conv,
            relu,
            flatten,
            linear,
            output_spec_layer,
        )

        try:
            TorchToACT(wrapped).run()
        except ValueError as e:
            assert "padding string not supported" in str(e)
            return
        raise AssertionError("Expected ValueError for padding='same'")

    def test_maxpool2d_meta_has_shapes(self) -> None:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        labeled = LabeledInputTensor(x, label=0)
        input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
        input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
        input_spec_layer = InputSpecLayer(spec=input_spec)

        pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(4, 2)

        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
        output_spec_layer = OutputSpecLayer(spec=output_spec)

        wrapped = torch.nn.Sequential(
            input_layer,
            input_spec_layer,
            pool,
            flatten,
            linear,
            output_spec_layer,
        )

        net = TorchToACT(wrapped).run()
        pools = [layer for layer in net.layers if layer.kind == "MAXPOOL2D"]
        assert len(pools) == 1
        assert "kernel_size" in pools[0].meta
        assert "output_size" in pools[0].meta
        assert "stride" in pools[0].meta

    def test_avgpool2d_meta_has_shapes(self) -> None:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        labeled = LabeledInputTensor(x, label=0)
        input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
        input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
        input_spec_layer = InputSpecLayer(spec=input_spec)

        pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(4, 2)

        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
        output_spec_layer = OutputSpecLayer(spec=output_spec)

        wrapped = torch.nn.Sequential(
            input_layer,
            input_spec_layer,
            pool,
            flatten,
            linear,
            output_spec_layer,
        )

        net = TorchToACT(wrapped).run()
        pools = [layer for layer in net.layers if layer.kind == "AVGPOOL2D"]
        assert len(pools) == 1
        assert "kernel_size" in pools[0].meta
        assert "output_size" in pools[0].meta
        assert "stride" in pools[0].meta

    def test_adaptiveavgpool2d_meta_has_shapes(self) -> None:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        labeled = LabeledInputTensor(x, label=0)
        input_layer = InputLayer(labeled, shape=(1, 1, 4, 4), dtype=torch.float64)
        input_spec = InputSpec(kind=InKind.BOX, lb=x.clone(), ub=(x + 1.0))
        input_spec_layer = InputSpecLayer(spec=input_spec)

        pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(4, 2)

        output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=0)
        output_spec_layer = OutputSpecLayer(spec=output_spec)

        wrapped = torch.nn.Sequential(
            input_layer,
            input_spec_layer,
            pool,
            flatten,
            linear,
            output_spec_layer,
        )

        net = TorchToACT(wrapped).run()
        pools = [layer for layer in net.layers if layer.kind == "AVGPOOL2D"]
        assert len(pools) == 1
        assert "kernel_size" in pools[0].meta
        assert "output_size" in pools[0].meta
        assert "stride" in pools[0].meta
        assert pools[0].meta.get("output_size") == (2, 2)


class TestBoundsPerNeuron:
    def test_validate_bounds_per_neuron_interval_e2e(self) -> None:
        validator = VerificationValidator(device="cpu", dtype=torch.float64)
        networks = validator.factory.list_networks()
        target = networks[0]
        if "control_conservative" in networks:
            target = "control_conservative"

        summary = validator.validate_bounds_per_neuron(
            networks=[target],
            tf_modes=["interval"],
            num_samples=1,
            atol=1e-6,
            rtol=0.0,
            topk=5,
            strict=True,
        )
        assert summary["total"] == 1

        results = [
            r for r in validator.validation_results
            if r.get("validation_type") == "bounds_per_neuron"
        ]
        assert len(results) == 1
        result = results[0]
        assert result["validation_status"] in ("PASSED", "FAILED", "ERROR")
        assert "layerwise_stats" in result
        assert "violations_topk" in result
        assert "alignment" in result

    def test_validate_bounds_per_neuron_hybridz_smoke(self) -> None:
        validator = VerificationValidator(device="cpu", dtype=torch.float64)
        networks = validator.factory.list_networks()
        preferred = [
            "mnist_robust_easy",
            "mnist_mlp_small",
            "control_conservative",
            "extra_noce_10_linearle",
        ]
        target = next((name for name in preferred if name in networks), networks[0])

        summary = validator.validate_bounds_per_neuron(
            networks=[target],
            tf_modes=["hybridz"],
            num_samples=1,
            atol=1e-6,
            rtol=0.0,
            topk=5,
            strict=True,
        )
        assert summary["total"] == 1
        results = [
            r for r in validator.validation_results
            if r.get("validation_type") == "bounds_per_neuron"
        ]
        assert len(results) == 1
        result = results[0]
        assert result["tf_mode"] == "hybridz"
        assert result["validation_status"] in ("PASSED", "FAILED", "ERROR")
