#!/usr/bin/env python3
# act/pipeline/verification/config_net_runtime.py
"""
Runtime ConfigNet: build ACT Nets in memory (no JSON) and convert to torch.

Provides a sampling interface that constructs a spec dict compatible with
NetFactory.create_network(), seeds deterministically, and returns a
ConfigSample containing ACT Net, torch model, and specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from act.pipeline.verification.utils_seed_logging import set_global_seed
from act.pipeline.verification.config_net import ConfigSample
from act.pipeline.verification.act2torch import ACTToTorch
from act.back_end.net_factory import NetFactory
from act.back_end.verifier import gather_input_spec_layers, get_assert_layer


def _make_mlp_spec(
    input_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_classes: int,
    spec_kind: str,
    eps: float,
    assert_kind: str,
    y_true: int,
    margin: float,
) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    layers.append(
        {
            "kind": "INPUT",
            "params": {},
            "meta": {
                "shape": list(input_shape),
                "dtype": "torch.float64",
                "desc": "runtime-mlp-input",
            },
        }
    )
    layers.append(
        {
            "kind": "INPUT_SPEC",
            "params": {},
            "meta": {
                "kind": spec_kind,
                "center_val": 0.5,
                "eps": eps,
            },
        }
    )

    in_features = input_shape[-1]
    for idx, h in enumerate(hidden_sizes):
        layers.append(
            {
                "kind": "DENSE",
                "params": {},
                "meta": {
                    "in_features": in_features,
                    "out_features": h,
                    "bias_enabled": True,
                },
            }
        )
        layers.append({"kind": "RELU", "params": {}, "meta": {}})
        in_features = h

    layers.append(
        {
            "kind": "DENSE",
            "params": {},
            "meta": {
                "in_features": in_features,
                "out_features": num_classes,
                "bias_enabled": True,
            },
        }
    )

    assert_meta = {"kind": assert_kind}
    if assert_kind in ("TOP1_ROBUST", "MARGIN_ROBUST"):
        assert_meta["y_true"] = y_true
    if assert_kind == "MARGIN_ROBUST":
        assert_meta["margin"] = margin

    layers.append({"kind": "ASSERT", "params": {}, "meta": assert_meta})

    spec = {
        "description": "runtime-mlp",
        "architecture_type": "mlp",
        "input_shape": list(input_shape),
        "layers": layers,
    }
    return spec


def _make_cnn_spec(
    input_shape: Sequence[int],
    conv_channels: int,
    kernel_size: int,
    num_classes: int,
    spec_kind: str,
    eps: float,
    assert_kind: str,
    y_true: int,
    margin: float,
) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    layers.append(
        {
            "kind": "INPUT",
            "params": {},
            "meta": {
                "shape": list(input_shape),
                "dtype": "torch.float64",
                "desc": "runtime-cnn-input",
            },
        }
    )
    layers.append(
        {
            "kind": "INPUT_SPEC",
            "params": {},
            "meta": {
                "kind": spec_kind,
                "center_val": 0.5,
                "eps": eps,
            },
        }
    )

    batch, in_channels, h, w = input_shape
    conv_out_shape = [batch, conv_channels, h, w]
    layers.append(
        {
            "kind": "CONV2D",
            "params": {},  # weight auto-generated; bias set below after import
            "meta": {
                "in_channels": in_channels,
                "out_channels": conv_channels,
                "kernel_size": kernel_size,
                "stride": 1,
                "padding": kernel_size // 2,
                "input_shape": list(input_shape),
                "output_shape": conv_out_shape,
            },
        }
    )
    layers.append({"kind": "RELU", "params": {}, "meta": {}})
    layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})

    flat_features = conv_channels * h * w
    layers.append(
        {
            "kind": "DENSE",
            "params": {},
            "meta": {
                "in_features": flat_features,
                "out_features": num_classes,
                "bias_enabled": True,
            },
        }
    )

    assert_meta = {"kind": assert_kind}
    if assert_kind in ("TOP1_ROBUST", "MARGIN_ROBUST"):
        assert_meta["y_true"] = y_true
    if assert_kind == "MARGIN_ROBUST":
        assert_meta["margin"] = margin

    layers.append({"kind": "ASSERT", "params": {}, "meta": assert_meta})

    spec = {
        "description": "runtime-cnn",
        "architecture_type": "cnn",
        "input_shape": list(input_shape),
        "layers": layers,
    }
    return spec


class RuntimeConfigNet:
    """
    Runtime sampler that builds nets in memory (no JSON) using NetFactory.
    """

    def __init__(self):
        self.factory = NetFactory()

    def sample(
        self,
        seed: int = 0,
        arch: str = "mlp",
        spec_kind: str = "LINF_BALL",
        assert_kind: str = "TOP1_ROBUST",
        y_true: int = 0,
        margin: float = 0.1,
        name: Optional[str] = None,
    ) -> ConfigSample:
        """
        Build an ACT Net in memory, convert to torch, and return ConfigSample.

        Args:
            seed: global seed for deterministic weights.
            arch: "mlp" or "cnn".
            spec_kind: INPUT_SPEC kind ("LINF_BALL"/"BOX").
            assert_kind: ASSERT kind ("TOP1_ROBUST"/"MARGIN_ROBUST"/...).
            y_true: target class for robustness specs.
            margin: margin value for MARGIN_ROBUST.
            name: optional explicit name; default derived from arch/seed.
        """
        set_global_seed(seed)

        if name is None:
            name = f"runtime_{arch}_{seed}"
        elif not name.startswith("runtime_"):
            name = f"runtime_{name}"

        if arch == "mlp":
            spec = _make_mlp_spec(
                input_shape=[1, 784],
                hidden_sizes=[64],
                num_classes=10,
                spec_kind=spec_kind,
                eps=0.01,
                assert_kind=assert_kind,
                y_true=y_true,
                margin=margin,
            )
        elif arch == "cnn":
            spec = _make_cnn_spec(
                input_shape=[1, 3, 32, 32],
                conv_channels=8,
                kernel_size=3,
                num_classes=10,
                spec_kind=spec_kind,
                eps=0.03,
                assert_kind=assert_kind,
                y_true=y_true,
                margin=margin,
            )
        else:
            raise ValueError(f"Unsupported arch '{arch}'")

        act_net = self.factory.create_network(name, spec)
        torch_model = ACTToTorch(act_net).run()
        input_specs = gather_input_spec_layers(act_net)
        assert_layer = get_assert_layer(act_net)

        metadata = {
            "seed": seed,
            "source": "runtime",
            "arch": arch,
            "spec_kind": spec_kind,
            "assert_kind": assert_kind,
        }

        return ConfigSample(
            name=name,
            act_net=act_net,
            torch_model=torch_model,
            input_specs=input_specs,
            assert_layer=assert_layer,
            metadata=metadata,
        )
