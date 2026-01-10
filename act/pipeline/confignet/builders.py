#!/usr/bin/env python3
#===- act/pipeline/confignet/builders.py - ConfigNet Builders -----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Build wrapped PyTorch models from InstanceSpec:
#     build_wrapped_model(instance_spec, device, dtype) -> nn.Module
#
# Notes:
#   - Depends on act.front_end.verifiable_model wrapper layers.
#   - Does NOT directly import ACT back_end. (Wrapper may validate schema internally.)
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)

from .schema import (
    ModelFamily,
    MLPConfig,
    CNN2DConfig,
    TemplateConfig,
    InputSpecConfig,
    OutputSpecConfig,
    InstanceSpec,
    GeneratedInstance,
)
from .seeds import seeded, derive_seed

logger = logging.getLogger(__name__)


# --------------------------
# Helpers
# --------------------------

def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


def _ensure_batch1(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) < 2:
        raise ValueError(f"shape must include batch dim, got {shape}")
    if shape[0] != 1:
        raise ValueError(f"ConfigNet assumes batch=1, got shape={shape}")
    return tuple(int(x) for x in shape)


def _make_labeled_input(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    label: int,
    value_range: Tuple[float, float],
) -> LabeledInputTensor:
    """
    Create a reproducible labeled input tensor for InputLayer.

    This tensor is a *single sample* (batch=1), consistent with the wrapper assumption.
    """
    lo, hi = float(value_range[0]), float(value_range[1])
    x = torch.empty(*shape, device=device, dtype=dtype).uniform_(lo, hi)
    return LabeledInputTensor(tensor=x, label=int(label))


def _build_input_spec_tensor_fields(
    cfg: InputSpecConfig,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Build tensor fields for InputSpec based on InputSpecConfig.

    Returns a dict containing keys among: lb, ub, center, A, b, eps.
    """
    kind = cfg.kind
    shape = _ensure_batch1(input_shape)

    fields: Dict[str, Any] = {}

    if kind == InKind.BOX:
        if cfg.lb is not None and cfg.ub is not None:
            lb = cfg.lb.to(device=device, dtype=dtype).reshape(shape)
            ub = cfg.ub.to(device=device, dtype=dtype).reshape(shape)
        else:
            lb_val = float(cfg.lb_val if cfg.lb_val is not None else cfg.value_range[0])
            ub_val = float(cfg.ub_val if cfg.ub_val is not None else cfg.value_range[1])
            lb = torch.full(shape, lb_val, device=device, dtype=dtype)
            ub = torch.full(shape, ub_val, device=device, dtype=dtype)

        # InputSpecLayer expects lb/ub as 1D and reshapes to x.shape internally
        fields["lb"] = lb.reshape(-1)
        fields["ub"] = ub.reshape(-1)
        return fields

    if kind == InKind.LINF_BALL:
        eps = float(cfg.eps if cfg.eps is not None else 0.0)
        if cfg.center is not None:
            center = cfg.center.to(device=device, dtype=dtype).reshape(shape)
        else:
            center_val = float(cfg.center_val if cfg.center_val is not None else sum(cfg.value_range) / 2.0)
            center = torch.full(shape, center_val, device=device, dtype=dtype)

        fields["center"] = center
        fields["eps"] = eps
        return fields

    if kind == InKind.LIN_POLY:
        # If user provided A/b, trust them
        if cfg.A is not None and cfg.b is not None:
            fields["A"] = cfg.A.to(device=device, dtype=dtype)
            fields["b"] = cfg.b.to(device=device, dtype=dtype)
            return fields

        # Derive a simple polytope from box bounds if requested
        if cfg.derive_poly_from_box:
            lb_val = float(cfg.lb_val if cfg.lb_val is not None else cfg.value_range[0])
            ub_val = float(cfg.ub_val if cfg.ub_val is not None else cfg.value_range[1])

            n = _prod(shape[1:])  # exclude batch
            I = torch.eye(n, device=device, dtype=dtype)
            A = torch.cat([I, -I], dim=0)
            b = torch.cat(
                [
                    torch.full((n,), ub_val, device=device, dtype=dtype),
                    torch.full((n,), -lb_val, device=device, dtype=dtype),
                ],
                dim=0,
            )
            fields["A"] = A
            fields["b"] = b
            return fields

        # Fallback: empty constraints
        n = _prod(shape[1:])
        fields["A"] = torch.zeros((0, n), device=device, dtype=dtype)
        fields["b"] = torch.zeros((0,), device=device, dtype=dtype)
        return fields

    raise ValueError(f"Unsupported input spec kind: {kind}")


def _build_output_spec_tensor_fields(
    cfg: OutputSpecConfig,
    num_classes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Build tensor fields for OutputSpec based on OutputSpecConfig.
    """
    kind = cfg.kind
    fields: Dict[str, Any] = {}

    if kind == OutKind.TOP1_ROBUST:
        fields["y_true"] = int(cfg.y_true if cfg.y_true is not None else 0)
        return fields

    if kind == OutKind.MARGIN_ROBUST:
        fields["y_true"] = int(cfg.y_true if cfg.y_true is not None else 0)
        fields["margin"] = float(cfg.margin)
        return fields

    if kind == OutKind.LINEAR_LE:
        # c^T y <= d
        if cfg.c is None:
            c = torch.ones((num_classes,), device=device, dtype=dtype)
        else:
            c = cfg.c.to(device=device, dtype=dtype).reshape(-1)
        d = float(cfg.d if cfg.d is not None else 0.0)
        fields["c"] = c
        fields["d"] = d
        return fields

    if kind == OutKind.RANGE:
        if cfg.lb is None or cfg.ub is None:
            lb = torch.full((num_classes,), -1e9, device=device, dtype=dtype)
            ub = torch.full((num_classes,), +1e9, device=device, dtype=dtype)
        else:
            lb = cfg.lb.to(device=device, dtype=dtype).reshape(-1)
            ub = cfg.ub.to(device=device, dtype=dtype).reshape(-1)
        fields["lb"] = lb
        fields["ub"] = ub
        return fields

    # Fallback: TOP1
    fields["y_true"] = int(cfg.y_true if cfg.y_true is not None else 0)
    return fields


def _as_block_param(v: Union[int, Tuple[int, ...]], i: int, n_blocks: int, name: str) -> int:
    """
    Resolve per-block parameter.
      - if v is int: use it for all blocks
      - if v is tuple:
          * len==1: broadcast
          * len==n_blocks: index by i
          * otherwise: error
    """
    if isinstance(v, int):
        return int(v)

    t = tuple(int(x) for x in v)
    if len(t) == 1:
        return int(t[0])
    if len(t) == n_blocks:
        return int(t[i])
    raise ValueError(f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}")


# --------------------------
# Body builders
# --------------------------

def _build_mlp_body(cfg: MLPConfig) -> nn.Module:
    layers: list[nn.Module] = []

    input_shape = _ensure_batch1(cfg.input_shape)

    # If input is image-like, include Flatten(1)
    if len(input_shape) > 2:
        layers.append(nn.Flatten(start_dim=1))
        in_features = _prod(input_shape[1:])
    else:
        in_features = int(input_shape[1])

    prev = in_features
    for h in cfg.hidden_sizes:
        layers.append(nn.Linear(prev, int(h), bias=bool(cfg.use_bias)))
        layers.append(_make_activation(cfg.activation))
        if cfg.dropout_p and cfg.dropout_p > 0.0:
            layers.append(nn.Dropout(p=float(cfg.dropout_p)))
        prev = int(h)

    layers.append(nn.Linear(prev, int(cfg.num_classes), bias=True))
    return nn.Sequential(*layers)


def _extract_body_layers(model: nn.Module) -> nn.Sequential:
    """
    Strip InputSpec/OutputSpec layers from a VerifiableModel/sequential body.
    """
    from act.front_end.verifiable_model import InputSpecLayer, OutputSpecLayer
    if isinstance(model, nn.Sequential):
        layers = list(model)
    else:
        layers = list(model.children())

    body_layers = [
        layer for layer in layers
        if not isinstance(layer, (InputSpecLayer, OutputSpecLayer))
    ]
    if not body_layers:
        raise ValueError("Template model produced no executable body layers")
    return nn.Sequential(*body_layers)


def _infer_input_shape_from_act_net(act_net: Any) -> Tuple[int, ...]:
    for layer in getattr(act_net, "layers", []):
        if getattr(layer, "kind", None) == "INPUT":
            meta = getattr(layer, "meta", {})
            shape = meta.get("shape", None)
            if shape is not None:
                return tuple(int(x) for x in shape)
    raise ValueError("ACT Net missing INPUT layer shape metadata")


def _infer_num_classes_from_act_net(act_net: Any) -> Optional[int]:
    for layer in getattr(act_net, "layers", []):
        if getattr(layer, "kind", None) == "INPUT":
            meta = getattr(layer, "meta", {})
            num_classes = meta.get("num_classes", None)
            if num_classes is not None:
                return int(num_classes)
    return None


def _infer_conv2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    # PyTorch conv2d output: floor((H + 2p - d*(k-1) - 1)/s + 1)
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)

    return out_dim(H), out_dim(W)


def _infer_maxpool2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    # same formula
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)

    return out_dim(H), out_dim(W)


def _build_cnn2d_body(cfg: CNN2DConfig) -> nn.Module:
    layers: list[nn.Module] = []

    shape = _ensure_batch1(cfg.input_shape)
    if len(shape) != 4:
        raise ValueError(f"CNN2D expects input_shape=(1,C,H,W), got {shape}")

    _, C, H, W = shape
    in_ch = int(C)
    act_name = cfg.activation

    n_blocks = len(cfg.conv_channels)
    for i, out_ch in enumerate(cfg.conv_channels):
        k = _as_block_param(cfg.kernel_sizes, i, n_blocks, "kernel_sizes")
        s = _as_block_param(cfg.strides, i, n_blocks, "strides")
        p = _as_block_param(cfg.paddings, i, n_blocks, "paddings")

        layers.append(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=int(out_ch),
                kernel_size=int(k),
                stride=int(s),
                padding=int(p),
                bias=bool(cfg.use_bias),
            )
        )
        layers.append(_make_activation(act_name))

        H, W = _infer_conv2d_output_hw(H, W, kernel=int(k), stride=int(s), padding=int(p))
        if H <= 0 or W <= 0:
            raise ValueError(
                f"Invalid conv output spatial dims after block {i}: H={H}, W={W} "
                f"(k={k}, s={s}, p={p}, in_shape={shape})"
            )

        in_ch = int(out_ch)

        if cfg.use_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=int(cfg.maxpool_kernel), stride=int(cfg.maxpool_stride)))
            H, W = _infer_maxpool2d_output_hw(H, W, kernel=int(cfg.maxpool_kernel), stride=int(cfg.maxpool_stride))
            if H <= 0 or W <= 0:
                raise ValueError(
                    f"Invalid maxpool output spatial dims after block {i}: H={H}, W={W} "
                    f"(k={cfg.maxpool_kernel}, s={cfg.maxpool_stride}, in_shape={shape})"
                )

    layers.append(nn.Flatten(start_dim=1))
    feat = int(in_ch * H * W)

    layers.append(nn.Linear(feat, int(cfg.fc_hidden), bias=True))
    layers.append(_make_activation(act_name))
    layers.append(nn.Linear(int(cfg.fc_hidden), int(cfg.num_classes), bias=True))

    return nn.Sequential(*layers)


# --------------------------
# Public API
# --------------------------

def build_wrapped_model(
    instance_spec: InstanceSpec,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    deterministic_algos: bool = False,
) -> nn.Module:
    """
    Build wrapped PyTorch model with InputLayer/InputSpecLayer/OutputSpecLayer.

    Returns:
        VerifiableModel that outputs dict with constraint status.
    """
    dev = torch.device(device)
    seed_model = derive_seed(int(instance_spec.seed), 0, "model")
    seed_input = derive_seed(int(instance_spec.seed), 0, "inputs")
    seed_spec = derive_seed(int(instance_spec.seed), 0, "spec")

    # Determine input shape + num_classes / build body
    with seeded(int(seed_model), deterministic=deterministic_algos):
        if instance_spec.family == ModelFamily.MLP:
            assert isinstance(instance_spec.model_cfg, MLPConfig)
            input_shape = _ensure_batch1(instance_spec.model_cfg.input_shape)
            num_classes = int(instance_spec.model_cfg.num_classes)
            body = _build_mlp_body(instance_spec.model_cfg)

        elif instance_spec.family == ModelFamily.CNN2D:
            assert isinstance(instance_spec.model_cfg, CNN2DConfig)
            input_shape = _ensure_batch1(instance_spec.model_cfg.input_shape)
            num_classes = int(instance_spec.model_cfg.num_classes)
            body = _build_cnn2d_body(instance_spec.model_cfg)

        elif instance_spec.family == ModelFamily.TEMPLATE:
            assert isinstance(instance_spec.model_cfg, TemplateConfig)
            from act.pipeline.verification.model_factory import ModelFactory
            from act.pipeline.verification.act2torch import ACTToTorch

            factory = ModelFactory()
            act_net = factory.get_act_net(
                instance_spec.model_cfg.template_name,
                spec_overrides=instance_spec.model_cfg.overrides,
            )
            input_shape = instance_spec.model_cfg.input_shape
            if input_shape is None:
                input_shape = _infer_input_shape_from_act_net(act_net)
            input_shape = _ensure_batch1(input_shape)
            num_classes = instance_spec.model_cfg.num_classes
            if num_classes is None:
                num_classes = _infer_num_classes_from_act_net(act_net) or 2
            model = ACTToTorch(act_net).run()
            body = _extract_body_layers(model)

        else:
            raise ValueError(f"Unsupported model family: {instance_spec.family}")

    # Prepare labeled input (for InputLayer)
    with seeded(int(seed_input), deterministic=deterministic_algos):
        label = int(instance_spec.output_spec.y_true if instance_spec.output_spec.y_true is not None else 0)
        labeled_input = _make_labeled_input(
            shape=input_shape,
            dtype=dtype,
            device=dev,
            label=label,
            value_range=instance_spec.input_spec.value_range,
        )

    # Build InputSpec / OutputSpec objects with tensors
    with seeded(int(seed_spec), deterministic=deterministic_algos):
        in_fields = _build_input_spec_tensor_fields(
            instance_spec.input_spec,
            input_shape,
            dtype=dtype,
            device=dev,
        )

        input_spec_obj = InputSpec(
            kind=instance_spec.input_spec.kind,
            **{k: v for k, v in in_fields.items() if k in ("lb", "ub", "center", "A", "b")},
            eps=in_fields.get("eps", None),
        )

        out_fields = _build_output_spec_tensor_fields(
            instance_spec.output_spec,
            num_classes=num_classes,
            dtype=dtype,
            device=dev,
        )

        output_spec_obj = OutputSpec(
            kind=instance_spec.output_spec.kind,
            y_true=out_fields.get("y_true", None),
            margin=float(out_fields.get("margin", instance_spec.output_spec.margin)),
            d=out_fields.get("d", None),
            meta=dict(instance_spec.output_spec.meta),
            c=out_fields.get("c", None),
            lb=out_fields.get("lb", None),
            ub=out_fields.get("ub", None),
        )

    # Wrapper layers
    in_layer = InputLayer(
        labeled_input=labeled_input,
        shape=input_shape,
        dtype=dtype,  # REQUIRED in your InputLayer
        desc="confignet_input",
        value_range=instance_spec.input_spec.value_range,
        dataset_name=instance_spec.meta.get("dataset_name", None),
        num_classes=num_classes,
        distribution=instance_spec.meta.get("input_distribution", "uniform"),
        sample_id=instance_spec.instance_id,
    )

    in_spec_layer = InputSpecLayer(input_spec_obj)
    out_spec_layer = OutputSpecLayer(output_spec_obj)

    model = VerifiableModel(
        in_layer,
        in_spec_layer,
        body,
        out_spec_layer,
    )
    model = model.to(device=dev, dtype=dtype)
    model.eval()

    logger.info(
        "Built wrapped model: %s (family=%s, id=%s)",
        type(model).__name__,
        instance_spec.family.value,
        instance_spec.instance_id,
    )
    return model


def build_generated_instance(
    instance_spec: InstanceSpec,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    deterministic_algos: bool = False,
) -> GeneratedInstance:
    """
    Convenience: build both wrapped model and bundle as GeneratedInstance.
    """
    model = build_wrapped_model(
        instance_spec,
        device=device,
        dtype=dtype,
        deterministic_algos=deterministic_algos,
    )
    return GeneratedInstance(instance_spec=instance_spec, wrapped_model=model)
