#!/usr/bin/env python3
#===- act/pipeline/confignet/schema.py - ConfigNet Schema ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Normalized dataclasses for ConfigNet-generated instances.
#   Schema only describes "what was generated", and does NOT perform verification.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from act.front_end.specs import InKind, OutKind

from .jsonl import canonical_hash, tensor_digest

# --------------------------
# Model Configs
# --------------------------

class ModelFamily(str, Enum):
    MLP = "mlp"
    CNN2D = "cnn2d"
    TEMPLATE = "template"


def _json_safe(obj: Any, *, strict: bool = False) -> Any:
    """Convert objects to JSON-safe values for auditing."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if torch.is_tensor(obj):
        return tensor_digest(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, strict=strict) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, strict=strict) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _json_safe(obj.to_dict(), strict=strict)
    if is_dataclass(obj):
        return _json_safe(asdict(obj), strict=strict)
    if strict:
        raise TypeError(f"Object is not JSON-serializable: {type(obj)}")
    return str(obj)


def _enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


@dataclass(frozen=True)
class MLPConfig:
    """
    MLP architecture configuration.

    Notes:
      - input_shape is the model input tensor shape including batch dimension.
        e.g. (1, 784) or (1, 1, 28, 28) if you want to include a Flatten layer.
      - If input_shape has >2 dims, builders will insert nn.Flatten(start_dim=1).
    """
    input_shape: Tuple[int, ...]
    hidden_sizes: Tuple[int, ...]
    activation: str = "relu"  # "relu", "tanh", "sigmoid", "silu", "gelu"
    use_bias: bool = True
    dropout_p: float = 0.0
    num_classes: int = 10

    # optional: end with extra linear layers? keep simple.
    def __post_init__(self):
        if len(self.input_shape) < 2:
            raise ValueError(f"MLPConfig.input_shape must include batch dim, got {self.input_shape}")
        if self.input_shape[0] != 1:
            raise ValueError(f"ConfigNet assumes batch=1, got input_shape={self.input_shape}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0,1), got {self.dropout_p}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "use_bias": bool(self.use_bias),
            "dropout_p": float(self.dropout_p),
            "num_classes": int(self.num_classes),
        }


@dataclass(frozen=True)
class CNN2DConfig:
    """
    Simple CNN2D architecture configuration.

    Notes:
      - input_shape includes batch: (1, C, H, W)
      - conv_channels defines conv blocks' out_channels, e.g. (8, 16)
      - kernel_sizes/strides/paddings can be scalar or per-block tuple
    """
    input_shape: Tuple[int, int, int, int]  # (1, C, H, W)
    conv_channels: Tuple[int, ...]          # out_channels per block
    kernel_sizes: Union[int, Tuple[int, ...]] = 3
    strides: Union[int, Tuple[int, ...]] = 1
    paddings: Union[int, Tuple[int, ...]] = 1
    activation: str = "relu"
    use_bias: bool = True
    use_maxpool: bool = False
    maxpool_kernel: int = 2
    maxpool_stride: int = 2
    num_classes: int = 10
    fc_hidden: int = 64  # one FC hidden layer after conv

    def __post_init__(self):
        if len(self.input_shape) != 4:
            raise ValueError(f"CNN2DConfig.input_shape must be (1,C,H,W), got {self.input_shape}")
        if self.input_shape[0] != 1:
            raise ValueError(f"ConfigNet assumes batch=1, got input_shape={self.input_shape}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError(f"conv_channels must be positive, got {self.conv_channels}")
        if self.maxpool_kernel <= 0 or self.maxpool_stride <= 0:
            raise ValueError("maxpool params must be positive")
        if self.fc_hidden <= 0:
            raise ValueError("fc_hidden must be positive")

    def to_dict(self) -> Dict[str, Any]:
        def _val(v: Union[int, Tuple[int, ...]]) -> Union[int, List[int]]:
            return int(v) if isinstance(v, int) else [int(x) for x in v]

        return {
            "input_shape": list(self.input_shape),
            "conv_channels": list(self.conv_channels),
            "kernel_sizes": _val(self.kernel_sizes),
            "strides": _val(self.strides),
            "paddings": _val(self.paddings),
            "activation": self.activation,
            "use_bias": bool(self.use_bias),
            "use_maxpool": bool(self.use_maxpool),
            "maxpool_kernel": int(self.maxpool_kernel),
            "maxpool_stride": int(self.maxpool_stride),
            "num_classes": int(self.num_classes),
            "fc_hidden": int(self.fc_hidden),
        }


@dataclass(frozen=True)
class TemplateConfig:
    """
    Template network configuration (built from ACT examples).
    """
    template_name: str
    dataset: Optional[str] = None
    activation: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": str(self.template_name),
            "dataset": self.dataset,
            "activation": self.activation,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "num_classes": int(self.num_classes) if self.num_classes is not None else None,
            "overrides": _json_safe(self.overrides),
        }


# --------------------------
# Spec Configs
# --------------------------

@dataclass(frozen=True)
class InputSpecConfig:
    """
    Input spec configuration for InputSpecLayer construction.

    Supported:
      - BOX: lb/ub via scalar range OR explicit tensors
      - LINF_BALL: center + eps (center can be scalar or tensor)
      - LIN_POLY: Ax <= b (can be derived from lb/ub if requested)
    """
    kind: InKind

    # Common numeric range hint (used for sampling center/lb/ub if tensors not provided)
    value_range: Tuple[float, float] = (0.0, 1.0)

    # BOX fields
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    lb_val: Optional[float] = None
    ub_val: Optional[float] = None

    # LINF_BALL fields
    center: Optional[torch.Tensor] = None
    center_val: Optional[float] = 0.5
    eps: Optional[float] = 0.03

    # LIN_POLY fields
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    derive_poly_from_box: bool = True  # if A/b missing and BOX available -> derive

    # misc
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": _enum_value(self.kind),
            "value_range": list(self.value_range),
            "lb": tensor_digest(self.lb) if self.lb is not None else None,
            "ub": tensor_digest(self.ub) if self.ub is not None else None,
            "lb_val": self.lb_val,
            "ub_val": self.ub_val,
            "center": tensor_digest(self.center) if self.center is not None else None,
            "center_val": self.center_val,
            "eps": self.eps,
            "A": tensor_digest(self.A) if self.A is not None else None,
            "b": tensor_digest(self.b) if self.b is not None else None,
            "derive_poly_from_box": bool(self.derive_poly_from_box),
            "meta": _json_safe(self.meta),
        }


@dataclass(frozen=True)
class OutputSpecConfig:
    """
    Output spec configuration for OutputSpecLayer construction.

    Supported:
      - TOP1_ROBUST: y_true
      - MARGIN_ROBUST: y_true + margin
      - LINEAR_LE: c^T y <= d
      - RANGE: lb <= y <= ub
    """
    kind: OutKind
    y_true: Optional[int] = None
    margin: float = 0.0

    # LINEAR_LE
    c: Optional[torch.Tensor] = None
    d: Optional[float] = None

    # RANGE
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": _enum_value(self.kind),
            "y_true": self.y_true,
            "margin": float(self.margin),
            "c": tensor_digest(self.c) if self.c is not None else None,
            "d": self.d,
            "lb": tensor_digest(self.lb) if self.lb is not None else None,
            "ub": tensor_digest(self.ub) if self.ub is not None else None,
            "meta": _json_safe(self.meta),
        }


# --------------------------
# Instance-level containers
# --------------------------

@dataclass(frozen=True)
class InstanceSpec:
    """
    Fully specified instance: id + seed + model config + spec config.

    meta: free-form dict for logging (e.g., sampler decisions, derived seeds).
    """
    instance_id: str
    seed: int
    family: ModelFamily
    model_cfg: Union[MLPConfig, CNN2DConfig, TemplateConfig]
    input_spec: InputSpecConfig
    output_spec: OutputSpecConfig
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "family": self.family.value if isinstance(self.family, Enum) else str(self.family),
            "model_cfg": _json_safe(self.model_cfg),
            "input_spec": _json_safe(self.input_spec),
            "output_spec": _json_safe(self.output_spec),
            "meta": _json_safe(self.meta),
        }

    def stable_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass
class GeneratedInstance:
    """
    Generated instance bundle: spec + wrapped torch model.

    wrapped_model is typically act.front_end.verifiable_model.VerifiableModel or nn.Sequential
    that includes InputLayer/InputSpecLayer/OutputSpecLayer.
    """
    instance_spec: InstanceSpec
    wrapped_model: torch.nn.Module


# --------------------------
# Sampler configuration
# --------------------------

@dataclass(frozen=True)
class ConfigNetConfig:
    """
    Sampler configuration.

    This controls distributions/ranges. sampler.py consumes this config and produces InstanceSpec list.
    """
    num_instances: int = 10
    base_seed: int = 0
    instance_id_prefix: str = "cfg"

    # Which families to sample from
    families: Tuple[ModelFamily, ...] = (ModelFamily.MLP, ModelFamily.CNN2D)
    p_mlp: float = 0.5  # when both families enabled

    # General
    num_classes_choices: Tuple[int, ...] = (10,)

    # MLP sampling
    mlp_input_shapes: Tuple[Tuple[int, ...], ...] = ((1, 784), (1, 1, 28, 28))
    mlp_depth_range: Tuple[int, int] = (2, 4)            # number of hidden layers
    mlp_width_choices: Tuple[int, ...] = (32, 64, 128)
    mlp_activation_choices: Tuple[str, ...] = ("relu", "tanh", "sigmoid")
    mlp_dropout_p_choices: Tuple[float, ...] = (0.0,)

    # CNN sampling
    cnn_input_shapes: Tuple[Tuple[int, int, int, int], ...] = ((1, 1, 28, 28), (1, 3, 32, 32))
    cnn_num_blocks_range: Tuple[int, int] = (1, 3)
    cnn_channels_choices: Tuple[int, ...] = (8, 16, 32)
    cnn_kernel_choices: Tuple[int, ...] = (3,)
    cnn_stride_choices: Tuple[int, ...] = (1,)
    cnn_padding_choices: Tuple[int, ...] = (1,)
    cnn_use_maxpool_p: float = 0.3
    cnn_fc_hidden_choices: Tuple[int, ...] = (32, 64, 128)
    cnn_activation_choices: Tuple[str, ...] = ("relu", "tanh", "sigmoid")

    # Input spec sampling
    input_kind_choices: Tuple[InKind, ...] = (InKind.BOX, InKind.LINF_BALL)
    p_box: float = 0.5
    value_range_choices: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)
    eps_choices: Tuple[float, ...] = (0.03, 0.05, 0.1)

    # Output spec sampling
    output_kind_choices: Tuple[OutKind, ...] = (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST, OutKind.LINEAR_LE, OutKind.RANGE)
    p_top1: float = 0.5
    margin_choices: Tuple[float, ...] = (0.0, 0.1, 1.0)
    output_linear_le_c_range: Tuple[float, float] = (-1.0, 1.0)
    output_linear_le_d_range: Tuple[float, float] = (-1.0, 1.0)
    output_range_choices: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0),)

    # Template sampling (optional)
    template_names: Optional[Tuple[str, ...]] = None
