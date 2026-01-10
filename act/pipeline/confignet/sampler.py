#!/usr/bin/env python3
#===- act/pipeline/confignet/sampler.py - ConfigNet Sampler -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Pure sampler for ConfigNet:
#     - Input:  ConfigNetConfig (sampling space + distributions)
#     - Output: List[InstanceSpec] (only config/spec objects)
#
#   This module:
#     - DOES: sample model/spec configs, produce normalized InstanceSpec
#     - DOES NOT: build torch model, run verification, import back_end
#
# Reproducibility:
#   Per-instance RNG is derived from (base_seed, idx, instance_id) using seeds.py
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from act.front_end.specs import InKind, OutKind

from .seeds import derive_seed
from .schema import (
    ConfigNetConfig,
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    CNN2DConfig,
    TemplateConfig,
    InputSpecConfig,
    OutputSpecConfig,
)

_TEMPLATE_FACTORY = None


# -----------------------------
# Helpers
# -----------------------------

def _to_plain(obj: Any) -> Any:
    """Convert dataclasses/enums/containers to plain python types for meta logging."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    # Enum etc.
    return str(obj)


def _randint_inclusive(rng: random.Random, lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _choose(rng: random.Random, items: Sequence[Any], *, name: str) -> Any:
    if not items:
        raise ValueError(f"ConfigNetConfig.{name} must be non-empty")
    return rng.choice(list(items))


def _sample_family(rng: random.Random, cfg: ConfigNetConfig) -> ModelFamily:
    fams = list(cfg.families)
    if not fams:
        raise ValueError("ConfigNetConfig.families must be non-empty")

    if len(fams) == 1:
        return fams[0]

    has_mlp = ModelFamily.MLP in fams
    has_cnn = ModelFamily.CNN2D in fams

    # If both present, use p_mlp
    if has_mlp and has_cnn:
        return ModelFamily.MLP if (rng.random() < float(cfg.p_mlp)) else ModelFamily.CNN2D

    # Otherwise uniform over whatever is provided
    return rng.choice(fams)


def _get_template_factory():
    global _TEMPLATE_FACTORY
    if _TEMPLATE_FACTORY is None:
        from act.pipeline.verification.model_factory import ModelFactory
        _TEMPLATE_FACTORY = ModelFactory()
    return _TEMPLATE_FACTORY


def _resolve_template_catalog(cfg: ConfigNetConfig) -> Dict[str, Any]:
    factory = _get_template_factory()
    names = list(factory.config.get("networks", {}).keys())
    if cfg.template_names:
        names = [n for n in cfg.template_names if n in names] or list(cfg.template_names)
    if not names:
        raise ValueError("No template networks available for ModelFamily.TEMPLATE sampling")
    return {"factory": factory, "names": names}


def _get_template_info(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    input_shape = spec.get("input_shape", None)
    dataset = None
    num_classes = None
    activation = None

    if input_shape is None:
        for layer in spec.get("layers", []):
            if layer.get("kind") == "INPUT":
                meta = layer.get("meta", {})
                input_shape = meta.get("shape", None)
                dataset = meta.get("dataset_name", None)
                num_classes = meta.get("num_classes", None)
                break
    else:
        input_shape = list(input_shape)
        for layer in spec.get("layers", []):
            if layer.get("kind") == "INPUT":
                meta = layer.get("meta", {})
                dataset = meta.get("dataset_name", None)
                num_classes = meta.get("num_classes", None)
                break

    return {
        "template_name": name,
        "input_shape": tuple(int(x) for x in input_shape) if input_shape is not None else None,
        "dataset": dataset,
        "num_classes": int(num_classes) if num_classes is not None else None,
        "activation": activation,
    }


def _build_template_overrides(input_spec: InputSpecConfig, output_spec: OutputSpecConfig) -> Dict[str, Any]:
    eps = 0.0
    if input_spec.kind == InKind.LINF_BALL and input_spec.eps is not None:
        eps = float(input_spec.eps)
    return {
        "eps": float(eps),
        "norm": "inf",
        "assert_kind": output_spec.kind.value,
        "y_true": int(output_spec.y_true if output_spec.y_true is not None else 0),
        "targeted": False,
        "target_label": None,
    }


def _make_instance_id(prefix: str, base_seed: int, idx: int) -> str:
    return f"{prefix}_{int(base_seed)}_{int(idx):05d}"


def _sample_mlp(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[MLPConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.mlp_input_shapes, name="mlp_input_shapes")

    depth = _randint_inclusive(rng, cfg.mlp_depth_range)
    if depth <= 0:
        raise ValueError(f"mlp_depth_range produced non-positive depth={depth}")

    widths = []
    for _ in range(depth):
        widths.append(int(_choose(rng, cfg.mlp_width_choices, name="mlp_width_choices")))
    hidden_sizes = tuple(widths)

    activation = str(_choose(rng, cfg.mlp_activation_choices, name="mlp_activation_choices"))
    dropout_p = float(_choose(rng, cfg.mlp_dropout_p_choices, name="mlp_dropout_p_choices"))

    model_cfg = MLPConfig(
        input_shape=tuple(int(x) for x in input_shape),
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_bias=True,
        dropout_p=dropout_p,
        num_classes=int(num_classes),
    )
    meta = {
        "depth": depth,
        "hidden_sizes": list(hidden_sizes),
        "activation": activation,
        "dropout_p": dropout_p,
        "input_shape": list(model_cfg.input_shape),
        "num_classes": num_classes,
    }
    return model_cfg, meta


def _sample_cnn2d(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[CNN2DConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.cnn_input_shapes, name="cnn_input_shapes")
    blocks = _randint_inclusive(rng, cfg.cnn_num_blocks_range)
    if blocks <= 0:
        raise ValueError(f"cnn_num_blocks_range produced non-positive blocks={blocks}")

    # Stronger sampling: non-decreasing channels (more realistic)
    ch_choices = sorted(int(x) for x in cfg.cnn_channels_choices)
    if not ch_choices:
        raise ValueError("cnn_channels_choices must be non-empty")

    conv_channels: List[int] = []
    prev = int(rng.choice(ch_choices))
    conv_channels.append(prev)
    for _ in range(1, blocks):
        eligible = [c for c in ch_choices if c >= prev]
        if not eligible:
            eligible = ch_choices
        prev = int(rng.choice(eligible))
        conv_channels.append(prev)

    # "Most powerful": allow scalar OR per-block tuples.
    def sample_scalar_or_tuple(choices: Tuple[int, ...], p_tuple: float = 0.5):
        if not choices:
            raise ValueError("kernel/stride/padding choices must be non-empty")
        if blocks == 1 or rng.random() >= p_tuple:
            return int(rng.choice(list(choices)))
        return tuple(int(rng.choice(list(choices))) for _ in range(blocks))

    kernel_sizes = sample_scalar_or_tuple(cfg.cnn_kernel_choices, p_tuple=0.5)
    strides = sample_scalar_or_tuple(cfg.cnn_stride_choices, p_tuple=0.4)
    paddings = sample_scalar_or_tuple(cfg.cnn_padding_choices, p_tuple=0.5)

    activation = str(_choose(rng, cfg.cnn_activation_choices, name="cnn_activation_choices"))
    use_maxpool = bool(rng.random() < float(cfg.cnn_use_maxpool_p))
    fc_hidden = int(_choose(rng, cfg.cnn_fc_hidden_choices, name="cnn_fc_hidden_choices"))

    model_cfg = CNN2DConfig(
        input_shape=tuple(int(x) for x in input_shape),  # (1,C,H,W)
        conv_channels=tuple(conv_channels),
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation=activation,
        use_bias=True,
        use_maxpool=use_maxpool,
        maxpool_kernel=2,
        maxpool_stride=2,
        num_classes=int(num_classes),
        fc_hidden=int(fc_hidden),
    )
    meta = {
        "blocks": blocks,
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes if isinstance(kernel_sizes, int) else list(kernel_sizes),
        "strides": strides if isinstance(strides, int) else list(strides),
        "paddings": paddings if isinstance(paddings, int) else list(paddings),
        "activation": activation,
        "use_maxpool": use_maxpool,
        "fc_hidden": fc_hidden,
        "input_shape": list(model_cfg.input_shape),
        "num_classes": num_classes,
    }
    return model_cfg, meta


def _sample_input_spec(rng: random.Random, cfg: ConfigNetConfig, *, input_shape: Tuple[int, ...]) -> Tuple[InputSpecConfig, Dict[str, Any]]:
    # choose kind
    kinds = list(cfg.input_kind_choices)
    if not kinds:
        raise ValueError("input_kind_choices must be non-empty")

    if len(kinds) == 1:
        kind = kinds[0]
    else:
        has_box = InKind.BOX in kinds
        has_linf = InKind.LINF_BALL in kinds
        if has_box and has_linf and len(kinds) == 2:
            kind = InKind.BOX if (rng.random() < float(cfg.p_box)) else InKind.LINF_BALL
        else:
            kind = rng.choice(kinds)

    value_range = _choose(rng, cfg.value_range_choices, name="value_range_choices")
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi < lo:
        lo, hi = hi, lo

    if kind == InKind.BOX:
        # "stronger": sample a narrower box within [lo,hi] (still feasible)
        # keep it stable and simple (builders will expand scalar to tensor)
        span = hi - lo
        shrink_a = rng.random() * 0.2  # up to 20% shrink from left
        shrink_b = rng.random() * 0.2  # up to 20% shrink from right
        lb_val = lo + span * shrink_a
        ub_val = hi - span * shrink_b
        if ub_val < lb_val:
            lb_val, ub_val = lo, hi  # fallback

        in_cfg = InputSpecConfig(
            kind=InKind.BOX,
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "BOX", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    if kind == InKind.LINF_BALL:
        center_val = lo + (hi - lo) * rng.random()
        eps = float(_choose(rng, cfg.eps_choices, name="eps_choices"))
        # cap eps to range width
        eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0

        in_cfg = InputSpecConfig(
            kind=InKind.LINF_BALL,
            value_range=(lo, hi),
            center_val=float(center_val),
            eps=float(eps),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LINF_BALL", "value_range": (lo, hi), "center_val": float(center_val), "eps": float(eps)}
        return in_cfg, meta

    if kind == InKind.LIN_POLY:
        # Derive A/b from box by default (builders will materialize A/b)
        span = hi - lo
        shrink_a = rng.random() * 0.2
        shrink_b = rng.random() * 0.2
        lb_val = lo + span * shrink_a
        ub_val = hi - span * shrink_b
        if ub_val < lb_val:
            lb_val, ub_val = lo, hi

        in_cfg = InputSpecConfig(
            kind=InKind.LIN_POLY,
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            derive_poly_from_box=True,
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LIN_POLY", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    # Fallback
    in_cfg = InputSpecConfig(
        kind=InKind.BOX,
        value_range=(lo, hi),
        lb_val=lo,
        ub_val=hi,
        meta={"fallback": True, "reason": f"unsupported kind={kind}"},
    )
    meta = {"kind": "FALLBACK_BOX", "value_range": (lo, hi), "lb_val": lo, "ub_val": hi}
    return in_cfg, meta


def _sample_output_spec(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[OutputSpecConfig, Dict[str, Any]]:
    kinds = list(cfg.output_kind_choices)
    if not kinds:
        raise ValueError("output_kind_choices must be non-empty")

    if len(kinds) == 1:
        kind = kinds[0]
    else:
        has_top1 = OutKind.TOP1_ROBUST in kinds
        has_margin = OutKind.MARGIN_ROBUST in kinds
        if has_top1 and has_margin and len(kinds) == 2:
            kind = OutKind.TOP1_ROBUST if (rng.random() < float(cfg.p_top1)) else OutKind.MARGIN_ROBUST
        else:
            kind = rng.choice(kinds)

    y_true = int(rng.randrange(int(num_classes)))

    if kind == OutKind.TOP1_ROBUST:
        out_cfg = OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=y_true,
            margin=0.0,
            meta={},
        )
        meta = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
        return out_cfg, meta

    if kind == OutKind.MARGIN_ROBUST:
        margin = float(_choose(rng, cfg.margin_choices, name="margin_choices"))
        out_cfg = OutputSpecConfig(
            kind=OutKind.MARGIN_ROBUST,
            y_true=y_true,
            margin=float(margin),
            meta={},
        )
        meta = {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}
        return out_cfg, meta

    if kind == OutKind.LINEAR_LE:
        c_lo, c_hi = cfg.output_linear_le_c_range
        d_lo, d_hi = cfg.output_linear_le_d_range
        c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
        d_val = d_lo + (d_hi - d_lo) * rng.random()
        c = torch.tensor(c_vals, dtype=torch.float32)
        out_cfg = OutputSpecConfig(
            kind=OutKind.LINEAR_LE,
            c=c,
            d=float(d_val),
            meta={},
        )
        meta = {"kind": "LINEAR_LE", "c": list(c_vals), "d": float(d_val)}
        return out_cfg, meta

    if kind == OutKind.RANGE:
        lo, hi = _choose(rng, cfg.output_range_choices, name="output_range_choices")
        lb_vals = []
        ub_vals = []
        for _ in range(int(num_classes)):
            a = lo + (hi - lo) * rng.random()
            b = lo + (hi - lo) * rng.random()
            lb_vals.append(min(a, b))
            ub_vals.append(max(a, b))
        lb = torch.tensor(lb_vals, dtype=torch.float32)
        ub = torch.tensor(ub_vals, dtype=torch.float32)
        out_cfg = OutputSpecConfig(
            kind=OutKind.RANGE,
            lb=lb,
            ub=ub,
            meta={},
        )
        meta = {"kind": "RANGE", "lb": list(lb_vals), "ub": list(ub_vals)}
        return out_cfg, meta

    # fallback
    out_cfg = OutputSpecConfig(kind=OutKind.TOP1_ROBUST, y_true=y_true, margin=0.0, meta={"fallback": True})
    meta = {"kind": "FALLBACK_TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
    return out_cfg, meta


# -----------------------------
# Public API
# -----------------------------

def sample_instances(cfg: ConfigNetConfig) -> List[InstanceSpec]:
    """
    Determinism contract:
      - instance_id = f"{prefix}_{base_seed}_{idx:05d}"
      - seed = derive_seed(base_seed, idx, instance_id)  (stable u32)
      - per-instance RNG = random.Random(seed)
    """
    base_seed = int(cfg.base_seed)
    num_instances = int(cfg.num_instances)
    prefix = str(cfg.instance_id_prefix)

    if num_instances < 0:
        raise ValueError(f"num_instances must be >= 0, got {num_instances}")

    out: List[InstanceSpec] = []

    for idx in range(num_instances):
        instance_id = _make_instance_id(prefix, base_seed, idx)
        seed = int(derive_seed(base_seed, idx, instance_id))
        rng = random.Random(seed)
        rng_family = random.Random(derive_seed(seed, 0, "family"))
        rng_model = random.Random(derive_seed(seed, 0, "model"))
        rng_spec = random.Random(derive_seed(seed, 0, "spec"))

        family = _sample_family(rng_family, cfg)
        num_classes = int(_choose(rng_family, cfg.num_classes_choices, name="num_classes_choices"))

        if family == ModelFamily.MLP:
            model_cfg, model_meta = _sample_mlp(rng_model, cfg, num_classes=num_classes)
            input_shape = tuple(model_cfg.input_shape)
        elif family == ModelFamily.CNN2D:
            model_cfg, model_meta = _sample_cnn2d(rng_model, cfg, num_classes=num_classes)
            input_shape = tuple(model_cfg.input_shape)
        elif family == ModelFamily.TEMPLATE:
            catalog = _resolve_template_catalog(cfg)
            name = str(rng_model.choice(catalog["names"]))
            spec = catalog["factory"].config["networks"].get(name, {})
            info = _get_template_info(name, spec)
            if info["input_shape"] is None:
                raise ValueError(f"Template '{name}' is missing input_shape metadata")
            if info["num_classes"] is None:
                info["num_classes"] = int(_choose(rng, cfg.num_classes_choices, name="num_classes_choices"))
            input_shape = tuple(info["input_shape"])
            num_classes = int(info["num_classes"])
            model_cfg = TemplateConfig(
                template_name=info["template_name"],
                dataset=info["dataset"],
                activation=info["activation"],
                input_shape=input_shape,
                num_classes=num_classes,
                overrides={},
            )
            model_meta = {
                "template_name": info["template_name"],
                "input_shape": list(input_shape),
                "num_classes": num_classes,
                "dataset": info["dataset"],
            }
        else:
            raise ValueError(f"Unknown family: {family}")

        input_spec, in_meta = _sample_input_spec(rng_spec, cfg, input_shape=input_shape)
        output_spec, out_meta = _sample_output_spec(rng_spec, cfg, num_classes=num_classes)

        if family == ModelFamily.TEMPLATE:
            overrides = _build_template_overrides(input_spec, output_spec)
            model_cfg = TemplateConfig(
                template_name=model_cfg.template_name,
                dataset=model_cfg.dataset,
                activation=model_cfg.activation,
                input_shape=model_cfg.input_shape,
                num_classes=model_cfg.num_classes,
                overrides=overrides,
            )
            model_meta = dict(model_meta)
            model_meta["overrides"] = _to_plain(overrides)

        meta: Dict[str, Any] = {
            "base_seed": base_seed,
            "idx": idx,
            "instance_id": instance_id,
            "seed": seed,
            "family": family.value,
            "seed_family": derive_seed(seed, 0, "family"),
            "seed_model": derive_seed(seed, 0, "model"),
            "seed_spec": derive_seed(seed, 0, "spec"),
            "model_meta": _to_plain(model_meta),
            "input_meta": _to_plain(in_meta),
            "output_meta": _to_plain(out_meta),
        }

        out.append(
            InstanceSpec(
                instance_id=instance_id,
                seed=seed,
                family=family,
                model_cfg=model_cfg,
                input_spec=input_spec,
                output_spec=output_spec,
                meta=meta,
            )
        )

    return out
