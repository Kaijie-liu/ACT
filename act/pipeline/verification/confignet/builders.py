from __future__ import annotations

from typing import Any

from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
from act.pipeline.verification.confignet.seeds import set_global_seeds


def build_act_net(model_cfg: ModelConfig, spec_cfg: SpecConfig, name: str) -> Any:
    """
    Build ACT Net by reusing existing ModelFactory/NetFactory templates.
    Notes:
      - Uses template_name from ModelConfig; no custom layer authoring here.
      - Spec overrides (eps, assert kind, y_true) are applied via ModelFactory.
    """
    set_global_seeds(model_cfg.seed)
    from act.pipeline.verification.model_factory import ModelFactory

    factory = ModelFactory()
    overrides = {
        "INPUT_SPEC": {
            "eps": spec_cfg.eps,
            "norm": spec_cfg.norm,
        },
        "ASSERT": {
            "kind": spec_cfg.assert_kind,
            "y_true": spec_cfg.true_label,
            "targeted": spec_cfg.targeted,
        },
    }
    return factory.get_act_net(model_cfg.template_name, spec_overrides=overrides)


def build_torch_model(act_net: Any, device: Any = None, dtype: Any = None):
    """
    Build torch model from an ACT net via ACTToTorch, optionally moving to device/dtype.
    """
    from act.pipeline.verification.act2torch import ACTToTorch

    model = ACTToTorch(act_net).run()

    if device is None and dtype is None:
        return model

    try:
        import torch
    except Exception:
        return model

    # Align model parameters to requested device/dtype when provided
    target_device = device
    target_dtype = dtype
    if target_device is None:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            target_device = first_param.device
    if target_dtype is None:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            target_dtype = first_param.dtype

    return model.to(device=target_device, dtype=target_dtype)
