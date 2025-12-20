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
        "eps": spec_cfg.eps,
        "y_true": spec_cfg.true_label,
        "assert_kind": spec_cfg.assert_kind,
    }
    # Clone to avoid side effects; template already encodes specs.
    return factory.get_act_net(model_cfg.template_name, spec_overrides=overrides)


def build_torch_model(model_cfg: ModelConfig, spec_cfg: SpecConfig, name: str):
    """
    Build torch model from existing template via ModelFactory/ACTToTorch.
    """
    set_global_seeds(model_cfg.seed)
    from act.pipeline.verification.model_factory import ModelFactory

    factory = ModelFactory()
    return factory.create_model(model_cfg.template_name, load_weights=True)
