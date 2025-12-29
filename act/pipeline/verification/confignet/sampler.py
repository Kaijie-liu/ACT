from __future__ import annotations

import random
import dataclasses
from typing import Dict, List, Sequence, Tuple, Optional

from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
from act.pipeline.verification.confignet.seeds import set_global_seeds
from act.pipeline.verification.confignet.utils import extract_effective_spec

EPS_CHOICES = (0.01, 0.02, 0.03)


def _classify_arch(name: str, spec: Dict[str, any]) -> str:
    arch_type = (spec.get("architecture_type") or "").lower()
    if arch_type in ("cnn", "conv", "convolutional"):
        return "cnn"
    if arch_type in ("mlp", "fc", "dense"):
        return "mlp"
    lower = name.lower()
    if "cnn" in lower or "conv" in lower:
        return "cnn"
    return "mlp"


def _list_templates_by_arch(factory) -> Dict[str, List[str]]:
    """Return available template names grouped by inferred architecture."""
    grouped: Dict[str, List[str]] = {"mlp": [], "cnn": []}
    for name, spec in factory.config.get("networks", {}).items():
        arch = _classify_arch(name, spec)
        grouped.setdefault(arch, []).append(name)
    for arch in grouped:
        grouped[arch] = sorted(grouped[arch])
    return grouped


def _infer_model_meta(factory, template_name: str) -> ModelConfig:
    """Create a ModelConfig from a template, inferring metadata from the spec."""
    spec = factory.config.get("networks", {}).get(template_name, {}) or {}
    arch = _classify_arch(template_name, spec)
    return ModelConfig(
        arch=arch,
        template_name=template_name,
        input_shape=spec.get("input_shape", []),
        num_classes=spec.get("num_classes", spec.get("output_dim", 0)) or 0,
        activation="relu",
        dataset=spec.get("dataset", "runtime"),
        seed=0,
    )


def _default_true_label(factory, name: str) -> Optional[int]:
    act_net = factory.get_act_net(name)
    eff = extract_effective_spec(act_net)
    return int(eff["y_true"]) if eff["y_true"] is not None else None


def _sample_spec(rng: random.Random, factory: ModelFactory, name: str) -> SpecConfig:
    eps = rng.choice(EPS_CHOICES)
    true_label = _default_true_label(factory, name)
    # If template had no y_true, sample
    if true_label is None:
        true_label = rng.choice(list(range(10)))
    return SpecConfig(
        eps=eps,
        norm="linf",
        targeted=False,
        true_label=true_label,
        target_label=None,
        assert_kind="TOP1_ROBUST",
    )


def sample_configs(
    seed: int,
    n: int,
    arch_choices: Sequence[str] = ("mlp", "cnn"),
) -> List[tuple[ModelConfig, SpecConfig]]:
    """
    Deterministically sample model/spec configs.

    Args:
        seed: global seed
        n: number of samples
        arch_choices: allowed architectures
    """
    set_global_seeds(seed)
    rng = random.Random(seed)
    from act.pipeline.verification.model_factory import ModelFactory

    factory = ModelFactory()
    templates_by_arch = _list_templates_by_arch(factory)

    results: List[tuple[ModelConfig, SpecConfig]] = []
    for i in range(n):
        case_seed = seed + i
        arch_pool_choices = [a for a in arch_choices if templates_by_arch.get(a)]
        if not arch_pool_choices:
            raise ValueError(f"No templates available for requested arch choices: {arch_choices}")
        arch = rng.choice(list(arch_pool_choices))
        pool = templates_by_arch.get(arch, [])
        if not pool:
            raise ValueError(f"No templates available for arch '{arch}'")
        chosen_name = rng.choice(pool)
        mcfg = dataclasses.replace(_infer_model_meta(factory, chosen_name), seed=case_seed)
        rng_case = random.Random(case_seed)
        scfg = _sample_spec(rng_case, factory, chosen_name)
        results.append((mcfg, scfg))
    return results
