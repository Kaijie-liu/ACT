from __future__ import annotations

import random
from typing import List, Sequence

from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
from act.pipeline.verification.confignet.seeds import set_global_seeds


TEMPLATE_BY_ARCH = {
    "mlp": "mnist_robust_easy",   # existing examples_config.yaml entry (MLP)
    "cnn": "cifar_margin_moderate",  # existing examples_config.yaml entry (CNN)
}


def _sample_mlp(rng: random.Random, seed: int) -> ModelConfig:
    return ModelConfig(
        arch="mlp",
        template_name=TEMPLATE_BY_ARCH["mlp"],
        input_shape=[1, 784],
        num_classes=10,
        activation="relu",
        dataset="mnist",
        seed=seed,
    )


def _sample_cnn(rng: random.Random, seed: int) -> ModelConfig:
    return ModelConfig(
        arch="cnn",
        template_name=TEMPLATE_BY_ARCH["cnn"],
        input_shape=[1, 3, 32, 32],
        num_classes=10,
        activation="relu",
        dataset="cifar10",
        seed=seed,
    )


def _sample_spec(rng: random.Random, seed: int) -> SpecConfig:
    eps = rng.choice([0.01, 0.02, 0.03])
    true_label = rng.choice(list(range(10)))
    return SpecConfig(
        eps=eps,
        norm="linf",
        targeted=False,  # Week1: untargeted only
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
    results: List[tuple[ModelConfig, SpecConfig]] = []
    for i in range(n):
        case_seed = seed + i
        arch = rng.choice(list(arch_choices))
        rng_case = random.Random(case_seed)
        if arch == "mlp":
            mcfg = _sample_mlp(rng_case, case_seed)
        elif arch == "cnn":
            mcfg = _sample_cnn(rng_case, case_seed)
        else:
            raise ValueError(f"Unsupported arch {arch}")
        scfg = _sample_spec(rng_case, case_seed)
        results.append((mcfg, scfg))
    return results
