from __future__ import annotations

from act.pipeline.verification.utils_seed_logging import set_global_seed


def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """Alias to the canonical seeding utility."""
    set_global_seed(seed, deterministic=deterministic)
