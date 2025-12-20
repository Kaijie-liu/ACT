from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds across random/numpy/torch (if available) for reproducibility.
    Torch is imported lazily inside the function to avoid heavy imports at
    module import time.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True  # type: ignore
                    torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        # Torch may be unavailable in minimal environments; ignore silently.
        return
