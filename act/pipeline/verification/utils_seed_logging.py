#!/usr/bin/env python3
"""
Seed and logging utilities for verification pipeline.

Provides a single place to set random seeds across python/random/numpy/torch
and to initialize a consistent logger format.
"""

from __future__ import annotations

import logging
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Integer seed value.
        deterministic: If True, set torch deterministic flags when available.
    """
    # Core seeding logic kept here to avoid duplicated implementations.
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
        # Torch may be unavailable or broken; ignore silently to stay import-light.
        return


def init_logging(level: int = logging.INFO, name: str = "act.confignet") -> logging.Logger:
    """
    Initialize logging with a consistent format and return a named logger.

    Args:
        level: Logging level (default: INFO).
        name: Logger name (default: "act.confignet").

    Returns:
        Configured logger instance.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Only configure root once to avoid duplicate handlers in pytest runs
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=log_format, datefmt=datefmt)
    root_logger.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
