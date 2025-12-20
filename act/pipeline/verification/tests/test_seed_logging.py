#`test_seed_logging.py`
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.utils_seed_logging import set_global_seed, init_logging


def test_torch_reproducibility_cpu():
    set_global_seed(123)
    a = torch.rand(3)
    set_global_seed(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_numpy_reproducibility():
    set_global_seed(321)
    a = np.random.rand(4)
    set_global_seed(321)
    b = np.random.rand(4)
    assert np.allclose(a, b)


def test_init_logging_no_crash():
    logger = init_logging(level=logging.DEBUG)
    logger.debug("init ok")
    assert isinstance(logger, logging.Logger)
