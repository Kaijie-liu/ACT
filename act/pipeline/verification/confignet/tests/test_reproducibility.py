import pytest

pytest.importorskip("torch")

from act.pipeline.verification.confignet import sample_configs


def _hash_list(seed: int, n: int):
    configs = sample_configs(seed=seed, n=n)
    return [mcfg.stable_hash() + "-" + scfg.stable_hash() for mcfg, scfg in configs]


def test_confignet_reproducibility_same_seed():
    h1 = _hash_list(0, 10)
    h2 = _hash_list(0, 10)
    assert h1 == h2


def test_confignet_reproducibility_different_seed():
    h1 = _hash_list(1, 5)
    h2 = _hash_list(2, 5)
    assert h1 != h2
