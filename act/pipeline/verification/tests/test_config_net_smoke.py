#test_config_net_smoke.py
import pytest

pytest.importorskip("torch")

from act.pipeline.verification.config_net import ConfigNet


def test_sample_returns_valid_fields():
    cfg = ConfigNet()
    sample = cfg.sample(seed=0)
    available = cfg._available
    assert sample.name in available
    assert sample.act_net is not None
    assert sample.torch_model is not None
    assert sample.input_specs is not None
    assert sample.assert_layer is not None


def test_sampling_is_deterministic_with_seed():
    cfg = ConfigNet()
    s1 = cfg.sample(seed=42)
    s2 = cfg.sample(seed=42)
    assert s1.name == s2.name


def test_sampling_by_name():
    cfg = ConfigNet()
    first_name = sorted(cfg._available)[0]
    sample = cfg.sample(name=first_name)
    assert sample.name == first_name
