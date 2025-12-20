import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.config_net_runtime import RuntimeConfigNet


def test_runtime_mlp_sample():
    cfg = RuntimeConfigNet()
    sample = cfg.sample(seed=1, arch="mlp")
    assert sample.act_net is not None
    assert sample.torch_model is not None
    assert sample.input_specs
    assert sample.assert_layer.meta.get("kind") in ("TOP1_ROBUST", "MARGIN_ROBUST")
    # deterministic naming
    assert sample.name == "runtime_mlp_1"


def test_runtime_cnn_sample():
    cfg = RuntimeConfigNet()
    sample = cfg.sample(seed=2, arch="cnn")
    assert sample.act_net is not None
    assert sample.torch_model is not None
    assert sample.input_specs
    assert sample.assert_layer.meta.get("kind") in ("TOP1_ROBUST", "MARGIN_ROBUST")
