import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.config_net_runtime import RuntimeConfigNet


def _first_layer_of_kind(layers, kind):
    for L in layers:
        if L.kind == kind:
            return L
    raise AssertionError(f"Layer kind '{kind}' not found")


def test_runtime_mlp_deterministic_weights():
    cfg = RuntimeConfigNet()
    s1 = cfg.sample(seed=123, arch="mlp")
    s2 = cfg.sample(seed=123, arch="mlp")

    dense1 = _first_layer_of_kind(s1.act_net.layers, "DENSE")
    dense2 = _first_layer_of_kind(s2.act_net.layers, "DENSE")

    assert torch.equal(dense1.params["W"], dense2.params["W"])
    if "b" in dense1.params or "b" in dense2.params:
        assert "b" in dense1.params and "b" in dense2.params
        assert torch.equal(dense1.params["b"], dense2.params["b"])


def test_runtime_cnn_deterministic_weights():
    cfg = RuntimeConfigNet()
    s1 = cfg.sample(seed=456, arch="cnn")
    s2 = cfg.sample(seed=456, arch="cnn")

    conv1 = _first_layer_of_kind(s1.act_net.layers, "CONV2D")
    conv2 = _first_layer_of_kind(s2.act_net.layers, "CONV2D")

    assert torch.equal(conv1.params["weight"], conv2.params["weight"])
    if "bias" in conv1.params or "bias" in conv2.params:
        assert "bias" in conv1.params and "bias" in conv2.params
        assert torch.equal(conv1.params["bias"], conv2.params["bias"])
