import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
from act.pipeline.verification.confignet.builders import build_act_net
from act.pipeline.verification.confignet.utils import extract_effective_spec

def test_spec_overrides_applied_and_cached_net_intact():
    try:
        from act.pipeline.verification.model_factory import ModelFactory
    except Exception as e:
        pytest.skip(f"ModelFactory import failed (likely torch missing/broken): {e}")
    try:
        factory = ModelFactory()
    except Exception as e:
        pytest.skip(f"ModelFactory unavailable (likely torch missing/broken): {e}")

    name = "mnist_robust_easy"
    base_net = factory.get_act_net(name)
    base_eff = extract_effective_spec(base_net)
    base_eps = base_eff["eps"] or 0.01
    base_y = base_eff["y_true"] or 0

    override_y = (base_y + 3) % 10  # ensure different
    override_eps = base_eps + 0.123

    model_cfg = ModelConfig(
        arch="mlp",
        template_name=name,
        input_shape=[1, 784],
        num_classes=10,
        activation="relu",
        dataset="mnist",
        seed=42,
    )
    spec_cfg = SpecConfig(eps=override_eps, norm="linf", targeted=False, true_label=override_y, assert_kind="TOP1_ROBUST")

    # Build with overrides
    act_net = build_act_net(model_cfg, spec_cfg, name="sample")
    eff = extract_effective_spec(act_net)
    assert eff["eps"] == pytest.approx(spec_cfg.eps)
    assert eff["assert_kind"] == spec_cfg.assert_kind
    assert eff["y_true"] == spec_cfg.true_label

    # Cached net should remain unchanged (different object)
    cached_net = factory.get_act_net(name)
    assert cached_net is not act_net
    cached_eff = extract_effective_spec(cached_net)
    # Ensure cached values differ to prove no mutation
    assert cached_eff["eps"] != eff["eps"] or cached_eff["y_true"] != eff["y_true"]
