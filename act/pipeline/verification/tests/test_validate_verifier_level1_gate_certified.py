# act/pipeline/verification/tests/test_validate_verifier_level1_gate_certified.py
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.config_net_runtime import RuntimeConfigNet
from act.pipeline.verification.validation import (
    InputSamplingPlan,
    assert_satisfies_input_spec,
    hash_tensor,
    run_level1_counterexample_check,
    sample_concrete_inputs,
)


def _spec_meta(spec_layer):
    meta = (spec_layer.meta or {}).copy()
    params = getattr(spec_layer, "params", None) or {}
    for key in ("lb", "ub", "center", "eps"):
        if key in params:
            meta[key] = params[key]
    return meta


def test_sampling_deterministic_and_satisfies_spec():
    cfg = RuntimeConfigNet()
    sample = cfg.sample(seed=0, arch="mlp")
    plan = InputSamplingPlan(k_center=1, k_random=2, k_boundary=1)

    inputs1 = sample_concrete_inputs(
        sample.act_net, seed=123, plan=plan, device="cpu", dtype=torch.float64, require_satisfy=True
    )
    inputs2 = sample_concrete_inputs(
        sample.act_net, seed=123, plan=plan, device="cpu", dtype=torch.float64, require_satisfy=True
    )

    assert [hash_tensor(x) for x in inputs1] == [hash_tensor(x) for x in inputs2]

    spec_meta = _spec_meta(sample.input_specs[0])
    for tensor in inputs1:
        assert_satisfies_input_spec(tensor, spec_meta)


def test_counterexample_gate_marks_non_certified():
    cfg = RuntimeConfigNet()
    sample = cfg.sample(seed=1, arch="mlp")
    plan = InputSamplingPlan(k_center=1, k_random=0, k_boundary=0)

    center_input = sample_concrete_inputs(
        sample.act_net, seed=7, plan=plan, device="cpu", dtype=torch.float64, require_satisfy=True
    )[0]

    model = sample.torch_model.to(device="cpu", dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        logits = model(center_input)
    if isinstance(logits, dict):
        if "output" in logits:
            logits = logits["output"]
        elif "logits" in logits:
            logits = logits["logits"]
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = int(torch.argmax(logits, dim=1 if logits.dim() > 1 else 0).item())
    wrong_label = (pred + 1) % logits.shape[-1]

    sample.assert_layer.meta = dict(sample.assert_layer.meta or {})
    sample.assert_layer.meta["kind"] = "TOP1_ROBUST"
    sample.assert_layer.meta["y_true"] = wrong_label

    result = run_level1_counterexample_check(
        sample,
        seed=99,
        plan=plan,
        device="cpu",
        dtype=torch.float64,
    )

    assert result.status == "COUNTEREXAMPLE_FOUND"
    assert result.counterexamples
    ce = result.counterexamples[0]
    assert ce.input_hash
    assert ce.true_label == wrong_label
    assert ce.kind
