# act/pipeline/verification/tests/test_validate_verifier_level2_interval_bounds.py
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.config_net_runtime import RuntimeConfigNet
from act.pipeline.verification.validation import InputSamplingPlan, run_level2_bounds_check


def test_interval_bounds_basic_schema():
    sample = RuntimeConfigNet().sample(seed=0, arch="mlp")
    plan = InputSamplingPlan(k_center=1, k_random=1, k_boundary=1)

    result = run_level2_bounds_check(
        sample,
        seed=123,
        plan=plan,
        device="cpu",
        dtype=torch.float64,
        tf_modes=("interval",),
    )

    assert result["case_name"] == sample.name
    assert result["overall_status"] in {"PASS", "VIOLATION_FOUND"}
    assert len(result["per_mode"]) == 1
    pm = result["per_mode"][0]
    assert pm["metadata"]["tf_mode"] == "interval"
    if pm["violations"]:
        v = pm["violations"][0]
        for key in (
            "layer_id",
            "layer_kind",
            "num_violations",
            "total_neurons",
            "concrete_min",
            "concrete_max",
            "abstract_lb_min",
            "abstract_ub_max",
        ):
            assert key in v


def test_unsupported_mode_errors():
    sample = RuntimeConfigNet().sample(seed=1, arch="mlp")
    plan = InputSamplingPlan(k_center=1, k_random=0, k_boundary=0)

    result = run_level2_bounds_check(
        sample,
        seed=999,
        plan=plan,
        device="cpu",
        dtype=torch.float64,
        tf_modes=("zonotope",),
    )

    assert result["overall_status"] == "ERROR"
    pm = result["per_mode"][0]
    assert pm["status"] == "ERROR"
    assert pm["metadata"]["error_type"] == "UnsupportedTFMode"
    assert "interval" in pm["metadata"]["supported_modes"]
