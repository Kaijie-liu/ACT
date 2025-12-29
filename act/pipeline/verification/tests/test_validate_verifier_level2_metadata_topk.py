# act/pipeline/verification/tests/test_validate_verifier_level2_metadata_topk.py
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.config_net_runtime import RuntimeConfigNet
from act.pipeline.verification.validation import InputSamplingPlan, run_level2_bounds_check


def test_level2_metadata_and_topk_fields():
    sample = RuntimeConfigNet().sample(seed=0, arch="mlp")
    plan = InputSamplingPlan(k_center=1, k_random=1, k_boundary=1)
    res = run_level2_bounds_check(
        sample,
        seed=42,
        plan=plan,
        device="cpu",
        dtype=torch.float64,
        tf_modes=("interval",),
        atol=1e-6,
    )
    pm = res["per_mode"][0]
    meta = pm["metadata"]
    assert meta["tf_mode"] == "interval"
    assert "confignet_source" in meta
    assert "num_samples" in meta
    assert "plan" in meta
    assert "atol" in meta
    assert meta["num_samples"] == plan.k_center + plan.k_random + plan.k_boundary

    viols = pm["violations"]
    if viols:
        v = viols[0]
        for key in ("sample_idx", "layer_id", "num_violations", "total_neurons", "concrete_min", "concrete_max", "lb_min", "ub_max"):
            assert key in v
        assert "topk" in v
        if v["topk"]:
            top = v["topk"][0]
            for key in ("idx", "value", "lb", "ub", "gap"):
                assert key in top
