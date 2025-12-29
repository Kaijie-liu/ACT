# act/pipeline/verification/tests/test_validate_verifier_runtime_bounds_source.py
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.validate_verifier import VerificationValidator


def test_validate_bounds_runtime_uses_runtime_names():
    validator = VerificationValidator(device="cpu", dtype=torch.float64)
    summary = validator.validate_bounds(
        networks=["mlp"],
        tf_modes=["interval"],
        num_samples=2,
        confignet_source="runtime",
        seed=0,
    )
    first = summary["results"][0]
    assert str(first["case_name"]).startswith("runtime_")
