import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.validate_verifier import VerificationValidator


def test_runtime_smoke_runs():
    validator = VerificationValidator(device="cpu", dtype=torch.float32)
    validator.runtime_smoke()  # Should not raise
