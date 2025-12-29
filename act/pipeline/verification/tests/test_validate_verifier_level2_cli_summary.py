# act/pipeline/verification/tests/test_validate_verifier_level2_cli_summary.py
import pytest

torch = pytest.importorskip("torch")

from act.pipeline.verification.validate_verifier import (
    VerificationValidator,
    _print_level2_bounds_summary,
)


def test_level2_bounds_cli_summary_runtime(capsys):
    validator = VerificationValidator(device="cpu", dtype=torch.float64)
    summary = validator.validate_bounds(
        networks=None,
        tf_modes=["interval"],
        num_samples=3,
        confignet_source="runtime",
        seed=0,
    )

    _print_level2_bounds_summary(summary)
    output = capsys.readouterr().out

    assert "case=" in output
    assert "overall=" in output
    assert "interval:" in output
