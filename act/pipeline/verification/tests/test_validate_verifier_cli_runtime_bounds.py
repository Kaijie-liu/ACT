import argparse

import pytest

torch = pytest.importorskip("torch")

from act.pipeline import cli as cli_mod
from act.pipeline.verification.validate_verifier import VerificationValidator, _print_level2_bounds_summary


def test_cli_bounds_runtime_runs_validation_and_prints_summary(capsys):
    args = argparse.Namespace(
        mode="bounds",
        device="cpu",
        dtype="float64",
        networks=None,
        solvers=["gurobi", "torchlp"],
        tf_modes=["interval"],
        samples=2,
        ignore_errors=True,
        confignet_source="runtime",
        runtime_smoke=False,
        seed=0,
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_mod.cmd_validate_verifier(args)

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "[L2][bounds]" in out
    assert "overall=" in out
    assert "interval:" in out


def test_validate_comprehensive_uses_runtime_cases():
    validator = VerificationValidator(device="cpu", dtype=torch.float64)
    summary = validator.validate_comprehensive(
        networks=None,
        solvers=["torchlp"],
        tf_modes=["interval"],
        num_samples=2,
        confignet_source="runtime",
        seed=0,
    )
    bounds_results = summary["level3_bounds"]["results"]
    assert any(r.get("case_name", "").startswith("runtime_") for r in bounds_results)
