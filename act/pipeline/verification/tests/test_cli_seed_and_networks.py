import sys
import subprocess


def test_cli_accepts_seed_and_space_networks():
    cmd = [
        sys.executable,
        "-m",
        "act.pipeline",
        "--validate-verifier",
        "--mode",
        "bounds",
        "--confignet-source",
        "runtime",
        "--networks",
        "mlp",
        "--tf-modes",
        "interval",
        "--samples",
        "2",
        "--seed",
        "7",
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--runtime-smoke",  # fast exit path, still parses args
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "unrecognized arguments: --seed" not in (proc.stderr or "")
