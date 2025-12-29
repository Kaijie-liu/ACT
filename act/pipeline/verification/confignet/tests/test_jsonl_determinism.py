import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")

from act.pipeline.verification.run_confignet_sample import main as run_main


def _run_cli(out_path: Path, seed: int):
    argv_backup = list(__import__("sys").argv)
    try:
        __import__("sys").argv = [
            "run_confignet_sample",
            "--seed",
            str(seed),
            "--n",
            "3",
            "--out",
            str(out_path),
            "--deterministic-jsonl",
        ]
        run_main()
    finally:
        __import__("sys").argv = argv_backup


def test_deterministic_jsonl_stable():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "a.jsonl"
        f2 = Path(tmpdir) / "b.jsonl"
        _run_cli(f1, seed=0)
        _run_cli(f2, seed=0)
        assert f1.read_bytes() == f2.read_bytes()
        line = json.loads(f1.read_text().strip().splitlines()[0])
        for key in ("run_seed", "case_id", "model_config", "spec_config", "hash", "git_sha"):
            assert key in line
        assert "timestamp" not in line


def test_different_seed_changes_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "a.jsonl"
        f2 = Path(tmpdir) / "b.jsonl"
        _run_cli(f1, seed=0)
        _run_cli(f2, seed=1)
        assert f1.read_bytes() != f2.read_bytes()
