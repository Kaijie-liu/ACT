#!/usr/bin/env python3
#===- tests/test_confignet_cli_level1.py - CLI Level1 Test ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from pathlib import Path

from act.pipeline.confignet.cli import main


def test_cli_level1_runs(tmp_path: Path) -> None:
    out_jsonl = tmp_path / "level1.jsonl"
    code = main([
        "level1",
        "--num", "1",
        "--seed", "0",
        "--n-inputs", "2",
        "--device", "cpu",
        "--dtype", "float64",
        "--out_jsonl", str(out_jsonl),
    ])
    assert code == 0
    assert out_jsonl.exists()
