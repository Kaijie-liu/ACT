#!/usr/bin/env python3
#===- tests/test_cli_confignet_routing.py - CLI Routing ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations


def test_pipeline_cli_routes_to_confignet(monkeypatch) -> None:
    called = {}

    def _fake_confignet_main(argv=None):
        called["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr("act.pipeline.confignet.main", _fake_confignet_main)

    from act.pipeline.cli import main as pipeline_main

    code = pipeline_main(
        [
            "confignet",
            "l1l2",
            "--num",
            "1",
            "--seed",
            "0",
            "--n-inputs",
            "1",
            "--tf-modes",
            "interval",
            "--out_jsonl",
            "/tmp/x.jsonl",
        ]
    )
    assert code == 0
    assert called["argv"][0] == "l1l2"


def test_pipeline_cli_non_confignet_routes_to_pipeline(monkeypatch) -> None:
    called = {"list": 0}

    def _fake_initialize(_args):
        return None

    def _fake_list_available(_creator):
        called["list"] += 1

    monkeypatch.setattr("act.pipeline.cli.initialize_from_args", _fake_initialize)
    monkeypatch.setattr("act.pipeline.cli.cmd_list_available", _fake_list_available)

    from act.pipeline.cli import main as pipeline_main

    code = pipeline_main(["--list"])
    assert code == 0
    assert called["list"] == 1
