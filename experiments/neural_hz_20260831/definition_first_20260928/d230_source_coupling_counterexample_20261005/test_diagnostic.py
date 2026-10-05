"""One frozen mathematical diagnostic; not model/native-runtime qualification.

Do not import, collect, compile, or execute before the D230 freeze.
"""

import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d230_source_coupling_counterexample_20261005 import diagnostic


RUN = (Path(__file__).resolve().parents[2]
       / "results/d230_source_coupling_counterexample_20261005_v1")


def _record_file(name, payload):
    """Isolated evidence writer, relocatable without changing mathematics."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def test_source_mean_binding_counterexample():
    with pytest.raises(diagnostic.bridge.Rejected):
        diagnostic.run()
    result = diagnostic.run(enabled=True)
    assert result["d229_complete_receiver_upper"] == "5/8"
    assert result["d062_four_plane_supports"] == ["0", "0", "1/8", "1/8"]
    assert result["fake_readout"] == "3/20"
    assert result["shared_source_coordinate_control"]["upper"] == "1/8"
    assert result["shared_source_coordinate_control"]["restored_old_reference_only"]
    assert result["original_h_relaxation_fake_feasible"]
    assert not result["formal_score_changed"]
    _record_file("summary.json", result)
