import numpy as np
import pytest

from experiments.neural_hz_20260831.run_corrected_phase_support_census_v1 import phase_support


def test_disjoint_phase_partition_and_channel_union():
    result = phase_support([-2, 0, -1, 1], [-1, 0, 2, 3], 2)
    assert (result["negative"], result["positive"], result["unstable"]) == (2, 1, 1)
    assert result["selected_rows"] == 2 and result["selected_channels"] == [1]


def test_positive_probe_is_first_eight_and_never_omits_unstable():
    result = phase_support([1.] * 12 + [-1.], [2.] * 13, 13)
    assert result["selected_channels"] == list(range(8)) + [12]


@pytest.mark.parametrize("lb,ub,channels", [([1], [0], 1), ([np.nan], [1], 1),
    ([0], [np.inf], 1), ([0], [1, 2], 1), ([0, 0], [1, 1], 3)])
def test_invalid_intervals_fail_closed(lb, ub, channels):
    with pytest.raises(ValueError):
        phase_support(lb, ub, channels)
