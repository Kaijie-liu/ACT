"""Ordinary original bounds and packed input-column reconstruction."""
import numpy as np
import pytest
import torch
from act.back_end.core import Bounds
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831.c100_original_input_v2 import build_input
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('radius',[[.125,.25,.375],[.125,0.,.25],[0.,.125,0.]])
def test_complete_fresh_input_matches_actual_TF_seed(radius):
    center=torch.tensor([[.11,.33,.71]],dtype=torch.float64);r=torch.tensor([radius],dtype=torch.float64)
    bounds=Bounds(center-r,center+r);tf=HybridzTF();actual=tf._sparse_from_bounds(bounds)
    hz,proof=build_input(bounds,actual.frame_id,pool=WorkPool(256_000_000),enabled=True)
    assert source_digest(hz)==source_digest(actual)
    assert proof['complete_input_coordinates']==3 and proof['active_input_columns']==np.count_nonzero(radius)


def test_invalid_box_default_off_and_budget_fail_closed():
    assert build_input(None,None,pool=None) is None
    b=Bounds(torch.tensor([[1.]],dtype=torch.float64),torch.tensor([[0.]],dtype=torch.float64))
    with pytest.raises(ValueError):build_input(b,1,pool=WorkPool(256_000_000),enabled=True)
    with pytest.raises(MemoryError):build_input(b,1,pool=WorkPool(0),enabled=True)
