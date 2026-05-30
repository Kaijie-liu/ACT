from __future__ import annotations

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    _zero_factor_hz_feasible,
    hz_from_bounds,
    lp_witness_to_input,
)
from act.back_end.hybridz_tf.representations import BoxHZ, LazyChainHZ, SparseGcZ


def _interval_bounds(hz):
    rad = hz.Gc.abs().sum(dim=1, keepdim=True) if hz.Gc.numel() else torch.zeros_like(hz.c)
    return hz.c - rad, hz.c + rad


def test_hz_from_bounds_drops_zero_width_generators_exactly():
    lb = torch.tensor([0.0, -1.0, 2.0, 5.0], dtype=torch.float64)
    ub = torch.tensor([0.0, 3.0, 2.0, 7.0], dtype=torch.float64)
    hz = hz_from_bounds(Bounds(lb=lb, ub=ub), torch.float64, torch.device("cpu"))

    assert hz.Gc.shape == (4, 2)
    assert hz.Ac.shape == (0, 2)
    got_lb, got_ub = _interval_bounds(hz)
    assert torch.allclose(got_lb.flatten(), lb)
    assert torch.allclose(got_ub.flatten(), ub)


def test_hz_from_bounds_all_zero_width_has_no_generators():
    lb = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    hz = hz_from_bounds(Bounds(lb=lb, ub=lb), torch.float64, torch.device("cpu"))

    assert hz.Gc.shape == (3, 0)
    assert hz.Ac.shape == (0, 0)
    got_lb, got_ub = _interval_bounds(hz)
    assert torch.allclose(got_lb.flatten(), lb)
    assert torch.allclose(got_ub.flatten(), lb)


def test_zero_factor_hz_feasibility_respects_eq_and_le_rows():
    lb = torch.tensor([1.0, 2.0], dtype=torch.float64)
    hz = hz_from_bounds(Bounds(lb=lb, ub=lb), torch.float64, torch.device("cpu"))

    assert _zero_factor_hz_feasible(hz)

    hz_bad_eq = hz_from_bounds(Bounds(lb=lb, ub=lb), torch.float64, torch.device("cpu"))
    hz_bad_eq.Ac = torch.zeros((1, 0), dtype=torch.float64)
    hz_bad_eq.Ab = torch.zeros((1, 0), dtype=torch.float64)
    hz_bad_eq.b = torch.tensor([[1.0]], dtype=torch.float64)
    hz_bad_eq.eq_mask = torch.tensor([True])
    assert not _zero_factor_hz_feasible(hz_bad_eq)

    hz_bad_le = hz_from_bounds(Bounds(lb=lb, ub=lb), torch.float64, torch.device("cpu"))
    hz_bad_le.Ac = torch.zeros((1, 0), dtype=torch.float64)
    hz_bad_le.Ab = torch.zeros((1, 0), dtype=torch.float64)
    hz_bad_le.b = torch.tensor([[-1.0]], dtype=torch.float64)
    hz_bad_le.eq_mask = torch.tensor([False])
    assert not _zero_factor_hz_feasible(hz_bad_le)


def test_lp_witness_to_input_with_pruned_root_generators():
    lb = torch.tensor([0.0, -1.0, 2.0, 5.0], dtype=torch.float64)
    ub = torch.tensor([0.0, 3.0, 2.0, 7.0], dtype=torch.float64)
    hz = hz_from_bounds(Bounds(lb=lb, ub=ub), torch.float64, torch.device("cpu"))

    x = lp_witness_to_input(np.array([-1.0, 1.0], dtype=np.float64), hz)
    assert np.allclose(x, [0.0, -1.0, 2.0, 7.0])


def test_boxhz_and_lazychain_report_active_generator_count():
    lb = torch.tensor([0.0, -1.0, 2.0, 5.0], dtype=torch.float64)
    ub = torch.tensor([0.0, 3.0, 2.0, 7.0], dtype=torch.float64)
    box = BoxHZ(lb, ub, dtype=torch.float64, device=torch.device("cpu"))
    chain = LazyChainHZ.from_box(box)

    assert box.ng == 2
    assert chain.n_root == 4
    assert chain.ng == 2
    hz = chain.to_full_hzono()
    assert hz.Gc.shape == (4, 2)


def test_lazychain_sparse_conv_uses_compact_active_columns():
    lb = torch.zeros(9, dtype=torch.float64)
    ub = torch.zeros(9, dtype=torch.float64)
    lb[4] = -0.1
    ub[4] = 0.1
    box = BoxHZ(lb, ub, dtype=torch.float64, device=torch.device("cpu"))
    weight = torch.ones((1, 1, 3, 3), dtype=torch.float64)

    chain = LazyChainHZ.from_box(box).with_conv(
        weight, None, stride=(1, 1), pad=0,
        in_shape=(1, 3, 3), out_shape=(1, 1, 1), out_dim=1,
    )
    sparse = chain.to_sparse_gc_z()

    assert isinstance(sparse, SparseGcZ)
    assert sparse.Gc_sparse.shape == (1, 1)
    lb_out, ub_out = sparse.bounds()
    assert torch.allclose(lb_out, torch.tensor([-0.1], dtype=torch.float64))
    assert torch.allclose(ub_out, torch.tensor([0.1], dtype=torch.float64))


if __name__ == "__main__":
    test_hz_from_bounds_drops_zero_width_generators_exactly()
    test_hz_from_bounds_all_zero_width_has_no_generators()
    test_zero_factor_hz_feasibility_respects_eq_and_le_rows()
    test_lp_witness_to_input_with_pruned_root_generators()
    test_boxhz_and_lazychain_report_active_generator_count()
    test_lazychain_sparse_conv_uses_compact_active_columns()
    print("OK: zero-width input prune tests pass")
