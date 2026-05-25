#===- tests/test_hz_representations.py - HZ representation flavors ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Smoke tests for the four HZ representation flavors (HZono / BoxHZ /
#   LazyChainHZ / SparseGcZ) in act/back_end/hybridz_tf/representations.py.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import numpy as np
import torch

from act.back_end.solver.solver_hz import HZono, lp_witness_to_input
from act.back_end.hybridz_tf.representations import (
    BoxHZ,
    LazyChainHZ,
    SparseGcZ,
)


def test_boxhz_basic():
    box = BoxHZ(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    assert box.dim == 2 and box.ng == 2 and box.nb == 0 and box.nc == 0
    lb, ub = box.bounds()
    assert lb.tolist() == [-1.0, 0.0] and ub.tolist() == [1.0, 2.0]
    hz = box.to_hzono()
    assert isinstance(hz, HZono)
    assert hz.Gc.shape == (2, 2)


def test_boxhz_witness_replay_candidate_is_in_box_center():
    box = BoxHZ(torch.tensor([-2.0, 0.0]), torch.tensor([1.0, 4.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    # BoxHZ has no final-factor-to-input inverse after downstream reduction.
    # A center proposal is concrete and replay-safe regardless of LP layout.
    x = lp_witness_to_input(np.array([1.0]), box)
    assert np.allclose(x, [-0.5, 2.0])
    assert np.all(x >= np.array([-2.0, 0.0]))
    assert np.all(x <= np.array([1.0, 4.0]))


def test_lazychain_dense():
    box = BoxHZ(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    chain = LazyChainHZ.from_box(box)
    W = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    chain2 = chain.with_dense(W, None)
    lb, ub = chain2.bounds()
    # y = W x with x in [-1,1]^2 → y_0 in [-2,2], y_1 in [-1,1]
    assert lb.tolist() == [-2.0, -1.0]
    assert ub.tolist() == [2.0, 1.0]


def test_lazychain_materialize_when_small():
    box = BoxHZ(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    chain = LazyChainHZ.from_box(box).with_dense(
        torch.eye(2, dtype=torch.float64), None
    )
    # dim=2 << materialize_dim_cap; should materialise
    assert chain.can_materialize()
    hz = chain.to_full_hzono()
    assert isinstance(hz, HZono)
    assert hz.Gc.shape == (2, 2)


def test_sparse_gc_bounds():
    c = torch.tensor([0.0, 0.0], dtype=torch.float64)
    ind = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    val = torch.tensor([1.0, 1.0], dtype=torch.float64)
    Gc_sp = torch.sparse_coo_tensor(ind, val, (2, 2))
    s = SparseGcZ(c=c, Gc_sparse=Gc_sp,
                  dtype=torch.float64, device=torch.device("cpu"))
    lb, ub = s.bounds()
    assert lb.tolist() == [-1.0, -1.0] and ub.tolist() == [1.0, 1.0]


if __name__ == "__main__":
    test_boxhz_basic()
    test_boxhz_witness_replay_candidate_is_in_box_center()
    test_lazychain_dense()
    test_lazychain_materialize_when_small()
    test_sparse_gc_bounds()
    print("OK: representations tests pass")
