import torch
from types import SimpleNamespace

from act.back_end.solver.solver_hz import _assert_is_order_only_zero_threshold


def _assert_layer(kind, **params):
    return SimpleNamespace(params={"kind": kind, **params})


def test_pairwise_zero_threshold_unsafe_linear_is_order_only():
    C = torch.tensor([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    d = torch.zeros(2)
    assert _assert_is_order_only_zero_threshold(
        _assert_layer("UNSAFE_LINEAR", c=C, d=d)
    )


def test_nonzero_margin_is_not_order_only():
    C = torch.tensor([[1.0, -1.0, 0.0]])
    d = torch.tensor([0.01])
    assert not _assert_is_order_only_zero_threshold(
        _assert_layer("UNSAFE_LINEAR", c=C, d=d)
    )


def test_arbitrary_linear_row_is_not_order_only():
    C = torch.tensor([[1.0, -0.5, -0.5]])
    d = torch.zeros(1)
    assert not _assert_is_order_only_zero_threshold(
        _assert_layer("UNSAFE_LINEAR", c=C, d=d)
    )

