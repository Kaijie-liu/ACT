import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    _eq_mask_of,
    _hz_sign_convex_hull,
    hz_from_bounds,
)


def _holds(hz, xi):
    xi = torch.tensor(xi, dtype=hz.c.dtype, device=hz.c.device).view(-1, 1)
    p = hz.ng
    q = hz.nb
    lhs = hz.Ac @ xi[:p] + hz.Ab @ xi[p:p + q]
    em = _eq_mask_of(hz).view(-1)
    if em.numel() == 0:
        return True
    eq_ok = bool((torch.abs(lhs[em] - hz.b[em]) <= 1e-9).all().item()) if bool(em.any().item()) else True
    le = ~em
    le_ok = bool((lhs[le] <= hz.b[le] + 1e-9).all().item()) if bool(le.any().item()) else True
    return eq_ok and le_ok


def test_sign_hull_contains_exact_sign_points():
    bounds = Bounds(
        lb=torch.tensor([[-2.0]], dtype=torch.float64),
        ub=torch.tensor([[3.0]], dtype=torch.float64),
    )
    hz = hz_from_bounds(bounds, dtype=torch.float64, device=torch.device("cpu"))
    out = _hz_sign_convex_hull(hz, bounds)
    # Factors are [input_xi, sign_eta].  x=-2 -> xi=-1, y=-1.
    assert _holds(out, [-1.0, -1.0])
    # x=3 -> xi=1, y=1.
    assert _holds(out, [1.0, 1.0])
    # torch.sign(0)=0; xi solves 0 = 0.5 + 2.5*xi.
    assert _holds(out, [-0.2, 0.0])


def test_sign_hull_rejects_opposite_stable_corner():
    bounds = Bounds(
        lb=torch.tensor([[-2.0]], dtype=torch.float64),
        ub=torch.tensor([[3.0]], dtype=torch.float64),
    )
    hz = hz_from_bounds(bounds, dtype=torch.float64, device=torch.device("cpu"))
    out = _hz_sign_convex_hull(hz, bounds)
    # At x=3 the hull should not allow y=-1.
    assert not _holds(out, [1.0, -1.0])

