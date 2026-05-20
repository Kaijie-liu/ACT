"""Shared Generator Merge (SGM) for ResNet-style residual ``y = x + f(x)``.

Background
----------
Standard HZ addition (``hz_minkowski_sum``) treats the two summands as
independent and concatenates their generator matrices, which doubles
``ng`` at every skip-connection. When ``x`` and ``f(x)`` descend from a
common upstream HZ and inherit the same continuous-generator factors,
that doubling is over-approximation: the sum ``x + f(x)`` should still
reference the SAME underlying generators, not two independent copies.

This module provides the SGM detection (``shares_generator``) and a
merge routine (``hz_sgm_add``) that keeps a single shared block when
the two summands' ``Gc`` matrices are pointwise equal (the strong form
of "fully shared", appropriate for HZ representations that have not
been transformed apart on either branch). For the broader notion of
shared ancestry that survives transformation, callers must track
``_base_ng`` / ``_base_nb`` and call into the HZ class's own ``add``
method (HyZor's HybridZonotope has this; ACT's HZono does not yet).
"""

from __future__ import annotations
import torch
from act.back_end.solver.solver_hz import HZono, hz_minkowski_sum


def shares_generator(hz_x: HZono, hz_y: HZono, *, tol: float = 1e-12) -> bool:
    """Detect whether two HZs share the entire continuous-generator block.

    Returns True iff ``hz_x.Gc`` and ``hz_y.Gc`` have the same shape AND
    are pointwise equal to within ``tol``. An all-zero (or empty) Gc on
    either side returns False (no meaningful sharing).

    The all-zero / empty-Gc guard is required because torch's
    ``.abs().max()`` raises ``RuntimeError`` on a zero-element tensor;
    the equivalent guard in HyZor's repo (``__init__.py`` line ~1314)
    was a latent crash discovered when the ResNet ``add`` dispatch hit
    a degenerate HZ. Keep this guard in any SGM port.
    """
    if hz_x.Gc.shape != hz_y.Gc.shape:
        return False
    if hz_x.Gc.numel() == 0:
        return False
    return bool((hz_x.Gc - hz_y.Gc).abs().max().item() < tol)


def hz_sgm_add(hz_x: HZono, hz_y: HZono) -> HZono:
    """Sum two HZs, sharing the continuous-generator block when possible.

    Falls back to ``hz_minkowski_sum`` when ``shares_generator`` is False.

    When the Gc matrices match exactly:
      - ``c <- hz_x.c + hz_y.c``                 (centers add)
      - ``Gc <- hz_x.Gc``                        (single shared block kept)
      - ``Gb <- cat([hz_x.Gb, hz_y.Gb], dim=1)`` (binary tails independent)
      - ``Ac``, ``Ab``, ``b`` stacked block-diagonally across the two
        branches' constraints (continuous part shared columns, binary
        part disjoint columns), matching the contract Minkowski sum
        would have produced minus the duplicated Gc.

    Soundness: when ``hz_x.Gc == hz_y.Gc`` as matrices, substituting any
    feasible factor assignment into both branches yields the same
    contribution from Gc; the formula above is the algebraic identity,
    not an over-approximation.
    """
    if not shares_generator(hz_x, hz_y):
        return hz_minkowski_sum(hz_x, hz_y)

    dtype = hz_x.c.dtype
    device = hz_x.c.device

    new_c = hz_x.c + hz_y.c.to(dtype=dtype, device=device)
    new_Gc = hz_x.Gc  # single shared block

    nb_x = hz_x.Gb.shape[1]
    nb_y = hz_y.Gb.shape[1]
    new_Gb = torch.cat(
        [hz_x.Gb, hz_y.Gb.to(dtype=dtype, device=device)], dim=1
    )

    nc_x = hz_x.Ac.shape[0]
    nc_y = hz_y.Ac.shape[0]
    ng = new_Gc.shape[1]

    # Ac: shared continuous columns means both branches' Ac rows
    # already index into the same continuous factor space.
    # Stack vertically (each branch contributes its own equality rows).
    Ac_x = hz_x.Ac if nc_x else hz_x.Ac.new_zeros(0, ng)
    Ac_y = hz_y.Ac.to(dtype=dtype, device=device) if nc_y \
        else hz_y.Ac.new_zeros(0, ng)
    new_Ac = torch.cat([Ac_x, Ac_y], dim=0)

    # Ab: x's binary cols are [0:nb_x), y's are [nb_x:nb_x+nb_y).
    # Pad x's Ab on right with zeros for y's binaries, and vice versa.
    Ab_x_pad = torch.cat(
        [hz_x.Ab, hz_x.Ab.new_zeros(nc_x, nb_y)], dim=1
    ) if nc_x else hz_x.Ab.new_zeros(0, nb_x + nb_y)
    Ab_y_pad = torch.cat(
        [hz_y.Ab.new_zeros(nc_y, nb_x),
         hz_y.Ab.to(dtype=dtype, device=device)], dim=1
    ) if nc_y else hz_y.Ab.new_zeros(0, nb_x + nb_y)
    new_Ab = torch.cat([Ab_x_pad, Ab_y_pad], dim=0)

    new_b = torch.cat(
        [hz_x.b, hz_y.b.to(dtype=dtype, device=device)], dim=0
    )

    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=new_Ac, Ab=new_Ab, b=new_b,
    )


# --- Self-tests (run with: python -m act.back_end.hybridz_tf.algorithms.sgm) ---


def _make_hz(c, Gc, Gb=None, Ac=None, Ab=None, b=None):
    n = int(c.shape[0])
    ng = int(Gc.shape[1])
    Gb = Gb if Gb is not None else c.new_zeros(n, 0)
    Ac = Ac if Ac is not None else c.new_zeros(0, ng)
    Ab = Ab if Ab is not None else c.new_zeros(0, int(Gb.shape[1]))
    b = b if b is not None else c.new_zeros(0, 1)
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)


def _test_shares_generator_match():
    c = torch.zeros(3, 1)
    Gc = torch.eye(3)
    hz_x = _make_hz(c, Gc)
    hz_y = _make_hz(c.clone(), Gc.clone())
    assert shares_generator(hz_x, hz_y) is True


def _test_shares_generator_shape_mismatch():
    Gc1 = torch.eye(3)
    Gc2 = torch.eye(4)
    hz_x = _make_hz(torch.zeros(3, 1), Gc1)
    hz_y = _make_hz(torch.zeros(4, 1), Gc2)
    assert shares_generator(hz_x, hz_y) is False


def _test_shares_generator_value_mismatch():
    Gc = torch.eye(3)
    Gc_d = Gc.clone(); Gc_d[0, 0] = 2.0
    hz_x = _make_hz(torch.zeros(3, 1), Gc)
    hz_y = _make_hz(torch.zeros(3, 1), Gc_d)
    assert shares_generator(hz_x, hz_y) is False


def _test_shares_generator_empty_gc():
    """Regression test for the empty-Gc `.max()` crash."""
    Gc_empty = torch.zeros(3, 0)
    hz_x = _make_hz(torch.zeros(3, 1), Gc_empty)
    hz_y = _make_hz(torch.zeros(3, 1), Gc_empty)
    assert shares_generator(hz_x, hz_y) is False  # not a crash


def _test_hz_sgm_add_shared():
    Gc = torch.eye(3)
    c_x = torch.ones(3, 1)
    c_y = 2 * torch.ones(3, 1)
    hz_x = _make_hz(c_x, Gc.clone())
    hz_y = _make_hz(c_y, Gc.clone())
    out = hz_sgm_add(hz_x, hz_y)
    assert out.c.allclose(torch.full_like(c_x, 3.0))
    assert out.Gc.shape == (3, 3)  # NOT 3, 6 — that's the SGM win
    assert out.Gb.shape == (3, 0)


def _test_hz_sgm_add_independent_falls_back_to_minkowski():
    hz_x = _make_hz(torch.zeros(3, 1), torch.eye(3))
    hz_y = _make_hz(torch.zeros(3, 1), 2 * torch.eye(3))
    out = hz_sgm_add(hz_x, hz_y)
    # falls back to Minkowski sum: ng doubles
    assert out.Gc.shape == (3, 6)


if __name__ == "__main__":
    _test_shares_generator_match()
    _test_shares_generator_shape_mismatch()
    _test_shares_generator_value_mismatch()
    _test_shares_generator_empty_gc()
    _test_hz_sgm_add_shared()
    _test_hz_sgm_add_independent_falls_back_to_minkowski()
    print("OK: sgm tests pass")
