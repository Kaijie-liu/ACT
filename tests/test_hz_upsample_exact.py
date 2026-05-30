import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.representations import BoxHZ, LazyChainHZ
from act.back_end.solver.solver_hz import (
    _hz_upsample_nearest_nchw,
    hz_from_bounds,
)


def test_nearest_upsample_preserves_hz_factors_exactly():
    lb = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    ub = torch.tensor([0.2, 1.4, 2.6, 3.8], dtype=torch.float64)
    hz = hz_from_bounds(Bounds(lb=lb, ub=ub), torch.float64, torch.device("cpu"))

    out = _hz_upsample_nearest_nchw(
        hz,
        {
            "mode": "nearest",
            "input_shape": (1, 1, 2, 2),
            "output_shape": (1, 1, 4, 4),
        },
    )

    idx = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
        dtype=torch.long,
    )
    assert out.dim == 16
    assert out.ng == hz.ng
    assert torch.allclose(out.c[:, 0], hz.c[idx, 0])
    assert torch.allclose(out.Gc, hz.Gc[idx])
    assert out.nc == hz.nc


def test_nearest_upsample_materializes_small_lazy_chain_without_boxing():
    box = BoxHZ(
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    W = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    lazy = LazyChainHZ.from_box(box).with_dense(W, None)

    out = _hz_upsample_nearest_nchw(
        lazy,
        {
            "mode": "nearest",
            "input_shape": (1, 1, 2, 2),
            "output_shape": (1, 1, 4, 4),
        },
    )

    # Only two root dimensions have non-zero radius. Exact upsample must
    # duplicate rows while preserving those two shared factors, not turn the
    # 16 output pixels into 16 independent interval generators.
    assert out.dim == 16
    assert out.ng == 2
    assert out.Gc.shape == (16, 2)


if __name__ == "__main__":
    test_nearest_upsample_preserves_hz_factors_exactly()
    test_nearest_upsample_materializes_small_lazy_chain_without_boxing()
    print("OK: exact nearest upsample tests pass")
