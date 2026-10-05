import numpy as np
import pytest

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import contract as v1
from experiments.neural_hz_20260831.c5_live_value_union_contraction_v2 import contract as v2
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


@pytest.mark.parametrize("divisor", [8., 7.3])
@pytest.mark.parametrize("stride,dilation,padding", [(1, 1, 1), (2, 1, 1), (1, 2, 2)])
def test_every_scalar_matches_v1_bitwise_including_nondyadic_payload(divisor, stride, dilation, padding):
    inner = ImplicitConv2DOp((np.arange(36).reshape(2, 2, 3, 3) % 7 - 3) / divisor,
                            (1, 2, 5, 5), stride=stride, dilation=dilation, padding=padding)
    outer = ImplicitConv2DOp((np.arange(54).reshape(3, 2, 3, 3) % 5 - 2) / divisor, inner.output_shape, padding=1)
    s = source(inner.shape[1])
    scale = np.broadcast_to(np.array([.123, -.731])[None, :, None, None], inner.output_shape).copy().reshape(-1)
    rows = np.arange(outer.shape[0])
    old, new = v1(s, inner, scale, outer, rows), v2(s, inner, scale, outer, rows)
    assert np.array_equal(old.matrix.data, new.matrix.data)
    assert np.array_equal(old.matrix.indices, new.matrix.indices)
    assert np.array_equal(old.matrix.indptr, new.matrix.indptr)
    assert new.stats["actual_channel_products"] <= old.stats["actual_channel_products"]
    assert new.stats["v1_spatial_products"] == old.stats["actual_channel_products"]
    assert np.array_equal(old.apply().Gc.toarray(), new.apply().Gc.toarray())


def test_repeated_spatial_coefficients_are_computed_once():
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 8, 8))
    s = source(op.shape[1])
    s.c[:] = 1.
    out = v2(s, op, np.ones(op.shape[0]), op, np.arange(op.shape[0]))
    assert out.stats["actual_channel_products"] == 8
    assert out.stats["unrestricted_spatial_channel_products"] == 512
    assert out.stats["quarter_product_gate"] and not out.stats["compile_cache_retained_after_return"]


@pytest.mark.parametrize("mode", ["scale", "mask", "cap"])
def test_nonstationarity_and_caps_reject(mode):
    mask = np.ones(8, dtype=bool)
    if mode == "mask":
        mask[1] = False
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 2, 2), row_mask=mask)
    scale = np.ones(8)
    if mode == "scale":
        scale[1] = np.nextafter(1., 2.)
    with pytest.raises((ValueError, MemoryError)):
        v2(source(8), op, scale, op, np.arange(8), max_products=0 if mode == "cap" else 200_000_000)
