from types import SimpleNamespace

import numpy as np
import pytest

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import geometry_cost, screen, stationary_scale


def graph():
    def node(lid, kind, **params):
        return SimpleNamespace(id=lid, kind=kind, params=params, out_vars=list(range(16)))
    layers = [node(0, "RELU"), node(1, "CONV2D", weight=np.ones((4, 2, 1, 1)), input_shape=(1, 2, 2, 2)),
              node(2, "SCALE", a=np.full(16, 2.)), node(3, "BIAS", c=np.ones(16)),
              node(4, "ADD"), node(5, "ADD"),
              node(6, "CONV2D", weight=np.ones((3, 4, 1, 1)), input_shape=(1, 4, 2, 2))]
    return layers, {0: [], 1: [0], 2: [1], 3: [2], 4: [3, 3], 5: [4, 3], 6: [5]}


def test_nested_add_expands_every_operand_and_never_authorizes():
    layers, preds = graph()
    result = screen(layers, preds, 6)
    assert result["complete_path_count"] == 3 and result["source_multiplicities"] == {0: 3}
    assert [o["adds"] for o in result["occurrences"]] == [[4, 5], [4, 5], [5]]
    assert all(o["explicit_bias_events"] == [3] for o in result["occurrences"])
    assert result["total_contraction_products_no_cache_discount"] == 72
    assert result["necessary_resource_caps_pass"] and not result["positive_authorization"]


@pytest.mark.parametrize("mutation", ["post_outer_add", "nonstationary", "cycle", "third_conv", "unknown"])
def test_invalid_core_rejects_completely(mutation):
    layers, preds = graph()
    terminal = 6
    if mutation == "post_outer_add":
        layers.append(SimpleNamespace(id=7, kind="ADD"))
        preds[7], terminal = [6, 6], 7
    elif mutation == "nonstationary":
        layers[2].params["a"][1] = 3.
    elif mutation == "cycle":
        preds[1] = [5]
    elif mutation == "third_conv":
        layers[2].kind = "CONV2D"
    else:
        layers[3].kind = "RESHAPE"
    with pytest.raises(ValueError):
        screen(layers, preds, terminal)


def test_channel_group_intersection_formula_matches_explicit_channel_sum():
    inner = ImplicitConv2DOp(np.ones((12, 2, 1, 1)), (1, 6, 2, 2), groups=3)
    outer = ImplicitConv2DOp(np.ones((8, 3, 1, 1)), (1, 12, 2, 2), groups=4)
    cost = geometry_cost(inner, outer)
    explicit = sum(1 for o in range(8) for m in range(12) for i in range(6)
                   if m // 3 == o // 2 and i // 2 == m // 4)
    assert cost["contraction_products"] == explicit == 48
    assert cost["coefficient_bytes_lower_bound"] == 8 * cost["coefficient_entries"]


def test_stationarity_requires_exact_values_not_allclose():
    value = np.ones((1, 2, 2, 2))
    value[0, 0, 0, 1] = np.nextafter(1., 2.)
    with pytest.raises(ValueError, match="nonstationary"):
        stationary_scale(value, value.shape)
