"""Explicit source-bound tests; do not run before the D015 freeze."""
from fractions import Fraction as F
import pytest
from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import census_worker_v1 as w


def spec(claims):
    return ('(declare-const X_0 Real)\n(declare-const Y_0 Real)\n' + claims).encode()


def test_decimal_tokens_are_exact():
    b = w.input_box(spec('(assert (>= X_0 -0.1))(assert (<= X_0 0.2))'), (1, 1, 1, 1))
    assert b == {0: (F(-1, 10), F(1, 5))}


def test_missing_bound_rejected():
    with pytest.raises(ValueError):
        w.input_box(spec('(assert (>= X_0 0))'), (1, 1, 1, 1))


def test_input_disjunction_rejected():
    with pytest.raises(ValueError):
        w.input_box(spec('(assert (or (<= X_0 1) (>= X_0 0)))'), (1, 1, 1, 1))


def test_duplicate_and_inverted_bounds_rejected():
    with pytest.raises(ValueError):
        w.input_box(spec('(assert (>= X_0 2))(assert (<= X_0 1))'), (1, 1, 1, 1))
    with pytest.raises(ValueError):
        w.input_box(spec('(assert (>= X_0 0))(assert (>= X_0 0))(assert (<= X_0 1))'), (1, 1, 1, 1))


def test_output_only_clause_is_not_a_property_solve():
    raw = spec('(assert (>= X_0 0))(assert (<= X_0 1))(assert (or (<= Y_0 0) (>= Y_0 1)))')
    assert w.input_box(raw, (1, 1, 1, 1)) == {0: (F(0), F(1))}


def test_spatial_anchors_include_all_corners_and_center():
    assert w.anchors(5, 7) == ((0, 0), (0, 6), (2, 3), (4, 0), (4, 6))
    assert w.anchors(1, 1) == ((0, 0),)


def test_padding_omits_outside_input_before_affine_bias():
    conv = dict(weight_shape=(2, 3, 3, 3), group=1, strides=(2, 2), dilations=(1, 1), pads=(1, 1, 1, 1))
    assert w.conv_shape(conv, (1, 3, 4, 4)) == (1, 2, 2, 2)
    terms = w.receptive(conv, (1, 3, 4, 4), 0, 0)
    assert len(terms) == 12 and all(0 <= t[1] < 2 and 0 <= t[2] < 2 for t in terms)


def test_evidence_ledger_accounts_shared_fraction_and_bytes():
    value = F(1, 3); raw = b'original bytes'; root = {'a': [value, value], 'raw': raw}
    report = w.ledger(root)
    assert report['retained_entries'] >= 4 and report['held_instance_bytes'] >= len(raw)
    assert w.encoded(root)['raw']['sha256'] == w.hashlib.sha256(raw).hexdigest()


def test_complete_synthetic_source_shielding_and_side_branch_accounting():
    from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
    geometry = dict(group=1, strides=(1, 1), dilations=(1, 1), pads=(0, 0, 0, 0))
    first = dict(geometry, weight_shape=(2, 3, 1, 1), weights=tuple(map(F, (1, 0, 0, -1, 0, 0))),
                 bias=(F(-3, 4), F(1, 4)))
    following = dict(geometry, weight_shape=(1, 2, 1, 1), weights=(F(1), F(-1)), bias=(F(-1, 10),))
    packet = dict(input_shape=(1, 3, 1, 1), pre_affine=dict(scale=(F(1),) * 3, bias=(F(0),) * 3),
        first_conv=first, first_post_ops=[], first_relu=dict(output='first_relu'),
        branches=[dict(conv=following, post_ops=[], target_relu=dict(output='second_relu')),
                  dict(conv=following, post_ops=[], target_relu=None)])
    result = w.census(packet, {i: (F(0), F(1)) for i in range(3)}, k, k.WorkBudget(enabled=True))
    assert result['rows'] == result['shielded_terms'] == result['outer_unstable_rows_shielded'] == 1
    assert result['certified_pairs'] == 1 and result['original_bits_deleted'] == 0
    row = result['windows'][0]['rows'][0]
    assert row['baseline_bounds'] == (F(-1, 10), F(-1, 10))
    assert row['shielded_positions'] == [1] and row['negative_tests'] == [(1, F(-1, 10))]


def test_raw_batchnorm_enclosure_preserves_negative_scale():
    from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
    op = dict(kind='batchnorm', gamma=(F(-3),), beta=(F(1),), mean=(F(2),),
              variance=(F(4),), epsilon=F(0))
    scale, bias = w.post_affine([op], 0, k, k.WorkBudget(enabled=True))
    assert scale == (F(-3, 2), F(-3, 2)) and bias == (F(4), F(4))


def test_shared_baseline_cancellation_enables_unstable_shielding():
    from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
    geometry = dict(group=1, strides=(1, 1), dilations=(1, 1), pads=(0, 0, 0, 0))
    first = dict(geometry, weight_shape=(3, 3, 1, 1),
        weights=tuple(map(F, (1, 0, 0, -2, 0, 0, -1, 0, 0))),
        bias=(F(1, 4), F(7, 4), F(1, 2)))
    following = dict(geometry, weight_shape=(1, 3, 1, 1),
        weights=(F(1), F(1), F(-1)), bias=(F(-7, 4),))
    packet = dict(input_shape=(1, 3, 1, 1),
        pre_affine=dict(scale=(F(1),) * 3, bias=(F(0),) * 3),
        first_conv=first, first_post_ops=[], first_relu=dict(output='shared_first'),
        branches=[dict(conv=following, post_ops=[], target_relu=dict(output='next'))])
    result = w.census(packet, {i: (F(-1), F(1)) for i in range(3)}, k,
                      k.WorkBudget(enabled=True))
    row = result['windows'][0]['rows'][0]
    assert all(v['tau'] == 1 for v in result['source_forms'].values())
    assert row['baseline_form'] == ((F(-1, 4), F(-1, 4)), {})
    assert row['baseline_bounds'] == (F(-1, 4), F(-1, 4))
    assert row['negative_tests'] == [(2, F(0))] and row['shielded_positions'] == [2]
    assert result['outer_unstable_rows_shielded'] == 1
    # Independently witness both signs for this synthetic control ONLY.
    def direct(x):
        return max(F(0), x + F(1, 4)) + max(F(0), -2*x + F(7, 4)) - max(F(0), -x + F(1, 2)) - F(7, 4)
    assert direct(F(-1)) == F(1, 2) and direct(F(1)) == F(-1, 2)
