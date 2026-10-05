"""Exact integer F4 preparation and literal-identical ordinary HZ packets.

The transform reference below is the full independent Fraction contraction,
not the separable implementation under test.  C119's unchanged independent
oracle then checks actual rows, coverage, normalization and redundant boxes.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c119_denominator_f4_v1 import construct as fraction_construct
from experiments.neural_hz_20260831.c119_f4_oracle_v1 import prove
from experiments.neural_hz_20260831.c120_word_f4_v1 import construct, prepare_words


# Deliberately independent of the constructor's constants and arithmetic.
_G = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
      (1, 2, 4), (1, -2, 4), (0, 0, 1))


def _kernels(mode):
    weights = (np.arange(36).reshape(2, 2, 3, 3) % 13-6).astype(np.float32)/16
    if mode == 'positive':
        weights = np.abs(weights)+np.float32(1/32)
    elif mode == 'mixed_powers':
        powers = (np.arange(36).reshape(weights.shape) % 9-4).astype(np.int32)
        weights = np.ldexp(weights, powers)
    elif mode == 'sparse':
        weights[:] = 0
        weights[:, :, 0, 2] = np.float32(-3/8)
        weights[0, 1, 1, 1] = np.float32(5/16)
    elif mode == 'zero':
        weights[:] = 0
        weights[0, 0, 0, 0] = np.float32(-0.0)
    return weights


def _fraction_kernel_reference(weights):
    result = np.empty((*weights.shape[:2], 6, 6), dtype=object)
    for k, c, t, u in np.ndindex(result.shape):
        result[k, c, t, u] = sum(
            (F(float(weights[k, c, i, j]))*_G[t][i]*_G[u][j]
             for i in range(3) for j in range(3)), F(0))
    return result


def _assert_exact_words(weights, words):
    assert set(words) == {'numerator', 'exponent'}
    numerator, exponent = words['numerator'], words['exponent']
    assert numerator.dtype == np.dtype(np.int64)
    assert exponent.dtype == np.dtype(np.int32)
    assert numerator.shape == (*weights.shape[:2], 6, 6)
    assert exponent.shape == weights.shape[:2]
    reference = _fraction_kernel_reference(weights)
    for k, c, t, u in np.ndindex(numerator.shape):
        assert F(int(numerator[k, c, t, u]))*F(2)**int(exponent[k, c]) == reference[k, c, t, u]


@pytest.mark.parametrize('mode', ['signed', 'positive', 'mixed_powers', 'sparse', 'zero'])
def test_complete_integer_transform_matches_independent_fraction_contraction(mode):
    weights = _kernels(mode)
    original = weights.copy()
    _, words = prepare_words(weights, pool=WorkPool(256_000_000), enabled=True)
    _assert_exact_words(weights, words)
    assert np.array_equal(weights.view(np.uint32), original.view(np.uint32))


def _fixture(mode='dense'):
    channels = 2 if mode == 'shared' else 1
    ids = np.arange(channels*36, dtype=np.int64).reshape(channels, 6, 6)
    powers = np.zeros(ids.shape, dtype=np.int32)
    outputs = np.arange(ids.size, ids.size+16, dtype=np.int64).reshape(1, 4, 4)
    outpowers = np.full(outputs.shape, 8, dtype=np.int32)
    weights = ((np.arange(9*channels).reshape(1, channels, 3, 3) % 11)-5).astype(np.float64)/16
    if mode == 'masked':
        ids[:, 2::2, 1::2] = -1
        outputs[:, 1::2, ::2] = -1
    elif mode == 'shared':
        ids[:, 1] = ids[:, 0]
        ids[1] = ids[0]
    elif mode == 'scaled':
        powers[:] = np.arange(ids.size).reshape(ids.shape) % 5-2
        outpowers[:] = np.arange(outputs.size).reshape(outputs.shape) % 3+8
    elif mode == 'sparse':
        weights[:] = 0
        weights[0, 0, 1, 1] = 3/8
    elif mode == 'zero':
        weights[:] = 0
    elif mode == 'unused_outputs':
        outputs[:] = -1
    return weights, ids, powers, outputs, outpowers, ids.size+16


@pytest.mark.parametrize('mode', ['dense', 'masked', 'shared', 'scaled',
                                  'sparse', 'zero', 'unused_outputs'])
def test_packet_is_literal_identical_to_fraction_constructor_and_passes_unchanged_oracle(mode):
    args = _fixture(mode)
    snapshots = [value.copy() for value in args[:-1]]
    report, packet = construct(*args, pool=WorkPool(256_000_000), enabled=True)
    old_report, old_packet = fraction_construct(*args, pool=WorkPool(256_000_000), enabled=True)
    assert packet.keys() == old_packet.keys()
    for name in packet:
        assert packet[name].dtype == old_packet[name].dtype
        assert np.array_equal(packet[name], old_packet[name]), name
    for name in ('kept_v', 'kept_m', 'new_factors', 'rows', 'nnz',
                 'base_n_cont', 'n_cont', 'whole_circuit_emission'):
        assert report[name] == old_report[name]
    proof = prove(packet, *args, pool=WorkPool(256_000_000), enabled=True)
    assert proof['all_1d_bilinear_coefficients_proved'] == 72
    assert proof['all_native_nnz_proved'] == report['nnz']
    assert proof['all_auxiliary_equations_and_redundant_boxes_proved'] == report['new_factors']
    assert proof['exact_required_support_coverage']
    assert proof['formal_gain'] == 0 and not proof['actual_network_source_bound']
    assert not report['live_admission'] and report['score_gain'] == 0
    assert report['row_construction_prepaid'] >= report['whole_circuit_emission']
    assert all(np.array_equal(snapshot, value)
               for snapshot, value in zip(snapshots, args[:-1], strict=True))


def test_default_off_is_uninspected_and_unpaid():
    pool = WorkPool(0)
    assert prepare_words(None, pool=pool) is None
    assert construct(None, None, None, None, None, None, pool=pool) is None
    assert pool.used == 0


def test_full_transform_tariff_is_prepaid_before_source_conversion_checks():
    # Invalid original float64 must not trigger an unpaid scan or a fallback.
    weights = np.full((1, 1, 3, 3), 0.1, dtype=np.float64)
    pool = WorkPool(2767)
    with pytest.raises(MemoryError):
        prepare_words(weights, pool=pool, enabled=True)
    assert pool.used == 0
    finite = np.ones_like(weights)
    pool = WorkPool(2768)
    _, words = prepare_words(finite, pool=pool, enabled=True)
    assert pool.used == 2768
    _assert_exact_words(finite, words)


def test_construct_rejects_unpaid_topology_before_examining_numeric_source():
    args = list(_fixture())
    args[0][0, 0, 0, 0] = np.nan
    pool = WorkPool(0)
    with pytest.raises(MemoryError):
        construct(*args, pool=pool, enabled=True)
    assert pool.used == 0


def test_exact_binary32_lift_in_float64_preserves_all_transform_words():
    weights = _kernels('mixed_powers')
    _, first = prepare_words(weights, pool=WorkPool(256_000_000), enabled=True)
    _, second = prepare_words(weights.astype(np.float64), pool=WorkPool(256_000_000), enabled=True)
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_inexact_binary64_source_rejects_without_fraction_fallback():
    args = list(_fixture())
    args[0][0, 0, 1, 1] = 0.1
    original = args[0].copy()
    with pytest.raises(ValueError):
        prepare_words(args[0], pool=WorkPool(256_000_000), enabled=True)
    with pytest.raises(ValueError):
        construct(*args, pool=WorkPool(256_000_000), enabled=True)
    assert np.array_equal(args[0], original)


def test_nonfinite_original_coefficients_reject_after_prepayment():
    for invalid in (np.nan, np.inf):
        weights = np.ones((1, 1, 3, 3), dtype=np.float32)
        weights[0, 0, 0, 0] = invalid
        pool = WorkPool(2768)
        with pytest.raises(ValueError):
            prepare_words(weights, pool=pool, enabled=True)
        assert pool.used == 2768


def test_alignment_33_envelope_preserves_all_coefficients_exactly():
    weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = np.float32(2**-33)
    weights[0, 0, 2, 2] = np.float32(1)
    report, words = prepare_words(weights, pool=WorkPool(256_000_000), enabled=True)
    assert report['maximum_alignment_shift'] == 33
    _assert_exact_words(weights, words)


def test_alignment_outside_proved_int64_envelope_rejects():
    weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = np.float32(2**-34)
    weights[0, 0, 2, 2] = np.float32(1)
    with pytest.raises(ValueError):
        prepare_words(weights, pool=WorkPool(256_000_000), enabled=True)


def test_unchanged_complete_native_row_window_remains_fail_closed():
    args = list(_fixture())
    args[4] = np.full(args[4].shape, 80, dtype=np.int32)
    for build in (construct, fraction_construct):
        with pytest.raises(ValueError, match='coefficient window'):
            build(*args, pool=WorkPool(256_000_000), enabled=True)
