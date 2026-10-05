"""Independent full Fraction checks for owned exact all-36 kernel evidence.

No constructor transform or inverse is used by the reference below.  Source
snapshots establish the source for this invocation only, never cache reuse or
native source admission after mutation of a caller-owned kernel.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c120_word_f4_v1 import prepare_words as old_words
from experiments.neural_hz_20260831.c124_mixed_f4_v1 import construct
from experiments.neural_hz_20260831.c124_mixed_f4_oracle_v1 import prove as old_prove
from experiments.neural_hz_20260831.test_c124_mixed_f4_v1 import _fixture, _direct, _rows
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import project_outputs
from experiments.neural_hz_20260831.c125_exact_kernel_proof_v1 import (
    HEADER_FEE, KERNEL_FEE, prepare, full_fraction_reference)
from experiments.neural_hz_20260831.c125_mixed_f4_oracle_v1 import prove


_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))


def _kernel(mode='signed'):
    weights = (np.arange(36).reshape(2, 2, 3, 3) % 13-6).astype(np.float32)/16
    if mode == 'positive':
        weights = np.abs(weights)+np.float32(1/32)
    elif mode == 'scaled':
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


def _reference(weights):
    result = {}
    for k in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            for y in range(6):
                for x in range(6):
                    value = F(0)
                    for a in range(3):
                        for b in range(3):
                            value += F(float(weights[k, c, a, b]))*_GN[y][a]*_GN[x][b]
                    result[k, c, y, x] = value
    return result


def _all_arrays(evidence):
    if isinstance(evidence, dict):
        return [array for value in evidence.values() for array in _all_arrays(value)]
    assert isinstance(evidence, np.ndarray)
    assert evidence.dtype.kind in 'biuf' and evidence.dtype.itemsize <= 8
    return [evidence]


def _assert_all_words(evidence, weights):
    expected = _reference(weights)
    assert evidence['numerator'].dtype == np.dtype(np.int64)
    assert evidence['exponent'].dtype == np.dtype(np.int32)
    assert evidence['numerator'].shape == (*weights.shape[:2], 6, 6)
    assert evidence['exponent'].shape == weights.shape[:2]
    for (k, c, y, x), value in expected.items():
        assert F(int(evidence['numerator'][k, c, y, x]))*F(2)**int(evidence['exponent'][k, c]) == value


def _assert_reference_words(evidence, weights):
    expected = _reference(weights)
    for location, value in expected.items():
        numerator = int(evidence['canonical_numerator'][location])
        exponent = int(evidence['canonical_exponent'][location])
        assert numerator == 0 or abs(numerator) % 2 == 1
        assert F(numerator)*F(2)**exponent == value


@pytest.mark.parametrize('mode', ['signed', 'positive', 'scaled', 'sparse', 'zero'])
def test_all_36_integer_words_match_independent_fraction_and_C120_literals(mode):
    weights = _kernel(mode)
    before = weights.copy()
    report, evidence = prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    _assert_all_words(evidence, weights)
    _, old = old_words(weights, pool=WorkPool(256_000_000), enabled=True)
    assert np.array_equal(evidence['numerator'], old['numerator'])
    assert np.array_equal(evidence['exponent'], old['exponent'])
    assert np.array_equal(evidence['source_snapshot'], weights)
    for index in np.ndindex(weights.shape):
        canonical = int(evidence['canonical_mantissa'][index])
        canonical_exp = int(evidence['canonical_exponent'][index])
        raw = int(evidence['raw_mantissa'][index])
        raw_exp = int(evidence['raw_exponent'][index])
        assert canonical == 0 or abs(canonical) % 2 == 1
        assert raw == 0 or 2**23 <= abs(raw) < 2**24
        assert F(canonical)*F(2)**canonical_exp == F(float(weights[index]))
        assert F(raw)*F(2)**raw_exp == F(float(weights[index]))
        scale = int(evidence['exponent'][index[:2]])
        assert F(int(evidence['aligned'][index]))*F(2)**scale == F(float(weights[index]))
    assert report['complete_all36_proved']
    assert report['all_transformed_kernel_coefficients_rebuilt'] == 36*weights.shape[0]*weights.shape[1]
    assert report['original_kernel_coefficients_observed'] == weights.size
    assert report['exact_original_binary32_domain'] and report['complete_signed_int64_envelope']
    assert report['no_prepared_cache_or_reuse'] and not report['live_admission']
    assert report['formal_gain'] == 0
    assert np.array_equal(weights.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize('mode', ['signed', 'scaled', 'zero'])
def test_retained_fraction_reference_is_complete_lossless_and_unchanged_paid_scope(mode):
    weights = _kernel(mode)
    kernels = weights.shape[0]*weights.shape[1]
    pool = WorkPool(256_000_000)
    report, evidence = full_fraction_reference(weights, pool=pool, enabled=True)
    _assert_reference_words(evidence, weights)
    assert set(evidence) == {'source_snapshot', 'canonical_numerator', 'canonical_exponent'}
    assert np.array_equal(evidence['source_snapshot'], weights)
    assert pool.used == (1024+32*weights.size)+64*196*kernels+64*36*kernels
    assert report['no_prepared_cache_or_reuse'] and not report['live_admission']
    assert report['formal_gain'] == 0
    for array in _all_arrays(evidence):
        assert array.flags.owndata and not np.shares_memory(array, weights)


def test_every_retained_array_is_numeric_owned_and_caller_mutation_cannot_change_snapshot():
    weights = _kernel('scaled')
    original = weights.copy()
    report, evidence = prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    assert set(evidence) == {'source_snapshot', 'canonical_mantissa', 'canonical_exponent',
                             'raw_mantissa', 'raw_exponent', 'aligned', 'numerator', 'exponent'}
    frozen_values = {name: array.copy() for name, array in evidence.items()}
    arrays = _all_arrays(evidence)
    for index, array in enumerate(arrays):
        assert array.flags.owndata and not np.shares_memory(array, weights)
        assert all(not np.shares_memory(array, previous) for previous in arrays[:index])
    weights[0, 0, 0, 0] += np.float32(.125)
    assert all(np.array_equal(evidence[name], before) for name, before in frozen_values.items())
    _assert_all_words(evidence, original)
    _, fresh = prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    _assert_all_words(fresh, weights)
    assert not np.array_equal(fresh['source_snapshot'], evidence['source_snapshot'])
    assert not np.array_equal(fresh['numerator'], evidence['numerator'])
    # There is deliberately no reuse API: old snapshot evidence is not a
    # receipt establishing equality to a caller's subsequently changed source.
    assert report['no_prepared_cache_or_reuse']


def test_exact_binary64_lift_and_binary32_have_identical_complete_word_evidence():
    weights = _kernel('scaled')
    _, first = prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    _, second = prepare(weights.astype(np.float64), pool=WorkPool(256_000_000), enabled=True)
    for name in first:
        assert np.array_equal(first[name], second[name]), name


def test_disabled_proof_paths_do_not_inspect_source_or_charge():
    pool = WorkPool(0)
    assert prepare(None, pool=pool) is None
    assert full_fraction_reference(None, pool=pool) is None
    assert prove(None, None, None, None, None, None, None,
                 selected_channels=None, pool=pool) is None
    assert pool.used == 0


def test_full_integer_tariff_is_prepaid_before_numeric_source_decode():
    weights = _kernel()
    fee = HEADER_FEE+KERNEL_FEE*weights.shape[0]*weights.shape[1]
    pool = WorkPool(fee-1)
    weights[0, 0, 0, 0] = np.nan
    with pytest.raises(MemoryError):
        prepare(weights, pool=pool, enabled=True)
    assert pool.used == 0
    weights = _kernel()
    pool = WorkPool(fee)
    _, evidence = prepare(weights, pool=pool, enabled=True)
    assert pool.used == fee
    _assert_all_words(evidence, weights)


@pytest.mark.parametrize('violation', ['inexact_binary64', 'nonfinite', 'alignment', 'subnormal'])
def test_unsupported_original_word_domains_fail_closed_without_fallback(violation):
    weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = 1
    if violation == 'inexact_binary64':
        weights = weights.astype(np.float64)
        weights[0, 0, 1, 1] = 0.1
    elif violation == 'nonfinite':
        weights[0, 0, 1, 1] = np.inf
    elif violation == 'alignment':
        weights[0, 0, 1, 1] = np.float32(2.**-34)
    else:
        weights[0, 0, 1, 1] = np.nextafter(np.float32(0), np.float32(1))
    pool = WorkPool(HEADER_FEE+KERNEL_FEE)
    with pytest.raises(ValueError):
        prepare(weights, pool=pool, enabled=True)
    assert pool.used == HEADER_FEE+KERNEL_FEE


def test_full_33_bit_alignment_envelope_retains_every_coefficient_exactly():
    weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = np.float32(2.**-33)
    weights[0, 0, 2, 2] = np.float32(1)
    report, evidence = prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    _assert_all_words(evidence, weights)
    assert report['complete_signed_int64_envelope']


@pytest.mark.parametrize('mode', ['dense', 'shared', 'cancel', 'scaled'])
def test_integer_kernel_oracle_retains_complete_mixed_native_source_proof(mode):
    args, selected = _fixture(mode)
    _, packet = construct(*args, selected_channels=selected,
                           pool=WorkPool(256_000_000), enabled=True)
    report, evidence = prove(packet, *args, selected_channels=selected,
                             pool=WorkPool(256_000_000), enabled=True)
    old = old_prove(packet, *args, selected_channels=selected,
                    pool=WorkPool(256_000_000), enabled=True)
    for name in ('all_native_rows_proved', 'all_native_nnz_proved',
                 'all_V_definitions_proved', 'all_M_definitions_proved',
                 'all_original_output_equations_proved',
                 'all_auxiliary_equations_and_redundant_boxes_proved',
                 'all_1d_bilinear_coefficients_proved',
                 'all_residual_scan_positions_proved', 'all_residual_nonzero_occurrences_rebuilt'):
        assert report[name] == old[name], name
    assert report['all_native_nnz_proved'] == len(packet['native'])
    assert report['exact_required_support_coverage']
    assert report['universal_unique_box_extension']
    assert report['compositional_original_source_equivalence']
    assert report['kernel_proof']['complete_all36_proved']
    assert not report['actual_network_source_bound'] and report['formal_gain'] == 0
    _assert_all_words(evidence, args[0])
    assert all(array.flags.owndata for array in _all_arrays(evidence))
    auxiliary, emitted = _rows(packet)
    assert project_outputs(auxiliary, emitted, args[-1]) == _direct(args)


@pytest.mark.parametrize('mutation', ['residual_coefficient', 'source_coefficient', 'mask', 'native_window'])
def test_complete_new_native_oracle_rejects_changed_residual_source_or_custody(mutation):
    args, selected = _fixture()
    _, packet = construct(*args, selected_channels=selected,
                           pool=WorkPool(256_000_000), enabled=True)
    if mutation == 'source_coefficient':
        args[0][0, 1, 0, 0] *= 2
    elif mutation == 'mask':
        packet['selected_channels'][:] = True
    elif mutation == 'native_window':
        packet['native'][0] = 2.**41
    else:
        row = int(np.flatnonzero(packet['roles'][:, 0] == 2)[0])
        begin, end = map(int, packet['indptr'][row:row+2])
        column = int(args[1][1, 0, 0])
        index = begin+int(np.flatnonzero(packet['columns'][begin:end] == column)[0])
        packet['native'][index] *= 2
    with pytest.raises(ValueError):
        prove(packet, *args, selected_channels=selected,
              pool=WorkPool(256_000_000), enabled=True)
