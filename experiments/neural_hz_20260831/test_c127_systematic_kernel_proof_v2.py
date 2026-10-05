"""Independent complete source/basis checks for systematic all36 rebuilding.

The reference contracts every literal tensor coefficient with Fraction; it
does not use constructor transforms, systematic anchors, expansion helpers or
an inverse certificate.  Every retained numeric array is compared with C125.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c125_exact_kernel_proof_v1 import (
    prepare as old_prepare, full_fraction_reference)
from experiments.neural_hz_20260831 import c127_systematic_kernel_proof_v2 as systematic


_GN = ((1, 0, 0), (-1, -1, -1), (-1, 1, -1),
       (1, 2, 4), (1, -2, 4), (0, 0, 1))
_EVIDENCE = {'source_snapshot', 'canonical_mantissa', 'canonical_exponent',
             'raw_mantissa', 'raw_exponent', 'aligned', 'numerator', 'exponent'}


def _kernel(mode='signed'):
    weights = ((np.arange(36).reshape(2, 2, 3, 3) % 13)-6).astype(np.float32)/16
    if mode == 'positive':
        weights = np.abs(weights)+np.float32(1/32)
    elif mode == 'scaled':
        powers = (np.arange(36).reshape(weights.shape) % 13-6).astype(np.int32)
        weights = np.ldexp(weights, powers)
    elif mode == 'sparse':
        weights[:] = 0
        weights[:, :, 0, 2] = np.float32(-3/8)
        weights[0, 1, 1, 1] = np.float32(5/16)
    elif mode == 'zero':
        weights[:] = 0
        weights[0, 0, 0, 0] = np.float32(-0.0)
    elif mode == 'mixed_zero':
        weights[0, 0] = 0
        weights[1, 1] = np.float32(-0.0)
    elif mode == 'random':
        rng = np.random.default_rng(127)
        words = rng.integers(1 << 23, 1 << 24, size=(2, 3, 3, 3), dtype=np.int64)
        words *= rng.choice(np.array([-1, 1], np.int64), size=words.shape)
        powers = rng.integers(-35, -13, size=words.shape, dtype=np.int32)
        weights = np.ldexp(words.astype(np.float32), powers)
    elif mode == 'noncontiguous':
        weights = weights[:, :, ::-1, ::-1]
        assert not weights.flags.c_contiguous
    elif mode == 'minnormal':
        weights = np.array([1, -1, 1.5, -1.5, 2, -2, 3, -3, 4], np.float32)
        weights = (weights*np.finfo(np.float32).tiny).reshape(1, 1, 3, 3)
    elif mode == 'maxnormal':
        maximum = np.finfo(np.float32).max
        weights = np.array([maximum, -maximum, maximum/2, -maximum/2,
                            maximum, -maximum, maximum/4, -maximum/4, 0], np.float32)
        weights = weights.reshape(1, 1, 3, 3)
    elif mode == 'span33':
        largest_mantissa = np.nextafter(np.float32(2), np.float32(0))
        weights = np.full((1, 1, 3, 3), largest_mantissa, np.float32)
        weights[0, 0, 0, 0] = np.float32(2.**-33)
    return weights


def _fraction_words(weights):
    result = {}
    for k in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            for a in range(6):
                for b in range(6):
                    value = F(0)
                    for i in range(3):
                        for j in range(3):
                            value += F(float(weights[k, c, i, j]))*_GN[a][i]*_GN[b][j]
                    result[k, c, a, b] = value
    return result


def _prepare(weights, pool=None):
    return systematic.prepare(weights, pool=pool if pool is not None else WorkPool(256_000_000),
                              enabled=True)


def _assert_complete_arrays(evidence, old):
    assert set(evidence) == set(old) == _EVIDENCE
    arrays = list(evidence.values())
    for name, value in evidence.items():
        assert type(value) is np.ndarray
        assert value.dtype == old[name].dtype and value.shape == old[name].shape, name
        assert value.tobytes(order='C') == old[name].tobytes(order='C'), name
        assert value.dtype.kind in 'if' and value.dtype.itemsize <= 8
        assert value.flags.owndata
    for index, value in enumerate(arrays):
        assert all(not np.shares_memory(value, previous) for previous in arrays[:index])


def _assert_every_source_and_transform_word(evidence, weights):
    expected = _fraction_words(weights)
    for location, value in expected.items():
        k, c, _, _ = location
        assert F(int(evidence['numerator'][location]))*F(2)**int(evidence['exponent'][k, c]) == value
    for index in np.ndindex(weights.shape):
        source = F(float(weights[index]))
        canonical = int(evidence['canonical_mantissa'][index])
        canonical_exp = int(evidence['canonical_exponent'][index])
        raw, raw_exp = int(evidence['raw_mantissa'][index]), int(evidence['raw_exponent'][index])
        assert canonical == 0 or abs(canonical) % 2 == 1
        assert raw == 0 or (1 << 23) <= abs(raw) < (1 << 24)
        assert F(canonical)*F(2)**canonical_exp == source
        assert F(raw)*F(2)**raw_exp == source
        assert F(int(evidence['aligned'][index]))*F(2)**int(evidence['exponent'][index[:2]]) == source
        if source == 0:
            assert (canonical, canonical_exp, raw, raw_exp) == (0, 0, 0, 0)
    return expected


@pytest.mark.parametrize('mode', [
    'signed', 'positive', 'scaled', 'sparse', 'zero', 'mixed_zero',
    'random', 'noncontiguous', 'minnormal', 'maxnormal', 'span33',
])
def test_every_owned_array_matches_C125_and_all36_match_full_Fraction(mode):
    weights = _kernel(mode)
    before = weights.tobytes(order='C')
    report, evidence = _prepare(weights)
    old_report, old = old_prepare(weights, pool=WorkPool(256_000_000), enabled=True)
    _assert_complete_arrays(evidence, old)
    expected = _assert_every_source_and_transform_word(evidence, weights)
    reference_report, reference = full_fraction_reference(weights, pool=WorkPool(256_000_000), enabled=True)
    for location, value in expected.items():
        assert (F(int(reference['canonical_numerator'][location]))
                *F(2)**int(reference['canonical_exponent'][location])) == value
    kernels = weights.shape[0]*weights.shape[1]
    for name in ('kernels', 'original_kernel_coefficients_observed', 'original_nonzero',
                 'all_transformed_kernel_coefficients_rebuilt', 'transformed_nonzero',
                 'maximum_raw_alignment_shift'):
        assert report[name] == old_report[name], name
    assert report['complete_all36_proved'] and report['exact_original_binary32_domain']
    assert report['complete_signed_int64_envelope'] and report['all_preparation_arrays_owned']
    assert report['systematic_source_derived_program']
    assert report['source_decoder'] == 'independent_vector_frexp_ldexp'
    assert not report['constructor_transform_or_inverse_used']
    assert not report['external_transformed_candidate_or_receipt_accepted']
    assert report['source_anchor_indices'] == [0, 1, 5]
    assert report['source_anchor_cells_per_kernel'] == 9
    assert report['source_derived_remaining_cells_per_kernel'] == 27
    assert report['full_basis_theorem_executed_each_call']
    assert report['all_source_basis_output_coefficients_proved'] == 324
    assert report['complete_reduced_512_bit_source_and_transform_domain']
    assert report['all_transformed_kernel_coefficients_rebuilt'] == 36*kernels
    assert report['original_kernel_coefficients_observed'] == weights.size
    assert report['no_prepared_cache_or_reuse'] and not report['current_source_reuse_authorized']
    assert report['formal_gain'] == 0 and not report['live_admission']
    assert reference_report['all_fraction_products_recomputed'] == 196*kernels
    assert reference_report['fraction_contraction_prepaid'] == 64*196*kernels
    assert reference_report['exact_lossless_numeric_reference_retained']
    assert weights.tobytes(order='C') == before
    assert all(not np.shares_memory(value, weights) for value in evidence.values())
    if mode in ('zero', 'mixed_zero'):
        empty = np.all(weights == 0, axis=(-1, -2))
        assert np.all(evidence['exponent'][empty] == 0)
        assert np.all(evidence['numerator'][empty] == 0)
    if mode == 'span33':
        assert report['maximum_raw_alignment_shift'] == 33
        # This valid source crosses2^62 in a transform coefficient.  A
        # premature loose-bound parity sum would not be safe signed int64.
        assert max(abs(int(value)) for value in evidence['numerator'].flat) > 1 << 62


def test_exact_binary64_lift_preserves_every_word_and_original_snapshot_dtype():
    weights = _kernel('random')
    _, binary32 = _prepare(weights)
    report, binary64 = _prepare(weights.astype(np.float64))
    _, old = old_prepare(weights.astype(np.float64), pool=WorkPool(256_000_000), enabled=True)
    _assert_complete_arrays(binary64, old)
    for name in _EVIDENCE-{'source_snapshot'}:
        assert binary32[name].dtype == binary64[name].dtype
        assert binary32[name].tobytes() == binary64[name].tobytes(), name
    assert binary32['source_snapshot'].dtype == np.dtype(np.float32)
    assert binary64['source_snapshot'].dtype == np.dtype(np.float64)
    assert np.array_equal(binary32['source_snapshot'], binary64['source_snapshot'])
    assert report['exact_original_binary32_domain']


def test_source_and_every_retained_array_are_owned_fresh_and_not_reusable_authority():
    weights = _kernel('scaled')
    original = weights.copy()
    report, evidence = _prepare(weights)
    held = {name: value.copy() for name, value in evidence.items()}
    weights[0, 0, 0, 0] += np.float32(1/8)
    for name, value in evidence.items():
        assert value.flags.owndata and not np.shares_memory(value, weights)
        assert value.tobytes() == held[name].tobytes(), name
    _assert_every_source_and_transform_word(evidence, original)
    _, fresh = _prepare(weights)
    _assert_every_source_and_transform_word(fresh, weights)
    assert not np.array_equal(fresh['numerator'], evidence['numerator'])
    assert all(not np.shares_memory(first, second)
               for first in evidence.values() for second in fresh.values())
    assert report['no_prepared_cache_or_reuse'] and not report['current_source_reuse_authorized']


def test_default_disabled_never_inspects_source_or_charges():
    pool = WorkPool(0)
    assert systematic.prepare(None, pool=pool) is None
    assert pool.used == 0 and pool.parts == {}


def test_entire_fixed_program_and_source_work_is_prepaid_before_numeric_inspection(monkeypatch):
    weights = _kernel()
    fee = systematic.HEADER_FEE+systematic.KERNEL_FEE*weights.shape[0]*weights.shape[1]
    invalid = weights.copy()
    invalid[0, 0, 0, 0] = np.nan
    pool = WorkPool(fee-1)
    def forbidden_basis():
        raise AssertionError('fixed numerical basis program ran before its complete payment')

    with monkeypatch.context() as patch:
        patch.setattr(systematic, '_prove_basis', forbidden_basis)
        with pytest.raises(MemoryError):
            _prepare(invalid, pool)
    assert pool.used == 0
    pool = WorkPool(fee)
    _, evidence = _prepare(weights, pool)
    assert pool.used == fee
    _assert_every_source_and_transform_word(evidence, weights)


@pytest.mark.parametrize('violation', [
    'inexact_binary64', 'nan', 'infinity', 'binary32_subnormal',
    'binary64_below_binary32_normal', 'span34',
])
def test_vector_decoder_rejects_every_unsupported_original_domain_without_fallback(violation):
    weights = np.zeros((1, 1, 3, 3), np.float32)
    weights[0, 0, 0, 0] = 1
    if violation == 'inexact_binary64':
        weights = weights.astype(np.float64)
        weights[0, 0, 1, 1] = .1
    elif violation == 'nan':
        weights[0, 0, 1, 1] = np.nan
    elif violation == 'infinity':
        weights[0, 0, 1, 1] = np.inf
    elif violation == 'binary32_subnormal':
        weights[0, 0, 1, 1] = np.nextafter(np.float32(0), np.float32(1))
    elif violation == 'binary64_below_binary32_normal':
        weights = weights.astype(np.float64)
        weights[0, 0, 1, 1] = 2.**-127
    else:
        weights[0, 0, 1, 1] = np.float32(2.**-34)
    fee = systematic.HEADER_FEE+systematic.KERNEL_FEE
    pool = WorkPool(fee)
    before = weights.tobytes()
    with pytest.raises(ValueError):
        _prepare(weights, pool)
    assert pool.used == fee
    assert weights.tobytes() == before


@pytest.mark.parametrize('violation', ['list', 'integer_dtype', 'wrong_geometry', 'empty_channel'])
def test_complete_exact_source_headers_are_required(violation):
    weights = np.ones((1, 1, 3, 3), np.float32)
    if violation == 'list':
        weights = weights.tolist()
    elif violation == 'integer_dtype':
        weights = weights.astype(np.int64)
    elif violation == 'wrong_geometry':
        weights = weights[:, :, :2]
    else:
        weights = weights[:, :0]
    with pytest.raises(ValueError):
        _prepare(weights)


def test_complete_nine_source_basis_images_prove_every_one_of_324_tensor_coefficients():
    weights = np.eye(9, dtype=np.float32).reshape(9, 1, 3, 3)
    report, evidence = _prepare(weights)
    compared = 0
    for basis in range(9):
        i, j = divmod(basis, 3)
        scale = F(2)**int(evidence['exponent'][basis, 0])
        for a in range(6):
            for b in range(6):
                actual = F(int(evidence['numerator'][basis, 0, a, b]))*scale
                assert actual == _GN[a][i]*_GN[b][j]
                compared += 1
    assert compared == 324
    assert report['complete_all36_proved']
    assert report['all_nine_source_basis_vectors_proved']
    assert report['all_source_basis_output_coefficients_proved'] == 324


@pytest.mark.parametrize('program_part', ['systematic', 'expansion'])
def test_fixed_independent_basis_proof_rejects_corrupted_implementation(monkeypatch, program_part):
    name = '_systematic' if program_part == 'systematic' else '_expand_three'
    original = getattr(systematic, name)

    def corrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        if program_part == 'expansion':
            entries = list(value)
            entries[4] = entries[4]+1
            return tuple(entries)
        value = value.copy()
        value[..., 5, 5] += 1
        return value

    monkeypatch.setattr(systematic, name, corrupted)
    with pytest.raises(ValueError):
        _prepare(np.ones((1, 1, 3, 3), np.float32))
