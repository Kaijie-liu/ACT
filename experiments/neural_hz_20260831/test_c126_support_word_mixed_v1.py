"""Complete native agreement and independent Fraction elimination for C126.

Tests preserve the C124 ordinary packet populations.  The reference eliminates
actual emitted equations rather than using either new implementation's word,
support, normalization, or transform helpers.  Ephemeral constructor supports
are not misrepresented as retained or reusable preparation evidence.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c124_mixed_f4_v1 import construct as old_construct
from experiments.neural_hz_20260831.c125_mixed_f4_oracle_v1 import prove as old_prove
from experiments.neural_hz_20260831.test_c124_mixed_f4_v1 import (
    _fixture as old_fixture, _direct, _rows, _satisfying_original)
from experiments.neural_hz_20260831 import c126_support_word_mixed_v1 as mixed
from experiments.neural_hz_20260831.c126_support_word_oracle_v1 import prove


def _fixture(mode='dense'):
    inherited = mode if mode in ('dense','masked','shared','cancel','scaled',
                                 'sparse_selected','zero_selected','no_outputs') else 'dense'
    args, selected = old_fixture(inherited)
    args = list(args)
    if mode == 'negative':
        args[0] *= -1
    elif mode == 'partial_demand':
        args[3].reshape(-1)[1:] = -1
    elif mode == 'residual_cancel':
        args[0] = np.concatenate((args[0], -args[0][:, 1:2]), axis=1)
        args[1] = np.concatenate((args[1], args[1][1:2]), axis=0)
        args[2] = np.concatenate((args[2], args[2][1:2]), axis=0)
        selected = np.array([True, False, False], dtype=bool)
    elif mode == 'positive_gauge':
        args[2][:] = -30
    elif mode == 'negative_gauge':
        args[2][:] = 40
        args[4][:] = 49
    return tuple(args), selected


def _build(args, selected, pool=None):
    return mixed.construct(*args, selected_channels=selected,
                           pool=pool if pool is not None else WorkPool(256_000_000),
                           enabled=True)


def _prove(packet, args, selected, pool=None):
    return prove(packet, *args, selected_channels=selected,
                 pool=pool if pool is not None else WorkPool(256_000_000), enabled=True)


def _copy_packet(packet):
    return {name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in packet.items()}


@pytest.mark.parametrize('mode', [
    'dense','masked','shared','cancel','scaled','negative','partial_demand',
    'sparse_selected','zero_selected','residual_cancel','positive_gauge',
    'negative_gauge','no_outputs',
])
def test_complete_word_packet_is_literal_old_packet_and_fraction_convolution(mode):
    args, selected = _fixture(mode)
    before = [array.copy() for array in args[:-1]]
    before_selected = selected.copy()
    report, packet = _build(args, selected)
    old_report, old_packet = old_construct(*args, selected_channels=selected,
                                          pool=WorkPool(256_000_000), enabled=True)
    assert packet is not None and old_packet is not None
    for name, value in old_packet.items():
        if isinstance(value, np.ndarray):
            assert packet[name].dtype == value.dtype
            assert np.array_equal(packet[name], value), name
        else:
            assert packet[name] == value
    for name in ('kept_v','kept_m','new_factors','rows','nnz','base_n_cont','n_cont',
                 'whole_circuit_emission'):
        assert report[name] == old_report[name], name
    proof, evidence = _prove(packet, args, selected)
    old_proof, old_evidence = old_prove(packet, *args, selected_channels=selected,
                                       pool=WorkPool(256_000_000), enabled=True)
    for name in ('all_native_rows_proved','all_native_nnz_proved','all_V_definitions_proved',
                 'all_M_definitions_proved','all_original_output_equations_proved',
                 'all_auxiliary_equations_and_redundant_boxes_proved',
                 'all_1d_bilinear_coefficients_proved',
                 'all_transformed_kernel_coefficients_rebuilt'):
        assert proof[name] == old_proof[name], name
    assert proof['exact_required_support_coverage'] and proof['universal_unique_box_extension']
    assert proof['compositional_original_source_equivalence']
    assert proof['all_native_nnz_proved'] == report['nnz'] == len(packet['native'])
    auxiliary, emitted = _rows(packet)
    direct = _direct(args)
    assert project_outputs(auxiliary, emitted, args[-1]) == direct
    original = _satisfying_original(args, emitted, direct)
    expanded = extend_actual(auxiliary, original)
    assert expanded[:args[-1]] == original
    for row in [*auxiliary,*emitted]:
        assert sum((F(value)*expanded[column] for column,value in row['coefficients']),F(0)) == F(row['rhs'])
    assert evidence.keys() == old_evidence.keys()
    for name,array in evidence.items():
        assert array.dtype.kind in 'biuf' and array.dtype.itemsize <= 8
        assert array.flags.owndata and np.array_equal(array, old_evidence[name]), name
        assert not any(np.shares_memory(array, old) for old in args[:-1])
    assert packet['selected_channels'].dtype == np.dtype(bool)
    assert packet['selected_channels'].flags.owndata
    assert not np.shares_memory(packet['selected_channels'], selected)
    assert all(np.array_equal(old, new) for old,new in zip(before,args[:-1],strict=True))
    assert np.array_equal(selected,before_selected)
    assert not report['live_admission'] and report['score_gain'] == 0
    assert not proof['actual_network_source_bound'] and proof['formal_gain'] == 0
    if mode == 'positive_gauge':
        assert np.any(packet['gauges'] > 0)
    elif mode == 'negative_gauge':
        assert np.any(packet['gauges'] < 0)
    elif mode == 'no_outputs':
        assert report['rows'] == report['nnz'] == report['new_factors'] == 0
    elif mode == 'zero_selected':
        assert report['new_factors'] == 0 and report['rows'] == len(direct)


@pytest.mark.parametrize('mutation', [
    'residual_coefficient','denominator','gauge','role','mask','rhs','binary',
    'mask_custody','auxiliary_count',
])
def test_complete_independent_word_oracle_rejects_native_or_custody_corruption(mutation):
    args, selected = _fixture()
    _, packet = _build(args, selected)
    packet = _copy_packet(packet)
    output_row = int(np.flatnonzero(packet['roles'][:,0] == 2)[0])
    mrow = int(np.flatnonzero(packet['roles'][:,0] == 1)[0])
    if mutation == 'residual_coefficient':
        begin,end = map(int,packet['indptr'][output_row:output_row+2])
        column = int(args[1][1,0,0])
        index = begin+int(np.flatnonzero(packet['columns'][begin:end] == column)[0])
        packet['native'][index] *= 2
    elif mutation == 'denominator':
        packet['defining_denominators'][mrow] += 1
    elif mutation == 'gauge':
        packet['gauges'][output_row] += 1
    elif mutation == 'role':
        packet['roles'][mrow,2] = 100
    elif mutation == 'mask':
        packet['selected_channels'][:] = True
    elif mutation == 'rhs':
        packet['rhs'][output_row] = .125
    elif mutation == 'binary':
        packet['ab_indptr'][-1] = 1
    elif mutation == 'mask_custody':
        packet['selected_channels'] = packet['selected_channels'].view()
    else:
        packet['new_factors'] += 1
    with pytest.raises(ValueError):
        _prove(packet,args,selected)


def test_all_selected_remains_literal_full_F4_with_no_direct_tail():
    args, selected = _fixture('scaled')
    selected[:] = True
    report,packet = _build(args,selected)
    _,old_packet = old_construct(*args,selected_channels=selected,
                                 pool=WorkPool(256_000_000),enabled=True)
    for name,value in old_packet.items():
        assert np.array_equal(packet[name],value),name
    proof,_ = _prove(packet,args,selected)
    assert proof['residual_channels'] == 0
    assert report['new_factors'] == packet['new_factors']


def test_empty_channel_selection_is_literal_noop_before_kernel_preparation(monkeypatch):
    args,selected = _fixture()
    selected[:] = False
    args[0][:] = np.nan
    def forbidden_preparation(*values,**keywords):
        raise AssertionError('no-op attempted kernel preparation')
    monkeypatch.setattr(mixed,'prepare_words',forbidden_preparation)
    report,packet = _build(args,selected)
    assert packet is None and report['literal_noop']
    assert report['rows'] == report['nnz'] == report['new_factors'] == 0
    assert report['kernel_transform_prepaid'] == report['whole_circuit_emission'] == 0
    assert report['selection_mask_bytes'] == report['selection_mask_entries'] == 0


def test_default_off_and_zero_budget_never_decode_numeric_source(monkeypatch):
    pool = WorkPool(0)
    assert mixed.construct(None,None,None,None,None,None,selected_channels=None,pool=pool) is None
    assert prove(None,None,None,None,None,None,None,selected_channels=None,pool=pool) is None
    args,selected = _fixture()
    args[0][0,0,0,0] = np.nan
    with pytest.raises(MemoryError):
        _build(args,selected,pool)
    assert pool.used == 0
    args,selected = _fixture()
    _,packet = _build(args,selected)
    args[0][0,0,0,0] = np.nan
    proof_pool = WorkPool(0)
    with pytest.raises(MemoryError):
        _prove(packet,args,selected,proof_pool)
    assert proof_pool.used == 0
    class RefuseSupport(WorkPool):
        def charge(self,name,amount):
            if name == 'c126_complete_residual_support_compilation':
                raise MemoryError('complete support compilation must be prepaid')
            return super().charge(name,amount)
    def forbidden_decode(*values,**keywords):
        raise AssertionError('unpaid support reached independent source-word decoding')
    monkeypatch.setattr(mixed,'_decode',forbidden_decode)
    args,selected = _fixture()
    with pytest.raises(MemoryError):
        _build(args,selected,RefuseSupport(256_000_000))


def test_changed_source_or_parent_support_cannot_reuse_an_old_native_packet():
    args,selected = _fixture()
    _,packet = _build(args,selected)
    args[0][0,1,0,0] *= 2
    with pytest.raises(ValueError):
        _prove(packet,args,selected)
    args,selected = _fixture()
    args[1][1,0,0] = -1
    with pytest.raises(ValueError):
        _prove(packet,args,selected)


@pytest.mark.parametrize('malformation',['dtype','rank','list'])
def test_selection_must_be_a_complete_boolean_channel_vector(malformation):
    args,selected = _fixture()
    if malformation == 'dtype':
        selected = selected.astype(np.uint8)
    elif malformation == 'rank':
        selected = selected.reshape(2,1)
    else:
        selected = selected.tolist()
    with pytest.raises((TypeError,ValueError)):
        _build(args,selected)


def test_complete_native_window_is_not_relaxed_by_word_arithmetic():
    args,selected = _fixture()
    args[4][:] = 80
    with pytest.raises(ValueError):
        _build(args,selected)


@pytest.mark.parametrize('invalid',['inexact_binary64','nonfinite'])
def test_non_word_original_kernel_rejects_without_fraction_fallback(invalid):
    args,selected = _fixture()
    args[0][0,0,0,0] = .1 if invalid == 'inexact_binary64' else np.nan
    with pytest.raises(ValueError):
        _build(args,selected)
