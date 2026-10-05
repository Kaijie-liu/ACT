from fractions import Fraction as F
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, _lower_hz_milp
from experiments.neural_hz_20260831 import c15_unit_row_splice_v1 as unit
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c10_alias_quotient_v1 import feasible
from experiments.neural_hz_20260831.c15_unit_row_splice_audit_v1 import audit


def fixture(count=512, mixed=False, subtract=False):
    nc = 2 + 2 * count
    kinds = [bool(mixed and i % 2) for i in range(count)]
    eq_rows = 1 + count + kinds.count(False)
    ac, ab = sp.lil_matrix((eq_rows, nc)), sp.lil_matrix((eq_rows, 1))
    auc, aub = sp.lil_matrix((kinds.count(True), nc)), sp.lil_matrix((kinds.count(True), 1))
    b, ub = np.zeros(eq_rows), np.zeros(kinds.count(True))
    gc = sp.lil_matrix((count, nc))
    ac[0, 0], ab[0, 0] = 1., -1.
    eq, le, info = count + 1, 0, []
    for i, inequality in enumerate(kinds):
        col, out, d = 2 + i, 2 + count + i, 1 + i
        pivot = 2. if i % 2 else 1.
        sign = -1 if subtract and i % 2 else 1
        h = .25 if i % 3 == 0 else 0.
        ac[d, 0], ac[d, col] = -.25, pivot
        if i % 2:
            ac[d, 1] = -.125
        b[d] = h
        target, cm, bm, rhs = (le, auc, aub, ub) if inequality else (eq, ac, ab, b)
        cm[target, col], cm[target, out] = -sign * pivot, 1.
        beta = .125 if i % 5 == 0 else 0.
        bm[target, 0], rhs[target] = beta, .125
        gc[i, out] = 1.
        info.append((col, out, d, inequality, target, pivot, sign, h, beta))
        le += inequality
        eq += not inequality
    hz = SparseHZono(np.zeros(count), gc.tocsr(), sp.csr_matrix((count, 1)),
        ac.tocsr(), ab.tocsr(), b, auc.tocsr(), aub.tocsr(), ub, frame_id=51)
    kw = dict(old_n_cont=2, logical_n_cont=2 + count, old_n_eq=1,
        eq_roots=np.arange(1 + count, dtype=np.int64), eq_scales=np.zeros(1 + count, np.int64),
        def_rows=np.zeros(0, np.int64))
    return hz, kw, info


def test_default_off_does_not_inspect_input():
    assert unit.splice(object()) is None


@pytest.mark.parametrize('mixed,subtract', [(False, False), (True, False), (False, True), (True, True)])
def test_actual_compact_transform_and_exact_reconstruction(mixed, subtract):
    hz, kw, info = fixture(mixed=mixed, subtract=subtract)
    before = source_digest(hz)
    result, metrics = unit.splice(hz, enabled=True, **kw)
    proof = audit(hz, result, old_n_eq=kw['old_n_eq'], eq_roots=kw['eq_roots'])
    assert proof['definitions_checked'] == len(info) and proof['reconstruction_from_surviving_rows_proved']
    assert result.hz.n_cont == hz.n_cont and result.hz.n_bin == hz.n_bin
    assert result.hz.frame_id == hz.frame_id and result.hz.exact
    assert metrics['strict_nnz_decrease'] and result.summary['unit_pairs'] == len(info)
    assert result.summary['original_coefficient_nnz'] - result.summary['spliced_coefficient_nnz'] == 2 * len(info)
    assert metrics['spliced_numeric_entries'] < metrics['original_numeric_entries']
    assert metrics['spliced_controlled_bytes'] < metrics['original_controlled_bytes']
    assert set(result.numeric_roots()) == {'hz', 'columns', 'descriptors', 'offsets'}
    assert not any(value is hz for value in vars(result).values())
    decoded = list(result.decoded())
    assert len(decoded) == len(info)
    for z in (-1, 1):
        for x1 in map(F, [-1., -.5, 0., .5, 1.]):
            point = [F(0)] * hz.n_cont
            point[0], point[1] = F(z), x1
            for i, (col, out, d, inequality, target, pivot, sign, h, beta) in enumerate(info):
                p = -F(.25) * point[0] - (F(.125) * point[1] if i % 2 else 0)
                point[out] = F(.125) + sign * (F(h) - p) - F(beta) * z
            extended = result.reconstruct_fraction(point)
            assert extended[:2] == point[:2]
            assert feasible(result.hz, point, (z,)) and feasible(hz, extended, (z,))
            for i, (col, out, d, inequality, target, pivot, sign, h, beta) in enumerate(info):
                p = -F(.25) * point[0] - (F(.125) * point[1] if i % 2 else 0)
                assert extended[col] == (F(h) - p) / F(pivot)
                assert abs(extended[col]) <= 1
            # Perturbing an equality-boundary point violates the same selected
            # consumer in both representations; this is a toy proof fixture.
            point[info[0][1]] += F(1, 16)
            assert not feasible(result.hz, point, (z,))
            assert not feasible(hz, result.reconstruct_fraction(point), (z,))
    assert source_digest(hz) == before
    model = _lower_hz_milp(result.hz, prune_unused=True, coalesce_rows=True,
        project_inactive_cont=False, fix_implied_binary=False)
    assert model.n_cont == hz.n_cont - len(info) and model.n_bin == 1


def test_certificate_serialization_and_sparse_offsets_preserve_content():
    hz, kw, info = fixture(mixed=True, subtract=True)
    result, _ = unit.splice(hz, enabled=True, **kw)
    assert result.columns.dtype == np.int32 and result.descriptors.dtype == np.uint64
    assert result.offsets.size == sum(bool(v[7]) for v in info)
    restored = pickle.loads(pickle.dumps(result, protocol=5))
    restored.validate()
    assert list(restored.decoded()) == list(result.decoded())
    assert source_digest(restored.hz) == source_digest(result.hz)


def test_actual_small_component_must_pay_python_and_certificate_overhead():
    hz, kw, _ = fixture(count=1)
    with pytest.raises(ValueError, match='including certificate/Python'):
        unit.splice(hz, enabled=True, **kw)


def test_no_unit_hit_returns_none_and_preserves_source():
    hz, kw, info = fixture(count=4)
    for col, out, d, inequality, target, *_ in info:
        start, stop = hz.Ac.indptr[target:target + 2]
        position = int(start + np.searchsorted(hz.Ac.indices[start:stop], col))
        hz.Ac.data[position] *= .5
    before = source_digest(hz)
    assert unit.splice(hz, enabled=True, **kw) is None
    assert source_digest(hz) == before


@pytest.mark.parametrize('change', ['column', 'descriptor', 'offset', 'reserved', 'hz', 'summary', 'extra', 'hidden'])
def test_mutation_or_hidden_certificate_payload_fails_closed(change):
    hz, kw, _ = fixture()
    result, _ = unit.splice(hz, enabled=True, **kw)
    if change == 'column': result.columns[0] += 1
    elif change == 'descriptor': result.descriptors[0] ^= np.uint64(1)
    elif change == 'offset': result.offsets[0] += .125
    elif change == 'reserved': result.descriptors[0] |= np.uint64(1 << 50)
    elif change == 'hz': result.hz.Ac.data[0] += .125
    elif change == 'summary': result.summary['formal_gain'] = 1
    elif change == 'extra': result.original = hz
    else: result.summary['hidden'] = np.zeros(4)
    with pytest.raises(ValueError):
        result.numeric_roots()


@pytest.mark.parametrize('change', ['map', 'tag', 'dtype', 'nonfinite', 'window', 'binary_definition',
    'rhs', 'box', 'work', 'increased_work', 'entries', 'increased_entries'])
def test_invalid_structure_arithmetic_or_caps_reject_without_input_change(change):
    hz, kw, info = fixture(count=4)
    if change == 'map': kw['eq_roots'][1] = 0
    elif change == 'tag': kw['eq_roots'][0] = -1
    elif change == 'dtype': hz.Ac = hz.Ac.astype(np.float32)
    elif change == 'nonfinite': hz.b[0] = np.nan
    elif change == 'window': hz.Ac.data[0] = 2.**-21
    elif change == 'binary_definition':
        ab = hz.Ab.tolil()
        ab[1, 0] = .125
        hz.Ab = ab.tocsr()
    elif change == 'rhs': hz.b[1], hz.b[info[0][4]] = .3, .7
    elif change == 'box': hz.b[1] = .875
    elif change == 'work': kw['max_work'] = 0
    elif change == 'increased_work': kw['max_work'] = 256_000_001
    elif change == 'entries': kw['max_entries'] = 0
    else: kw['max_entries'] = 64_000_001
    before = None if change == 'nonfinite' else source_digest(hz)
    with pytest.raises((ValueError, MemoryError)):
        unit.splice(hz, enabled=True, **kw)
    if before is not None:
        assert source_digest(hz) == before


def test_dependent_selected_definitions_are_rejected_as_a_whole():
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-.25, 0., 1., 0., 0.], [0., 0., -1., 1., 0.], [0., 0., 0., -1., 1.]]),
        sp.csr_matrix((3, 1)), np.zeros(3), frame_id=4)
    with pytest.raises(ValueError, match='selected-definition dependencies'):
        unit.splice(hz, enabled=True, old_n_cont=2, logical_n_cont=4, old_n_eq=0,
            eq_roots=np.arange(2, dtype=np.int64), eq_scales=np.zeros(2, np.int64), def_rows=np.zeros(0, np.int64))


def test_complete_cost_is_checked_before_emission(monkeypatch):
    hz, kw, _ = fixture()
    result, _ = unit.splice(hz, enabled=True, **kw)
    def forbidden(*args, **kwargs):
        raise AssertionError('emission happened before complete budget acceptance')
    monkeypatch.setattr(unit, '_emit', forbidden)
    with pytest.raises(MemoryError, match='complete work ceiling'):
        unit.splice(hz, enabled=True, **kw, max_work=result.summary['logical_work_upper'] - 1)


@pytest.mark.parametrize('wrong', ['source', 'map', 'input'])
def test_independent_audit_rejects_wrong_source_mapping_or_input_prefix(wrong):
    hz, kw, _ = fixture()
    result, _ = unit.splice(hz, enabled=True, **kw)
    inp = None
    if wrong == 'source': hz.b[0] += .125
    elif wrong == 'map': kw['eq_roots'][1] = 0
    else:
        inp = SparseHZono(np.zeros(1), sp.csr_matrix((1, 3)), sp.csr_matrix((1, 0)),
            sp.csr_matrix((0, 3)), sp.csr_matrix((0, 0)), np.zeros(0), frame_id=hz.frame_id)
    with pytest.raises(ValueError):
        audit(hz, result, old_n_eq=kw['old_n_eq'], eq_roots=kw['eq_roots'], input_hz=inp)
