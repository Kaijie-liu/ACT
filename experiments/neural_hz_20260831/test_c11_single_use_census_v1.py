from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c11_single_use_census_v1 import census, redundant_box, overlaps, row
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def fixture(inequality=False):
    ac = np.array([[1., 0., 0., 0., 0.], [-.25, -.125, 2., 0., 0.],
                   [-.5, 0., -.5, 4., 0.], [-.5, 0., 0., -.25, 1.]])
    if inequality:
        ac[3, 3] = 0.
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix(ac), sp.csr_matrix([[-1.], [.25], [0.], [.125]]), np.array([0., .25, .125, 0.]),
        sp.csr_matrix([[0., 0., 0., .5, 0.]]) if inequality else sp.csr_matrix((0, 5)),
        sp.csr_matrix([[.25]]) if inequality else sp.csr_matrix((0, 1)),
        np.ones(1) if inequality else np.zeros(0), frame_id=81, exact=True)
    return hz, dict(old_n_cont=2, logical_n_cont=5, old_n_eq=1,
        eq_roots=np.arange(4, dtype=np.int64), eq_scales=np.zeros(4, np.int64), def_rows=np.zeros(0, np.int64))


@pytest.mark.parametrize('inequality', [False, True])
def test_complete_affine_fraction_identity_and_redundant_box(inequality):
    hz, kw = fixture(inequality)
    before = source_digest(hz)
    report, table = census(hz, **kw)
    assert table['column'].tolist() == [2, 3]
    assert report['individually_admissible'] == 2
    assert report['admissible_nonzero_offsets'] == 2
    assert report['admissible_binary_definitions'] == 1
    assert report['admissible_inequality_consumers'] == int(inequality)
    assert report['simultaneous_substitution_proved'] is False
    for k, col in enumerate(table['column']):
        d, t = int(table['defining_row'][k]), int(table['consumer_row'][k])
        le = bool(table['consumer_inequality'][k])
        cmat, bmat, rhs = (hz.Auc, hz.Aub, hz.ub) if le else (hz.Ac, hz.Ab, hz.b)
        definition = {int(i): F(float(v)) for i, v in zip(*row(hz.Ac, d))}
        binary_def = {int(i): F(float(v)) for i, v in zip(*row(hz.Ab, d))}
        consumer = {int(i): F(float(v)) for i, v in zip(*row(cmat, t))}
        binary_consumer = {int(i): F(float(v)) for i, v in zip(*row(bmat, t))}
        pivot, u = definition.pop(int(col)), consumer.pop(int(col))
        ratio = -u / pivot
        assert F(float(table['consumer_multiplier'][k])) == ratio
        target = F(float(rhs[t])) + ratio * F(float(hz.b[d]))
        rewritten_c = dict(consumer)
        rewritten_b = dict(binary_consumer)
        for dst, src in ((rewritten_c, definition), (rewritten_b, binary_def)):
            for i, v in src.items():
                dst[i] = dst.get(i, F(0)) + ratio * v
        original_nnz = len(definition) + 1 + len(binary_def) + len(consumer) + 1 + len(binary_consumer)
        new_nnz = sum(v != 0 for v in rewritten_c.values()) + sum(v != 0 for v in rewritten_b.values())
        assert new_nnz - original_nnz == table['individual_nnz_delta'][k]
        for values in product((-1, 0, 1), repeat=hz.n_cont - 1):
            x = [F(0)] * hz.n_cont
            for i, value in zip([j for j in range(hz.n_cont) if j != col], values):
                x[i] = F(value)
            for z in (-1, 1):
                x[col] = (F(float(hz.b[d])) - sum(v * x[i] for i, v in definition.items())
                          - sum(v * z for v in binary_def.values())) / pivot
                assert abs(x[col]) <= 1
                old_residual = (sum(v * x[i] for i, v in consumer.items()) + u * x[col]
                    + sum(v * z for v in binary_consumer.values()) - F(float(rhs[t])))
                new_residual = (sum(v * x[i] for i, v in rewritten_c.items())
                    + sum(v * z for v in rewritten_b.values()) - target)
                assert old_residual == new_residual
                assert (old_residual <= 0 if le else old_residual == 0) == (new_residual <= 0 if le else new_residual == 0)
    assert source_digest(hz) == before


def small(parent=.3, consumer=-1., overlap=-.7, constant=0., consumer_rhs=0.):
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-parent, 1., 0.], [overlap, consumer, 1.]]), sp.csr_matrix((2, 1)),
        np.array([constant, consumer_rhs]), frame_id=9)
    return hz, dict(old_n_cont=1, logical_n_cont=3, old_n_eq=0,
        eq_roots=np.arange(2, dtype=np.int64), eq_scales=np.zeros(2, np.int64), def_rows=np.zeros(0, np.int64))


@pytest.mark.parametrize('change,guard', [
    ({}, 'collision_sums_exact'),
    ({'parent': .3, 'consumer': -.3, 'overlap': 0.}, 'all_products_exact'),
    ({'parent': 2.**-20, 'consumer': -.5, 'overlap': 0.}, 'products_window_safe'),
    ({'parent': .125, 'constant': .3, 'consumer_rhs': .7, 'overlap': 0.}, 'rhs_exact'),
    ({'parent': .75, 'constant': .5, 'overlap': 0.}, 'redundant_box')])
def test_rejects_each_individual_guard(change, guard):
    hz, kw = small(**change)
    report, table = census(hz, **kw)
    assert report['direct_dead_single_use_definitions'] == 1
    assert not table[guard][0] and report['individually_admissible'] == 0


def test_exact_cancellation_delta_and_constant_only():
    hz, kw = small(parent=.25, overlap=.25)
    report, table = census(hz, **kw)
    assert report['individually_admissible'] == 1
    assert table['cancelled_columns'].tolist() == [1]
    assert table['individual_nnz_delta'].tolist() == [-4]
    for constant in (0., .25):
        hz, kw = small(parent=0., overlap=0., constant=constant)
        report, table = census(hz, **kw)
        assert report['individually_admissible'] == 1
        assert table['definition_width'].tolist() == [1]


def test_existing_alias_tags_and_retained_original_input():
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-.25, 0., 1., 0.], [0., 0., -.5, 1.]]), sp.csr_matrix((2, 1)),
        np.zeros(2), frame_id=27)
    scales = np.zeros(3, np.int64)
    scales.view(np.float64)[0] = .5
    report, table = census(hz, old_n_cont=1, logical_n_cont=4, old_n_eq=0,
        eq_roots=np.array([-1, 0, 1], np.int64), eq_scales=scales, def_rows=np.zeros(0, np.int64))
    assert report['already_eliminated_main_aliases'] == 1
    assert table['column'].tolist() == [2]


@pytest.mark.parametrize('mode', ['map', 'tag', 'shape', 'radix', 'prefix', 'work', 'entries',
    'increased_work', 'increased_entries', 'inexact_domain', 'no_frame', 'nonfinite', 'window'])
def test_fail_closed_structure_and_caps(mode):
    hz, kw = fixture()
    if mode == 'map': kw['eq_roots'][1] = 0
    elif mode == 'tag': kw['eq_roots'][0] = -1
    elif mode == 'shape': kw['eq_roots'] = kw['eq_roots'].reshape(2, 2)
    elif mode == 'radix': kw['def_rows'] = np.array([3], np.int64)
    elif mode == 'prefix': kw['old_n_cont'] = 6
    elif mode == 'work': kw['max_work'] = 0
    elif mode == 'entries': kw['max_entries'] = 0
    elif mode == 'increased_work': kw['max_work'] = 256_000_001
    elif mode == 'increased_entries': kw['max_entries'] = 64_000_001
    elif mode == 'inexact_domain': hz.exact = False
    elif mode == 'no_frame': hz.frame_id = None
    elif mode == 'nonfinite': hz.b[0] = np.nan
    else: hz.Ac.data[0] = 2.**-21
    with pytest.raises((ValueError, MemoryError)):
        census(hz, **kw)


def test_complete_preflight_rejects_before_any_individual_arithmetic():
    hz, kw = fixture()
    report, _ = census(hz, **kw)
    seen = []
    with pytest.raises(MemoryError, match='complete individual arithmetic'):
        census(hz, **kw, max_work=report['logical_work_upper'] - 1, observe=seen.append)
    assert len(seen) == 1 and not seen[0]['arithmetic_cap_fits']
    assert seen[0]['direct_dead_single_use_definitions'] == 2


@pytest.mark.parametrize('coefficients,binary,rhs,pivot', [
    ([.25, -.125], [.25], .25, 2.), ([.25], [], 0., 1.), ([], [], .25, 1.), ([], [], 0., 1.)])
def test_integer_box_envelope_is_sufficient(coefficients, binary, rhs, pivot):
    assert redundant_box(coefficients, binary, rhs, pivot)
    assert sum(map(lambda x: abs(F(x)), coefficients + binary + [rhs]), F(0)) <= F(pivot)


def test_live_output_and_multiple_consumers_not_candidates():
    hz, kw = fixture()
    hz.Gc = sp.csr_matrix([[0., 0., 1., 0., 1.]])
    hz.Auc = sp.csr_matrix([[0., 0., 0., .5, 0.]])
    hz.Aub = sp.csr_matrix((1, 1))
    hz.ub = np.ones(1)
    report, table = census(hz, **kw)
    assert report['direct_dead_single_use_definitions'] == 0
    assert not table['individually_admissible'].size
