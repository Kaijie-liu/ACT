import copy
import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c20_product_census_v1 import census
from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift as fused_lift
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def fixture():
    from act.back_end.solver.solver_hz import SparseHZono
    # Two local aliases, the second defined by the first. Both must be counted,
    # even though an independent selected frontier cannot contain them both.
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-.5, 1., 0., 0.], [0., -.3, 1., 0.], [0., .5, -1., 1.]]),
        sp.csr_matrix((3, 1)), np.zeros(3), sp.csr_matrix([[0., .25, .5, 0.]]),
        sp.csr_matrix([[.5]]), np.ones(1), frame_id=9)
    return hz, dict(old_nc=1, logical_nc=4, old_eq=0, eq_roots=np.arange(3, dtype=np.int64),
        def_rows=np.empty(0, np.int64), output_slots=np.array([3]))


def test_complete_local_including_own_alias_parent_and_inequality():
    hz, kwargs = fixture()
    before = source_digest(hz)
    r = census(hz, **kwargs)
    assert r['local_aliases'] == 2 and r['local_right_power_two_aliases'] == 1
    assert r['counts']['hits'] == 5 and r['counts']['right_power_two_hits'] == 3
    assert r['counts']['general_right_power_two_left_hits'] == 2
    assert r['row_classes'] == {'all_dyadic': 1, 'mixed': 2}
    assert source_digest(hz) == before and not r['selected_frontier_only']


@pytest.mark.parametrize('wide', [False, True])
def test_counts_match_fresh_complete_original_fused_fixture(wide):
    expr = expr_fixture(wide)
    old, new = (f(expr, np.ones(expr.n_out, bool), enabled=True) for f in (original_lift, fused_lift))
    root = old.nodes[old.root]
    report = census(old.hz, old_nc=old.old_n_cont, logical_nc=old.logical_n_cont,
        old_eq=old.old_n_eq, eq_roots=old.eq_roots, def_rows=old.def_rows,
        output_slots=root['slots'][root['needed']])
    assert report['local_aliases'] == new.report['alias_quotient']['local_aliases']
    assert report['counts']['hits'] == new.report['alias_quotient']['alias_products_checked']


@pytest.mark.parametrize('cap', [0, 256_000_001])
def test_before_operation_caps(cap, monkeypatch):
    hz, kw = fixture()
    def forbidden(*args, **kwargs): raise AssertionError('allocation before work acceptance')
    monkeypatch.setattr(np, 'zeros', forbidden)
    with pytest.raises((ValueError, MemoryError)): census(hz, **kw, max_work=cap)


@pytest.mark.parametrize('bad', ['roots', 'pivot', 'nan', 'frame', 'output'])
def test_bad_proof_domain_fails_closed(bad):
    hz, kw = fixture()
    if bad == 'roots': kw['eq_roots'][-1] = 0
    if bad == 'pivot': hz.Ac.data[1] = .3
    if bad == 'nan': hz.Ac.data[0] = np.nan
    if bad == 'frame': kw['logical_nc'] = 3
    if bad == 'output': kw['output_slots'][0] = 4
    with pytest.raises(ValueError): census(hz, **kw)
