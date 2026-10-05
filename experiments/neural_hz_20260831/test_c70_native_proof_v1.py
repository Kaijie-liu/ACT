"""Ordinary complete phase, mixed-row, local-inverse and population checks."""
from dataclasses import asdict
from types import SimpleNamespace
import pickle
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.c70_native_proof_v1 import extract, bind_phase, digest, factor_plans, verify, verify_inverse
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c68_local_splice_v1 import compile_journal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA, encode
from experiments.neural_hz_20260831.c69_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c65_full_source_audit_v1 import audit
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from, full_reference
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete, old_fields


def pool(): return WorkPool(256_000_000)


def journal(c, plans):
    return compile_journal(c.eq_roots, c.eq_scales, plans, old_n_cont=c.old_n_cont,
        old_n_eq=c.old_n_eq, source_n_cont=c.hz.n_cont, source_schema=SCHEMA, pool=pool(), enabled=True)


def packet(c, h):
    return extract(c.hz, h, old_n_cont=c.old_n_cont, old_n_eq=c.old_n_eq,
        logical_n_cont=c.logical_n_cont, first_uid=c.report['radix_uid_base']+16384,
        provenance={'test_only': True}, pool=pool(), enabled=True)


def test_default_off_does_not_read_any_input():
    assert bind_phase(object(), object(), pool=object()) is None
    assert extract(object(), object(), old_n_cont=None, old_n_eq=None,
        logical_n_cont=None, first_uid=None, provenance=None, pool=object()) is None


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('subtract', [False, True])
@pytest.mark.parametrize('phase', [False, True])
def test_all_rows_factor_population_and_inverse(mixed, subtract, phase):
    c, h, overlay, *_ = source(mixed=mixed, subtract=subtract, phase=phase)
    p = packet(c, h); before = digest(p)
    p = pickle.loads(pickle.dumps(p, protocol=5))
    assert digest(p) == before
    view = bind_phase(c, p, pool=pool(), enabled=True)
    plans, _ = discover_append(c, view, overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view, plans, pool=pool(), enabled=True)
    j = journal(c, plans)
    proof = verify(c, view, overlay, plans, new, j, pool=pool())
    ref = full_reference(h, plans)
    for k, expected in ref.items():
        got = getattr(new, k)
        assert np.array_equal(got.toarray() if sp.issparse(got) else got, expected)
    assert proof['complete_plans'] == len(plans) == 12
    assert proof['all_written_predicate_coefficients'] == sum(getattr(new,k).nnz for k in ('Ac','Ab','Auc','Aub'))
    assert verify_inverse(c, new, j, plans, pool=pool())['unit_equations'] == 12
    assert digest(p) == before


@pytest.mark.parametrize('change', ['no_pivot', 'output_live', 'binary_definition',
    'lower_consumer_head', 'extra_old_consumer', 'empty_consumer_row', 'longer_tail'])
def test_complete_factor_rule_preserves_ordinary_exclusions(change):
    c, h, overlay, *_ = source(change=change)
    view = view_from(c, h)
    plans, _ = discover_append(c, view, overlay, pool=pool(), enabled=True)
    expected, *_ = factor_plans(c, view, first_uid=overlay.old_uid_ceiling, pool=pool())
    assert [asdict(p) for p in plans] == [asdict(p) for p in expected]


@pytest.mark.parametrize('old_consumers', [0, 6, 12])
def test_all_old_and_new_consumers_are_included(old_consumers):
    c, h, overlay, *_ = source(phase=True, old_consumers=old_consumers)
    view = view_from(c, h)
    plans, stats = discover_append(c, view, overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view, plans, pool=pool(), enabled=True)
    proof = verify(c, view, overlay, plans, new, journal(c, plans), pool=pool())
    assert stats['selected_old_consumers'] == old_consumers
    assert stats['selected_new_consumers'] == 12-old_consumers
    assert proof['complete_plans'] == 12


@pytest.mark.parametrize('key', ['frame_id', 'source_n_cont', 'source_n_bin',
    'old_n_cont', 'old_n_eq', 'logical_n_cont', 'first_uid', 'pre_c'])
def test_complete_phase_binding_rejects_mismatch(key):
    c, h, *_ = source(phase=True)
    p = packet(c, h)
    if key == 'pre_c': p[key][0] += .125
    else: p[key] += 1
    with pytest.raises(ValueError, match='binding'):
        bind_phase(c, p, pool=pool(), enabled=True)


@pytest.mark.parametrize('key', ['Ac', 'Ab', 'Auc', 'Aub', 'b', 'ub'])
def test_all_written_predicate_components_are_independently_checked(key):
    c, h, overlay, *_ = source(mixed=True, phase=True)
    view = view_from(c, h)
    plans, _ = discover_append(c, view, overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view, plans, pool=pool(), enabled=True)
    value = getattr(new, key)
    if sp.issparse(value):
        assert value.nnz > 0
        value.data[0] += .125
    else: value[0] += .125
    with pytest.raises(ValueError): verify(c, view, overlay, plans, new, journal(c, plans), pool=pool())


@pytest.mark.parametrize('subtract', [False, True])
def test_nonempty_unit_and_chained_local_inverse(subtract):
    c, h, overlay, *_ = source(redirect=True, subtract=subtract)
    view = view_from(c, h)
    plans, _ = discover_append(c, view, overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view, plans, pool=pool(), enabled=True)
    old_width = h.n_cont
    tags, nums = zip(encode(plans[0].column, (3,-2)), encode(old_width, (-1,-1)))
    def widen(m):
        return sp.csr_matrix((m.data,m.indices,m.indptr),shape=(m.shape[0],old_width+2))
    def wide(hz):
        return SparseHZono(hz.c,widen(hz.Gc),hz.Gb,widen(hz.Ac),hz.Ab,hz.b,
            widen(hz.Auc),hz.Aub,hz.ub,frame_id=hz.frame_id,exact=True)
    fields = dict(vars(c)); fields['hz'] = wide(c.hz)
    fields['eq_roots'] = np.r_[c.eq_roots,np.asarray(tags,np.int64)]
    fields['eq_scales'] = np.r_[c.eq_scales,np.asarray(nums,np.float64).view(np.int64)]
    c = SimpleNamespace(**fields)
    proof = verify_inverse(c, wide(new), journal(c, plans), plans, pool=pool())
    assert proof['unit_equations'] == 12 and proof['local_equations'] == 2


@pytest.mark.parametrize('kind', ['chain', 'shared', 'conv_disjoint'])
def test_fresh_C69_complete_source_and_actual_native_phase(kind):
    _, saved = complete(kind)
    legacy, _ = old_fields(saved)
    fresh = lift(saved['expression'], saved['keep'], enabled=True)
    source_proof = audit(saved, legacy, fresh, pool=pool(), enabled=True)
    c = SimpleNamespace(**fresh['fields']); h = c.hz
    slots = [(h.n_cont+2*i,h.n_cont+2*i+1,h.n_bin+i) for i in range(h.n_out)]
    post = sparse_hz_apply_relu_exact(h,np.full(h.n_out,-8.),np.full(h.n_out,8.),
        slots,h.n_cont+2*h.n_out,h.n_bin+h.n_out)
    p = packet(c, post); view = bind_phase(c,p,pool=pool(),enabled=True)
    first = p['first_uid']
    overlay, _ = build(c.owners,[(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
        old_n_cont=c.old_n_cont,old_uid_ceiling=first,pool=pool(),enabled=True)
    try:
        expected, *_ = factor_plans(c,view,first_uid=first,pool=pool())
    except ValueError as exc:
        assert 'dependencies' in str(exc)
        with pytest.raises(ValueError,match='simultaneous independent'):
            discover_append(c,view,overlay,pool=pool(),enabled=True)
        return
    plans, _ = discover_append(c,view,overlay,pool=pool(),enabled=True)
    assert [asdict(v) for v in plans] == [asdict(v) for v in expected]
    assert source_proof['local_inverse_equations'] > 0
    if plans:
        new, _ = splice_append(view,plans,pool=pool(),enabled=True)
        assert verify(c,view,overlay,plans,new,journal(c,plans),pool=pool())['complete_plans'] == len(plans)


def test_missing_plan_and_work_exhaustion_fail_closed():
    c,h,o,*_ = source()
    view = view_from(c,h)
    plans,_ = discover_append(c,view,o,pool=pool(),enabled=True)
    new,_ = splice_append(view,plans,pool=pool(),enabled=True)
    with pytest.raises(ValueError,match='populations'):
        verify(c,view,o,plans[:-1],new,journal(c,plans),pool=pool())
    with pytest.raises(MemoryError):
        factor_plans(c,view,first_uid=o.old_uid_ceiling,pool=WorkPool(0))
