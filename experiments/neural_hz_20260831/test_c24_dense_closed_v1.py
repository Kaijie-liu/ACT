import copy
import gc
import hashlib
import pickle
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c24_dense_ownership_v1 import DenseOwnerEngine, RADIX, UID_LIMIT, retag
from experiments.neural_hz_20260831.c24_dense_emission_v1 import lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import Draft, Closed, close, export, restore, _Receipt
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import build_slabs, node_uid_tables, closed_uid_tables, resolve
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as overlay_build
from experiments.neural_hz_20260831.c22_uid_runs_v1 import unpack, row_for_uid, uid_for_row
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool as WholePool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool, check_append, incidence_oracle, verify_all_and_discover
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift as fused_lift
from experiments.neural_hz_20260831.c10_portable_binding_v1 import reconstruct_fraction
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz
from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture as conv_fixture


def fixture(kind='ordinary'):
    expr = conv_fixture()[0] if kind == 'shared_conv' else expr_fixture(kind == 'old_radix')
    if kind in ('main_radix', 'main_binary'):
        source = expr.terms[0].source
        matrix = (source.Gc if kind == 'main_radix' else source.Gb).tolil()
        matrix[0, 1 if kind == 'main_radix' else 0] = 1e-40 if kind == 'main_radix' else .25
        if kind == 'main_radix': source.Gc = matrix.tocsr()
        else: source.Gb = matrix.tocsr()
    mask = np.ones(expr.n_out, bool)
    draft = lift(expr, mask, enabled=True)
    original = original_lift(expr, mask, enabled=True)
    maps = {k: getattr(original, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    return draft, original, maps


def test_default_off_no_input_access():
    assert lift(object(), object()) is None
    assert close(object(), object(), object()) is None
    assert overlay_build(object(), object(), pool=object()) is None


@pytest.mark.parametrize('mask', [[False, True, False, True], [True, True, True, True], [False]*4])
def test_dense_active_rank_reverse_exact(mask):
    matrix = sp.csr_matrix([[1., 0., .5], [0., -.5, 1.], [1., 1., 0.], [0., 1., 1.]])
    mask = np.array(mask)
    engine = DenseOwnerEngine()
    result = engine.owners(matrix, mask)
    expected = np.zeros(3, np.int64)
    for rank, row in enumerate(np.flatnonzero(mask)):
        a, b = matrix.indptr[row:row+2]
        expected[matrix.indices[a:b]] += RADIX + rank
    assert np.array_equal(result, expected)
    before = engine.visits
    assert engine.owners(matrix, mask) is result and engine.visits == before


@pytest.mark.parametrize('counts', [[], [(5, 0)], [(3, 3), (2, 2)], [(5, 2), (4, 3), (2, 0), (3, 1)]])
def test_slabs_cover_dense_slots_and_all_reserved_holes(counts):
    nodes = [dict(width=w, auxiliaries=n) for w, n in counts]
    pool = WorkPool(0, 0)
    words = build_slabs(nodes, 4, pool=pool)
    uid, main, expected = 4, 0, {}
    for width, count in counts:
        for rank in range(count): expected[uid+rank] = main+rank
        uid += width; main += count
    for key in range(uid+1): assert row_for_uid(words, key, pool=pool) == expected.get(key)
    inverse = {m:u for u,m in expected.items()}
    for key in range(main+1): assert uid_for_row(words, key, pool=pool) == inverse.get(key)
    if counts == [(3, 3), (2, 2)]: assert len(words) == 1


@pytest.mark.parametrize('counts,first', [([(1, 2)], 0), ([(-1, 0)], 0), ([(1, 1)], UID_LIMIT), ([(1, 1)], -1)])
def test_invalid_slab_domains(counts, first):
    with pytest.raises(ValueError): build_slabs([dict(width=w, auxiliaries=n) for w,n in counts], first, pool=WorkPool(0,0))


def test_slab_charge_before_pack():
    with pytest.raises(MemoryError): build_slabs([dict(width=1, auxiliaries=1)], 0, pool=WorkPool(0,0,max_work=0))


@pytest.mark.parametrize('kind', ['ordinary', 'old_radix', 'main_radix', 'main_binary', 'shared_conv'])
def test_fresh_dense_generator_and_closed_source_proof(kind):
    draft, original, maps = fixture(kind)
    fused = fused_lift(draft.expression, draft.keep, enabled=True)
    assert all(compare_hz(draft.hz, fused.hz).values())
    eq, le, main = node_uid_tables(draft)
    assert np.array_equal(actual_words(draft.hz, draft.old_n_cont, draft.logical_n_cont, eq, le), draft.owners)
    closed, proof = close(draft, original.hz, maps, enabled=True)
    assert proof['all_MAIN_ownership_checked'] == len(draft.owners)
    assert all(compare_hz(closed.hz, draft.hz).values())
    assert not hasattr(closed, 'nodes') and closed.owners is draft.owners
    assert np.array_equal(closed_uid_tables(closed)[0], eq)
    pool = WorkPool(0,0)
    expected = {int(u):(False,r) for r,u in enumerate(eq)}
    expected.update({int(u):(True,r) for r,u in enumerate(le)})
    for uid in range(closed.report['radix_uid_base'] + len(closed.def_rows) + 1):
        assert resolve(closed, uid, pool=pool) == expected.get(uid)
    assert reconstruct_fraction(closed, np.zeros(closed.hz.n_cont)) == reconstruct_fraction(draft, np.zeros(draft.hz.n_cont))


@pytest.mark.parametrize('change', ['hz', 'source', 'owners', 'slabs', 'maps', 'hidden', 'receipt'])
def test_changed_state_and_outer_reseal_cannot_forge_receipt(change):
    draft, original, maps = fixture()
    closed, proof = close(draft, original.hz, maps, enabled=True)
    if change == 'hz': closed.hz.b[0] += .125
    elif change == 'source': closed.expression.bias[0] += .125
    elif change == 'owners': closed.owners[0] += 1
    elif change == 'slabs': closed.uid_slabs[0] += np.uint64(1)
    elif change == 'maps': closed.eq_scales[0] += 1
    elif change == 'hidden': closed.extra = np.zeros(4)
    else: closed.receipt = object()
    try: closed.seal = closed.fingerprint()
    except ValueError: pass
    with pytest.raises(ValueError): closed.validate()


def test_receipt_not_serializable_or_issued_from_exact_flag():
    with pytest.raises(ValueError): _Receipt(object())
    draft, original, maps = fixture()
    with pytest.raises(ValueError): overlay_build(draft, [], pool=WorkPool(0,0), enabled=True)
    closed, _ = close(draft, original.hz, maps, enabled=True)
    with pytest.raises(TypeError): pickle.dumps(closed.receipt)


def test_proof_failure_never_retires_or_mints_closed_state():
    draft, original, maps = fixture()
    draft.owners[0] += 1
    draft.seal = draft.fingerprint()
    with pytest.raises(ValueError, match='ownership'): close(draft, original.hz, maps, enabled=True)
    assert draft.nodes and 'slots' in draft.nodes[0]


def test_registry_keeps_no_graph_array_and_authenticated_portable_restore():
    draft, original, maps = fixture()
    removed = [weakref.ref(n[k]) for n in draft.nodes for k in ('support', 'needed', 'slots', 'exponents')]
    closed, proof = close(draft, original.hz, maps, enabled=True)
    payload, raw = export(closed)
    del draft
    gc.collect()
    assert all(ref() is None for ref in removed)
    restored_fields = pickle.loads(pickle.dumps(payload))
    restored = restore(restored_fields, raw, expected_proof_sha256=hashlib.sha256(raw).hexdigest())
    restored.validate()
    assert all(compare_hz(restored.hz, closed.hz).values())
    with pytest.raises(ValueError): restore(payload, raw, expected_proof_sha256='0'*64)
    restored_fields['owners'][0] += 1
    with pytest.raises(ValueError): restore(restored_fields, raw, expected_proof_sha256=hashlib.sha256(raw).hexdigest())


def test_complete_source_math_cannot_be_skipped_by_exact_flag_and_reseal():
    draft, original, maps = fixture()
    draft.hz.b[0] += .125
    draft.hz.exact = True
    draft.seal = draft.fingerprint()
    with pytest.raises(ValueError): close(draft, original.hz, maps, enabled=True)


def test_same_dense_mask_rank_is_shared_across_sum_parents():
    # A shared Conv/Sum fixture exercises parents with different live support;
    # the independent row oracle, not a per-parent rank, decides every UID.
    draft, original, maps = fixture('shared_conv')
    sums = [n for n in draft.nodes if n['kind'] == 'sum']
    assert sums
    assert draft.report['sum_dense_rank_extra_work'] == 2 * sum(n['width'] for n in sums)
    closed, proof = close(draft, original.hz, maps, enabled=True)
    assert proof['complete_graph_free_row_maps_equal']
    assert 'rewrite_order_check_emit' not in draft.report['alias_quotient']['work_parts']
    assert draft.report['alias_quotient']['normalization_counts']['uniform_full_stable_sort'] > 0


@pytest.mark.parametrize('stable_first', [False, True])
def test_checked_overlay_native_phase_full_incidence(stable_first):
    from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
    draft, original, maps = fixture()
    closed, proof = close(draft, original.hz, maps, enabled=True)
    pre = closed.hz
    lower = np.array([0. if stable_first else -256., -256.])
    k = int(np.count_nonzero(lower < 0.))
    slots = [(pre.n_cont+2*i, pre.n_cont+2*i+1, pre.n_bin+i) for i in range(k)]
    post = sparse_hz_apply_relu_exact(pre, lower, np.full(2,256.), slots, pre.n_cont+2*k, pre.n_bin+k)
    check_append(pre, post)
    first = closed.report['radix_uid_base'] + 16384
    pool = WorkPool(0,0)
    overlay, report = overlay_build(closed, [(post.Ac[pre.n_eq:],first),(post.Auc[pre.n_ineq:],first+k)], pool=pool, enabled=True)
    assert 'overlay_base_validation' not in pool.parts
    whole = WholePool(256_000_000)
    eq, le = closed_uid_tables(closed)
    eq, le = np.r_[eq,np.arange(first,first+k)], np.r_[le,np.arange(first+k,first+3*k)]
    actual = incidence_oracle(post,eq,le,closed.old_n_cont,closed.logical_n_cont,pool=whole)
    assert np.array_equal(list(overlay.iter_words(pool=pool)),actual)
    columns, report = verify_all_and_discover(closed,post,overlay,actual,eq,le,whole=whole,branch=BranchPool(whole))
    assert report['complete_post_incidence_equal'] and post.n_bin == pre.n_bin+k


@pytest.mark.parametrize('kwargs', [{'max_work':0},{'max_branch_work':0},{'max_entries':0},
    {'max_work':256_000_001},{'max_branch_work':200_000_001},{'max_entries':64_000_001}])
def test_caps_unchanged(kwargs):
    with pytest.raises((MemoryError,ValueError)): lift(expr_fixture(),np.ones(2,bool),enabled=True,**kwargs)
