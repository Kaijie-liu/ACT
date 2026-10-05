import copy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import (
    RADIX, UID_LIMIT, OwnerEngine, Ledger, word, unique_other, retag, validate_words, append_owned_rows)
from experiments.neural_hz_20260831.c18_owned_emission_v1 import lift
from experiments.neural_hz_20260831.c18_owned_rows_v1 import OwnedRowEncoder, WorkPool, fold_rows
from experiments.neural_hz_20260831.c17_owned_graph_v1 import graph
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import audit as ownership_audit, actual_words, row_uid_tables, phase_event_audit
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift as prior_lift
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit as exact_audit
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz
from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture as conv_fixture
from experiments.neural_hz_20260831.c6_support_affine_plan_v2 import SupportEngine


def test_default_off_does_not_read_input():
    assert lift(object(), object()) is None


@pytest.mark.parametrize('uids', [[], [0], [UID_LIMIT - 1], [1, 5], [0, 7, UID_LIMIT - 1]])
def test_packed_count_sum_exact_and_remove_own(uids):
    packed = sum(map(word, uids))
    values = np.array([packed], np.int64)
    validate_words(values)
    assert packed // RADIX == len(uids) and packed % RADIX == sum(uids)
    if len(uids) == 2:
        assert unique_other(packed, uids[0]) == uids[1]


@pytest.mark.parametrize('uid', [-1, UID_LIMIT, 1.5])
def test_invalid_uid_fails_closed(uid):
    with pytest.raises(ValueError): word(uid)


def test_add_remove_relocate_and_overflow_guard():
    pool = WorkPool(0, 0)
    ledger = Ledger(np.array([word(1) + word(5), word(7)], np.int64), 2, pool=pool)
    ledger.change(np.array([2]), 1, -1)
    ledger.change(np.array([2, 3]), 9, 1)
    ledger.relocate_radix(np.array([2]), 5, 11)
    ledger.finish()
    assert ledger.words.tolist() == [word(11) + word(9), word(7) + word(9)]
    extreme = Ledger(np.array([np.iinfo(np.int64).max], np.int64), 0, pool=WorkPool(0, 0))
    before = extreme.words.copy()
    with pytest.raises(ValueError, match='overflow'):
        extreme.change(np.array([0]), 0, 1)
    assert np.array_equal(extreme.words, before)


def test_owner_engine_and_retag_fail_closed_at_fixed_domains():
    for budget in (-1, 256_000_001):
        with pytest.raises(ValueError): OwnerEngine(budget)
    for values in (np.array([-RADIX], np.int64), np.ones(2, np.float64)):
        with pytest.raises(ValueError): retag(values, UID_LIMIT - 1)


@pytest.mark.parametrize('indices', [np.array([-1, 0]), np.array([1, 0], np.uint64), np.array([0, 0])])
def test_retirement_requires_the_complete_sorted_frontier(indices):
    ledger = Ledger(np.array([word(1), word(2)], np.int64), 0, pool=WorkPool(0, 0))
    before = ledger.words.copy()
    with pytest.raises(ValueError): ledger.retire_verified_frontier(indices)
    assert np.array_equal(ledger.words, before)


def test_sum_is_not_a_membership_proof_and_known_events_are_required():
    # Deliberately demonstrates why an arbitrary external remove operation is
    # NOT authenticated by aggregate words. Only owned source row events may
    # update a generation ledger, followed by a complete independent audit.
    packed = word(1) + word(5)
    assert unique_other(packed, 3) == 3


def test_csr_packed_transport_reuses_local_labels_and_preserves_old_mask_api():
    op = sp.csr_matrix([[1., 0., .5], [0., -.5, 1.], [1., 1., 0.]])
    mask = np.array([True, False, True])
    engine = OwnerEngine()
    local = engine.owners(op, mask)
    expected = np.array([word(0) + word(2), word(2), word(0)], np.int64)
    assert np.array_equal(local, expected)
    work = engine.visits
    assert engine.owners(op, mask) is local and engine.visits == work
    assert np.array_equal(retag(local, 100), np.array([word(100) + word(102), word(102), word(100)]))
    with pytest.raises(ValueError):
        SupportEngine().compute(op, (RADIX + np.arange(3)).astype(np.int64), transpose=True)


@pytest.mark.parametrize('groups,stride,dilation,padding,batch', [
    (1, 1, 1, 1, 1), (2, 1, 1, 1, 2), (1, 2, 1, 1, 1), (2, 1, 2, 2, 1)])
def test_conv_packed_owner_matches_explicit_canonical_rows(groups, stride, dilation, padding, batch, monkeypatch):
    kernel = (np.arange(4 * (4 // groups) * 9).reshape(4, 4 // groups, 3, 3) % 5 - 2) / 8.
    op = ImplicitConv2DOp(kernel, (batch, 4, 5, 5), stride=stride, dilation=dilation, padding=padding, groups=groups)
    op = ImplicitConv2DOp(kernel, op.input_shape, stride=stride, dilation=dilation, padding=padding,
        groups=groups, row_mask=np.arange(op.shape[0]) % 3 != 0)
    matrix = op.to_csr_reference()
    mask = np.arange(op.shape[0]) % 4 != 0
    expected = np.zeros(op.shape[1], np.int64)
    for r in np.flatnonzero(mask):
        a, b = matrix.indptr[r:r + 2]
        expected[matrix.indices[a:b][matrix.data[a:b] != 0.]] += word(int(r))
    def forbidden(*args, **kwargs): raise AssertionError('ownership expanded a convolution')
    monkeypatch.setattr(ImplicitConv2DOp, '_row', forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, 'to_csr_reference', forbidden)
    actual = OwnerEngine().owners(op, mask)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize('kind', ['ordinary', 'old_radix', 'main_radix', 'main_binary', 'shared_conv'])
def test_fresh_owned_generation_matches_exact_quotient_and_full_incidence(kind):
    expr = conv_fixture()[0] if kind == 'shared_conv' else expr_fixture(kind == 'old_radix')
    if kind in ('main_radix', 'main_binary'):
        source = expr.terms[0].source
        matrix = (source.Gc if kind == 'main_radix' else source.Gb).tolil()
        matrix[0, 1 if kind == 'main_radix' else 0] = 1e-40 if kind == 'main_radix' else .25
        if kind == 'main_radix': source.Gc = matrix.tocsr()
        else: source.Gb = matrix.tocsr()
    keep = np.ones(expr.n_out, bool)
    candidate = lift(expr, keep, enabled=True)
    prior = prior_lift(expr, keep, enabled=True)
    original = original_lift(expr, keep, enabled=True)
    assert all(compare_hz(candidate.hz, prior.hz).values())
    assert ownership_audit(candidate)['integer_count_and_UID_sum_equal']
    maps = {k: getattr(original, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    assert exact_audit(candidate, original.hz, maps)['tagged_physical_lineage_checked']
    assert candidate.numeric_roots()['owners'] is candidate.owners
    assert candidate.report['alias_quotient']['already_ordered_rewrites'] >= 0
    if kind == 'main_radix': assert candidate.report['ownership_event_updates'] > 0


@pytest.mark.parametrize('weight', [.3, -.3])
def test_owned_collision_and_cancellation_update_the_same_incidence(weight):
    # MAIN factors1,2, with source column0 protected. Consumer row1 initially
    # uses MAIN1 and MAIN2; the alias defining row0 also uses MAIN1.
    pool = WorkPool(0, 0)
    ledger = Ledger(np.array([word(10) + word(11), word(11)], np.int64), 1, pool=pool)
    enc = OwnedRowEncoder(3, 1, 100, pool=pool, ledger=ledger, radix_uid_base=20)
    roots, scales = [], []
    for uid, cc, cv in ((10, [0, 1], [-.3, 1.]), (11, [0, 1, 2], [-weight, -1., 1.])):
        r, s = enc.encode_uid(uid, np.array(cc), np.array(cv), np.empty(0, np.int64), np.empty(0), 0.)
        roots.append(r); scales.append(s)
    roots, scales, report = fold_rows(enc, roots, scales, old_nc=1, old_eq=0, output_slots=[2])
    assert ledger.words.tolist() == [0, word(11)]
    assert report['collision_groups'] == 1 and report['already_ordered_rewrites'] == 1


def test_nonmonotone_alias_rewrite_keeps_the_full_sort_and_correct_ownership():
    pool = WorkPool(0, 0)
    ledger = Ledger(np.array([word(10) + word(12), word(11) + word(12), word(12)], np.int64), 1, pool=pool)
    enc = OwnedRowEncoder(4, 1, 100, pool=pool, ledger=ledger, radix_uid_base=20)
    roots, scales = [], []
    for uid, cc, cv, rhs in ((10, [0, 1], [-.25, 1.], .125),
            (11, [0, 2], [-.5, 1.], 0.), (12, [1, 2, 3], [-1., -1., 2.], 0.)):
        r, s = enc.encode_uid(uid, np.array(cc), np.array(cv), np.empty(0, np.int64), np.empty(0), rhs)
        roots.append(r); scales.append(s)
    roots, scales, report = fold_rows(enc, roots, scales, old_nc=1, old_eq=0, output_slots=[3])
    assert report['actual_sort_rewrites'] == 1 and report['already_ordered_rewrites'] == 0
    assert ledger.words.tolist() == [word(10) + word(12), 0, word(12)]
    assert enc.eq[-1][0].tolist() == [0, 1, 3]


@pytest.mark.parametrize('stable_first', [False, True])
def test_native_relu_consumption_adds_only_new_predicate_events(stable_first):
    from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
    expr = expr_fixture()
    expr.bias[:] = 100.
    candidate = lift(expr, np.ones(2, bool), enabled=True)
    hz = candidate.hz
    lower = np.array([0. if stable_first else -256., -256.])
    upper = np.full(2, 256.)
    k = int(np.count_nonzero(lower < 0.))
    slots = [(hz.n_cont + 2 * i, hz.n_cont + 2 * i + 1, hz.n_bin + i) for i in range(k)]
    post = sparse_hz_apply_relu_exact(hz, lower, upper, slots, hz.n_cont + 2 * k, hz.n_bin + k)
    ledger = Ledger(candidate.owners.copy(), candidate.old_n_cont, pool=WorkPool(0, 0))
    first = candidate.report['radix_uid_base'] + 16_384
    append_owned_rows(ledger, post.Ac[hz.n_eq:], first)
    append_owned_rows(ledger, post.Auc[hz.n_ineq:], first + k)
    ledger.finish()
    eq, le = row_uid_tables(candidate)
    actual = actual_words(post, candidate.old_n_cont, candidate.logical_n_cont,
        np.r_[eq, np.arange(first, first + k)], np.r_[le, np.arange(first + k, first + 3 * k)])
    assert np.array_equal(ledger.words, actual)
    assert post.n_bin == hz.n_bin + k
    assert ownership_audit(candidate)['integer_count_and_UID_sum_equal']
    original = original_lift(expr, np.ones(2, bool), enabled=True)
    maps = {key: getattr(original, key) for key in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    proof = exact_audit(candidate, original.hz, maps)
    report, columns, words = phase_event_audit(candidate, post, exact_proof=proof)
    assert report['complete_post_incidence_equal'] and np.array_equal(words, actual)
    assert report['new_EQ_rows'] == k and not report['new_native_relu_executed']
    bad = copy.deepcopy(post)
    bad.b[0] += .125
    with pytest.raises(ValueError, match='RHS'):
        phase_event_audit(candidate, bad, exact_proof=proof)


@pytest.mark.parametrize('corrupt', ['words', 'hidden', 'phase'])
def test_mutation_of_owned_state_fails_closed(corrupt):
    candidate = lift(expr_fixture(), np.ones(2, bool), enabled=True)
    if corrupt == 'words': candidate.owners[0] += 1
    elif corrupt == 'phase': candidate.hz.Ab.data[0] += .25
    else: candidate.hidden = np.zeros(8)
    with pytest.raises(ValueError): candidate.numeric_roots()


@pytest.mark.parametrize('kwargs', [{'max_work': 0}, {'max_branch_work': 0}, {'max_entries': 0},
    {'max_work': 256_000_001}, {'max_branch_work': 200_000_001}, {'max_entries': 64_000_001}])
def test_fixed_caps_are_not_relaxed(kwargs):
    with pytest.raises((ValueError, MemoryError)):
        lift(expr_fixture(), np.ones(2, bool), enabled=True, **kwargs)


def test_owner_uid_preflight_before_any_structure_traversal(monkeypatch):
    def forbidden(*args): raise AssertionError('ownership ran before UID acceptance')
    monkeypatch.setattr(OwnerEngine, 'compute', forbidden)
    with pytest.raises(MemoryError, match='UID reservation'):
        graph(expr_fixture(), np.ones(2, bool), 256_000_000, uid_start=UID_LIMIT)


def test_independent_audit_rejects_resealed_ownership_corruption():
    candidate = lift(expr_fixture(), np.ones(2, bool), enabled=True)
    candidate.owners[0] += 1
    candidate.seal = candidate.fingerprint()
    with pytest.raises(ValueError, match='actual sparse incidence'):
        ownership_audit(candidate)


@pytest.mark.parametrize('collision_weight', [None, -.5, .5])
@pytest.mark.parametrize('inequality', [False, True])
def test_wide_sparse_merge_tracks_main_parent_binary_consumer_and_cancellation(collision_weight, inequality):
    pool = WorkPool(0, 0)
    old_nc = 128
    owner = np.array([word(10) + word(12), word(11) + word(14),
        word(12) + word(14), word(13) + word(14)], np.int64)
    if collision_weight is not None: owner[0] += word(14)
    ledger = Ledger(owner, old_nc, pool=pool)
    enc = OwnedRowEncoder(132, 1, 1000, pool=pool, ledger=ledger, radix_uid_base=20)
    roots, scales = [], []
    empty_c, empty_v = np.empty(0, np.int64), np.empty(0)
    for uid, cc, cv, rhs in ((10, [0, 128], [-.25, 1.], .125),
            (11, [1, 129], [-.25, 1.], .125), (12, [128, 130], [-.5, 1.], 0.),
            (13, [2, 131], [-.25, 1.], .125)):
        r, s = enc.encode_uid(uid, np.array(cc), np.array(cv), empty_c, empty_v, rhs)
        roots.append(r); scales.append(s)
    cols = np.r_[np.arange(128), [128] if collision_weight is not None else [], [129, 130, 131]].astype(np.int64)
    vals = np.full(len(cols), -.25)
    vals[cols == 130] = -1.
    if collision_weight is not None: vals[cols == 128] = collision_weight
    # This existing non-defining row is included in old_eq when an EQ; arrange
    # its map first so MAIN definitions still occupy the required suffix.
    r, s = enc.encode_uid(14, cols, vals, np.array([0]), np.array([.5]), .125, inequality=inequality)
    old_eq = 0
    if not inequality:
        roots, scales = [r] + roots, [s] + scales
        old_eq = 1
    new_roots, new_scales, report = fold_rows(enc, roots, scales, old_nc=old_nc,
        old_eq=old_eq, output_slots=[131])
    assert report['normalization_counts'] == {'merge_ordered': 1}
    assert report['selected_aliases'] == 1
    expected_parent = word(10) + (0 if collision_weight == .5 else word(14))
    assert ledger.words.tolist() == [expected_parent, word(11) + word(14), 0, word(13) + word(14)]
    consumer = enc.ineq[-1] if inequality else enc.eq[-1]
    assert consumer[2].tolist() == [0] and consumer[3].tolist() == [.5] and consumer[4] == .125
    assert np.all(np.diff(consumer[0]) > 0) and 130 not in consumer[0]
    assert (128 in consumer[0]) == (collision_weight != .5)
