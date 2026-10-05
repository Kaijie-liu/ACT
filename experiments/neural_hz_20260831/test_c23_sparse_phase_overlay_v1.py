import copy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import (
    RADIX, UID_LIMIT, Overlay, build, pack, validate_events, _checked_sum)
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import (
    BranchPool, check_append, incidence_oracle, verify_all_and_discover)
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def make(base, blocks, old=0, ceiling=10, cap=256_000_000):
    pool = WorkPool(cap)
    return (*build(base, blocks, old_n_cont=old, old_uid_ceiling=ceiling, pool=pool, enabled=True), pool)


def test_default_off_does_not_read_inputs():
    assert build(object(), object(), old_n_cont=object(), old_uid_ceiling=object(), pool=object()) is None


@pytest.mark.parametrize('index,uid', [(0, 0), (UID_LIMIT - 1, UID_LIMIT - 1), (5, 10)])
def test_exact_pair(index, uid):
    value = pack(index, uid)
    assert value >> 20 == index and value % UID_LIMIT == uid and value < RADIX


@pytest.mark.parametrize('index,uid', [(-1, 0), (0, -1), (UID_LIMIT, 0), (0, UID_LIMIT), (True, 0), (0, 1.)])
def test_pair_domains(index, uid):
    with pytest.raises(ValueError): pack(index, uid)


@pytest.mark.parametrize('size', [0, 1, 5])
def test_zero_event_no_copy(size):
    base = np.zeros(size, np.int64)
    overlay, report, pool = make(base, [])
    assert overlay.base is base and overlay.numeric_roots()['base'] is base
    assert overlay.events.nbytes == 0 and not report['dense_phase_copy_allocated']
    assert list(overlay.iter_words(pool=pool)) == base.tolist()
    for i in range(size): assert overlay.query(i, pool=pool) == 0


@pytest.mark.parametrize('sign', [-1., 1.])
def test_repeated_columns_disjoint_EQ_INEQ_rows_and_prefix_filter(sign):
    base = np.array([RADIX + 1, 0, RADIX + 7], np.int64)
    before = base.copy()
    eq = sp.csr_matrix(sign * np.array([[1., 2., 0., 3., 4.], [0., 5., 1., 0., 0.]]))
    le = sp.csr_matrix([[0., -1., 0., 2., 0.]])
    overlay, report, pool = make(base, [(eq, 10), (le, 12)], old=1)
    expected = [4 * RADIX + 1 + 10 + 11 + 12, RADIX + 11, 3 * RADIX + 7 + 10 + 12]
    assert report['event_count'] == 6
    assert list(overlay.iter_words(pool=pool)) == expected
    assert [overlay.query(i, pool=pool) for i in range(3)] == expected
    assert np.array_equal(base, before) and base.flags.writeable
    assert not overlay.events.flags.writeable


@pytest.mark.parametrize('bad', ['duplicate', 'unsorted', 'highbits', 'old_uid', 'outside', 'wrongdtype'])
def test_event_validation(bad):
    events = np.array([pack(0, 10), pack(1, 11)], np.uint64)
    if bad == 'duplicate': events[1] = events[0]
    if bad == 'unsorted': events = events[::-1]
    if bad == 'highbits': events[1] = RADIX
    if bad == 'old_uid': events[0] = pack(0, 9)
    if bad == 'outside': events[1] = pack(2, 11)
    if bad == 'wrongdtype': events = events.astype(np.int64)
    with pytest.raises(ValueError): validate_events(events, 2, 10)


@pytest.mark.parametrize('bad', ['overlap', 'negativeuid', 'limituid', 'narrow', 'zero', 'nan', 'inf', 'duplicate', 'unsorted'])
def test_invalid_append_fails_without_base_mutation(bad):
    base = np.array([RADIX + 1, 0], np.int64)
    original = base.copy()
    matrix = sp.csr_matrix([[1., 1.]])
    first = 10
    if bad == 'overlap': first = 9
    if bad == 'negativeuid': first = -1
    if bad == 'limituid': first = UID_LIMIT
    if bad == 'narrow': matrix = sp.csr_matrix([[1.]])
    if bad == 'zero': matrix.data[0] = 0.
    if bad == 'nan': matrix.data[0] = np.nan
    if bad == 'inf': matrix.data[0] = np.inf
    if bad in ('duplicate', 'unsorted'):
        assert matrix.has_canonical_format  # Prime the scipy cache deliberately.
        matrix.indices[:] = [0, 0] if bad == 'duplicate' else [1, 0]
    with pytest.raises(ValueError): make(base, [(matrix, first)])
    assert np.array_equal(base, original)


def test_two_blocks_cannot_reuse_UID_even_if_different_columns():
    base = np.zeros(2, np.int64)
    with pytest.raises(ValueError):
        make(base, [(sp.csr_matrix([[1., 0.]]), 10), (sp.csr_matrix([[0., 1.]]), 10)])


@pytest.mark.parametrize('base', [np.array([-1], np.int64), np.array([11 * RADIX], np.int64),
    np.array([RADIX + 10], np.int64), np.zeros((1, 1), np.int64), np.zeros(1)])
def test_base_domain_guards(base):
    with pytest.raises(ValueError): make(base, [])


def test_maximum_valid_sum_and_carry_guards():
    assert _checked_sum(RADIX + UID_LIMIT - 1, 0, 0) == RADIX + UID_LIMIT - 1
    with pytest.raises(ValueError): _checked_sum(UID_LIMIT * RADIX, 1, 0)
    with pytest.raises(ValueError): _checked_sum(RADIX - 1, 1, 1)
    with pytest.raises(ValueError): _checked_sum(-1, 0, 0)


@pytest.mark.parametrize('stage', ['base', 'scan', 'pack', 'array', 'sort', 'validation'])
def test_cap_before_every_build_stage(stage):
    class Stop:
        used = 0
        def charge(self, name, amount):
            target = {'base': 'overlay_base_validation', 'scan': 'overlay_appended_rows_scan',
                'pack': 'overlay_event_pack', 'array': 'overlay_event_array',
                'sort': 'overlay_event_sort', 'validation': 'overlay_event_validation'}[stage]
            if name == target: raise MemoryError(name)
            self.used += amount
    base = np.array([RADIX + 1], np.int64)
    with pytest.raises(MemoryError):
        build(base, [(sp.csr_matrix([[1.]]), 10)], old_n_cont=0, old_uid_ceiling=10, pool=Stop(), enabled=True)
    assert base.tolist() == [RADIX + 1]


def test_query_caps_and_boundaries():
    overlay, report, _ = make(np.zeros(1, np.int64), [(sp.csr_matrix([[1.]]), 10)])
    for key in (-1, 1, 0.):
        with pytest.raises(ValueError): overlay.query(key, pool=WorkPool(0))
    with pytest.raises(MemoryError): overlay.query(0, pool=WorkPool(0))
    with pytest.raises(MemoryError): list(overlay.iter_words(pool=WorkPool(0)))
    with pytest.raises(MemoryError): overlay.query(0, pool=WorkPool(32))
    with pytest.raises(MemoryError): list(overlay.iter_words(pool=WorkPool(16)))


def test_branch_and_whole_caps_precede_mutation():
    for whole, branch in [(0, 10), (10, 0)]:
        pool = WorkPool(whole)
        nested = BranchPool(pool, branch)
        with pytest.raises(MemoryError): nested.charge('x', 1)
        assert nested.used == 0 and pool.used == 0
    with pytest.raises(ValueError): BranchPool(WorkPool(10), 200_000_001)


def test_structural_reseal_is_not_a_source_incidence_proof():
    overlay, _, pool = make(np.zeros(1, np.int64), [(sp.csr_matrix([[1.]]), 10)])
    corrupted = Overlay(overlay.base, np.array([pack(0, 11)], np.uint64), 10)
    corrupted.validate()  # Deliberate counterexample to using a flag as proof.
    assert corrupted.query(0, pool=pool) != RADIX + 10


def test_complete_incidence_oracle_counts_both_predicate_kinds_and_signed_rows():
    hz = SimpleNamespace(n_cont=4, n_eq=2, n_ineq=2,
        Ac=sp.csr_matrix([[1., -1., 0., .5], [0., 2., -1., 0.]]),
        Auc=sp.csr_matrix([[0., -.5, 1., 0.], [0., 0., 0., 0.]]))
    eq, le = np.array([1, 2]), np.array([10, 11])
    result = incidence_oracle(hz, eq, le, 1, 3, pool=WorkPool(256_000_000))
    assert result.tolist() == [3 * RADIX + 13, 2 * RADIX + 12]
    with pytest.raises(ValueError): incidence_oracle(hz, eq, np.array([2, 11]), 1, 3, pool=WorkPool(256_000_000))
    with pytest.raises(MemoryError): incidence_oracle(hz, eq, le, 1, 3, pool=WorkPool(0))


def test_complete_stream_mismatch_is_not_accepted_as_partial_proof():
    base = np.zeros(2, np.int64)
    overlay, _, _ = make(base, [])
    candidate = SimpleNamespace(old_n_cont=0, logical_n_cont=2, old_n_eq=0,
        eq_roots=np.array([-1, -1]))
    post = SimpleNamespace(Gc=sp.csr_matrix((1, 2)), n_cont=2)
    whole = WorkPool(256_000_000)
    with pytest.raises(ValueError, match='differs'):
        verify_all_and_discover(candidate, post, overlay, np.array([0, RADIX]),
            np.zeros(0, np.int64), np.zeros(0, np.int64), whole=whole, branch=BranchPool(whole))


@pytest.mark.parametrize('stable_first', [False, True])
def test_native_phase_complete_oracle_and_all_pair_discovery(stable_first):
    from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
    from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture
    from experiments.neural_hz_20260831.c17_owned_emission_v1 import lift
    from experiments.neural_hz_20260831.c17_ownership_audit_v1 import row_uid_tables, phase_event_audit
    from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
    from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit
    expr = expr_fixture()
    expr.bias[:] = 100.
    candidate = lift(expr, np.ones(2, bool), enabled=True)
    original = original_lift(expr, np.ones(2, bool), enabled=True)
    maps = {k: getattr(original, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    proof = audit(candidate, original.hz, maps)
    pre = candidate.hz
    lower = np.array([0. if stable_first else -256., -256.])
    k = int(np.count_nonzero(lower < 0.))
    slots = [(pre.n_cont + 2*i, pre.n_cont + 2*i + 1, pre.n_bin + i) for i in range(k)]
    post = sparse_hz_apply_relu_exact(pre, lower, np.full(2, 256.), slots, pre.n_cont + 2*k, pre.n_bin + k)
    check_append(pre, post)
    whole = WorkPool(256_000_000)
    branch = BranchPool(whole)
    first = candidate.report['radix_uid_base'] + 16384
    eq, le = row_uid_tables(candidate)
    base = incidence_oracle(pre, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=whole)
    assert np.array_equal(base, candidate.owners)
    overlay, report = build(base, [(post.Ac[pre.n_eq:], first), (post.Auc[pre.n_ineq:], first + k)],
        old_n_cont=candidate.old_n_cont, old_uid_ceiling=first, pool=branch, enabled=True)
    eq, le = np.r_[eq, np.arange(first, first+k)], np.r_[le, np.arange(first+k, first+3*k)]
    actual = incidence_oracle(post, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=whole)
    columns, checked = verify_all_and_discover(candidate, post, overlay, actual, eq, le, whole=whole, branch=branch)
    prior, old_columns, old_words = phase_event_audit(candidate, post, exact_proof=proof)
    assert np.array_equal(actual, old_words) and np.array_equal(columns, old_columns)
    assert checked['all_post_MAIN_columns_checked'] == len(base)
    for name in ('frame', 'rhs', 'binary_prefix'):
        bad = copy.deepcopy(post)
        if name == 'frame': bad.frame_id = 'changed'
        elif name == 'rhs': bad.b[0] += .125
        else: bad.Ab.data[0] += .125
        with pytest.raises(ValueError): check_append(pre, bad)
