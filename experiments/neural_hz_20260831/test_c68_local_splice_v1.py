"""Ordinary full-row/source/owner proofs for the new explicit inverse reader."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import pickle
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.c68_local_splice_v1 import compile_journal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA, encode
from experiments.neural_hz_20260831.c67_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c65_full_source_audit_v1 import audit
from experiments.neural_hz_20260831.c32_native_blocks_v1 import phase_blocks
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import same_matrix, equal
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete, old_fields, residual
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from, full_reference
from experiments.neural_hz_20260831.test_c10_alias_quotient_v1 import feasible


def pool():
    return WorkPool(256_000_000)


def journal(c, plans):
    return compile_journal(c.eq_roots, c.eq_scales, plans, old_n_cont=c.old_n_cont,
        old_n_eq=c.old_n_eq, source_n_cont=c.hz.n_cont, source_schema=SCHEMA,
        pool=pool(), enabled=True)


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('subtract', [False, True])
@pytest.mark.parametrize('phase', [False, True])
def test_complete_native_splice_rows_owners_UIDs_and_original_input(mixed, subtract, phase):
    c, h, overlay, eq, le, words, info = source(mixed=mixed, subtract=subtract, phase=phase)
    before = (source_digest(h), c.eq_roots.tobytes(), c.eq_scales.tobytes())
    plans, stats = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, report = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    j = journal(c, plans)
    assert len(plans) == len(info) and j.eq_roots is c.eq_roots and j.eq_scales is c.eq_scales
    for name, expected in full_reference(h, plans).items():
        value = getattr(new, name)
        assert np.array_equal(value.toarray() if sp.issparse(value) else value, expected)
    keep = np.ones(h.n_eq, bool)
    keep[[p.definition for p in plans]] = False
    eq, le = eq.copy(), le.copy()
    for p in plans:
        (le if p.inequality else eq)[p.consumer] = p.producer_uid
        assert j.retired_to(p.consumer_uid, pool=pool()) == p.producer_uid
    expected = actual_words(new, c.old_n_cont, c.logical_n_cont, eq[keep], le)
    assert list(j.iter_words(overlay, pool=pool())) == expected.tolist()
    assert [j.owner_query(i, overlay, pool=pool()) for i in range(len(words))] == expected.tolist()
    assert [j.eq_row(i, pool=pool()) for i in range(h.n_eq)] == [int(keep[:i].sum()) if keep[i] else None for i in range(h.n_eq)]
    for z in (-1, 1):
        for x in (F(-1), F(0), F(1)):
            point = [F(0)] * h.n_cont
            point[0], point[1] = F(z), x
            for i, (col, out, d, kind, target, pivot, sign, offset, beta) in enumerate(info):
                prefix = -F(.25) * point[0] - (F(.125) * point[1] if i % 2 else 0)
                point[out] = F(.125) + sign * (F(offset) - prefix) - F(beta) * z
            restored = j.reconstruct_fraction(new, point, pool=pool())
            assert restored[:c.old_n_cont] == point[:c.old_n_cont]
            assert feasible(new, point, (z,)) and feasible(h, restored, (z,))
    assert before == (source_digest(h), c.eq_roots.tobytes(), c.eq_scales.tobytes())
    assert sum(getattr(j, k).nbytes for k in ('columns', 'tags', 'offsets', 'retired', 'tails')) == 28 * len(plans) + 8 * len(j.tails)


@pytest.mark.parametrize('subtract', [False, True])
def test_local_children_of_unit_producers_restore_after_splice_and_keep_shared_maps(subtract):
    c, h, overlay, eq, le, words, info = source(redirect=True, subtract=subtract)
    plans, _ = discover_append(c, view_from(c, h), overlay, pool=pool(), enabled=True)
    new, _ = splice_append(view_from(c, h), plans, pool=pool(), enabled=True)
    old_width = h.n_cont
    tags, nums = zip(encode(plans[0].column, (3, -2)), encode(old_width, (-1, -1)))
    roots = np.r_[c.eq_roots, np.array(tags, np.int64)]
    scales = np.r_[c.eq_scales, np.array(nums, np.float64).view(np.int64)]
    def widen(m):
        return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], old_width + 2))
    expanded = SparseHZono(new.c, widen(new.Gc), new.Gb, widen(new.Ac), new.Ab, new.b,
        widen(new.Auc), new.Aub, new.ub, frame_id=new.frame_id, exact=True)
    j = compile_journal(roots, scales, plans, old_n_cont=c.old_n_cont, old_n_eq=c.old_n_eq,
        source_n_cont=old_width + 2, source_schema=SCHEMA, pool=pool(), enabled=True)
    point = [F(0)] * expanded.n_cont
    point[0] = F(1)
    full = j.reconstruct_fraction(expanded, point, pool=pool())
    assert full[old_width] == F(3, 4) * full[plans[0].column]
    assert full[old_width + 1] == -F(1, 2) * full[old_width]
    assert full[plans[0].column] != 0 and full[:2] == point[:2]
    # The original two removed local equations both have exactly zero residual.
    assert 4 * full[old_width] - 3 * full[plans[0].column] == 0
    assert 2 * full[old_width + 1] + full[old_width] == 0
    restored = pickle.loads(pickle.dumps(dict(journal=j, source_roots=roots, source_scales=scales), protocol=5))
    assert restored['journal'].eq_roots is restored['source_roots']
    assert restored['journal'].eq_scales is restored['source_scales']
    assert restored['journal'].reconstruct_fraction(expanded, point, pool=pool()) == full


@pytest.mark.parametrize('kind', ['chain', 'shared', 'conv_disjoint'])
def test_fresh_C67_full_source_native_phase_and_exact_all_row_inverse(kind):
    _, saved = complete(kind)
    legacy, _ = old_fields(saved)
    candidate = lift(saved['expression'], saved['keep'], enabled=True)
    proof = audit(saved, legacy, candidate, pool=pool(), enabled=True)
    fields = candidate['fields']
    c = SimpleNamespace(**fields)
    hz = c.hz
    before = source_digest(hz)
    lb, ub = np.full(hz.n_out, -8.), np.full(hz.n_out, 8.)
    slots = [(hz.n_cont + 2 * i, hz.n_cont + 2 * i + 1, hz.n_bin + i) for i in range(hz.n_out)]
    nc, nb = hz.n_cont + 2 * hz.n_out, hz.n_bin + hz.n_out
    view = phase_blocks(hz, lb, ub, slots, nc, nb, source_pre=hz, enabled=True)
    new = sparse_hz_apply_relu_exact(hz, lb, ub, slots, nc, nb)
    original = sparse_hz_apply_relu_exact(saved['hz'], lb, ub, slots, nc, nb)
    for name, matrix in [('Ac', view.eq_c), ('Ab', view.eq_b), ('Auc', view.le_c), ('Aub', view.le_b)]:
        start = hz.n_ineq if name.startswith('Au') else hz.n_eq
        assert same_matrix(getattr(new, name)[start:], matrix)
    first = c.report['radix_uid_base'] + 16384
    overlay, _ = build(c.owners, [(view.eq_c, first), (view.le_c, first + view.eq_c.shape[0])],
        old_n_cont=c.old_n_cont, old_uid_ceiling=first, pool=pool(), enabled=True)
    plans, _ = discover_append(c, view, overlay, pool=pool(), enabled=True)
    if plans:
        spliced, _ = splice_append(view, plans, pool=pool(), enabled=True)
        for name, expected in full_reference(new, plans).items():
            value = getattr(spliced, name)
            assert np.array_equal(value.toarray() if sp.issparse(value) else value, expected)
    # This comparison isolates the entire native append/local inverse relation;
    # the tests above prove the additional nonempty unit/local composition.
    j = journal(c, [])
    vals = [F(i % 5 - 2, 5) for i in range(nc)]
    binary = [F(-1 if i % 2 else 1) for i in range(nb)]
    full = j.reconstruct_fraction(new, vals, pool=pool())
    assert full[:c.old_n_cont] == vals[:c.old_n_cont] and proof['local_inverse_equations'] > 0
    erased = np.zeros(saved['hz'].n_eq, bool)
    gone = fields['eq_roots'] < 0
    erased[saved['eq_roots'][gone]] = True
    rowmap = np.cumsum(~erased) - 1
    gauges = np.zeros(saved['hz'].n_eq, np.int64)
    gauges[saved['eq_roots'][~gone]] = fields['eq_scales'][~gone] - saved['eq_scales'][~gone]
    gauges[saved['def_rows']] = fields['radix_gauges']
    for row in range(original.n_eq):
        old = residual(original, 'Ac', 'Ab', 'b', row, full, binary)
        if row < len(erased) and erased[row]:
            assert old == 0
        else:
            target = int(rowmap[row]) if row < len(erased) else new.n_eq - original.n_eq + row
            gauge = int(gauges[row]) if row < len(gauges) else 0
            assert residual(new, 'Ac', 'Ab', 'b', target, vals, binary) == old * F(2) ** gauge
    for row in range(original.n_ineq):
        assert residual(original, 'Auc', 'Aub', 'ub', row, full, binary) == residual(new, 'Auc', 'Aub', 'ub', row, vals, binary)
    assert new.n_bin == original.n_bin and equal(new.c, original.c)
    assert source_digest(hz) == before


def test_default_off_and_wrong_schema_fail_closed():
    assert compile_journal(None, None, None, old_n_cont=None, old_n_eq=None,
        source_n_cont=None, source_schema=None, pool=None) is None
    c, h, o, *_ = source()
    with pytest.raises(ValueError, match='explicit local'):
        compile_journal(c.eq_roots, c.eq_scales, [], old_n_cont=c.old_n_cont,
            old_n_eq=c.old_n_eq, source_n_cont=h.n_cont, source_schema='legacy_alias', pool=pool(), enabled=True)
    with pytest.raises(MemoryError):
        compile_journal(c.eq_roots, c.eq_scales, [], old_n_cont=c.old_n_cont,
            old_n_eq=c.old_n_eq, source_n_cont=h.n_cont, source_schema=SCHEMA, pool=WorkPool(0), enabled=True)
