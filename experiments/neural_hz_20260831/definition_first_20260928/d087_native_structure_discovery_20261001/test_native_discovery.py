"""Four fixed mathematical fixtures; no model, LP, GPU or archive execution."""

from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.definition_first_20260928.d086_native_shared_transfer_20261001 import test_native_shared_transfer as old
from experiments.neural_hz_20260831.definition_first_20260928.d087_native_structure_discovery_20261001 import native_discovery as nd

ZERO, ONE = F(0), F(1)


class _Pool:
    def __init__(self, limit=10**12):
        self.limit, self.used, self.charges = limit, 0, []

    def charge(self, name, amount):
        assert type(name) is str and type(amount) is int and amount >= 0
        if self.used+amount > self.limit:
            raise nd.KernelError('fixture budget exceeded before work')
        self.used += amount
        self.charges.append((name, amount))


def _discover(hz, pool=None):
    return nd.discover(hz, pool=_Pool() if pool is None else pool, enabled=True)


def _append_le(hz, cont, binary, rhs=0.0):
    ac = sp.csr_matrix(([a for _, a in cont], ([0]*len(cont), [i for i, _ in cont])),
                       shape=(1, hz.n_cont))
    ab = sp.csr_matrix(([a for _, a in binary], ([0]*len(binary), [i for i, _ in binary])),
                       shape=(1, hz.n_bin))
    return replace(hz, Auc=sp.vstack((hz.Auc, ac), format='csr'),
                   Aub=sp.vstack((hz.Aub, ab), format='csr'),
                   ub=np.concatenate((hz.ub, np.array([rhs]))))


def _receipt_view(applied, receipt):
    return SimpleNamespace(hz=applied.hz, shared_columns=receipt.shared_columns,
                           residual_bindings=receipt.residual_bindings)


def _five_gates(fixture):
    """One ordinary next mixed gate creates a second uniform parent bank."""
    hz, parents, consumers, _ = fixture
    eta0, etar = parents[0].slots[1], consumers[0][0].slots[1]
    qscale = nd.nb._extract_graph(hz, consumers[0][0])[1].bias
    pre = replace(hz, c=np.array([float(F(1, 2)+qscale-F(1, 4))]),
                  Gc=sp.csr_matrix(([-.5, -float(qscale)], ([0, 0], [eta0, etar])),
                                   shape=(1, hz.n_cont)),
                  Gb=sp.csr_matrix((1, hz.n_bin)))
    slots = ((hz.n_cont, hz.n_cont+1, hz.n_bin),)
    return old.sparse_hz_apply_relu_exact(pre, [-.25], [2.], slots,
                                         hz.n_cont+2, hz.n_bin+1)


def test_guard_catalog_and_uniform_groups():
    fixture = old._fixture()
    hz, parents, consumers, _ = fixture
    before = old._state(hz)
    pool = _Pool()
    found = _discover(hz, pool)
    assert found.graphs == (*parents, *(c[0] for c in consumers))
    assert found.groups == ((parents, consumers),)
    assert found.summary['graphs'] == 4 and found.summary['groups'] == 1
    assert found.summary['covered_children'] == found.summary['uncovered_children'] == 2
    assert found.summary['eq_rows_scanned'] == hz.n_eq
    assert found.summary['le_rows_scanned'] == hz.n_ineq
    assert found.summary['extended_only'] == 1
    assert all(type(v) is int and v >= 0 for v in found.summary.values())
    assert pool.used > 0 and old._state(hz) == before
    # Identical LE and EQ duplicates are scanned, authenticated and charged,
    # but the lowest original row is the canonical descriptor.
    duplicate = replace(hz, Auc=sp.vstack((hz.Auc, hz.Auc[:1]), format='csr'),
                        Aub=sp.vstack((hz.Aub, hz.Aub[:1]), format='csr'),
                        ub=np.concatenate((hz.ub, hz.ub[:1])),
                        Ac=sp.vstack((hz.Ac, hz.Ac[:1]), format='csr'),
                        Ab=sp.vstack((hz.Ab, hz.Ab[:1]), format='csr'),
                        b=np.concatenate((hz.b, hz.b[:1])))
    same = _discover(duplicate)
    assert same.graphs == found.graphs and same.groups == found.groups
    assert same.summary['duplicate_guard_rows'] == 1
    assert same.summary['duplicate_equation_rows'] == 1
    assert same.summary['candidate_relations'] == 5
    # Full bank: fifth gate is not selected because of an LP result. Its two
    # earliest matching eta columns deterministically create another group.
    bank = _five_gates(fixture)
    all_found = _discover(bank)
    assert all_found.summary['graphs'] == 5
    assert all_found.summary['groups'] == 2
    assert all_found.summary['covered_children'] == 3
    assert all_found.groups[0] == (parents, consumers)
    assert all_found.groups[1][0] == (parents[0], consumers[0][0])
    assert all_found.groups[1][1][0][1:] == (ONE, 2*F(5, 8))


def test_full_bank_transfer_and_projection():
    for wide in (False, True):
        fixture = old._fixture(wide=wide)
        hz, parents, consumers, _ = fixture
        before = old._state(hz)
        found = _discover(hz)
        pool = _Pool()
        applied = nd.apply_groups(hz, found, pool=pool, enabled=True)
        direct = nd.nt.append_shared_upper(hz, parents, consumers, enabled=True)
        assert len(applied.receipts) == 1
        receipt = applied.receipts[0]
        assert not hasattr(receipt, 'hz')
        assert old._state(applied.hz) == old._state(direct.hz)
        assert receipt.exact_rows == direct.exact_rows
        assert receipt.residual_bindings == direct.residual_bindings
        assert receipt.row_errors == direct.row_errors
        assert applied.summary['groups_applied'] == 1
        assert applied.summary['consumers_applied'] == 2
        assert applied.summary['original_bits_preserved'] == 1
        assert applied.summary['added_rows'] <= found.summary['added_rows_upper']
        assert applied.summary['added_cont'] <= found.summary['added_cont_upper']
        assert receipt.nnz['added'] <= found.summary['added_nnz_upper']
        assert pool.charges == [('apply_binding_preflight', 32),
                               ('apply_entire_bank_work_and_allocation',
                                found.summary['apply_total_charge'])]
        zero_labels = set()
        points = ((-ONE, -ONE), (ZERO, ZERO), (ONE, ONE),
                  (F(1, 2), F(1, 4)), (ZERO, F(1, 4)))
        for xy in points:
            point = (*xy, F(1, 2)) if wide else xy
            for upstream in ((-ONE, ONE) if wide else (ONE,)):
                for cont, bits in old._original_states(fixture, point, upstream):
                    assert old._holds(hz, cont, bits)
                    extension = old._extend(fixture, _receipt_view(applied, receipt), cont, bits)
                    assert old._holds(applied.hz, extension, bits)
                    assert extension[:len(point)] == point
                    assert all(old._eval(row, extension, bits) <= rhs
                               for row, rhs in receipt.exact_rows)
                    if xy == (ZERO, ZERO):
                        offset = 1 if wide else 0
                        zero_labels.add(bits[offset:offset+2])
        assert len(zero_labels) == 4
        assert old._state(hz) == before
    # More than one group is applied; no receipt retains the preceding HZ.
    bank = _five_gates(old._fixture())
    found = _discover(bank)
    applied = nd.apply_groups(bank, found, pool=_Pool(), enabled=True)
    expected = bank
    for parents, consumers in found.groups:
        expected = nd.nt.append_shared_upper(expected, parents, consumers, enabled=True).hz
    assert len(applied.receipts) == 2
    assert applied.summary['consumers_applied'] == 3
    assert old._state(applied.hz) == old._state(expected)
    assert applied.hz.n_bin == bank.n_bin and applied.hz.n_eq == bank.n_eq


def test_missing_ambiguous_and_compact_scope():
    hz, parents, _, _ = old._fixture()
    keep = np.arange(hz.n_ineq) != parents[0].le_rows[0]
    missing = replace(hz, Auc=hz.Auc[keep].tocsr(), Aub=hz.Aub[keep].tocsr(), ub=hz.ub[keep])
    found = _discover(missing)
    assert found.summary['missing_guard_bits'] == 1
    assert found.summary['graphs'] == 3 and not found.groups
    assert all(g.slots[2] != parents[0].slots[2] for g in found.graphs)
    ambiguous = _append_le(hz, ((0, -1.),), ((parents[0].slots[2], -1.),))
    found = _discover(ambiguous)
    assert found.summary['ambiguous_guard_bits'] == 1
    assert found.summary['graphs'] == 3 and not found.groups
    contradictory = replace(hz, Ac=sp.vstack((hz.Ac, hz.Ac[:1]), format='csr'),
                            Ab=sp.vstack((hz.Ab, hz.Ab[:1]), format='csr'),
                            b=np.concatenate((hz.b, hz.b[:1]+.125)))
    found = _discover(contradictory)
    assert found.summary['ambiguous_relation_bits'] == 1
    assert found.summary['graphs'] == 3 and not found.groups
    # An ambiguous native relationship is simply uncovered, never UNSAT.
    assert 'unsat' not in found.summary
    compact = old._fixture(compact=True)[0]
    found = _discover(compact)
    assert found.graphs == found.groups == ()
    assert found.summary['extended_only'] == 1 and found.summary['eq_rows_scanned'] == 0
    applied = nd.apply_groups(compact, found, pool=_Pool(), enabled=True)
    assert applied.hz is compact and not applied.receipts
    assert applied.summary['groups_applied'] == 0


def test_default_off_and_budget_rejection():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled argument inspected')
    poison = Poison()
    assert nd.discover(poison, pool=poison) is None
    assert nd.apply_groups(poison, poison, pool=poison) is None
    with pytest.raises(nd.KernelError):
        nd.discover(poison, pool=poison, enabled=1)
    with pytest.raises(nd.KernelError):
        nd.apply_groups(poison, poison, pool=poison, enabled=1)
    hz = old._fixture()[0]
    before = old._state(hz)
    with pytest.raises(nd.KernelError):
        nd.discover(hz, pool=_Pool(0), enabled=True)
    with pytest.raises(nd.KernelError):
        nd.discover(hz, pool=object(), enabled=True)
    with pytest.raises(nd.KernelError):
        nd.discover(replace(hz, frame_id=None), pool=_Pool(), enabled=True)
    bad = hz.b.copy()
    bad[0] = np.inf
    with pytest.raises(nd.KernelError):
        nd.discover(replace(hz, b=bad), pool=_Pool(), enabled=True)
    found = _discover(hz)
    pool = _Pool(32)
    with pytest.raises(nd.KernelError):
        nd.apply_groups(hz, found, pool=pool, enabled=True)
    assert pool.used == 32
    with pytest.raises(nd.KernelError):
        nd.apply_groups(replace(hz), found, pool=_Pool(), enabled=True)
    assert old._state(hz) == before
