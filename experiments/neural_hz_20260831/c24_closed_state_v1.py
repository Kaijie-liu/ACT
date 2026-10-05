"""Checker-issued closed HZ state: graph proof fields are construction-only.

The process-local receipt registry retains ONLY immutable textual proof data,
never a draft graph or numeric witness. Unissued/copied tokens cannot validate
an object. Persisted restoration requires a hash anchored in an independent
enclosing archive manifest, as C16, not a caller's self-seal or exact flag.
"""

from dataclasses import dataclass, fields
import hashlib
import json
import weakref

import numpy as np
from experiments.neural_hz_20260831.c17_owned_emission_v1 import OwnedIntegrated
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays, operator_digest
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit as exact_audit
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c22_uid_runs_v1 import validate as validate_slabs, uid_for_row, row_for_uid
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import node_uid_tables, closed_uid_tables
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool

MAPS = ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')


@dataclass
class Draft:
    expression: object
    origin_binding: tuple
    hz: object
    nodes: list
    root: int
    old_n_cont: int
    old_n_bin: int
    old_n_eq: int
    logical_n_cont: int
    keep: np.ndarray
    report: dict
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    ineq_roots: np.ndarray
    ineq_scales: np.ndarray
    def_rows: np.ndarray
    owners: np.ndarray
    uid_slabs: np.ndarray
    seal: str = ''

    def _old(self):
        if set(vars(self)) != {f.name for f in fields(Draft)}:
            raise ValueError('unregistered dense draft payload')
        return OwnedIntegrated(**{f.name: getattr(self, f.name) for f in fields(OwnedIntegrated) if f.name != 'seal'})

    def fingerprint(self):
        validate_slabs(self.uid_slabs)
        return hashlib.sha256((self._old().fingerprint() + digest_arrays(self.uid_slabs)).encode()).hexdigest()

    def validate(self):
        aliases(self)
        if expression_binding(self.expression) != self.origin_binding or self.fingerprint() != self.seal:
            raise ValueError('dense draft source/graph/HZ/UIDs changed')

    def numeric_roots(self):
        self.validate()
        old = self._old()
        old.seal = old.fingerprint()
        return {**old.numeric_roots(), 'uid_slabs': self.uid_slabs}


_ISSUER = object()
_RECEIPTS = weakref.WeakKeyDictionary()


class _Receipt:
    __slots__ = ('__weakref__',)

    def __new__(cls, issuer):
        if issuer is not _ISSUER: raise ValueError('receipt must be issued by the full checker')
        return super().__new__(cls)

    def __reduce__(self):
        raise TypeError('runtime proof tokens are not serializable; use an authenticated archive')


@dataclass
class Closed:
    expression: object
    origin_binding: tuple
    hz: object
    old_n_cont: int
    old_n_bin: int
    old_n_eq: int
    logical_n_cont: int
    keep: np.ndarray
    report: dict
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    ineq_roots: np.ndarray
    ineq_scales: np.ndarray
    def_rows: np.ndarray
    owners: np.ndarray
    uid_slabs: np.ndarray
    receipt: object = None
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {f.name for f in fields(Closed)}:
            raise ValueError('unregistered closed HZ payload')
        h = hashlib.sha256(b'c24_checked_closed_nonconvex_HZ_v1')
        def token(v): h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode() + b'\0')
        sources, operators = {}, {}
        token((source_digest(self.hz), self.old_n_cont, self.old_n_bin, self.old_n_eq,
            self.logical_n_cont, self.expression.frame_id, self.expression.n_out,
            digest_arrays(self.expression.bias, self.keep), self.report))
        for term in self.expression.terms:
            source = term.source
            sid = sources.setdefault(id(source), len(sources))
            ops = [(operators.setdefault(id(op), len(operators)), operator_digest(op)) for op in term.operators]
            token((sid, source_digest(source), ops))
        token(digest_arrays(*(getattr(self, k) for k in (*MAPS, 'owners', 'uid_slabs'))))
        return h.hexdigest()

    def validate(self):
        if type(self.receipt) is not _Receipt or self.receipt not in _RECEIPTS:
            raise ValueError('closed HZ lacks an independently issued proof receipt')
        expected, unused = _RECEIPTS[self.receipt]
        aliases(self)
        if (expression_binding(self.expression) != self.origin_binding
                or self.fingerprint() != expected or self.seal != expected):
            raise ValueError('closed source/HZ/lineage/ownership changed, including resealing')

    def numeric_roots(self):
        self.validate()
        # The registry contains no graph/arrays; expose its complete textual
        # record too, so Python shallow storage is not silently hidden.
        return {**{k: getattr(self, k) for k in ('expression', 'hz', 'keep', *MAPS, 'owners', 'uid_slabs')},
            'checked_proof_record': _RECEIPTS[self.receipt]}


def _issue(closed, proof):
    raw = json.dumps(proof, sort_keys=True, allow_nan=False).encode()
    receipt = _Receipt(_ISSUER)
    _RECEIPTS[receipt] = (proof['closed_identity'], raw)
    closed.receipt, closed.seal = receipt, proof['closed_identity']
    closed.validate()
    return closed


def close(candidate, original_hz, maps, *, enabled=False):
    """Perform ALL original math and ownership/UID checks before minting proof."""
    if not enabled: return None
    if type(candidate) is not Draft: raise ValueError('full dense draft required')
    candidate.validate()
    before = candidate.fingerprint()
    exact = exact_audit(candidate, original_hz, maps)
    eq, le, expected_uids = node_uid_tables(candidate)
    expected = actual_words(candidate.hz, candidate.old_n_cont, candidate.logical_n_cont, eq, le)
    if not np.array_equal(expected, candidate.owners):
        raise ValueError('all generated ownership differs from complete actual incidence')
    pool = WorkPool(256_000_000)
    # Proof-only queries have their own explicit ledger; none constructs the
    # already-generated slabs. No source reconstruction flag can skip this.
    for main, uid in enumerate(expected_uids):
        if uid_for_row(candidate.uid_slabs, main, pool=pool) != int(uid):
            raise ValueError('dense MAIN-to-UID index disagrees with original needed slots')
    first = candidate.old_n_eq + len(candidate.ineq_roots)
    width = candidate.report['radix_uid_base'] - first
    pool.charge('independent_all_reserved_UID_inverse', 4 * (width + len(expected_uids)))
    inverse = np.full(width, -1, np.int64)
    inverse[expected_uids - first] = np.arange(len(expected_uids))
    for offset, expected_main in enumerate(inverse):
        found = row_for_uid(candidate.uid_slabs, first + offset, pool=pool)
        if found != (None if expected_main < 0 else int(expected_main)):
            raise ValueError('dense UID index disagrees on a reserved UID or hole')
    if candidate.fingerprint() != before:
        raise ValueError('draft mutated during complete source/owner proof')
    values = {f.name: getattr(candidate, f.name) for f in fields(Closed) if f.name not in {'receipt', 'seal'}}
    closed = Closed(**values)
    eq2, le2 = closed_uid_tables(closed)
    if not np.array_equal(eq, eq2) or not np.array_equal(le, le2):
        raise ValueError('graph-free physical row maps disagree with complete original graph')
    proof = {'schema': 'c24_independent_closed_source_proof_v1', 'completed': True,
        'identity': exact, 'closed_identity': closed.fingerprint(), 'draft_identity': before,
        'all_MAIN_ownership_checked': len(expected), 'all_MAIN_UID_queries_checked': len(expected_uids),
        'all_reserved_MAIN_UIDs_checked': width, 'all_physical_EQ_checked': len(eq),
        'all_physical_INEQ_checked': len(le), 'complete_graph_free_row_maps_equal': True,
        'uid_query_proof_work': pool.used, 'uid_query_proof_work_parts': dict(pool.parts),
        'main_boxes_and_alias_extension_proved': True, 'formal_gain': 0, 'whole_live_path_proved': False}
    return _issue(closed, proof), proof


def export(closed):
    closed.validate()
    return ({f.name: getattr(closed, f.name) for f in fields(Closed)
        if f.name not in {'receipt', 'seal', 'origin_binding'}}, _RECEIPTS[closed.receipt][1])


def restore(fields, raw_proof, *, expected_proof_sha256):
    """Expected hash MUST come from an independent enclosing artifact manifest."""
    if hashlib.sha256(raw_proof).hexdigest() != expected_proof_sha256:
        raise ValueError('independent closed-proof archive hash mismatch')
    proof = json.loads(raw_proof)
    original = proof.get('identity', {}).get('original_affine_proof', {})
    if (proof.get('schema') != 'c24_independent_closed_source_proof_v1'
            or proof.get('completed') is not True
            or proof.get('identity', {}).get('status') != 'EXACT_ORIGINAL_DAG_AND_QUOTIENT'
            or original.get('all_redundant_main_and_radix_boxes_proved') is not True
            or proof.get('complete_graph_free_row_maps_equal') is not True):
        raise ValueError('incomplete independent closed-state proof')
    closed = Closed(**fields, origin_binding=expression_binding(fields['expression']))
    if closed.fingerprint() != proof['closed_identity']:
        raise ValueError('restored source/HZ/sharing/maps differ from independent proof')
    return _issue(closed, proof)
