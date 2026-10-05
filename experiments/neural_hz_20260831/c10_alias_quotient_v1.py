"""Default-off, frame-preserving exact quotient of value-dead MAIN aliases.

Only this component is implemented here. No runtime hooks or solver calls.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c10_predicate_census_v1 import census, exact_products
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots

MATRICES = ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')
ARRAYS = ('columns', 'parents', 'ratios', 'defining_rows')
MINIMUM, MAXIMUM = 2.**-20, 2.**40


def exact_sum(values):
    """Exact finite float64 sum on the fixed coefficient-window dyadic grid."""
    total = 0
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or (value != 0. and not MINIMUM <= abs(value) <= MAXIMUM):
            raise ValueError('sum operand outside fixed coefficient window')
        numerator, denominator = value.as_integer_ratio()
        shift = 72 - (denominator.bit_length() - 1)
        if shift < 0:
            raise ValueError('sum operand outside fixed dyadic grid')
        total += numerator << shift
    if total == 0:
        return 0.
    result = math.ldexp(float(total), -72)
    numerator, denominator = result.as_integer_ratio()
    shift = 72 - (denominator.bit_length() - 1)
    if shift < 0 or (numerator << shift) != total:
        raise ValueError('collision sum is not exactly representable in float64')
    if not MINIMUM <= abs(result) <= MAXIMUM:
        raise ValueError('collision sum outside fixed coefficient window')
    return result


def frontier(table, n_cont):
    """Descending maximal independent set in the topological alias forest."""
    eligible = table['all_products_exact'] & table['products_window_safe']
    blocked = np.zeros(n_cont, dtype=bool)
    chosen = []
    for index in np.flatnonzero(eligible)[::-1]:
        col, parent = int(table['column'][index]), int(table['parent'][index])
        if not 0 <= parent < col < n_cont:
            raise ValueError('non-topological alias dependency')
        if not blocked[col]:
            chosen.append(index)
            blocked[parent] = True
    return np.asarray(chosen[::-1], dtype=np.int64)


def _plain(value):
    # Proof metadata may not hide arrays, source HZs or other runtime payload.
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _plain(item)
        return
    if type(value) in (tuple, list):
        for item in value:
            _plain(item)
        return
    raise ValueError('unregistered certificate metadata payload')


@dataclass
class Quotient:
    hz: SparseHZono
    columns: np.ndarray
    parents: np.ndarray
    ratios: np.ndarray
    defining_rows: np.ndarray
    old_n_cont: int
    logical_n_cont: int
    original_digest: str
    report: dict
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'hz', *ARRAYS, 'old_n_cont', 'logical_n_cont',
                              'original_digest', 'report', 'seal'}:
            raise ValueError('unregistered quotient fields')
        if set(vars(self.hz)) != {'c', *MATRICES, 'b', 'ub', 'frame_id', 'exact'}:
            raise ValueError('unregistered HZ fields')
        _plain(self.report)
        if type(self.old_n_cont) is not int or type(self.logical_n_cont) is not int:
            raise ValueError('non-scalar frame metadata')
        h = hashlib.sha256(source_digest(self.hz).encode())
        h.update(json.dumps([self.old_n_cont, self.logical_n_cont, self.original_digest,
                             self.report], sort_keys=True, allow_nan=False).encode())
        size = self.columns.size
        for name in ARRAYS:
            value = getattr(self, name)
            dtype = np.dtype(np.float64 if name == 'ratios' else np.int64)
            if type(value) is not np.ndarray or value.dtype != dtype or value.shape != (size,):
                raise ValueError('invalid quotient reconstruction array')
            h.update(name.encode())
            h.update(value.tobytes())
        return h.hexdigest()

    def validate(self):
        if self.fingerprint() != self.seal:
            raise ValueError('quotient HZ/certificate changed')

    def numeric_roots(self):
        self.validate()
        return {'hz': self.hz, **{name: getattr(self, name) for name in ARRAYS}}

    def reconstruct_fraction(self, continuous):
        """Exact extension, not a floating witness acceptance/repair routine."""
        self.validate()
        if len(continuous) != self.hz.n_cont:
            raise ValueError('reconstruction requires original global frame width')
        values = [Fraction(v) for v in continuous]
        if any(abs(v) > 1 for v in values):
            raise ValueError('continuous point outside latent box')
        for col, parent, ratio in zip(self.columns, self.parents, self.ratios):
            values[int(col)] = Fraction(float(ratio)) * values[int(parent)]
        return values


def storage(roots):
    return snapshot_partial_csr_owners(WholeStateRoots(active=roots, consumer_gc_enabled=False))


def _rewrite(matrix, selected, parents, ratios, keep=None):
    result = matrix.copy() if keep is None else matrix[keep].tocsr()
    positions = np.flatnonzero(selected[result.indices])
    if not positions.size:
        return result, {'changed_occurrences': 0, 'affected_rows': 0, 'collision_groups': 0}
    columns = result.indices[positions].copy()
    exact, products = exact_products(result.data[positions], ratios[columns])
    if not exact.all() or np.any(np.abs(products) < MINIMUM) or np.any(np.abs(products) > MAXIMUM):
        raise ValueError('selected incident product lost exactness/window')
    result.data[positions] = products
    result.indices[positions] = parents[columns]
    rows = np.unique(np.searchsorted(result.indptr, positions, side='right') - 1)
    collisions = 0
    for row in rows:
        start, stop = result.indptr[row:row + 2]
        order = np.argsort(result.indices[start:stop], kind='stable')
        cols, vals = result.indices[start:stop][order], result.data[start:stop][order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(cols)) + 1, len(cols)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            if end - begin > 1:
                try:
                    vals[begin] = exact_sum(vals[begin:end])
                except ValueError as exc:
                    raise ValueError(f'row {int(row)} column {int(cols[begin])}: {exc}') from exc
                vals[begin + 1:end] = 0.
                collisions += 1
        result.indices[start:stop], result.data[start:stop] = cols, vals
    # No unchecked duplicate sum; all duplicates after the first are EXACT zero.
    result.has_sorted_indices = True
    result.eliminate_zeros()
    # Own compact buffers, rather than retain the original larger CSR owners.
    result = sp.csr_matrix((result.data.copy(), result.indices.copy(), result.indptr.copy()), shape=result.shape)
    if not result.has_canonical_format or np.any(result.data == 0.):
        raise ValueError('quotient CSR is not canonical/zero-free')
    return result, {'changed_occurrences': int(positions.size), 'affected_rows': int(rows.size),
                    'collision_groups': collisions}


def quotient(hz, *, enabled=False, old_n_cont=None, logical_n_cont=None, old_n_eq=None,
             eq_roots=None, def_rows=None, max_work=256_000_000, max_entries=64_000_000,
             observe=None):
    if not enabled:
        return None
    frozen = source_digest(hz)
    report, table = census(hz, old_n_cont=old_n_cont, logical_n_cont=logical_n_cont,
        old_n_eq=old_n_eq, eq_roots=eq_roots, def_rows=def_rows,
        max_work=max_work, max_entries=max_entries)
    for name in MATRICES:
        data = getattr(hz, name).data
        if np.any(np.abs(data) < MINIMUM) or np.any(np.abs(data) > MAXIMUM):
            raise ValueError('input coefficients outside fixed component window')
    chosen = frontier(table, hz.n_cont)
    if chosen.size == 0:
        raise ValueError('no eligible independent alias frontier')
    cols = table['column'][chosen].copy()
    ps = table['parent'][chosen].copy()
    rs = table['ratio'][chosen].copy()
    removed = table['defining_row'][chosen].copy()
    selected = np.zeros(hz.n_cont, dtype=bool)
    selected[cols] = True
    parents = np.arange(hz.n_cont, dtype=np.int64)
    parents[cols] = ps
    ratios = np.ones(hz.n_cont)
    ratios[cols] = rs
    if np.any(selected[ps]):
        raise ValueError('selected alias depends on another selected alias')
    keep = np.ones(hz.n_eq, dtype=bool)
    keep[removed] = False
    incidence, sorting = 0, 0
    for name in ('Ac', 'Auc'):
        matrix = getattr(hz, name)
        positions = np.flatnonzero(selected[matrix.indices])
        rows = np.searchsorted(matrix.indptr, positions, side='right') - 1
        if name == 'Ac':
            rows = rows[keep[rows]]
        incidence += rows.size
        widths = np.diff(matrix.indptr)[np.unique(rows)].astype(np.int64)
        sorting += int(np.sum(widths * np.ceil(np.log2(np.maximum(2, widths))).astype(np.int64)))
    work = int(report['logical_work_upper'] + 8 * report['total_coefficient_nnz']
               + 32 * report['main_factors'] + 64 * incidence + 4 * sorting)
    event = {'event': 'quotient_preflight', 'eligible_aliases': report['aliases_exact_and_window_safe'],
        'selected_aliases': int(cols.size), 'remaining_selected_occurrences': int(incidence),
        'sort_work': sorting, 'logical_work_upper': work, 'max_work': max_work}
    if observe:
        observe(event)
    if work > max_work:
        raise MemoryError(f'quotient whole-component work cap exceeded: {event}')
    ac, eq_stats = _rewrite(hz.Ac, selected, parents, ratios, keep)
    auc, ineq_stats = _rewrite(hz.Auc, selected, parents, ratios)
    new = SparseHZono(hz.c, hz.Gc, hz.Gb, ac, hz.Ab[keep], hz.b[keep],
        auc, hz.Aub, hz.ub, frame_id=hz.frame_id, exact=True)
    old_nnz = sum(getattr(hz, name).nnz for name in MATRICES)
    new_nnz = sum(getattr(new, name).nnz for name in MATRICES)
    if new_nnz > old_nnz - 2 * cols.size:
        raise ValueError('strict no-fill total coefficient reduction failed')
    facts = {'schema': 'c10_alias_quotient_v1', 'default_off': True, 'formal_gain': 0,
        'eligible_aliases': report['aliases_exact_and_window_safe'], 'selected_aliases': int(cols.size),
        'original_n_eq': hz.n_eq, 'remaining_n_eq': new.n_eq, 'raw_frame_n_cont': hz.n_cont,
        'preserved_n_bin': hz.n_bin, 'original_coefficient_nnz': old_nnz,
        'quotient_coefficient_nnz': new_nnz, 'strict_nnz_decrease': True,
        'logical_work_upper': work, 'census_work_upper': report['logical_work_upper'],
        'remaining_selected_occurrences': int(incidence), 'sort_work': sorting,
        'equalities': eq_stats, 'inequalities': ineq_stats,
        'whole_live_reduction_proved': False, 'native_ingestion_executed': False, 'solver_executed': False}
    result = Quotient(new, cols, ps, rs, removed, old_n_cont, logical_n_cont, frozen, facts)
    result.seal = result.fingerprint()
    old_storage = storage({'hz': hz})
    new_storage = storage(result.numeric_roots())
    # Attribute names deliberately come from the same existing owner ledger.
    facts.update(original_component_bytes=old_storage.resident_bytes,
        quotient_with_certificate_bytes=new_storage.resident_bytes,
        original_component_entries=old_storage.resident_entries,
        quotient_with_certificate_entries=new_storage.resident_entries,
        certificate_array_bytes=sum(getattr(result, key).nbytes for key in ARRAYS))
    result.seal = result.fingerprint()
    if new_storage.resident_entries > max_entries:
        raise MemoryError('quotient plus reconstruction entries exceed cap')
    if new_storage.resident_bytes >= old_storage.resident_bytes:
        raise ValueError('quotient plus reconstruction has no strict resident byte reduction')
    if source_digest(hz) != frozen:
        raise ValueError('source mutated during quotient construction')
    if observe:
        observe({'event': 'quotient_constructed', **facts})
    return result
