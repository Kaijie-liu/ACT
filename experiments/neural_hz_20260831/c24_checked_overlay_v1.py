"""Append sparse ownership only to an independently checker-issued base."""

import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c24_closed_state_v1 import Closed
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay, pack, validate_events, UID_LIMIT


def build(closed, blocks, *, pool, enabled=False):
    """Blocks are NEW canonical continuous CSR rows with disjoint fresh UIDs.

The enclosing phase audit must first check the complete HZ prefix/frame/RHS
and bind the old words to actual old incidence. This primitive never infers
membership from a count/sum, never accepts a signed deletion and never reads
an iid, solver result, public label or historical unit-pair table.
    """
    if not enabled:
        return None
    if type(closed) is not Closed:
        raise ValueError('independently checked closed owner base required')
    closed.validate()
    base = closed.owners
    old_n_cont = closed.old_n_cont
    old_uid_ceiling = closed.report['radix_uid_base'] + 16384
    # The full source/UID/actual-incidence checker proves every old UID is
    # below this ceiling. Its receipt binds this complete base by content;
    # no caller boolean or self-seal can avoid the original proof.
    records, previous_end, scanned, rows = [], old_uid_ceiling, 0, 0
    before_events = pool.used
    for matrix, first in blocks:
        if (not sp.isspmatrix_csr(matrix) or type(first) is not int
                or first < previous_end or first + matrix.shape[0] > UID_LIMIT
                or matrix.shape[1] < old_n_cont + len(base)):
            raise ValueError('invalid/overlapping/new row UID block or global width')
        if matrix.nnz > 64_000_000:
            raise MemoryError('unchanged sparse coefficient entry ceiling exceeded')
        pool.charge('overlay_appended_rows_scan', 12 * int(matrix.nnz) + 8 * int(matrix.shape[0]))
        if (not matrix.has_canonical_format or not np.isfinite(matrix.data).all()
                or np.any(matrix.data == 0.)):
            raise ValueError('new rows must be canonical finite nonzero CSR')
        previous_end = first + matrix.shape[0]
        for row in range(matrix.shape[0]):
            a, b = map(int, matrix.indptr[row:row + 2])
            previous_col = -1
            for raw in matrix.indices[a:b]:
                col = int(raw)
                # Check actual order too, not only scipy's cached flag.
                if not previous_col < col < matrix.shape[1]:
                    raise ValueError('duplicate/unsorted/out-of-domain new incidence')
                previous_col = col
                if old_n_cont <= col < old_n_cont + len(base):
                    pool.charge('overlay_event_pack', 12)
                    if len(records) >= 64_000_000:
                        raise MemoryError('unchanged sparse event entry ceiling exceeded')
                    records.append(pack(col - old_n_cont, first + row))
        scanned += int(matrix.nnz)
        rows += int(matrix.shape[0])
    pool.charge('overlay_event_array', len(records))
    events = np.asarray(records, np.uint64)
    pool.charge('overlay_event_sort', 4 * len(events) * max(1, (len(events) - 1).bit_length()))
    events.sort(kind='stable')
    pool.charge('overlay_event_validation', 8 * len(events))
    validate_events(events, len(base), old_uid_ceiling)
    events.flags.writeable = False
    result = Overlay(base, events, old_uid_ceiling)
    return result, {'new_rows_scanned': rows, 'new_continuous_nnz_scanned': scanned,
        'event_count': len(events), 'retained_event_entries': len(events),
        'retained_event_bytes': events.nbytes, 'base_shared_by_identity': result.base is base,
        'dense_phase_copy_allocated': False, 'event_build_work': pool.used - before_events,
        'formal_gain': 0, 'new_phase_executed': False}
