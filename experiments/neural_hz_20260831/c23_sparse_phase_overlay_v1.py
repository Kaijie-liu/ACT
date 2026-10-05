"""Exact append-only MAIN ownership overlay; no HZ, phase or solver rewrite.

Queries are valid only inside a content-bound read transaction. Count/sum
words and structural validation are NOT source-incidence membership proofs.
The base is shared, never copied or mutated; only actual new incidences persist.
"""

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c17_packed_ownership_v1 import (
    RADIX, UID_LIMIT, validate_words)

MASK = UID_LIMIT - 1


def pack(index, uid):
    if (type(index) is not int or type(uid) is not int
            or not 0 <= index < UID_LIMIT or not 0 <= uid < UID_LIMIT):
        raise ValueError('event outside exact 20-bit column/UID domain')
    return (index << 20) | uid


def validate_events(events, size, old_uid_ceiling):
    if (type(events) is not np.ndarray or events.dtype != np.dtype(np.uint64)
            or events.ndim != 1 or type(size) is not int
            or type(old_uid_ceiling) is not int or not 0 <= size <= UID_LIMIT
            or not 0 <= old_uid_ceiling <= UID_LIMIT):
        raise ValueError('invalid sparse event owner/domain')
    if len(events) > 64_000_000:
        raise MemoryError('unchanged sparse event entry ceiling exceeded')
    previous = -1
    for raw in events:
        value = int(raw)
        if (not 0 <= value < RADIX or value <= previous
                or value >> 20 >= size or (value & MASK) < old_uid_ceiling):
            raise ValueError('noncanonical/duplicate/out-of-domain sparse event')
        previous = value


def _checked_sum(base_word, count, uid_sum):
    # Python integers prevent overflow before checking the exact packed range.
    value = int(base_word) + count * RADIX + uid_sum
    count, total = divmod(value, RADIX)
    if (value < 0 or count > UID_LIMIT or total > count * MASK
            or value > np.iinfo(np.int64).max):
        raise ValueError('overlaid ownership outside exact count/sum range')
    # A sum carry must never masquerade as an extra incidence count.
    if int(base_word) % RADIX + uid_sum >= RADIX:
        raise ValueError('overlaid UID sum carries into count')
    return value


@dataclass(frozen=True)
class Overlay:
    base: np.ndarray
    events: np.ndarray
    old_uid_ceiling: int

    def validate(self):
        if set(vars(self)) != {'base', 'events', 'old_uid_ceiling'}:
            raise ValueError('unregistered sparse overlay payload')
        validate_words(self.base)
        validate_events(self.events, len(self.base), self.old_uid_ceiling)

    def numeric_roots(self):
        self.validate()
        return {'base': self.base, 'events': self.events}

    def query(self, index, *, pool):
        if type(index) is not int or not 0 <= index < len(self.base):
            raise ValueError('query outside MAIN prefix')
        # Two lower-bound searches, before any packed event/base access.
        pool.charge('overlay_random_search', 16 * max(1, len(self.events).bit_length()) + 16)
        bounds = []
        for key in (index << 20, (index + 1) << 20):
            lo, hi = 0, len(self.events)
            while lo < hi:
                mid = (lo + hi) // 2
                if int(self.events[mid]) < key:
                    lo = mid + 1
                else:
                    hi = mid
            bounds.append(lo)
        start, stop = bounds
        pool.charge('overlay_random_event_sum', 12 * (stop - start))
        total = sum(int(raw) & MASK for raw in self.events[start:stop])
        return _checked_sum(self.base[index], stop - start, total)

    def iter_words(self, *, pool):
        """Complete ordered query stream, with no dense overlaid return buffer."""
        cursor = 0
        for index in range(len(self.base)):
            pool.charge('overlay_stream_column', 16)
            count = total = 0
            while cursor < len(self.events) and int(self.events[cursor]) >> 20 == index:
                pool.charge('overlay_stream_event', 12)
                total += int(self.events[cursor]) & MASK
                count += 1
                cursor += 1
            yield _checked_sum(self.base[index], count, total)
        if cursor != len(self.events):
            raise ValueError('unconsumed sparse events outside MAIN stream')


def build(base, blocks, *, old_n_cont, old_uid_ceiling, pool, enabled=False):
    """Blocks are NEW canonical continuous CSR rows with disjoint fresh UIDs.

The enclosing phase audit must first check the complete HZ prefix/frame/RHS
and bind the old words to actual old incidence. This primitive never infers
membership from a count/sum, never accepts a signed deletion and never reads
an iid, solver result, public label or historical unit-pair table.
    """
    if not enabled:
        return None
    if (type(base) is not np.ndarray or base.dtype != np.dtype(np.int64)
            or base.ndim != 1 or len(base) > UID_LIMIT
            or type(old_n_cont) is not int or old_n_cont < 0
            or type(old_uid_ceiling) is not int or not 0 <= old_uid_ceiling <= UID_LIMIT):
        raise ValueError('invalid immutable base/domain')
    pool.charge('overlay_base_validation', 8 * len(base))
    validate_words(base)
    count, total = base // RADIX, base % RADIX
    if (np.any(count > old_uid_ceiling)
            or np.any(total > count * max(0, old_uid_ceiling - 1))):
        raise ValueError('base words exceed the disjoint old UID domain')
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
