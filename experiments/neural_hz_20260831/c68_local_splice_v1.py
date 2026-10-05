"""Explicit-schema source-local inverses plus a sparse immutable-map journal.

This default-off primitive is not an admission receipt. Complete source,
discovery, written-row and ownership proofs remain mandatory at its caller.
Neither original lineage array is copied or mutated. Positive splice tags
are decoded only from the separate journal, never from the source maps.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, UID_LIMIT
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan, encode_splice, decode as decode_splice
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA as LOCAL_SCHEMA, decode as decode_local
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import fraction

SCHEMA = 'c68_immutable_source_local_splice_journal_v1'
MASK = UID_LIMIT - 1


@dataclass(frozen=True)
class LocalSpliceJournal:
    """Shared source maps and O(selected splices + affected tails) new storage."""
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    columns: np.ndarray
    tags: np.ndarray
    offsets: np.ndarray
    retired: np.ndarray
    tails: np.ndarray
    old_n_cont: int
    old_n_eq: int
    source_n_cont: int
    source_schema: str = LOCAL_SCHEMA
    schema: str = SCHEMA

    def numeric_roots(self):
        """Explicit complete numeric payload, including the shared source maps."""
        return {k: getattr(self, k) for k in
                ('eq_roots', 'eq_scales', 'columns', 'tags', 'offsets', 'retired', 'tails')}

    def eq_row(self, old_row, *, pool):
        """Map a physical pre-splice EQ index by ordered sparse deletion rank."""
        if type(old_row) is not int or old_row < 0:
            raise ValueError('nonnegative physical EQ row required')
        pool.charge('c68_deleted_EQ_search', 12 * max(1, len(self.columns).bit_length()) + 8)
        lo, hi = 0, len(self.columns)
        while lo < hi:
            mid = (lo + hi) // 2
            if decode_splice(self.tags[mid])[1] < old_row:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.tags) and decode_splice(self.tags[lo])[1] == old_row:
            return None
        return old_row - lo

    def retired_to(self, uid, *, pool):
        """Return the producer UID now owning a retired consumer's row."""
        pool.charge('c68_retired_UID_search', 8 * max(1, len(self.retired).bit_length()) + 8)
        lo, hi = 0, len(self.retired)
        while lo < hi:
            mid = (lo + hi) // 2
            if int(self.retired[mid]) >> 20 < uid:
                lo = mid + 1
            else:
                hi = mid
        return (int(self.retired[lo]) & MASK) if lo < len(self.retired) and int(self.retired[lo]) >> 20 == uid else None

    def owner_query(self, index, original, *, pool):
        """Exact degree/UID-sum reader, without interpreting source negatives."""
        main = len(self.eq_roots) - self.old_n_eq
        if type(index) is not int or not 0 <= index < main:
            raise ValueError('ownership query outside MAIN')
        pool.charge('c68_sparse_selected_column_search', 8 * max(1, len(self.columns).bit_length()) + 8)
        col = index + self.old_n_cont
        at = int(np.searchsorted(self.columns, col))
        if at < len(self.columns) and int(self.columns[at]) == col:
            return 0
        value = original.query(index, pool=pool)
        count = value // RADIX
        pool.charge('c68_tail_range_search', 16 * max(1, len(self.tails).bit_length()) + 16)
        a, b = np.searchsorted(self.tails, np.array([index << 40, (index + 1) << 40], np.uint64))
        for raw in self.tails[int(a):int(b)]:
            pool.charge('c68_tail_UID_change', 12)
            raw = int(raw)
            value += (raw & MASK) - ((raw >> 20) & MASK)
        if value < 0 or value // RADIX != count or value % RADIX > count * MASK:
            raise ValueError('tail update changed incidence degree or UID range')
        return value

    def iter_words(self, original, *, pool):
        """Complete ordered owner stream using sparse cursors, no dense vector."""
        tail = selected = count = 0
        for index, value in enumerate(original.iter_words(pool=pool)):
            pool.charge('c68_stream_sparse_cursors', 20)
            count = index + 1
            degree = value // RADIX
            while tail < len(self.tails) and int(self.tails[tail]) >> 40 == index:
                pool.charge('c68_stream_tail_UID_change', 12)
                raw = int(self.tails[tail])
                value += (raw & MASK) - ((raw >> 20) & MASK)
                tail += 1
            if selected < len(self.columns) and int(self.columns[selected]) == self.old_n_cont + index:
                value = 0
                selected += 1
            elif value < 0 or value // RADIX != degree or value % RADIX > degree * MASK:
                raise ValueError('owner stream outside original degree/UID range')
            yield value
        if count != len(self.eq_roots) - self.old_n_eq or tail != len(self.tails) or selected != len(self.columns):
            raise ValueError('incomplete original owner stream or journal')

    def reconstruct_fraction(self, hz, continuous, *, pool):
        """Restore unit producers, then exact topological local-edge children."""
        if (self.schema != SCHEMA or self.source_schema != LOCAL_SCHEMA
                or len(continuous) != hz.n_cont or hz.n_cont < self.source_n_cont):
            raise ValueError('explicit local/journal schema and original global frame required')
        pool.charge('c68_full_frame_fraction_and_box', 16 * hz.n_cont)
        result = [F(v) for v in continuous]
        if any(abs(v) > 1 for v in result):
            raise ValueError('point outside original latent box')
        for col, tag, offset in zip(self.columns, self.tags, self.offsets):
            pool.charge('c68_exact_splice_inverse', 128)
            col = int(col)
            kind, _, consumer, inequality, pivot, sign = decode_splice(tag)
            if kind != 'splice':
                raise ValueError('journal contains a non-splice descriptor')
            target = consumer if inequality else self.eq_row(consumer, pool=pool)
            matrix = hz.Auc if inequality else hz.Ac
            if target is None or not 0 <= target < matrix.shape[0]:
                raise ValueError('missing surviving inverse row')
            a, b = map(int, matrix.indptr[target:target + 2])
            cols, values = matrix.indices[a:b], matrix.data[a:b]
            cut = int(np.searchsorted(cols, col))
            pool.charge('c68_exact_surviving_prefix', 64 * cut)
            prefix = sum((F(float(v)) * result[int(k)] for k, v in zip(cols[:cut], values[:cut])), F(0))
            result[col] = (F(float(offset)) - sign * prefix) / F(pivot)
            if abs(result[col]) > 1:
                raise ValueError('unit inverse outside proved MAIN box')
        pool.charge('c68_all_source_local_tags', 8 * (len(self.eq_roots) - self.old_n_eq))
        scales = self.eq_scales.view(np.float64)
        for at in range(self.old_n_eq, len(self.eq_roots)):
            tag = self.eq_roots[at]
            if tag >= 0:
                continue
            pool.charge('c68_exact_local_inverse', 128)
            col = self.old_n_cont + at - self.old_n_eq
            parent, ratio = decode_local(tag, scales[at], column=col,
                                         n_cont=self.source_n_cont, schema=self.source_schema)
            result[col] = fraction(ratio) * result[parent]
            if abs(result[col]) > 1:
                raise ValueError('local inverse outside original box')
        return result


def compile_journal(roots, scales, plans, *, old_n_cont, old_n_eq, source_n_cont,
                    source_schema, pool, enabled=False):
    """Compile checked plans without source writes; no native admission claim."""
    if not enabled:
        return None
    pool.charge('c68_explicit_source_schema_header', 128)
    if (source_schema != LOCAL_SCHEMA or type(roots) is not np.ndarray
            or type(scales) is not np.ndarray or roots.dtype != np.dtype(np.int64)
            or scales.dtype != np.dtype(np.int64) or roots.ndim != 1
            or roots.shape != scales.shape or not roots.flags.c_contiguous
            or not scales.flags.c_contiguous or not 0 <= old_n_eq <= len(roots)
            or not 0 <= old_n_cont <= old_n_cont + len(roots) - old_n_eq <= source_n_cont < 2**31):
        raise ValueError('complete explicit local source maps/frame required')
    columns, tags, offsets, retired, tails = [], [], [], [], []
    definitions, consumers, producers = set(), set(), set()
    previous = previous_definition = -1
    main = len(roots) - old_n_eq
    for p in plans:
        pool.charge('c68_pair_descriptor_and_disjointness', 128)
        if (type(p) is not Plan or not old_n_cont <= p.column < old_n_cont + main
                or p.column <= previous or p.definition <= previous_definition
                or int(roots[old_n_eq + p.column - old_n_cont]) != p.definition
                or not math.isfinite(p.offset) or not 0 <= p.producer_uid < UID_LIMIT
                or not 0 <= p.consumer_uid < UID_LIMIT or p.producer_uid == p.consumer_uid
                or p.producer_uid in producers or (p.inequality, p.consumer) in consumers):
            raise ValueError('complete independent ordered source-bound plans required')
        if p.consumer_main is not None:
            at = old_n_eq + p.consumer_main - old_n_cont
            if p.inequality or not old_n_eq <= at < len(roots) or int(roots[at]) != p.consumer:
                raise ValueError('consumer MAIN definition differs from shared source')
        previous, previous_definition = p.column, p.definition
        definitions.add(p.definition)
        consumers.add((p.inequality, p.consumer))
        producers.add(p.producer_uid)
        columns.append(p.column)
        tags.append(encode_splice(p.definition, p.consumer, p.inequality, p.pivot, p.sign))
        offsets.append(p.offset)
        retired.append((p.consumer_uid << 20) | p.producer_uid)
        pool.charge('c68_tail_incidence_scan', 12 * len(p.tail))
        previous_tail = p.column
        for col in p.tail:
            if type(col) is not int or col <= previous_tail or col >= 2**31:
                raise ValueError('ordered tail above selected pivot required')
            previous_tail = col
            if old_n_cont <= col < old_n_cont + main:
                tails.append(((col - old_n_cont) << 40) | (p.consumer_uid << 20) | p.producer_uid)
    if any(not kind and row in definitions for kind, row in consumers):
        raise ValueError('selected-definition dependencies')
    if len(set(v >> 20 for v in retired)) != len(plans):
        raise ValueError('consumer UID reused by selected rows')
    n, t = len(columns), len(tails)
    pool.charge('c68_sparse_array_writes', 4 * n + t)
    pool.charge('c68_sparse_sort', 4 * (n * max(1, (n - 1).bit_length()) + t * max(1, (t - 1).bit_length())))
    return LocalSpliceJournal(roots, scales, np.asarray(columns, np.int32),
        np.asarray(tags, np.uint64), np.asarray(offsets, np.float64),
        np.sort(np.asarray(retired, np.uint64), kind='stable'),
        np.sort(np.asarray(tails, np.uint64), kind='stable'),
        old_n_cont, old_n_eq, source_n_cont)
