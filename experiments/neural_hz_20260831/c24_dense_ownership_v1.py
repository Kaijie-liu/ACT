"""C24 exact ownership with dense needed-row ranks inside fixed node UID ranges.

Only local logical row labels change. cumsum(mask) replaces arange(width),
with the same one-per-output-label traversal tariff, three vector operations
and one owned labels array. Zero rows are masked; active ranks are unique.
No HZ coefficient, binary phase, global factor slot or interval changes.
"""

import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c6_support_affine_plan_v2 import SupportEngine as BaseEngine
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import operator_digest, digest_arrays

RADIX = 1 << 40
UID_LIMIT = 1 << 20


def validate_words(words):
    if type(words) is not np.ndarray or words.dtype != np.dtype(np.int64) or words.ndim != 1:
        raise ValueError('ownership requires a flat int64 vector')
    count, total = words // RADIX, words % RADIX
    if (np.any(words < 0) or np.any(count > UID_LIMIT)
            or np.any(total > count * (UID_LIMIT - 1))):
        raise ValueError('ownership count/sum range or underflow violation')


def word(uid):
    if type(uid) is not int or not 0 <= uid < UID_LIMIT:
        raise ValueError('row UID outside fixed canonical domain')
    return RADIX + uid


def unique_other(packed, own_uid):
    remaining = int(packed) - word(own_uid)
    if remaining < 0:
        raise ValueError('missing own defining occurrence')
    count, total = divmod(remaining, RADIX)
    if count > UID_LIMIT or total > count * (UID_LIMIT - 1):
        raise ValueError('invalid remaining ownership')
    return int(total) if count == 1 else None


class DenseOwnerEngine(BaseEngine):
    """Replace reverse support count with packed count/local-row sum.

    Forward support is unchanged. Packed reverse results use distinct cache
    keys; local row IDs permit reuse for equal masks at different graph nodes.
    Every geometric visit is charged exactly as in the inherited traversal.
    """

    def __init__(self, max_visits=256_000_000):
        if type(max_visits) is not int or not 0 <= max_visits <= 256_000_000:
            raise ValueError('invalid/increased ownership work ceiling')
        super().__init__(max_visits)

    def owners(self, op, mask):
        sha = operator_digest(op)
        mask = np.asarray(mask)
        if (mask.dtype != np.dtype(bool) or mask.shape != (op.shape[0],)
                or not 0 <= op.shape[0] <= UID_LIMIT):
            raise ValueError('bounded boolean ownership rows required')
        if op.shape[1] > 64_000_000 or self.cache_bytes + 8 * op.shape[1] > 1024**3:
            raise MemoryError('ownership vector/cache preallocation exceeds unchanged ceiling')
        key = ('c24_packed_reverse_dense_needed_rank', id(op), sha, digest_arrays(mask), RADIX, UID_LIMIT)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if not mask.any():
            result = np.zeros(op.shape[1], np.int64)
        else:
            # Each canonical row contributes at most once per input column.
            # count<=2^20, local sum<2^40 => result<2^61, no int64 overflow.
            self.charge(op.shape[0])
            labels = (RADIX - 1 + np.cumsum(mask, dtype=np.int64)) * mask
            if type(op) is sp.csr_matrix:
                if not op.has_canonical_format:
                    raise ValueError('ownership needs unique row/column incidence')
                self.charge(op.nnz)
                values = np.repeat(labels, np.diff(op.indptr))
                values[op.data == 0.] = 0
                result = np.zeros(op.shape[1], np.int64)
                np.add.at(result, op.indices, values)
            else:
                # The inherited V2 convolution integer traversal is valid for
                # these labels by the stricter canonical signed-int64 proof.
                # Do not route labels through BaseEngine.compute's mask API.
                if (any(v <= 0 for v in (*op._stride, *op._dilation))
                        or op._groups <= 0 or any(v < 0 for v in op._padding)):
                    raise ValueError('noncanonical convolution incidence geometry')
                result = self._conv_counts(op, labels, True)
        # The canonical geometry and row-label bound prove this range without
        # another full vector pass. Independent tests/oracles check the theorem;
        # the complete generation ledger is range-checked again at publication.
        result.flags.writeable = False
        if self.cache_bytes + result.nbytes > 1024**3:
            raise MemoryError('unchanged support/ownership cache ceiling exceeded')
        self.cache_bytes += result.nbytes
        self.cache[key] = result
        return result


def retag(local, base):
    if type(local) is not np.ndarray or local.dtype != np.dtype(np.int64) or local.ndim != 1:
        raise ValueError('retag requires an authenticated flat int64 result')
    if type(base) is not int or not 0 <= base < UID_LIMIT:
        raise ValueError('invalid graph UID base')
    count, total = local // RADIX, local % RADIX
    tagged = total + count * base
    if np.any(count < 0) or np.any(count > UID_LIMIT) or np.any(tagged > count * (UID_LIMIT - 1)):
        raise ValueError('retagged consumers exceed UID domain')
    return local + count * base


def append_owned_rows(ledger, matrix, first_uid):
    """Consume only newly emitted canonical predicate rows, not an old HZ scan."""
    if (not sp.isspmatrix_csr(matrix) or not matrix.has_canonical_format
            or np.any(matrix.data == 0.) or not np.isfinite(matrix.data).all()
            or type(first_uid) is not int or first_uid < 0
            or first_uid + matrix.shape[0] > UID_LIMIT):
        raise ValueError('invalid new physical predicate ownership rows')
    for r in range(matrix.shape[0]):
        start, stop = matrix.indptr[r:r + 2]
        ledger.change(matrix.indices[start:stop], first_uid + r, 1)


class Ledger:
    """Owned generation metadata; deltas come from exact row-emission events.

    Aggregate words are NOT arbitrary-set membership proofs. The caller must
    supply canonical known incidences from its owned row transaction. Complete
    independent incidence audits and a sealed state guard publication.
    """

    def __init__(self, words, old_nc, *, pool):
        self.words = np.asarray(words)
        self.old_nc = int(old_nc)
        self.pool = pool
        self.updates = 0
        if self.words.dtype != np.dtype(np.int64) or self.words.ndim != 1:
            raise ValueError('invalid owned metadata vector')

    def change(self, columns, uid, sign):
        if sign not in (-1, 1):
            raise ValueError('only a known incidence add/remove is allowed')
        amount = sign * word(uid)
        columns = np.asarray(columns)
        if columns.ndim != 1 or columns.dtype.kind not in 'iu':
            raise ValueError('invalid incidence columns')
        active = (columns >= self.old_nc) & (columns < self.old_nc + len(self.words))
        ids = columns[active] - self.old_nc
        self.pool.charge('ownership_known_incidence_updates', 4 * int(ids.size))
        if ids.size:
            limit = int(ids.size) * abs(amount)
            current = self.words[ids]
            if ((sign > 0 and int(current.max()) + limit > np.iinfo(np.int64).max)
                    or (sign < 0 and int(current.min()) - limit < np.iinfo(np.int64).min)):
                raise ValueError('ownership event could overflow signed int64')
        # Repeats are deliberate when quotient collision records collapse them
        # later. The final canonical state is checked before publication.
        np.add.at(self.words, ids, amount)
        self.updates += int(ids.size)
        if np.any(self.words[ids] < 0):
            raise ValueError('ownership update removed absent incidence')

    def relocate_radix(self, columns, old_uid, new_uid):
        word(old_uid)
        word(new_uid)
        columns = np.asarray(columns)
        ids = columns[(columns >= self.old_nc) & (columns < self.old_nc + len(self.words))] - self.old_nc
        self.pool.charge('ownership_radix_relocations', 4 * int(ids.size))
        np.add.at(self.words, ids, new_uid - old_uid)
        self.updates += int(ids.size)

    def retire_verified_frontier(self, indices):
        """After the COMPLETE quotient rewrite, erased aliases have no rows.

        This is a trusted construction event, not arbitrary set subtraction:
        the independent alias frontier and full incidence rewrite must have
        completed. Intermediate alias words are intentionally not queried.
        The actual sparse-incidence oracle independently checks all final zeros.
        """
        indices = np.asarray(indices)
        self.pool.charge('ownership_verified_frontier_retirement', 3 * int(indices.size))
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(self.words))
                or np.any(np.diff(indices.astype(np.int64)) <= 0)):
            raise ValueError('invalid verified alias frontier index set')
        self.words[indices] = 0

    def finish(self):
        self.pool.charge('ownership_final_range_validation', 3 * len(self.words))
        validate_words(self.words)
