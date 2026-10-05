"""Prepaid full owned evidence; never change the live source or owner ledger.

All named primitive arrays follow the same copy rule, including CSR components.
Callers retain the complete original live objects separately and still compare
actual live source arrays. These copies are evidence, not a replacement HZ.
"""
import numpy as np

MAX_ARRAYS = 128
MAX_ENTRIES = 64000000
MAX_BYTES = 1073741824


def snapshot(arrays, *, pool):
    """Own a complete read-only, byte-preserving snapshot after full prepayment.

The1024 fixed units cover at most128 array/header entries and the fixed mapping
allocation. Eight units per logical element cover its read, copied allocation,
write and retention. No source coefficient arithmetic, dtype conversion or
coalescing occurs. Header checks precede all array allocation/numeric reads;
the element charge precedes even the first copy. A failed reservation cannot
leave a partially funded set of snapshots. Empty arrays retain their shapes.
"""
    if type(arrays) is not dict or not 0 < len(arrays) <= MAX_ARRAYS:
        raise ValueError('bounded nonempty complete named array mapping required')
    pool.charge('c129_complete_owned_evidence_headers', 1024)
    entries = size_bytes = 0
    for name, value in arrays.items():
        if (type(name) is not str or type(value) is not np.ndarray
            or value.dtype.kind not in 'biuf' or value.dtype.hasobject
            or value.dtype.itemsize > 8):
            raise ValueError('complete primitive ndarray evidence required')
        entries += int(value.size)
        size_bytes += int(value.nbytes)
        if entries > MAX_ENTRIES or size_bytes > MAX_BYTES:
            raise MemoryError('complete owned evidence entry/byte bound exceeded')
    pool.charge('c129_complete_owned_evidence_copy', 8*entries)
    owned = {}
    for name, value in arrays.items():
        copy = np.array(value, copy=True, order='C')
        if (copy.dtype != value.dtype or copy.shape != value.shape
            or copy.base is not None or not copy.flags.owndata
            or not copy.flags.c_contiguous):
            raise ValueError('full evidence copy did not acquire exact owned storage')
        copy.flags.writeable = False
        owned[name] = copy
    return owned
