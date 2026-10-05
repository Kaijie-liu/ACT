"""Exact monotone MAIN UID/physical-row runs; no HZ or proof-state deletion."""

import numpy as np

LIMIT = 1 << 20
MASK = LIMIT - 1


def pack(uid, row, length):
    if (any(type(v) is not int for v in (uid,row,length)) or not 0 <= uid < LIMIT
            or not 0 <= row < LIMIT or not 1 <= length <= LIMIT
            or uid+length > LIMIT or row+length > LIMIT):
        raise ValueError('run fields outside exact 20-bit UID/row domain')
    return (uid << 40) | (row << 20) | (length-1)


def unpack(word):
    value = int(word)
    if not 0 <= value < 1 << 60:
        raise ValueError('invalid run word')
    return value >> 40, (value >> 20) & MASK, (value & MASK)+1


def validate(words):
    if type(words) is not np.ndarray or words.dtype != np.dtype(np.uint64) or words.ndim != 1:
        raise ValueError('run index requires flat uint64 numeric ownership')
    if not len(words): return
    if np.any(words >= np.uint64(1 << 60)):
        raise ValueError('run payload has unregistered high bits')
    u = words >> np.uint64(40)
    p = (words >> np.uint64(20)) & np.uint64(MASK)
    n = (words & np.uint64(MASK)) + np.uint64(1)
    ue, pe = u+n, p+n
    if (np.any(ue > LIMIT) or np.any(pe > LIMIT)
            or np.any(u[1:] < ue[:-1]) or np.any(p[1:] < pe[:-1])
            or np.any((u[1:] == ue[:-1]) & (p[1:] == pe[:-1]))):
        raise ValueError('overlapping/nonmonotone/nonmaximal exact runs')


def build(eq_uids, main_first, radix_base, *, pool):
    """Standalone streaming prototype; its FULL scan is charged, not called free.

    Main UIDs increase with physical EQ row after alias erasure. Old prefix
    and radix UIDs interrupt runs and remain represented by existing maps.
    """
    if (type(eq_uids) is not np.ndarray or eq_uids.dtype != np.dtype(np.int64)
            or eq_uids.ndim != 1 or len(eq_uids) > LIMIT
            or type(main_first) is not int or type(radix_base) is not int
            or not 0 <= main_first <= radix_base <= LIMIT):
        raise ValueError('invalid complete UID stream/domain')
    records = []
    start_uid = start_row = previous = None
    length = tracked = 0
    last_main = -1
    def flush():
        if start_uid is not None:
            pool.charge('uid_run_pack', 16)
            records.append(pack(start_uid,start_row,length))
    for row, raw in enumerate(eq_uids):
        pool.charge('uid_run_stream_scan', 8)
        uid = int(raw)
        if not 0 <= uid < LIMIT:
            raise ValueError('physical row lacks a valid stable UID')
        if not main_first <= uid < radix_base:
            flush()
            start_uid = start_row = previous = None
            length = 0
            continue
        if uid <= last_main:
            raise ValueError('MAIN UID stream is not strictly increasing')
        last_main = uid
        tracked += 1
        if previous is not None and uid == previous+1:
            length += 1
        else:
            flush()
            start_uid, start_row, length = uid, row, 1
        previous = uid
    flush()
    pool.charge('uid_run_array', len(records))
    words = np.asarray(records,dtype=np.uint64)
    pool.charge('uid_run_full_validation', 12*len(words))
    validate(words)
    return words, {'tracked_MAIN_rows': tracked, 'run_count': len(words),
        'numeric_entries': len(words), 'numeric_bytes': words.nbytes,
        'standalone_build_work': 8*len(eq_uids)+29*len(words),
        'fused_generation_work_claimed': False, 'original_graph_fields_retired': False,
        'formal_gain': 0}


def _query(words, key, field, pool):
    """Query INSIDE a caller-validated immutable read transaction.

    Full content/geometry validation is done before and after the complete
    batch by the caller; an individual answer is not a proof receipt.
    """
    if type(key) is not int or not 0 <= key < LIMIT:
        raise ValueError('query outside fixed UID/row domain')
    if type(words) is not np.ndarray or words.dtype != np.dtype(np.uint64) or words.ndim != 1:
        raise ValueError('invalid run query owner')
    pool.charge('uid_run_query', 8*max(1,len(words).bit_length())+8)
    lo,hi = 0,len(words)
    while lo < hi:
        mid=(lo+hi)//2
        word=int(words[mid])
        start=word>>40 if field==0 else (word>>20)&MASK
        if start <= key: lo=mid+1
        else: hi=mid
    if lo==0: return None
    uid,row,length=unpack(words[lo-1])
    start=uid if field==0 else row
    if key-start >= length: return None
    return (row if field==0 else uid)+key-start


def row_for_uid(words, uid, *, pool):
    return _query(words,uid,0,pool)


def uid_for_row(words, row, *, pool):
    return _query(words,row,1,pool)
