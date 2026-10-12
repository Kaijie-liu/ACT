"""Untrusted opt-in sparse-row producer; no model, optimizer or device work.

Parsing/serialization are shared; rounding and residual construction are not
used by the independent inclusion checker. Old production APIs are unchanged.
"""
from fractions import Fraction as F
import math

from source_enclosure.format import pack
from scoped_source.check_hz_sparse_rows import (
    Work, REFERENCE, SCHEMA, MATRICES, VECTORS, bounded, capture, identity, label, parse_owned)


def nearest(q):
    try:
        f = float(q)
    except OverflowError:
        raise ValueError('no finite binary64 coefficient') from None
    if not math.isfinite(f):
        raise ValueError('no finite binary64 coefficient')
    return F(f)


def upward(q):
    f = float(nearest(q))
    if F(f) < q:
        f = math.nextafter(f, math.inf)
    if not math.isfinite(f):
        raise ValueError('no finite outward binary64 endpoint')
    return F(f)


def produce(reference, *, expected_reference_sha256, owner, deadline, chunk_size=256):
    work = Work(deadline, chunk_size)
    label(owner)
    owned, anchor = capture(reference, work, 'source_anchor')
    if anchor != expected_reference_sha256:
        raise ValueError('external reference binding')
    s, ci, bi = parse_owned(owned, work)
    target = {'frame_id': s['frame_id']}
    for key in ('c', 'b'):
        target[key] = [nearest(v) for v in work.entries(s[key], 'round_vectors')]
    for key in MATRICES:
        target[key] = []
        for row in s[key]:
            rounded = {}
            for j, v in work.entries(row.items(), 'round_chunk'):
                value = nearest(v)
                if value:
                    rounded[j] = value
            target[key].append(rounded)
    newids, records = list(ci), {}
    # The producer only creates rounded original columns, so its sum traverses
    # source entries. The checker independently covers source AND target union.
    def needed(i, ck, bk, initial):
        total = bounded(initial)
        for key in (ck, bk):
            for j, v in work.entries(s[key][i].items(), 'producer_residual_chunk'):
                total = bounded(total + abs(bounded(v - target[key][i].get(j, F(0)))))
        return total
    for kind, vector, ck, bk in (('output', 'c', 'Gc', 'Gb'), ('equality', 'b', 'Ac', 'Ab')):
        records[kind] = []
        for i, value in enumerate(s[vector]):
            radius = upward(needed(i, ck, bk, abs(bounded(value - target[vector][i]))))
            name = None
            if radius:
                name = f'{owner}/{anchor}/{kind}/{i}'
                if name in newids + bi:
                    raise ValueError('fresh factor collision')
                target[ck][i][len(newids)] = radius
                newids.append(name)
            records[kind].append({'factor': name, 'radius': str(radius)})
    target['ub'] = [upward(needed(i, 'Auc', 'Aub', rhs)) for i, rhs in enumerate(s['ub'])]
    result = {'schema': REFERENCE, 'state': pack(target, newids, bi),
              'ownership': {'continuous': owned['ownership']['continuous'] + [owner] * (len(newids) - len(ci)),
                            'binary': list(owned['ownership']['binary'])}}
    parse_owned(result, work, target=True)
    proof = {'schema': SCHEMA, 'reference_sha256': anchor, 'owner': owner,
             'target': result, 'target_sha256': identity(result),
             'maps': {'continuous': list(range(len(ci))), 'binary': list(range(len(bi))),
                      'flat': list(range(len(ci))) + [len(newids) + j for j in range(len(bi))]},
             'output_errors': records['output'], 'equality_errors': records['equality']}
    work.tick('before_final_identity')
    if identity(reference) != anchor:
        raise ValueError('reference changed during construction')
    work.tick('after_final_identity')
    return proof


def instantiate(target, *, expected_target_sha256, deadline):
    """Actual CPU SparseHZono, exact stored coefficient round trip only."""
    work = Work(deadline)
    owned, anchor = capture(target, work, 'live_anchor')
    if anchor != expected_target_sha256:
        raise ValueError('target identity')
    h, ci, bi = parse_owned(owned, work, target=True)
    import numpy as np
    import scipy.sparse as sp
    from act.back_end.solver.solver_hz import SparseHZono
    from act.back_end.solver.hz_lp_export import snapshot
    from upstream_source.checker import hz
    args = {k: np.array(list(map(float, h[k])), dtype=np.float64) for k in VECTORS}
    for key in MATRICES:
        data, indices, ptr = [], [], [0]
        for row in h[key]:
            for j, v in work.entries(row.items(), 'instantiate_chunk'):
                indices.append(j)
                data.append(float(v))
            ptr.append(len(data))
        args[key] = sp.csr_matrix((data, indices, ptr), shape=(len(h[key]), len(ci) if key.endswith('c') else len(bi)))
    live = SparseHZono(**args, frame_id=h['frame_id'], exact=False)
    snap = snapshot(live)
    if hz(snap) != h:
        raise ValueError('live sparse HZ coefficient mismatch')
    work.tick('before_final_identity')
    if identity(target) != anchor:
        raise ValueError('target changed during instantiation')
    work.tick('after_final_identity')
    return live, snap
