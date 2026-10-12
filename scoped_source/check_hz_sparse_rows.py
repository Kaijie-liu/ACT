"""Independent sparse-row inclusion checker, for a supplied exact HZ only.

Standard library only. One owned decode per caller source/proof, one coefficient
parse per source/target. Residuals cover sparse unions; fresh-column layout is
checked separately. The deadline is cooperative, not a hard memory/time bound.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import re
import time

REFERENCE = 'OWNED_EXACT_HZ_REFERENCE_V1'
SCHEMA = 'HZ_SPARSE_ROW_OUTER_ENCLOSURE_V1'
VECTORS = ('c', 'b', 'ub')
MATRICES = ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')
MAX_FACTORS, MAX_ROWS, MAX_NNZ = 4096, 16, 262144
STORED_BITS, WORK_BITS = 4096, 8192


class ResourceRefusal(ValueError):
    """A finite arithmetic/admission contract was exceeded, not unsafety."""


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(compact(value)).hexdigest()


def bounded(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > WORK_BITS:
        raise ResourceRefusal('exact intermediate capacity')
    return value


def label(value):
    if type(value) is not str or not 1 <= len(value) <= 48:
        raise ValueError('explicit finite owner/tag required')
    return value


class Work:
    def __init__(self, deadline, chunk_size=256):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('finite deadline required')
        if type(chunk_size) is not int or not 1 <= chunk_size <= 256:
            raise ValueError('chunk size must be in 1..256')
        self.deadline, self.chunk_size = deadline, chunk_size
        self.stats = {'source_parses': 0, 'target_parses': 0,
                      'source_matrix_coefficients': 0, 'target_matrix_coefficients': 0,
                      'source_vector_coefficients': 0, 'target_vector_coefficients': 0,
                      'residual_union_entries': 0, 'residual_chunks': 0,
                      'layout_entries': 0, 'layout_chunks': 0,
                      'sparse_sort_calls': 0, 'sparse_sort_keys': 0}
        self.tick('start')
        if deadline - time.monotonic() > 300:
            raise ValueError('at most 300 second cooperative budget')

    def tick(self, phase):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('sparse-row deadline: ' + phase)

    def entries(self, values, phase, counter=None):
        for i, value in enumerate(values):
            if i % self.chunk_size == 0:
                self.tick(phase)
                if counter:
                    self.stats[counter + '_chunks'] += 1
            if counter:
                key = 'residual_union_entries' if counter == 'residual' else 'layout_entries'
                self.stats[key] += 1
            yield value
        self.tick(phase + '_end')


def capture(value, work, phase):
    """Decoding the anchored bytes prevents ordinary caller row aliasing."""
    work.tick(phase)
    raw = compact(value)
    anchor = hashlib.sha256(raw).hexdigest()
    owned = json.loads(raw)
    work.tick(phase + '_end')
    return owned, anchor


def rational(value, *, target=False):
    # Length is bounded before Fraction's arbitrary-precision integer parsing.
    if type(value) is not str or len(value) > 2470:
        raise ResourceRefusal('stored rational string capacity/type')
    if re.fullmatch(r'-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?', value) is None:
        raise ValueError('canonical rational required')
    q = F(value)
    if max(q.numerator.bit_length(), q.denominator.bit_length()) > STORED_BITS:
        raise ResourceRefusal('stored rational capacity')
    if str(q) != value:
        raise ValueError('canonical rational required')
    if target:
        try:
            f = float(q)
        except OverflowError:
            raise ValueError('nonfinite binary64 target') from None
        if not math.isfinite(f) or F(f) != q:
            raise ValueError('target is not exact stored binary64')
    return q


def _keys(obj, keys, message):
    if type(obj) is not dict or set(obj) != set(keys):
        raise ValueError(message)


def parse_owned(reference, work, *, target=False):
    """Parse only caller-independent data from capture(), once per object."""
    kind = 'target' if target else 'source'
    work.stats[kind + '_parses'] += 1
    _keys(reference, ('schema', 'state', 'ownership'), 'owned reference schema')
    if reference['schema'] != REFERENCE:
        raise ValueError('owned reference schema')
    state = reference['state']
    _keys(state, ('schema', 'continuous_ids', 'binary_ids', 'hz'), 'source state schema')
    if state['schema'] != 'CHECKED_SOURCE_HZ_V1':
        raise ValueError('source state schema')
    ci, bi = state['continuous_ids'], state['binary_ids']
    if type(ci) is not list or type(bi) is not list:
        raise ValueError('factor inventory type')
    cap = MAX_FACTORS + 2 * MAX_ROWS if target else MAX_FACTORS
    if len(ci) + len(bi) > cap:
        raise ResourceRefusal('factor capacity')
    for name in work.entries(ci + bi, kind + '_ids'):
        if type(name) is not str or not 1 <= len(name) <= 200:
            raise ValueError('factor identity')
    if len(set(ci + bi)) != len(ci) + len(bi):
        raise ValueError('factor identity alias')
    ownership = reference['ownership']
    _keys(ownership, ('continuous', 'binary'), 'ownership schema')
    for key, ids in (('continuous', ci), ('binary', bi)):
        owners = ownership[key]
        if type(owners) is not list or len(owners) != len(ids):
            raise ValueError('factor ownership dimensions')
        for owner in work.entries(owners, kind + '_owners'):
            label(owner)
    raw = state['hz']
    _keys(raw, VECTORS + MATRICES + ('frame_id', 'exact'), 'HZ schema')
    if type(raw['frame_id']) is not int or raw['exact'] is not False:
        raise ValueError('frame/exact type')
    h = {'frame_id': raw['frame_id'], 'nc': len(ci), 'nb': len(bi)}
    for key in VECTORS:
        values = raw[key]
        if type(values) is not list or len(values) > MAX_ROWS or (key == 'c' and not values):
            raise ResourceRefusal('row capacity')
        h[key] = []
        for v in work.entries(values, kind + '_vectors'):
            h[key].append(rational(v, target=target))
            work.stats[kind + '_vector_coefficients'] += 1
    total = 0
    for key in MATRICES:
        a = raw[key]
        _keys(a, ('shape', 'data', 'indices', 'indptr'), 'canonical CSR fields')
        nr = len(h['c' if key.startswith('G') else 'b' if key in ('Ac', 'Ab') else 'ub'])
        nc = len(ci) if key.endswith('c') else len(bi)
        shape, data, ix, ptr = (a[k] for k in ('shape', 'data', 'indices', 'indptr'))
        if (type(shape) is not list or any(type(v) is not int for v in shape) or shape != [nr, nc]
                or type(data) is not list or type(ix) is not list or type(ptr) is not list):
            raise ValueError('canonical CSR shape/type')
        total += len(data)
        if total > MAX_NNZ:
            raise ResourceRefusal('stored nonzero capacity')
        if (len(data) != len(ix) or len(ptr) != nr + 1 or any(type(v) is not int for v in ptr)
                or ptr[0] != 0 or ptr[-1] != len(data)
                or any(not 0 <= x <= y <= len(data) for x, y in zip(ptr, ptr[1:]))):
            raise ValueError('canonical CSR pointers')
        rows = []
        for start, stop in zip(ptr, ptr[1:]):
            row, previous = {}, -1
            for p in work.entries(range(start, stop), kind + '_coefficients'):
                j = ix[p]
                if type(j) is not int or not previous < j < nc:
                    raise ValueError('canonical CSR coordinate')
                v = rational(data[p], target=target)
                if not v:
                    raise ValueError('explicit zero CSR coefficient')
                row[j] = v
                previous = j
                work.stats[kind + '_matrix_coefficients'] += 1
            rows.append(row)
        h[key] = rows
    work.tick(kind + '_parsed')
    return h, ci, bi


def residual(source, target, row, ck, bk, nc, work, initial=F(0)):
    answer = bounded(initial)
    for key in (ck, bk):
        left, right = source[key][row], target[key][row]
        keys = set(left) | {j for j in right if key.endswith('b') or j < nc}
        work.stats['sparse_sort_calls'] += 1
        work.stats['sparse_sort_keys'] += len(keys)
        for j in work.entries(sorted(keys), 'residual_chunk', 'residual'):
            delta = bounded(left.get(j, F(0)) - right.get(j, F(0)))
            answer = bounded(answer + abs(delta))
    return answer


def check(reference, proof, *, expected_reference_sha256, expected_owner, deadline, chunk_size=256):
    work = Work(deadline, chunk_size)
    label(expected_owner)
    source, source_hash = capture(reference, work, 'source_anchor')
    owned, anchor = capture(proof, work, 'proof_anchor')
    if source_hash != expected_reference_sha256:
        raise ValueError('external reference binding')
    s, ci, bi = parse_owned(source, work)
    _keys(owned, ('schema', 'reference_sha256', 'owner', 'target', 'target_sha256', 'maps',
                  'output_errors', 'equality_errors'), 'enclosure identity/schema')
    if (owned['schema'] != SCHEMA or owned['reference_sha256'] != source_hash
            or owned['owner'] != expected_owner or identity(owned['target']) != owned['target_sha256']):
        raise ValueError('enclosure identity/schema')
    t, ct, bt = parse_owned(owned['target'], work, target=True)
    if (t['frame_id'] != s['frame_id'] or bt != bi or ct[:len(ci)] != ci
            or owned['target']['ownership']['continuous'][:len(ci)] != source['ownership']['continuous']
            or owned['target']['ownership']['binary'] != source['ownership']['binary']
            or any(len(t[k]) != len(s[k]) for k in VECTORS)
            or len(ct) > len(ci) + len(s['c']) + len(s['b'])):
        raise ValueError('original factor/owner/frame/row inventory')
    maps = {'continuous': list(range(len(ci))), 'binary': list(range(len(bi))),
            'flat': list(range(len(ci))) + [len(ct) + j for j in range(len(bi))]}
    if (owned['maps'] != maps or type(owned['maps']) is not dict
            or any(type(v) is not int for values in owned['maps'].values() for v in values)):
        raise ValueError('continuous/binary/flat factor maps')
    fresh, layout, diagnostics = [], {}, []
    for kind, vector, ck, bk in (('output', 'c', 'Gc', 'Gb'), ('equality', 'b', 'Ac', 'Ab')):
        claims = owned[kind + '_errors']
        if type(claims) is not list or len(claims) != len(s[vector]):
            raise ValueError('compensation row inventory')
        for i, claim in enumerate(claims):
            work.tick('row')
            _keys(claim, ('factor', 'radius'), 'compensation fields')
            radius = rational(claim['radius'], target=True)
            if radius < 0:
                raise ValueError('compensation sign')
            needed = residual(s, t, i, ck, bk, len(ci), work, abs(bounded(s[vector][i] - t[vector][i])))
            if radius < needed:
                raise ValueError('inward compensation')
            expected = None
            if radius:
                expected = f'{expected_owner}/{source_hash}/{kind}/{i}'
                layout[(ck, i)] = {len(ci) + len(fresh): radius}
                fresh.append(expected)
            if claim['factor'] != expected:
                raise ValueError('compensation identity')
            diagnostics.append({'kind': kind, 'row': i, 'required': str(needed), 'stored': str(radius)})
    if (ct != ci + fresh or owned['target']['ownership']['continuous'] !=
            source['ownership']['continuous'] + [expected_owner] * len(fresh)):
        raise ValueError('fresh factor ownership/inventory')
    for key in MATRICES:
        for i, row in enumerate(t[key]):
            actual = {}
            for j, v in work.entries(row.items(), 'layout_chunk', 'layout'):
                if key.endswith('c') and j >= len(ci):
                    actual[j] = v
            if actual != layout.get((key, i), {}):
                raise ValueError('error/slack zero layout')
    inequalities = []
    for i, rhs in enumerate(s['ub']):
        needed = residual(s, t, i, 'Auc', 'Aub', len(ci), work, rhs)
        if t['ub'][i] < needed:
            raise ValueError('inward inequality/guard')
        inequalities.append({'row': i, 'required': str(needed), 'stored': str(t['ub'][i])})
    result = {'status': 'CHECKED_SPARSE_ROW_ENCLOSURE_GIVEN_REFERENCE',
              'reference_sha256': source_hash, 'proof_sha256': anchor, 'target_sha256': owned['target_sha256'],
              'added_continuous': len(fresh), 'binary_preserved': len(bi),
              'compensation': diagnostics, 'inequalities': inequalities, 'work': dict(work.stats),
              'chunk_size': chunk_size, 'set_equality_claimed': False, 'network_proof': False,
              'deployed_float_SAFE': False, 'hard_budget_supervision': False,
              'remaining_trust': ['given exact reference and declared ownership', 'exact checker implementation']}
    work.tick('before_final_identity')
    if identity(reference) != source_hash or identity(proof) != anchor:
        raise ValueError('enclosure input changed during checking')
    work.tick('after_final_identity')
    return result


def check_embedding(reference, proof, original, extended, *, expected_reference_sha256,
                    expected_owner, deadline, chunk_size=256):
    """Supplementary exact points; core parse counters exclude this separate call."""
    work = Work(deadline, chunk_size)
    anchors = [identity(v) for v in (reference, proof, original, extended)]
    check(reference, proof, expected_reference_sha256=expected_reference_sha256,
          expected_owner=expected_owner, deadline=deadline, chunk_size=chunk_size)
    refs = [capture(v, work, 'embedding_owned')[0] for v in (reference, proof['target'])]
    assignments = [capture(v, work, 'point_owned')[0] for v in (original, extended)]
    outputs, points = [], []
    for ref, assignment, target in zip(refs, assignments, (False, True)):
        h, ci, bi = parse_owned(ref, work, target=target)
        _keys(assignment, ('continuous', 'binary'), 'point fields')
        if any(type(assignment[k]) is not list for k in assignment):
            raise ValueError('point vectors')
        x, z = ([rational(v) for v in work.entries(assignment[k], 'point_coefficients')]
                for k in ('continuous', 'binary'))
        if len(x) != len(ci) or len(z) != len(bi) or any(abs(v) > 1 for v in x) or any(v not in (-1, 1) for v in z):
            raise ValueError('factor assignment domains')
        points.append((x, z))
        def values(ck, bk):
            result = []
            for a, b in zip(h[ck], h[bk]):
                total = F(0)
                for row, vals in ((a, x), (b, z)):
                    for j, v in work.entries(row.items(), 'point_chunk'):
                        total = bounded(total + bounded(v * vals[j]))
                result.append(total)
            return result
        if values('Ac', 'Ab') != h['b'] or any(a > b for a, b in zip(values('Auc', 'Aub'), h['ub'])):
            raise ValueError('infeasible factor assignment')
        outputs.append([bounded(c + v) for c, v in zip(h['c'], values('Gc', 'Gb'))])
    if points[1][0][:len(points[0][0])] != points[0][0] or points[1][1] != points[0][1]:
        raise ValueError('embedding changed original factors')
    if outputs[0] != outputs[1]:
        raise ValueError('embedding output mismatch')
    if anchors != [identity(v) for v in (reference, proof, original, extended)]:
        raise ValueError('embedding input changed')
    work.tick('embedding_final_identity')
    return {'status': 'CHECKED_EXACT_ASSIGNMENT_EMBEDDING', 'outputs': list(map(str, outputs[0]))}
