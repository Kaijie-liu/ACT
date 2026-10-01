"""Exact reception from original shared/private HZ blocks, never a joint LP.

Given-HZ factor provenance is a premise here; the source checker binds it. Every
row is parsed even for zero duals. No producer, optimizer or sparse library call.
"""
from fractions import Fraction as F

from act.back_end.moe.check_hz_endpoints import _source
from scoped_source.rowwise_bound import clock, identity, rational

SCHEMA = 'SHARED_TEMPLATE_BLOCK_SUPPORT_V1'
ALGORITHM = 'projected_dual_subgradient_blocks_v1'
NAMES = ('entry', 'a', 'b')


def structure(sources, tick):
    if set(sources) != {'common', 'entry', 'a', 'b'}:
        raise ValueError('four original HZ sources required')
    parsed = {k: _source(s, tick) for k, s in sources.items()}
    common, sc, sb = parsed['common']; entry, ec, eb = parsed['entry']
    frame = sources['common']['frame_id']
    if (ec, eb) != (sc, sb):
        raise ValueError('guard cannot add factors')
    for key in ('c', 'Gc', 'Gb', 'Ac', 'Ab', 'b'):
        if entry[key] != common[key]:
            raise ValueError('guard changed common output/equalities')
    for name in NAMES:
        h, nc, nb = parsed[name]
        if nc < sc or nb < sb or sources[name]['frame_id'] != frame:
            raise ValueError('factor dimensions/frame binding')
        for ck, bk, rhs in (('Ac', 'Ab', 'b'), ('Auc', 'Aub', 'ub')):
            n = len(common[rhs])
            if any(h[k][:n] != common[k] for k in (ck, bk, rhs)):
                raise ValueError('common prefix changed/lost/private pollution')
    a, ac, ab = parsed['a']; b, bc, bb = parsed['b']
    nc, nb = ac + bc - sc, ab + bb - sb
    if len(a['c']) != len(b['c']) or not 1 <= len(a['c']) * 2 <= 128:
        raise ValueError('expert output dimensions')
    if not 1 <= nc + nb <= 128:
        raise ValueError('global factor capacity')
    cm = {'entry': list(range(sc)), 'a': list(range(ac)),
          'b': list(range(sc)) + list(range(ac, nc))}
    bm = {'entry': list(range(sb)), 'a': list(range(ab)),
          'b': list(range(sb)) + list(range(ab, nb))}
    layout = []; parts = []
    for name in NAMES:
        h, lc, lb = parsed[name]
        es = 0 if name == 'entry' else len(common['b'])
        us = 0 if name == 'entry' else len(common['ub'])
        mapping = cm[name] + [nc + j for j in bm[name]]
        layout.append({'source': name, 'columns': mapping, 'equality_start': es,
                       'inequality_start': us})
        # Local coordinates only; no global row expansion.
        def local(ck, bk, start):
            return [dict(row) | {lc+j: v for j, v in brow.items()}
                    for row, brow in zip(h[ck][start:], h[bk][start:])]
        parts.append({'source': name, 'columns': mapping, 'width': lc+lb,
                      'A': local('Auc', 'Aub', us), 'b': h['ub'][us:],
                      'E': local('Ac', 'Ab', es), 'h': h['b'][es:]})
    if sum(len(p['b']) + len(p['h']) for p in parts) > 256:
        raise ValueError('global row capacity')
    return parsed, {'n_cont': nc, 'n_bin': nb, 'blocks': layout}, parts


def project(parsed, layout, query, tick):
    a = parsed['a'][0]; b = parsed['b'][0]; count = len(a['c'])
    if len(query['q']) != count * 2 or query['side'] not in ('min', 'max'):
        raise ValueError('objective shape/side')
    q = list(map(rational, query['q'])); offset = rational(query['offset'])
    sign = 1 if query['side'] == 'min' else -1
    c = [F(0)] * (layout['n_cont'] + layout['n_bin'])
    d = offset
    for h, weights, name, block in ((a, q[:count], 'a', 1), (b, q[count:], 'b', 2)):
        lc = parsed[name][1]; mapping = layout['blocks'][block]['columns']
        for weight, center, cr, br in zip(weights, h['c'], h['Gc'], h['Gb']):
            tick(); d += weight * center
            for j, v in cr.items(): c[mapping[j]] += sign * weight * v
            for j, v in br.items(): c[mapping[lc+j]] += sign * weight * v
    return c, sign*d


def validate_batch(batch, *, expected_batch_sha256, deadline):
    tick = clock(deadline)
    if (identity(batch) != expected_batch_sha256 or set(batch) !=
            {'schema', 'context', 'sources', 'layout', 'queries', 'relaxation'}
            or batch['schema'] != SCHEMA or batch['relaxation'] != 'BINARY_FACTORS_TO_CONTINUOUS_BOX'):
        raise ValueError('block batch identity/schema')
    ctx = batch['context']
    if type(ctx) is not dict or any(type(ctx.get(k)) is not str or not ctx[k]
                                  for k in ('request', 'domain', 'guard')):
        raise ValueError('caller context required')
    parsed, layout, parts = structure(batch['sources'], tick)
    if batch['layout'] != layout:
        raise ValueError('block columns/spans/private alias')
    queries = batch['queries']
    if type(queries) is not list or not 1 <= len(queries) <= 8:
        raise ValueError('finite complete query roster')
    ids = set(); targets = []
    for q in queries:
        if (set(q) != {'id', 'q', 'offset', 'side', 'c', 'constant'} or type(q['id']) is not str
                or not q['id'] or q['id'] in ids):
            raise ValueError('query identity/fields')
        ids.add(q['id'])
        for v in q['q'] + [q['offset']] + q['c'] + [q['constant']]:
            if type(v) is not str or str(rational(v)) != v:
                raise ValueError('canonical exact query required')
        c, d = project(parsed, layout, q, tick)
        if q['c'] != list(map(str, c)) or q['constant'] != str(d):
            raise ValueError('original property projection/offset')
        targets.append((c, d))
    if identity(batch) != expected_batch_sha256: raise ValueError('batch changed while checking')
    tick(); return parts, targets


def evaluate(parts, target, duals, tick):
    """Exact weak duality; accumulate shared columns before taking box support."""
    if type(duals) is not list or len(duals) != 3:
        raise ValueError('three complete dual blocks required')
    c, d = target; residual = c[:]; lower = d
    for p, dual in zip(parts, duals):
        if (set(dual) != {'source', 'y', 't'} or dual['source'] != p['source']
                or len(dual['y']) != len(p['b']) or len(dual['t']) != len(p['h'])):
            raise ValueError('dual block row/source coverage')
        y = list(map(rational, dual['y'])); t = list(map(rational, dual['t']))
        if any(v > 0 for v in y): raise ValueError('inequality dual must be nonpositive')
        for matrix, rhs, values in ((p['A'], p['b'], y), (p['E'], p['h'], t)):
            for row, bound, value in zip(matrix, rhs, values):
                tick(); lower += bound * value
                for col, coefficient in row.items():
                    residual[p['columns'][col]] -= coefficient * value
    lower -= sum(map(abs, residual), F(0))
    tick(); return lower, residual


def check_batch(batch, candidate, *, expected_batch_sha256, deadline):
    tick = clock(deadline); anchor = identity(candidate)
    parts, targets = validate_batch(batch, expected_batch_sha256=expected_batch_sha256, deadline=deadline)
    if (set(candidate) != {'batch_sha256', 'entries', 'algorithm', 'iterations', 'dtype', 'device'}
            or candidate['batch_sha256'] != expected_batch_sha256
            or candidate['algorithm'] != ALGORITHM or type(candidate['iterations']) is not int
            or candidate['iterations'] != 128 or candidate['dtype'] != 'float64'
            or candidate['device'] != 'cpu'
            or len(candidate['entries']) != len(batch['queries'])):
        raise ValueError('candidate binding/complete roster/configuration')
    results = []
    for q, target, item in zip(batch['queries'], targets, candidate['entries']):
        if set(item) != {'id', 'duals', 'claimed_lower_bound', 'zero_candidate_lower_bound'} or item['id'] != q['id']:
            raise ValueError('candidate query identity')
        bound, residual = evaluate(parts, target, item['duals'], tick)
        zero = target[1] - sum(map(abs, target[0]), F(0))
        if rational(item['claimed_lower_bound']) > bound or rational(item['zero_candidate_lower_bound']) != zero:
            raise ValueError('candidate claim/zero bound')
        results.append({'id': q['id'], 'side': q['side'], 'bound': str(bound if q['side']=='min' else -bound),
                        'checked_lower_bound': str(bound), 'residual': list(map(str, residual))})
    if identity(batch) != expected_batch_sha256 or identity(candidate) != anchor:
        raise ValueError('block reception input pollution')
    tick(); return {'batch_sha256': expected_batch_sha256, 'results': results}
