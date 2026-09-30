"""Untrusted H2 source adapter; whole pair-local source, no pruning study.

Both arms obtain the same independently checkable source ranges and guards.
All pair/property duties remain, including empty guards and interval reuse.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_source.graph import clock
from scoped_source.sparse_ir import index, row, lp_record
from scoped_source.endpoint_build import build as endpoints
from source_enclosure.format import identity
from upstream_source.checker import csr

THRESHOLD = F(1, 10_000_000)


def source_request(doc, expected, tick):
    r, nodes, outputs = index(doc, expected, tick)
    bank = {}; bounds = {}
    for name, node in nodes.items():
        tick(); rows = []
        if node['kind'] == 'input':
            lo, hi = node['bounds']
        elif node['kind'] == 'affine':
            lo = hi = node['bias']
            for parent, v in node['terms'].items():
                a, b = bounds[parent]; lo += min(v*a, v*b); hi += max(v*a, v*b)
            rows = [row('eq', {name: F(1), **{p: -v for p, v in node['terms'].items()}}, node['bias'])]
        else:
            p = node['parent']; a, b = bounds[p]; lo, hi = max(F(0), a), max(F(0), b)
            if b <= 0: rows = [row('eq', {name: F(1)}, F(0))]
            elif a >= 0: rows = [row('eq', {name: F(1), p: F(-1)}, F(0))]
            else:
                slope = b/(b-a)
                rows = [row('le', {name: F(-1)}, F(0)),
                        row('le', {p: F(1), name: F(-1)}, F(0)),
                        row('le', {name: F(1), p: -slope}, -slope*a)]
        bounds[name] = (lo, hi)
        bank[name] = {'bounds': [str(lo), str(hi)], 'rows': rows}
    duties = []; scopes = []
    for pair in combinations(range(r['experts']), 2):
        a, b = pair; ra, rb = outputs['router'][a], outputs['router'][b]
        low, high = bounds[ra][0]-bounds[rb][1], bounds[ra][1]-bounds[rb][0]
        gate = ['1/2', '1/2'] if low == high == 0 else ['0', '1/2'] if high <= 0 else \
               ['1/2', '1'] if low >= 0 else ['0', '1']
        names = [v for v in nodes if v.split('/')[0] in ('input', 'router', f'expert{a}', f'expert{b}')]
        rows = [line for v in names for line in bank[v]['rows']]
        conflict = None
        for selected in pair:
            for outside in range(r['experts']):
                if outside in pair: continue
                s, o = outputs['router'][selected], outputs['router'][outside]
                inequality_index = sum(v['sense'] == 'le' for v in rows)
                rows.append(row('le', {o: F(1), s: F(-1)}, F(0)))
                gap = bounds[o][0]-bounds[s][1]
                if conflict is None and gap > 0:
                    conflict = {'selected': selected, 'outside': outside,
                                'inequality_index': inequality_index, 'gap': str(gap)}
        base = lp_record(names, {v: bounds[v] for v in names}, rows, {}, F(0))
        for competitor in range(r['classes']):
            if competitor == r['label']: continue
            forms = []; facts = []
            for expert in pair:
                y, j = outputs[f'expert{expert}'][r['label']], outputs[f'expert{expert}'][competitor]
                forms.append({'c': [str(int(v == y)-int(v == j)) for v in names],
                              'offset': str(-F(r['margin']))})
                facts.append(str(bounds[y][0]-bounds[j][1]-F(r['margin'])))
            duties.append({'pair': list(pair), 'competitor': competitor, 'variables': names[:],
                           'base': deepcopy(base), 'a': forms[0], 'b': forms[1], 'gate': gate[:]})
            scopes.append({'pair': list(pair), 'competitor': competitor,
                           'router_margin': [str(low), str(high)], 'facts': facts,
                           'fact_domain': 'GLOBAL_INPUT_BOX', 'guard_conflict': conflict})
    request = {'schema': 'GIVEN_LP_TOP2_ENDPOINT_REQUEST_V1', 'experts': r['experts'],
               'classes': r['classes'], 'label': r['label'], 'duties': duties}
    tick(); return bank, request, scopes


def mc_lp(duty):
    names = duty['variables']; base = duty['base']; d0 = F(duty['a']['offset'])-F(duty['b']['offset'])
    delta = {v: F(a)-F(b) for v, a, b in zip(names, duty['a']['c'], duty['b']['c'])}
    dl = du = d0
    for v, low, high in zip(names, base['lower'], base['upper']):
        dl += min(delta[v]*F(low), delta[v]*F(high)); du += max(delta[v]*F(low), delta[v]*F(high))
    rows = []
    for key, rhs, sense in [('A', 'b', 'le'), ('E', 'h', 'eq')]:
        rows.extend(row(sense, {names[j]: v for j, v in r.items()}, F(b)) for r, b in zip(csr(base[key]), base[rhs]))
    a, b = map(F, duty['gate'])
    for g, d, sign in [(a, dl, 1), (b, du, 1), (b, dl, -1), (a, du, -1)]:
        terms = {v: sign*g*c for v, c in delta.items()}
        terms.update(gate=sign*d, product=F(-sign))
        rows.append(row('le', terms, sign*g*(d-d0)))
    bounds = {v: tuple(map(F, pair)) for v, pair in zip(names, zip(base['lower'], base['upper']))}
    corners = [g*d for g in (a, b) for d in (dl, du)]
    bounds.update(gate=(a, b), product=(min(corners), max(corners)))
    objective = {v: F(c) for v, c in zip(names, duty['b']['c'])}; objective['product'] = F(1)
    return lp_record(names+['gate', 'product'], bounds, rows, objective, F(duty['b']['offset']))


def common_certificate(lp, conflict=None):
    """Fresh exact candidate from a box or a strict guard/box conflict."""
    c = list(map(F, lp['c'])); low = list(map(F, lp['lower'])); high = list(map(F, lp['upper']))
    y = [F(0)]*len(lp['b']); lower = F(lp['offset']) + sum((min(v*a, v*b) for v, a, b in zip(c, low, high)), F(0))
    if conflict is not None:
        scale = max(F(0), (1+THRESHOLD-lower)/F(conflict['gap']))
        y[conflict['inequality_index']] = -scale
    value = F(lp['offset']); residual = c[:]
    for r, b, d in zip(csr(lp['A']), lp['b'], y):
        value += F(b)*d
        for j, a in r.items(): residual[j] -= d*a
    value += sum((min(v*a, v*b) for v, a, b in zip(residual, low, high)), F(0))
    return {'lp_sha256': identity(lp), 'inequality_dual': list(map(str, y)),
            'equality_dual': ['0']*len(lp['h']), 'claimed_lower_bound': str(value)}


def build(doc, *, expected_source_sha256, deadline, mode='endpoints', reuse_keys=(), proposer=None):
    tick = clock(deadline)
    if mode not in ('endpoints', 'mccormick'): raise ValueError('unknown H2 source arm')
    bank, request, scopes = source_request(doc, expected_source_sha256, tick)
    required = [(tuple(d['pair']), d['competitor']) for d in request['duties']]
    reuse = set(reuse_keys)
    if len(reuse) != len(reuse_keys) or not reuse <= set(required): raise ValueError('reuse duty inventory')
    origins = []; scope_by_key = {}
    for key, scope in zip(required, scopes):
        kind = 'EMPTY_GUARD_DUAL' if scope['guard_conflict'] is not None else \
               'SOURCE_BOX_REUSE' if key in reuse and min(map(F, scope['facts'])) > THRESHOLD else 'PROPOSED'
        origins.append(kind); scope_by_key[key] = (scope, kind)
    stats = {'native_proposals': 0, 'common_proposals': 0}; errors = []
    def proposal(duty, weight, lp):
        tick(); scope, kind = scope_by_key[(tuple(duty['pair']), duty['competitor'])]
        if kind != 'PROPOSED':
            stats['common_proposals'] += 1
            return common_certificate(lp, scope['guard_conflict'] if kind == 'EMPTY_GUARD_DUAL' else None)
        if proposer is None: return None
        stats['native_proposals'] += 1
        try:
            result = proposer(lp, time_limit=max(0., deadline-time.monotonic()))
        except (RuntimeError, ValueError) as error:
            errors.append({'pair': duty['pair'], 'competitor': duty['competitor'],
                           'weight': None if weight is None else str(weight), 'error': str(error)})
            result = None
        tick(); return result
    if mode == 'endpoints':
        proof = endpoints(request, expected_request_sha256=identity(request), deadline=deadline, proposer=proposal)
    else:
        records = []
        for duty in request['duties']:
            tick(); lp = mc_lp(duty)
            scope, kind = scope_by_key[(tuple(duty['pair']), duty['competitor'])]
            if kind == 'SOURCE_BOX_REUSE':
                # A sufficient mixture fact, NOT a claim about this MC LP's bound.
                stats['common_proposals'] += 1
                cert = {'kind': 'SOURCE_BOX_FACT', 'lower_bound': str(min(map(F, scope['facts'])))}
            else:
                cert = proposal(duty, None, deepcopy(lp))
            tick()
            records.append({'pair': duty['pair'], 'competitor': duty['competitor'],
                            'lp_sha256': identity(lp), 'certificate': cert})
        proof = {'schema': 'SOURCE_MCCORMICK_PROOF_V1', 'request_sha256': identity(request), 'duties': records}
    tick()
    return {'schema': 'SOURCE_H2_CONTROL_V1', 'source_sha256': expected_source_sha256, 'mode': mode,
            'bank': bank, 'request': request, 'scopes': scopes, 'origins': origins,
            'reuse_requested': [[list(p), j] for p, j in sorted(reuse)],
            'proof': proof, 'proposal_stats': stats, 'proposal_errors': errors}
