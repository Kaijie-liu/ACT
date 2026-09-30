"""Untrusted property-dependency LP producer; synthetic H1 interface only.

No old HZ, old certificate, route exclusion, production entry or automatic
real-request runner. This uses direct ReLU triangles, not old binary-HZ rows.
"""
from fractions import Fraction as F
from itertools import combinations
import time
from scoped_source.graph import clock
from scoped_source.sparse_ir import index, row, lp_record
from source_enclosure.format import identity

THRESHOLD = F(1, 10_000_000)


def build(doc, *, expected_source_sha256, deadline, mode='dependency',
          reuse_keys=(), omit_nodes=(), proposer=None):
    start = time.monotonic(); tick = clock(deadline)
    if mode not in ('dependency', 'full'):
        raise ValueError('unknown construction mode')
    r, nodes, outputs = index(doc, expected_source_sha256, tick)
    bank = {}; bounds = {}; visiting = set()
    omitted = set(omit_nodes)
    if len(omitted) != len(omit_nodes) or any(k not in nodes or nodes[k]['kind'] == 'input' for k in omitted):
        raise ValueError('invalid omitted source row block')
    keys = [(tuple(p), j) for p in combinations(range(r['experts']), 2)
            for j in range(r['classes']) if j != r['label']]
    reuse = set(reuse_keys)
    if len(reuse) != len(reuse_keys) or not reuse <= set(keys):
        raise ValueError('unknown/duplicate reuse obligation')

    def block(key):
        tick()
        if key in bank:
            return
        if key in visiting:
            raise ValueError('cyclic source range')
        visiting.add(key); node = nodes[key]; kind = node['kind']; rows = []
        if kind == 'input':
            interval = node['bounds']; parents = []
        elif kind == 'affine':
            parents = list(node['terms'])
            for p in parents: block(p)
            lo = hi = node['bias']
            for p, coefficient in node['terms'].items():
                a, b = bounds[p]; lo += min(coefficient*a, coefficient*b); hi += max(coefficient*a, coefficient*b)
            interval = (lo, hi)
            rows = [row('eq', {key: F(1), **{p: -v for p, v in node['terms'].items()}}, node['bias'])]
        else:
            p = node['parent']; parents = [p]; block(p); lo, hi = bounds[p]
            interval = (max(F(0), lo), max(F(0), hi))
            if hi <= 0: rows = [row('eq', {key: F(1)}, F(0))]
            elif lo >= 0: rows = [row('eq', {key: F(1), p: F(-1)}, F(0))]
            else:
                slope = hi/(hi-lo)
                rows = [row('le', {key: F(-1)}, F(0)),
                        row('le', {p: F(1), key: F(-1)}, F(0)),
                        row('le', {key: F(1), p: -slope}, -slope*lo)]
        bounds[key] = interval
        bank[key] = {'parents': parents, 'bounds': list(map(str, interval)), 'rows': rows}
        visiting.remove(key)

    def expr(items):
        terms = {}; offset = F(0)
        # One exact source-level affine substitution; no whole-HZ projection.
        for key, scale in items:
            n = nodes[key]
            if n['kind'] == 'affine':
                offset += scale*n['bias']; source = n['terms']
            else: source = {key: F(1)}
            for p, v in source.items(): terms[p] = terms.get(p, F(0)) + scale*v
        return {k: v for k, v in terms.items() if v}, offset

    def interval(terms, offset):
        lo = hi = offset
        for p, v in terms.items():
            block(p); a, b = bounds[p]; lo += min(v*a, v*b); hi += max(v*a, v*b)
        return lo, hi

    def closure(seeds):
        active = set(); stack = list(seeds)
        while stack:
            key = stack.pop(); block(key)
            if key not in active:
                active.add(key); stack.extend(bank[key]['parents'])
        return [k for k in nodes if k in active]

    records = []; query_count = 0; query_seconds = 0.; materialized_rows = 0
    for pair, competitor in keys:
        tick(); a, b = pair; y = r['label']
        ma, oa = expr([(outputs[f'expert{a}'][y], F(1)), (outputs[f'expert{a}'][competitor], F(-1))])
        u, u0 = expr([(outputs[f'expert{b}'][y], F(1)), (outputs[f'expert{b}'][competitor], F(-1))])
        d = {k: ma.get(k, F(0))-u.get(k, F(0)) for k in ma.keys() | u.keys()}
        d = {k: v for k, v in d.items() if v}; d0 = oa-u0
        margin = F(r['margin']); facts = [interval(ma, oa)[0]-margin, interval(u, u0)[0]-margin]
        if (pair, competitor) in reuse and min(facts) > THRESHOLD:
            active = closure(ma.keys() | u.keys())
            records.append({'pair': list(pair), 'competitor': competitor, 'kind': 'reused',
                            'variables': active, 'facts': list(map(str, facts))})
            continue
        dlo, dhi = interval(d, d0)
        guards = []
        for selected in pair:
            for outside in range(r['experts']):
                if outside not in pair:
                    g, g0 = expr([(outputs['router'][outside], F(1)), (outputs['router'][selected], F(-1))])
                    guards.append(row('le', g, -g0))
        seeds = set(u) | set(d)
        for g in guards: seeds.update(g['terms'])
        # Eager control is pair-local: do not burden it with unrelated experts.
        eager = [k for k in nodes if k.startswith(('input/', 'router/', f'expert{a}/', f'expert{b}/'))]
        active = closure(eager if mode == 'full' else seeds)
        used = [k for k in active if k not in omitted]
        rows = [v for k in used for v in bank[k]['rows']] + guards
        # w = lambda * (d0 + d*z), lambda in [0,1].
        for s, t, sign in [(F(0), dlo, 1), (F(1), dhi, 1), (F(1), dlo, -1), (F(0), dhi, -1)]:
            terms = {k: sign*s*v for k, v in d.items() if s*v}
            terms['gate'] = sign*t; terms['product'] = F(-sign)
            rows.append(row('le', terms, sign*s*(t-d0)))
        variables = active + ['gate', 'product']
        local_bounds = {k: bounds[k] for k in active}
        local_bounds.update(gate=(F(0), F(1)), product=(min(F(0), dlo, dhi), max(F(0), dlo, dhi)))
        objective = {**u, 'product': F(1)}
        lp = lp_record(variables, local_bounds, rows, objective, u0-margin)
        materialized_rows += len(rows)
        record = {'pair': list(pair), 'competitor': competitor, 'kind': 'weighted',
                  'variables': variables, 'blocks': used, 'difference': [str(dlo), str(dhi)],
                  'lp_sha256': identity(lp), 'certificate': None, 'proposal_error': None}
        if proposer is not None:
            tick(); query_count += 1; before = time.monotonic()
            try: record['certificate'] = proposer(lp, time_limit=max(0., deadline-time.monotonic()))
            except (ValueError, RuntimeError) as exc: record['proposal_error'] = type(exc).__name__ + ': ' + str(exc)
            query_seconds += time.monotonic()-before; tick()
        records.append(record)
    tick()
    result = {'schema': 'SOURCE_SPARSE_TOP2_CONTROL_V1', 'source_sha256': expected_source_sha256,
              'mode': mode, 'omitted_blocks': sorted(omitted), 'bank': bank, 'obligations': records,
              'stats': {'source_nodes': len(nodes), 'constructed_blocks': len(bank),
                        'materialized_lp_rows': materialized_rows, 'output_queries': query_count,
                        'proposal_seconds': query_seconds, 'generation_seconds': time.monotonic()-start}}
    return result
