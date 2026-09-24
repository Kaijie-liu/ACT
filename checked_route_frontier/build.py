"""Untrusted proposer and lazy builder. No native solver or historical proof use.

The only optimization is discharging empty pair domains using checked strict
router margins, before building experts. Retained output algebra is unchanged.
"""
from fractions import Fraction as F

from full_source.lift import affine
from full_source.obligations import build as output_lp
from scoped_source.build import guards, join, project
from scoped_source.graph import clock, operator, validate
from source_enclosure.format import identity, sparse, unpack
from source_enclosure.produce import box, relu


def network(root, graph, shape, tick):
    state = root
    steps = []
    for layer in graph['layers']:
        tick()
        before = shape
        shape, weights, bias = operator(shape, layer)
        old = state
        tag = graph['name'] + '/layer/' + str(layer['index'])
        if layer['kind'] == 'Flatten':
            proof = None
        elif layer['kind'] == 'ReLU':
            state, proof = relu(old, tag)
        else:
            state, proof = affine(old, weights, bias, tag)
        steps.append({'index': layer['index'], 'kind': layer['kind'],
                      'input_shape': before, 'output_shape': shape,
                      'source_sha256': identity(old), 'state': state, 'proof': proof})
    tick()
    return {'name': graph['name'], 'steps': steps}


def prefix(doc, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    request, lo, hi = validate(doc, expected_source_sha256, tick)
    root = box(lo, hi)
    router = network(root, doc['networks'][0], request['center']['shape'], tick)
    tick()
    return {'schema': 'CHECKED_ROUTE_PREFIX_V1', 'source_sha256': expected_source_sha256,
            'input': root, 'router': router}


def margin_lp(state, higher, lower):
    h, continuous, binary = unpack(state)
    nc, nb = len(continuous), len(binary)
    n = nc + nb
    objective = [F(0)] * n
    for key, shift in [('Gc', 0), ('Gb', nc)]:
        for index, sign in [(higher, 1), (lower, -1)]:
            for col, value in h[key][index].items():
                objective[col + shift] += sign * value
    result = {'matrix_format': 'csr_v1', 'c': list(map(str, objective)),
              'offset': str(h['c'][higher] - h['c'][lower]),
              'lower': [-1] * n, 'upper': [1] * n}
    for out, ac, ab, rhs, dest in [('A', 'Auc', 'Aub', 'ub', 'b'), ('E', 'Ac', 'Ab', 'b', 'h')]:
        rows = [{**a, **{j + nc: v for j, v in b.items()}}
                for a, b in zip(h[ac], h[ab])]
        result[out] = sparse(rows, n)
        result[dest] = list(map(str, h[rhs]))
    return result


def propose_final_affine(doc, router_prefix, *, deadline):
    """One fixed algebraic dual per ordered margin; NO LP optimization.

Pull back only the final affine lift, leaving every earlier equality unused.
This exposes shared penultimate factors. A poor bound retains the pair. The
independent checker validates the dual against the complete router LP, not
this proposal algorithm or its naming convention.
"""
    tick = clock(deadline)
    state = router_prefix['router']['steps'][-1]['state']
    h, ci, _ = unpack(state)
    final = doc['networks'][0]['layers'][-1]
    tag = 'router/layer/' + str(final['index']) + '/value/'
    source_hash, router_hash = identity(doc), identity(state)
    candidates = []
    for higher in range(doc['request']['experts']):
        for lower in range(doc['request']['experts']):
            if higher == lower:
                continue
            tick()
            lp = margin_lp(state, higher, lower)
            residual = list(map(F, lp['c']))
            dual = [F(0)] * len(h['b'])
            if final['kind'] == 'Linear':
                # Only the two final output factors are eliminated. No sparse
                # basis search, cross-prime arithmetic, or iterative tuning.
                for output in (higher, lower):
                    name = tag + str(output)
                    if name not in ci:
                        continue  # constant output
                    col = ci.index(name)
                    found = [i for i, row in enumerate(h['Ac']) if row.get(col)]
                    if len(found) != 1:
                        raise ValueError('final affine factor definition')
                    index = found[0]
                    value = residual[col] / h['Ac'][index][col]
                    dual[index] += value
                    for j, v in h['Ac'][index].items():
                        residual[j] -= value * v
                    for j, v in h['Ab'][index].items():
                        residual[len(ci) + j] -= value * v
            candidates.append({'schema': 'ROUTER_MARGIN_CANDIDATE_V1',
                'source_sha256': source_hash, 'router_sha256': router_hash,
                'higher': higher, 'lower': lower,
                'certificate': {'lp_sha256': identity(lp),
                    'inequality_dual': ['0'] * len(lp['b']),
                    'equality_dual': list(map(str, dual))}})
    tick()
    return candidates


def finish(doc, router_prefix, candidates, *, expected_source_sha256, deadline):
    # Fail closed before omitting any expert. Final checking repeats this in a
    # fresh process; this invocation is not accepted as proof by itself.
    from checked_route_frontier.check import check_frontier
    tick = clock(deadline)
    frontier = check_frontier(doc, router_prefix, candidates,
        expected_source_sha256=expected_source_sha256, deadline=deadline)
    request = doc['request']
    root = router_prefix['input']
    ends = {'router': router_prefix['router']['steps'][-1]['state']}
    experts = []
    for i in frontier['needed_experts']:
        trace = network(root, doc['networks'][i + 1], request['center']['shape'], tick)
        experts.append(trace)
        ends[trace['name']] = trace['steps'][-1]['state']
    pairs = []
    for decision in frontier['pairs']:
        if decision['status'] != 'RETAINED':
            continue
        tick()
        pair = decision['pair']
        merged = join(root, [ends['router']] + [ends[f'expert{i}'] for i in pair])
        guarded = guards(merged, pair, request['experts'])
        projected = project(guarded, request['experts'], request['classes'],
                            request['label'], request['margin'])
        base, rows = output_lp(projected, pair, request['classes'], request['label'])
        pairs.append({'pair': pair, 'joint': merged, 'guarded': guarded,
                      'projected': projected, 'base': base, 'obligations': rows})
    tick()
    return {'schema': 'CHECKED_ROUTE_CONSTRUCTION_V1',
            'source_sha256': expected_source_sha256, 'prefix': router_prefix,
            'route_candidates': candidates, 'frontier': frontier,
            'experts': experts, 'pairs': pairs, 'lower_bound_certificates': []}
