"""Untrusted lazy construction; mathematical primitives unchanged."""
from checked_route_frontier.build import network
from full_source.obligations import build as output_lp
from scoped_source.build import guards, join, project
from scoped_source.graph import clock

def finish(doc, router_prefix, candidates, *, mode, invocation, expected_source_sha256, deadline):
    # Fail closed before omitting any expert. Final checking repeats this in a
    # fresh process; this invocation is not accepted as proof by itself.
    from residual_proof.check import route_check
    tick = clock(deadline)
    frontier = route_check(doc, router_prefix, candidates, mode=mode, invocation=invocation,
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
    return {'schema': 'RESIDUAL_PROOF_CONSTRUCTION_V1', 'route_mode': mode, 'invocation': invocation,
            'source_sha256': expected_source_sha256, 'prefix': router_prefix,
            'route_candidates': candidates, 'frontier': frontier,
            'experts': experts, 'pairs': pairs, 'lower_bound_certificates': []}
