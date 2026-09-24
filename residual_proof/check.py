"""Independent source/route/output checking for both residual-evidence arms."""
from checked_route_frontier.check import check_network
from full_source.check_obligations import check as check_outputs
from scoped_source.check import check_guards, check_join, check_projection
from scoped_source.graph import clock

def route_check(doc, prefix, candidates, *, mode, invocation, expected_source_sha256, deadline):
    if mode == 'pairwise':
        from checked_route_frontier.check import check_frontier
        result = check_frontier(doc, prefix, candidates, expected_source_sha256=expected_source_sha256, deadline=deadline)
    elif mode == 'shared':
        from shared_route_residual.check import check
        result = check(doc, prefix, candidates, invocation=invocation, expected_source_sha256=expected_source_sha256, deadline=deadline)
    else:
        raise ValueError('unregistered residual mode')
    return result

def check(doc, bundle, *, invocation, expected_source_sha256, deadline):
    tick = clock(deadline)
    if (set(bundle) != {'schema', 'source_sha256', 'prefix', 'route_candidates', 'frontier', 'experts', 'pairs', 'lower_bound_certificates', 'route_mode', 'invocation'} or
            bundle['schema'] != 'RESIDUAL_PROOF_CONSTRUCTION_V1' or bundle['invocation'] != invocation or
            bundle['source_sha256'] != expected_source_sha256 or bundle['lower_bound_certificates'] != []):
        raise ValueError('construction binding; no borrowed output proofs')
    frontier = route_check(doc, bundle['prefix'], bundle['route_candidates'], mode=bundle['route_mode'], invocation=invocation,
                             expected_source_sha256=expected_source_sha256, deadline=deadline)
    if bundle['frontier'] != frontier:
        raise ValueError('changed/missing route coverage')
    needed = frontier['needed_experts']
    names = [f'expert{i}' for i in needed]
    if [t['name'] for t in bundle['experts']] != names:
        raise ValueError('required expert trace missing/duplicated')
    request = doc['request']
    root = bundle['prefix']['input']
    ends = {'router': bundle['prefix']['router']['steps'][-1]['state']}
    for i, trace in zip(needed, bundle['experts']):
        ends[trace['name']] = check_network(doc['networks'][i + 1], trace, root,
            request['center']['shape'], tick)
    retained = [p['pair'] for p in frontier['pairs'] if p['status'] == 'RETAINED']
    if [p['pair'] for p in bundle['pairs']] != retained:
        raise ValueError('retained pair inventory')
    outputs = 0
    for entry in bundle['pairs']:
        tick()
        if set(entry) != {'pair', 'joint', 'guarded', 'projected', 'base', 'obligations'}:
            raise ValueError('pair construction fields')
        pair = entry['pair']
        check_join(root, [ends['router']] + [ends[f'expert{i}'] for i in pair], entry['joint'])
        check_guards(entry['joint'], entry['guarded'], pair, request['experts'])
        check_projection(entry['guarded'], entry['projected'], request['experts'],
                         request['classes'], request['label'], request['margin'])
        checked = check_outputs(entry['projected'], entry['base'], entry['obligations'],
                                pair, request['classes'], request['label'])
        outputs += checked['obligations']
    tick()
    return {'status': 'CHECKED_ROUTE_COVER_AND_RETAINED_OUTPUT_CONSTRUCTIONS',
            'source_sha256': expected_source_sha256, 'frontier': frontier,
            'expert_traces_checked': len(needed), 'output_obligations': outputs,
            'original_output_obligations': frontier['total_pairs'] * (request['classes'] - 1),
            'excluded_output_obligations': frontier['excluded_pairs'] * (request['classes'] - 1),
            'lower_bounds_checked': 0, 'complete_output_positive_proof': False,
            'route_changing_established': False, 'native_float_proof': False,
            'trusted': ['declared graph/program correspondence', 'stored-center preprocessing',
                        'standard-library checker implementation/runtime']}
