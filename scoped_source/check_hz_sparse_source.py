"""Independent exact-source checks followed by sparse enclosure and block duties.

No producer/optimizer calls. Old source-global and block endpoint limits remain;
each lift also obeys the new row limit. This is not a portable deployment proof.
"""
from itertools import combinations

from source_enclosure.format import identity
from source_enclosure.check import check_box, check_affine
from scoped_source.graph import validate, operator, clock
from scoped_source.check_hz_row_enclosure import parse
from scoped_source.check_hz_lifted_source import owners, relu_reference, route_reference
from scoped_source.check_hz_templates import common_entry
from scoped_source.check_hz_sparse_rows import check as row_check
from scoped_source.hz_source_check import properties, gate_range, state_snapshot
from act.back_end.moe.check_block_endpoints import check_request

SCHEMA = 'CHECKED_SPARSE_HZ_BLOCK_SOURCE_V1'
CONTEXT_GUARD = 'ALL_TIE_LEGAL_SPARSE_BLOCK_TEMPLATES_V1'


def lift(record, tag, deadline):
    if set(record) != {'reference', 'enclosure', 'snapshot'}: raise ValueError('source lift schema')
    reference = record['reference']; h, _, _ = parse(reference)
    checked = row_check(reference, record['enclosure'], expected_reference_sha256=identity(reference),
                        expected_owner=tag, deadline=deadline)
    target = record['enclosure']['target']; parse(target, target=True)
    state_snapshot(target['state'], record['snapshot']); clock(deadline)()
    return target, {**checked, 'rows': sum(len(h[k]) for k in ('c','b','ub'))}


def network(doc, name, initial, trace, prefix, deadline):
    tick = clock(deadline); graph = next(g for g in doc['networks'] if g['name'] == name)
    if len(trace) != len(graph['layers']): raise ValueError('all source layers required')
    state = initial; shape = doc['center']['shape']; steps, lifts = [], []
    for i, (layer, step) in enumerate(zip(graph['layers'], trace)):
        tick(); tag = f'{prefix}/layer{i}'
        if (set(step) != {'index','kind','nominal','certificate','lift'}
                or type(step['index']) is not int or (step['index'],step['kind']) != (i,layer['kind'])):
            raise ValueError('source layer schema/order')
        ref = step['lift']['reference']; owners(state, ref, tag)
        shape, op, bias = operator(shape, layer)
        if layer['kind'] == 'Linear':
            result = check_affine(state['state'], ref['state'], op, bias, step['nominal'], step['certificate'], tag)
        elif layer['kind'] == 'ReLU':
            if step['nominal'] is not None: raise ValueError('unexpected ReLU nominal')
            result = relu_reference(state, ref, step['certificate'], tag)
        else:
            if step['nominal'] is not None or step['certificate'] is not None or ref != state:
                raise ValueError('flatten reference changed')
            result = {'status': 'CHECKED_IDENTITY_FLATTEN'}
        state, receipt = lift(step['lift'], tag, deadline); steps.append(result); lifts.append(receipt)
    return state, steps, lifts


def check(doc, package, *, expected_source_sha256, deadline):
    tick = clock(deadline); anchor = identity(package)
    r, lower, upper = validate(doc, expected_source_sha256, tick)
    if (set(package) != {'schema','source_sha256','input','router','common','templates','pairs','endpoint_request','proof'}
            or package['schema'] != SCHEMA or package['source_sha256'] != expected_source_sha256):
        raise ValueError('sparse block source package identity/schema')
    ref = package['input']['reference']; _, ci, bi = parse(ref)
    if ref['ownership'] != {'continuous': ['input']*len(ci), 'binary': []}: raise ValueError('input ownership')
    check_box(lower, upper, ref['state']); initial, first = lift(package['input'], 'input', deadline)
    router, steps, lifts = network(doc, 'router', initial, package['router'], 'router', deadline)
    lifts = [first]+lifts; base = package['common']; common_entry(initial, router, base)
    _, bc, bb = parse(base, target=True); e = r['experts']; saved = package['templates']
    if type(saved) is not list or [t['expert'] for t in saved] != list(range(e)):
        raise ValueError('all expert templates required in order')
    states, private = [], []
    for i, item in enumerate(saved):
        if set(item) != {'expert','trace'} or type(item['expert']) is not int: raise ValueError('template fields')
        state, ss, ll = network(doc, f'expert{i}', base, item['trace'], f'template/expert{i}', deadline)
        steps += ss; lifts += ll; states.append(state); _, tc, tb = parse(state, target=True)
        if tc[:len(bc)] != bc or tb[:len(bb)] != bb: raise ValueError('template shared namespace')
        private += tc[len(bc):]+tb[len(bb):]
    if len(set(private)) != len(private) or set(private)&set(bc+bb): raise ValueError('expert private factor alias')
    roster = [list(p) for p in combinations(range(e),2)]; req = package['endpoint_request']; props = properties(r)
    context = {'request': expected_source_sha256, 'domain': identity(r), 'guard': CONTEXT_GUARD}
    if ([p['pair'] for p in package['pairs']] != roster or req['context'] != context or req['properties'] != props
            or req['experts'] != e or req['classes'] != r['classes'] or [p['pair'] for p in req['pairs']] != roster):
        raise ValueError('all source pairs/properties binding')
    for p, endpoint in zip(package['pairs'], req['pairs']):
        tick()
        if set(p) != {'pair','entry','gate_evidence'}: raise ValueError('source pair fields')
        a, b = p['pair']; tag = f'pair{a}-{b}'
        route_reference(initial, router, p['entry']['reference'], (a,b), e)
        entry, receipt = lift(p['entry'], tag+'/guard', deadline); lifts.append(receipt)
        _, ec, eb = parse(entry, target=True)
        if ec != bc or eb != bb or entry['ownership'] != base['ownership']:
            raise ValueError('guard factor ownership/identity')
        for state, key in ((base,'common'),(entry,'entry'),(states[a],'a'),(states[b],'b')):
            state_snapshot(state['state'], endpoint['batch']['sources'][key])
        evidence = gate_range(router['state'], (a,b))
        if p['gate_evidence'] != evidence or endpoint['gate']['bounds'] != evidence['bounds']:
            raise ValueError('checked source gate binding')
    result = check_request(req, package['proof'], expected_request_sha256=identity(req), deadline=deadline)
    work = {k: sum(l['work'][k] for l in lifts) for k in lifts[0]['work']}
    output = {'status': 'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if result['positive']==result['required'] else result['status'],
              'source_sha256': expected_source_sha256, 'package_sha256': anchor,
              **{k: result[k] for k in ('required','positive','checked_endpoints','missing_endpoints','results')},
              'checked_source_steps': len(steps), 'checked_expert_templates': e,
              'checked_row_lifts': sum(l['rows'] for l in lifts), 'checked_sparse_lifts': len(lifts),
              'sparse_core_work': work,
              'nonzero_lift_compensations': sum(l['added_continuous'] for l in lifts),
              'source_lowering_checked': True, 'deployed_float_SAFE': False, 'hard_budget_supervision': False,
              'portable_distribution': False, 'real_model_claim': False,
              'remaining_trust': ['declaration_corresponds_to_intended_program','exact_checker_implementation']}
    if identity(doc) != expected_source_sha256 or identity(package) != anchor: raise ValueError('source/package changed')
    tick()
    return output
