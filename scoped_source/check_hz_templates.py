"""Check source templates and conditional pair views without propagation.

All templates share the checked router assignment, not independent input boxes.
The given finite declared graph is not a native floating-point execution proof.
"""
from copy import deepcopy
from itertools import combinations

from source_enclosure.format import unpack, pack, identity
from source_enclosure.check import check_box
from scoped_source.graph import validate, clock
from scoped_source.check_hz_binary64 import REFERENCE
from scoped_source.check_hz_row_enclosure import parse
from scoped_source.check_hz_lifted_source import lift, network, route_reference
from scoped_source.hz_source_check import properties, gate_range, state_snapshot
from act.back_end.moe.check_hz_endpoints import check_request

SCHEMA = 'CHECKED_HZ_EXPERT_TEMPLATES_V1'
VIEW = 'CHECKED_HZ_PAIR_TEMPLATE_VIEW_V1'
CONTEXT_GUARD = 'ALL_TIE_LEGAL_TEMPLATE_VIEWS_V1'


def common_entry(initial, router, common):
    x, xc, xb = parse(initial, target=True)
    r, rc, rb = parse(router, target=True)
    if rc[:len(xc)] != xc or rb[:len(xb)] != xb or x['frame_id'] != r['frame_id']:
        raise ValueError('common input/router identity')
    expected = deepcopy(r)
    for key in ('c', 'Gc', 'Gb'):
        expected[key] = deepcopy(x[key])
    wanted = {'schema': REFERENCE, 'state': pack(expected, rc, rb),
              'ownership': deepcopy(router['ownership'])}
    parse(common, target=True)
    if common != wanted:
        raise ValueError('common entry must restore input and retain full router')


def check_view(common, template, entry, record, *, pair, expert, deadline):
    """Identity column injection; shared guard rows precede all private rows."""
    tick = clock(deadline)
    anchors = [identity(v) for v in (common, template, entry, record)]
    h0, c0, b0 = parse(common, target=True)
    t, ct, bt = parse(template, target=True)
    g, cg, bg = parse(entry, target=True)
    if (set(record) != {'schema', 'pair', 'expert', 'common_sha256', 'template_sha256',
                        'entry_sha256', 'target'} or record['schema'] != VIEW
            or record['pair'] != pair or type(record['expert']) is not int
            or record['expert'] != expert or expert not in pair
            or [record[k] for k in ('common_sha256', 'template_sha256', 'entry_sha256')] != anchors[:3]):
        raise ValueError('template view source/pair/expert binding')
    if (cg != c0 or bg != b0 or ct[:len(c0)] != c0 or bt[:len(b0)] != b0
            or g['frame_id'] != h0['frame_id'] or t['frame_id'] != h0['frame_id']
            or entry['ownership'] != common['ownership']):
        raise ValueError('guard added factors or changed common identity')
    for kind, n in (('continuous', len(c0)), ('binary', len(b0))):
        if template['ownership'][kind][:n] != common['ownership'][kind]:
            raise ValueError('template shared ownership')
    for key in ('c', 'Gc', 'Gb', 'Ac', 'Ab', 'b'):
        if g[key] != h0[key]:
            raise ValueError('guard changed common output/equalities')
    expected = deepcopy(t)
    for ck, bk, rhs in (('Ac', 'Ab', 'b'), ('Auc', 'Aub', 'ub')):
        n = len(h0[rhs])
        if len(g[rhs]) < n:
            raise ValueError('guard lost common rows')
        for key in (ck, bk, rhs):
            if t[key][:n] != h0[key] or g[key][:n] != h0[key]:
                raise ValueError('template/guard common prefix')
            expected[key] = deepcopy(g[key]) + deepcopy(t[key][n:])
    wanted = {'schema': REFERENCE, 'state': pack(expected, ct, bt),
              'ownership': deepcopy(template['ownership'])}
    parse(record['target'], target=True)
    if record['target'] != wanted:
        raise ValueError('conditional view row/column/ownership composition')
    result = {'status': 'CHECKED_CONDITIONAL_VIEW_GIVEN_TEMPLATE_AND_GUARD',
              'target_sha256': identity(wanted), 'guard_rows': len(g['ub'])-len(h0['ub']),
              'private_continuous': len(ct)-len(c0), 'private_binary': len(bt)-len(b0)}
    if [identity(v) for v in (common, template, entry, record)] != anchors:
        raise ValueError('view input changed')
    tick()
    return result


def check(doc, package, *, expected_source_sha256, deadline):
    tick = clock(deadline); anchor = identity(package)
    r, lower, upper = validate(doc, expected_source_sha256, tick)
    if (set(package) != {'schema', 'source_sha256', 'input', 'router', 'common',
                        'templates', 'pairs', 'endpoint_request', 'proof'}
            or package['schema'] != SCHEMA or package['source_sha256'] != expected_source_sha256):
        raise ValueError('template source package identity/schema')
    ref = package['input']['reference']; _, ci, bi = parse(ref)
    if ref['ownership'] != {'continuous': ['input']*len(ci), 'binary': []}:
        raise ValueError('input ownership')
    check_box(lower, upper, ref['state'])
    initial, first = lift(package['input'], 'input', deadline)
    router, steps, lifts = network(doc, 'router', initial, package['router'], 'router', deadline)
    lifts = [first]+lifts
    base = package['common']; common_entry(initial, router, base)
    e = r['experts']; saved = package['templates']
    if type(saved) is not list or [t['expert'] for t in saved] != list(range(e)):
        raise ValueError('all expert templates required once and in order')
    states = []; private = []; _, bc, bb = parse(base, target=True)
    for i, item in enumerate(saved):
        tick()
        if set(item) != {'expert', 'trace'} or type(item['expert']) is not int:
            raise ValueError('template fields')
        state, ss, ll = network(doc, f'expert{i}', base, item['trace'], f'template/expert{i}', deadline)
        steps += ss; lifts += ll; states.append(state)
        _, tc, tb = parse(state, target=True)
        if tc[:len(bc)] != bc or tb[:len(bb)] != bb:
            raise ValueError('template shared namespace')
        private += tc[len(bc):]+tb[len(bb):]
    if len(set(private)) != len(private) or set(private)&set(bc+bb):
        raise ValueError('expert private factor alias')
    roster = [list(p) for p in combinations(range(e), 2)]
    req = package['endpoint_request']; props = properties(r)
    context = {'request': expected_source_sha256, 'domain': identity(r), 'guard': CONTEXT_GUARD}
    if ([p['pair'] for p in package['pairs']] != roster or req['context'] != context
            or req['properties'] != props or req['experts'] != e or req['classes'] != r['classes']
            or [p['pair'] for p in req['pairs']] != roster):
        raise ValueError('all source pairs/properties and endpoint binding')
    views = []
    for p, endpoint in zip(package['pairs'], req['pairs']):
        tick()
        if set(p) != {'pair', 'entry', 'views', 'gate_evidence'}:
            raise ValueError('pair view fields')
        a, b = p['pair']; tag = f'pair{a}-{b}'
        route_reference(initial, router, p['entry']['reference'], (a,b), e)
        entry, l = lift(p['entry'], tag+'/guard', deadline); lifts.append(l)
        if len(p['views']) != 2:
            raise ValueError('both expert views required')
        state_snapshot(entry['state'], endpoint['sources']['entry'])
        for i, key, record in zip((a,b), ('a','b'), p['views']):
            views.append(check_view(base, states[i], entry, record, pair=[a,b], expert=i, deadline=deadline))
            state_snapshot(record['target']['state'], endpoint['sources'][key])
        evidence = gate_range(router['state'], (a,b))
        if (p['gate_evidence'] != evidence or endpoint['gate']['bounds'] != evidence['bounds']
                or endpoint['relation_mode'] != 'shared_input'):
            raise ValueError('checked template router gate binding')
    result = check_request(req, package['proof'], expected_request_sha256=identity(req), deadline=deadline)
    output = {'status': 'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if result['positive']==result['required'] else result['status'],
              'source_sha256': expected_source_sha256, 'package_sha256': anchor,
              **{k: result[k] for k in ('required','positive','checked_endpoints','missing_endpoints','results')},
              'checked_source_steps': len(steps), 'checked_expert_templates': e, 'checked_views': len(views),
              'checked_row_lifts': sum(l['rows'] for l in lifts),
              'nonzero_lift_compensations': sum(l['added_continuous'] for l in lifts),
              'source_lowering_checked': True, 'deployed_float_SAFE': False, 'hard_budget_supervision': False,
              'portable_distribution': False, 'real_model_claim': False,
              'remaining_trust': ['declaration_corresponds_to_intended_program','exact_checker_implementation']}
    if identity(doc) != expected_source_sha256 or identity(package) != anchor:
        raise ValueError('source/package changed')
    tick(); return output
