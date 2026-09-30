"""Independent declared-source H2 checker; no producer or numerical imports.

Rebuild source ranges, all pair-local constraints, properties and gate signs.
All obligations must close on that source. Native program correspondence and
deployed floating point remain outside this theorem and checker interface.
"""
from fractions import Fraction as F
from itertools import combinations

from scoped_source.graph import clock
from scoped_source.sparse_ir import index, row, lp_record
from scoped_source.endpoint_check import check as check_endpoints, THRESHOLD
from scoped_source.sparse_check import check_bound
from source_enclosure.format import identity
from upstream_source.checker import csr


def reconstruct(doc, expected, tick):
    request, definitions, outputs = index(doc, expected, tick)
    proven = {}; blocks = {}
    for name, definition in definitions.items():
        tick(); kind = definition['kind']; constraints = []
        if kind == 'input':
            lower, upper = definition['bounds']
        elif kind == 'affine':
            lower = definition['bias']; upper = definition['bias']; terms = {name: F(1)}
            for parent, coefficient in definition['terms'].items():
                lo, hi = proven[parent]
                if coefficient >= 0: lower += coefficient*lo; upper += coefficient*hi
                else: lower += coefficient*hi; upper += coefficient*lo
                terms[parent] = -coefficient
            constraints = [row('eq', terms, definition['bias'])]
        elif kind == 'relu':
            parent = definition['parent']; lo, hi = proven[parent]
            lower, upper = max(F(0), lo), max(F(0), hi)
            if hi <= 0: constraints = [row('eq', {name: F(1)}, F(0))]
            elif lo >= 0: constraints = [row('eq', {parent: F(-1), name: F(1)}, F(0))]
            else:
                constraints = [row('le', {name: F(-1)}, F(0)),
                               row('le', {name: F(-1), parent: F(1)}, F(0)),
                               row('le', {name: F(1), parent: hi/(lo-hi)}, lo*hi/(lo-hi))]
        else: raise ValueError('unknown source rule')
        proven[name] = (lower, upper)
        blocks[name] = {'bounds': [str(lower), str(upper)], 'rows': constraints}
    duties = []; scopes = []
    for pair in combinations(range(request['experts']), 2):
        left, right = (outputs['router'][j] for j in pair)
        margin_lo = proven[left][0]-proven[right][1]
        margin_hi = proven[left][1]-proven[right][0]
        # Only independent, previously reconstructed forward ranges are used.
        if margin_lo == 0 and margin_hi == 0: gate = ['1/2', '1/2']
        elif margin_hi <= 0: gate = ['0', '1/2']
        elif margin_lo >= 0: gate = ['1/2', '1']
        else: gate = ['0', '1']
        families = {'input', 'router'} | {f'expert{j}' for j in pair}
        variables = [v for v in definitions if v.split('/')[0] in families]
        constraints = [r for v in variables for r in blocks[v]['rows']]
        first_conflict = None
        for selected in pair:
            for outside in range(request['experts']):
                if outside in pair: continue
                sel, out = outputs['router'][selected], outputs['router'][outside]
                index_ = len([c for c in constraints if c['sense'] == 'le'])
                constraints.append(row('le', {sel: F(-1), out: F(1)}, F(0)))
                if first_conflict is None and proven[out][0] > proven[sel][1]:
                    first_conflict = {'selected': selected, 'outside': outside,
                                      'inequality_index': index_, 'gap': str(proven[out][0]-proven[sel][1])}
        base = lp_record(variables, {v: proven[v] for v in variables}, constraints, {}, F(0))
        for competitor in range(request['classes']):
            if competitor == request['label']: continue
            forms = []; facts = []
            for expert in pair:
                endpoint = outputs[f'expert{expert}']; label_node = endpoint[request['label']]; other = endpoint[competitor]
                coefficient = {label_node: F(1), other: F(-1)}
                forms.append({'c': [str(coefficient.get(v, F(0))) for v in variables],
                              'offset': str(-F(request['margin']))})
                facts.append(str(proven[label_node][0]-proven[other][1]-F(request['margin'])))
            duties.append({'pair': list(pair), 'competitor': competitor, 'variables': variables[:],
                           'base': base, 'a': forms[0], 'b': forms[1], 'gate': gate[:]})
            scopes.append({'pair': list(pair), 'competitor': competitor,
                           'router_margin': [str(margin_lo), str(margin_hi)], 'facts': facts,
                           'fact_domain': 'GLOBAL_INPUT_BOX', 'guard_conflict': first_conflict})
    return blocks, {'schema': 'GIVEN_LP_TOP2_ENDPOINT_REQUEST_V1',
                    'experts': request['experts'], 'classes': request['classes'],
                    'label': request['label'], 'duties': duties}, scopes


def reconstruct_mc(duty):
    variables = duty['variables']; base = duty['base']
    delta = [F(a)-F(b) for a, b in zip(duty['a']['c'], duty['b']['c'])]
    offset = F(duty['a']['offset'])-F(duty['b']['offset'])
    lower = upper = offset
    for value, lo, hi in zip(delta, base['lower'], base['upper']):
        if value >= 0: lower += value*F(lo); upper += value*F(hi)
        else: lower += value*F(hi); upper += value*F(lo)
    constraints = []
    for matrix, rhs, sense in [('A', 'b', 'le'), ('E', 'h', 'eq')]:
        for coefficients, value in zip(csr(base[matrix]), base[rhs]):
            constraints.append(row(sense, {variables[j]: a for j, a in coefficients.items()}, F(value)))
    lo, hi = map(F, duty['gate'])
    def product_row(scale, gate_coefficient, product_coefficient, rhs):
        terms = {v: scale*a for v, a in zip(variables, delta)}
        terms['gate'] = gate_coefficient; terms['product'] = product_coefficient
        return row('le', terms, rhs)
    constraints.extend([
        product_row(lo, lower, F(-1), lo*(lower-offset)),
        product_row(hi, upper, F(-1), hi*(upper-offset)),
        product_row(-hi, -lower, F(1), hi*(offset-lower)),
        product_row(-lo, -upper, F(1), lo*(offset-upper))])
    corners = [lo*lower, lo*upper, hi*lower, hi*upper]
    bounds = {v: (F(l), F(h)) for v, l, h in zip(variables, base['lower'], base['upper'])}
    bounds['gate'] = (lo, hi); bounds['product'] = (min(corners), max(corners))
    objective = {v: F(c) for v, c in zip(variables, duty['b']['c'])}; objective['product'] = F(1)
    return lp_record(variables+['gate', 'product'], bounds, constraints, objective, F(duty['b']['offset']))


def check(doc, package, *, expected_source_sha256, expected_mode, deadline):
    tick = clock(deadline)
    if (set(package) != {'schema', 'source_sha256', 'mode', 'bank', 'request', 'scopes', 'origins',
                        'reuse_requested', 'proof', 'proposal_stats', 'proposal_errors'} or
            package['schema'] != 'SOURCE_H2_CONTROL_V1' or package['source_sha256'] != expected_source_sha256 or
            package['mode'] != expected_mode or expected_mode not in ('endpoints', 'mccormick')):
        raise ValueError('source package/arm identity')
    bank, request, scopes = reconstruct(doc, expected_source_sha256, tick)
    if package['bank'] != bank or package['request'] != request or package['scopes'] != scopes:
        raise ValueError('source, property, guard, factor or gate reconstruction mismatch')
    keys = [(tuple(d['pair']), d['competitor']) for d in request['duties']]
    reuse = []
    for record in package['reuse_requested']:
        if (len(record) != 2 or len(record[0]) != 2 or type(record[1]) is not int or
                any(type(v) is not int for v in record[0])): raise ValueError('reuse key syntax')
        reuse.append((tuple(record[0]), record[1]))
    if reuse != sorted(set(reuse)) or not set(reuse) <= set(keys): raise ValueError('reuse keys')
    origins = []
    for key, scope in zip(keys, scopes):
        origins.append('EMPTY_GUARD_DUAL' if scope['guard_conflict'] is not None else
                       'SOURCE_BOX_REUSE' if key in reuse and min(map(F, scope['facts'])) > THRESHOLD else 'PROPOSED')
    if package['origins'] != origins: raise ValueError('common fact scope/eligibility')
    proof = package['proof']; results = []; checked = 0; missing = 0
    if expected_mode == 'endpoints':
        result = check_endpoints(request, proof, expected_request_sha256=identity(request), deadline=deadline)
        results = result['duties']; checked = result['endpoint_bounds_checked']; missing = result['missing']
    else:
        if (set(proof) != {'schema', 'request_sha256', 'duties'} or
                proof['schema'] != 'SOURCE_MCCORMICK_PROOF_V1' or proof['request_sha256'] != identity(request) or
                len(proof['duties']) != len(keys)):
            raise ValueError('MC proof complete request identity')
        for duty, scope, origin, record in zip(request['duties'], scopes, origins, proof['duties']):
            tick(); lp = reconstruct_mc(duty)
            if (set(record) != {'pair', 'competitor', 'lp_sha256', 'certificate'} or
                    record['pair'] != duty['pair'] or record['competitor'] != duty['competitor'] or
                    type(record['competitor']) is not int or any(type(v) is not int for v in record['pair']) or
                    record['lp_sha256'] != identity(lp)):
                raise ValueError('MC duty/LP identity')
            certificate = record['certificate']
            if origin == 'SOURCE_BOX_REUSE':
                lower = min(map(F, scope['facts']))
                if certificate != {'kind': 'SOURCE_BOX_FACT', 'lower_bound': str(lower)}:
                    raise ValueError('MC source fact receipt')
            elif certificate is None:
                lower = None; missing += 1
            else:
                lower = F(check_bound(lp, certificate)['checked_lower_bound']); checked += 1
            tick()
            results.append({'pair': duty['pair'], 'competitor': duty['competitor'],
                            'lower_bound': None if lower is None else str(lower),
                            'positive': lower is not None and lower > THRESHOLD})
    tick(); positive = sum(d['positive'] for d in results)
    return {'status': 'CHECKED_DECLARED_SOURCE_POSITIVE' if positive == len(keys) else
            'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
            'source_sha256': expected_source_sha256, 'request_sha256': identity(request),
            'mode': expected_mode, 'required': len(keys), 'positive': positive,
            'lp_bounds_checked': checked, 'missing': missing, 'duties': results,
            'source_blocks_checked': len(bank), 'scopes': scopes, 'origins': origins,
            'declared_real_graph_source_checked': True,
            'routing_coverage': 'ALL_UNORDERED_TOP2_PAIRS_NO_DUTIES_DROPPED',
            'hard_budget_supervision': False, 'deployed_float_SAFE': False,
            'remaining_trust': ['declared_graph_correspondence_to_native_program',
                                'source_tensor_parser_and_exact_checker_implementation']}
