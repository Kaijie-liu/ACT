"""Independent source-rule and complete-obligation checker for H1 controls.

No producer or solver imports. The shared IR only parses source syntax and
serializes named rows; affine/ReLU/range/guard/product rules are rebuilt here.
Guarantee: declared real Linear/ReLU source, not the captured native program.
"""
from fractions import Fraction as F
from itertools import combinations
from scoped_source.graph import clock
from scoped_source.sparse_ir import index, row, lp_record
from source_enclosure.format import identity
from upstream_source.checker import csr, rational


def check_bound(lp, certificate):
    """Small stdlib dual kernel; no ACT package initializer or solver import."""
    if lp['matrix_format'] != 'csr_v1' or certificate['lp_sha256'] != identity(lp):
        raise ValueError('LP/certificate identity')
    c, low, high = ([rational(v) for v in lp[k]] for k in ('c', 'lower', 'upper'))
    n = len(c)
    if not n or len(low) != n or len(high) != n or any(a > b for a, b in zip(low, high)):
        raise ValueError('finite variable bounds')
    residual = c[:]; value = rational(lp['offset'])
    for matrix, rhs, dual, signed in [('A', 'b', 'inequality_dual', True),
                                      ('E', 'h', 'equality_dual', False)]:
        rows = csr(lp[matrix], [len(lp[rhs]), n])
        multipliers = [rational(v) for v in certificate[dual]]
        if len(multipliers) != len(rows) or signed and any(v > 0 for v in multipliers):
            raise ValueError('dual shape/sign')
        for coefficients, right, multiplier in zip(rows, lp[rhs], multipliers):
            value += multiplier*rational(right)
            for j, coefficient in coefficients.items(): residual[j] -= multiplier*coefficient
    value += sum((min(r*a, r*b) for r, a, b in zip(residual, low, high)), F(0))
    if rational(certificate['claimed_lower_bound']) > value:
        raise ValueError('claim exceeds exact residual-compensated bound')
    return {'checked_lower_bound': str(value)}


def check(doc, package, *, expected_source_sha256, deadline):
    tick = clock(deadline)
    request, source, endpoints = index(doc, expected_source_sha256, tick)
    if (package['schema'] != 'SOURCE_SPARSE_TOP2_CONTROL_V1' or
            package['source_sha256'] != expected_source_sha256 or
            package['mode'] not in ('dependency', 'full')):
        raise ValueError('package/source/mode binding')
    dropped = package['omitted_blocks']
    if (dropped != sorted(set(dropped)) or
            any(k not in source or source[k]['kind'] == 'input' for k in dropped)):
        raise ValueError('invalid discarded constraint')
    required = [(p, j) for p in combinations(range(request['experts']), 2)
                for j in range(request['classes']) if j != request['label']]
    records = package['obligations']
    if (any(type(r['competitor']) is not int or any(type(v) is not int for v in r['pair']) for r in records) or
            [(tuple(r['pair']), r['competitor']) for r in records] != required):
        raise ValueError('missing/duplicate/substituted route or property')
    memo = {}; needed_bank = set(); visiting = set()

    def verify_block(name):
        tick(); needed_bank.add(name)
        if name in memo: return memo[name]
        if name in visiting: raise ValueError('cyclic range proof')
        visiting.add(name); n = source[name]; kind = n['kind']
        if kind == 'input':
            low, high = n['bounds']; parents = []; constraints = []
        elif kind == 'affine':
            parents = list(n['terms']); low = n['bias']; high = n['bias']
            coefficients = {name: F(1)}
            for parent in parents:
                weight = n['terms'][parent]; a, b = verify_block(parent)
                if weight >= 0: low += weight*a; high += weight*b
                else: low += weight*b; high += weight*a
                coefficients[parent] = -weight
            constraints = [row('eq', coefficients, n['bias'])]
        elif kind == 'relu':
            parent = n['parent']; parents = [parent]; a, b = verify_block(parent)
            low, high = max(F(0), a), max(F(0), b)
            if b <= 0: constraints = [row('eq', {name: F(1)}, F(0))]
            elif a >= 0: constraints = [row('eq', {name: F(1), parent: F(-1)}, F(0))]
            else:
                constraints = [row('le', {name: F(-1)}, F(0)),
                               row('le', {name: F(-1), parent: F(1)}, F(0)),
                               row('le', {name: F(1), parent: b/(a-b)}, a*b/(a-b))]
        else: raise ValueError('unsupported range derivation')
        expected = {'parents': parents, 'bounds': [str(low), str(high)], 'rows': constraints}
        if package['bank'].get(name) != expected:
            raise ValueError('source range/row/dependency mismatch: ' + name)
        memo[name] = (low, high); visiting.remove(name)
        return low, high

    def linear_form(terms):
        result = {}; constant = F(0)
        for name, multiplier in terms:
            definition = source[name]
            if definition['kind'] == 'affine':
                constant += multiplier*definition['bias']
                for dep, weight in definition['terms'].items():
                    result[dep] = result.get(dep, F(0)) + multiplier*weight
            else: result[name] = result.get(name, F(0)) + multiplier
        return {k: v for k, v in result.items() if v}, constant

    def enclosure(coefficients, constant):
        low = high = constant
        for name, coefficient in coefficients.items():
            a, b = verify_block(name)
            if coefficient < 0: a, b = b, a
            low += coefficient*a; high += coefficient*b
        return low, high

    def source_closure(seeds):
        found = set(); queue = list(seeds)
        while queue:
            name = queue.pop()
            if name in found: continue
            found.add(name); verify_block(name)
            n = source[name]
            if n['kind'] == 'affine': queue.extend(n['terms'])
            elif n['kind'] == 'relu': queue.append(n['parent'])
        return [name for name in source if name in found]

    results = []; checked_lps = 0; reused = 0; duty_nodes = []; lp_rows = 0
    for record in records:
        tick(); pair = record['pair']; competitor = record['competitor']; label = request['label']
        expert_forms = []
        for expert in pair:
            out = endpoints[f'expert{expert}']
            expert_forms.append(linear_form([(out[label], F(1)), (out[competitor], F(-1))]))
        (left, lc), (right, rc) = expert_forms
        source_facts = [enclosure(c, d)[0]-F(request['margin']) for c, d in expert_forms]
        if record['kind'] == 'reused':
            active = source_closure(left.keys() | right.keys()); duty_nodes.append(set(active))
            if (set(record) != {'pair', 'competitor', 'kind', 'variables', 'facts'} or
                    record['variables'] != active or record['facts'] != list(map(str, source_facts)) or
                    min(source_facts) <= F(1, 10_000_000)):
                raise ValueError('invalid scoped source fact reuse')
            lower = min(source_facts); state = 'CHECKED_SOURCE_INTERVAL'; reused += 1
        elif record['kind'] == 'weighted':
            delta = {name: left.get(name, F(0))-right.get(name, F(0)) for name in left.keys() | right.keys()}
            delta = {k: v for k, v in delta.items() if v}; dc = lc-rc
            dl, du = enclosure(delta, dc)
            if record['difference'] != [str(dl), str(du)]: raise ValueError('difference range binding')
            guard_rows = []; seeds = set(delta) | set(right)
            for selected in pair:
                for other in range(request['experts']):
                    if other in pair: continue
                    c, d = linear_form([(endpoints['router'][other], F(1)),
                                        (endpoints['router'][selected], F(-1))])
                    guard_rows.append(row('le', c, -d)); seeds.update(c)
            eager = [name for name in source if name.split('/')[0] in
                     ('input', 'router', f'expert{pair[0]}', f'expert{pair[1]}')]
            active = source_closure(eager if package['mode'] == 'full' else seeds)
            duty_nodes.append(set(active))
            kept = [name for name in active if name not in dropped]
            if record['variables'] != active+['gate', 'product'] or record['blocks'] != kept:
                raise ValueError('variable identity or retained block inventory')
            constraints = [line for name in kept for line in package['bank'][name]['rows']]
            constraints += guard_rows
            # Independently spell out the four universal-gate McCormick rows.
            constraints += [row('le', {'gate': dl, 'product': F(-1)}, F(0)),
                            row('le', {**delta, 'gate': du, 'product': F(-1)}, du-dc),
                            row('le', {**{k: -v for k, v in delta.items()}, 'gate': -dl, 'product': F(1)}, dc-dl),
                            row('le', {'gate': -du, 'product': F(1)}, F(0))]
            variable_bounds = {name: memo[name] for name in active}
            variable_bounds['gate'] = (F(0), F(1))
            variable_bounds['product'] = (min(F(0), dl, du), max(F(0), dl, du))
            lp = lp_record(record['variables'], variable_bounds, constraints,
                           {**right, 'product': F(1)}, rc-F(request['margin']))
            lp_rows += len(constraints)
            if record['lp_sha256'] != identity(lp): raise ValueError('new LP identity')
            if record['certificate'] is None:
                lower = None; state = 'MISSING_EVIDENCE'
            else:
                checked = check_bound(lp, record['certificate']); tick(); checked_lps += 1
                lower = F(checked['checked_lower_bound']); state = 'CHECKED_LP'
        else: raise ValueError('unknown proof kind')
        results.append({'pair': pair, 'competitor': competitor, 'kind': state,
                        'lower_bound': None if lower is None else str(lower),
                        'positive': lower is not None and lower > F(1, 10_000_000)})
    if set(package['bank']) != needed_bank: raise ValueError('extra or missing source bank')
    tick(); positives = sum(r['positive'] for r in results)
    return {'status': 'CHECKED_DECLARED_SOURCE_POSITIVE' if positives == len(required)
            else 'UNKNOWN_MISSING_EVIDENCE' if any(r['lower_bound'] is None for r in results)
            else 'UNKNOWN_NONPOSITIVE',
            'source_sha256': expected_source_sha256, 'required': len(required), 'positive': positives,
            'lp_bounds_checked': checked_lps, 'reused': reused, 'obligations': results,
            'lp_rows_reconstructed': lp_rows,
            'source_nodes': len(source), 'source_blocks_checked': len(needed_bank),
            'duty_source_nodes': list(map(len, duty_nodes)),
            'duty_union_nodes': len(set().union(*duty_nodes)),
            'routing_coverage': 'ALL_UNORDERED_TOP2_PAIRS_NO_EXCLUSIONS',
            'route_change_witness_checked': False, 'deployed_float_SAFE': False,
            'remaining_trust': ['declared_graph_correspondence_to_native_program',
                                'source_tensor_parser_and_checker_implementation']}
