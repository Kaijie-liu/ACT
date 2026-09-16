"""Recompute result coverage/count/sign structure, not the underlying LP proof."""
from moe_evidence.schema import route_pairs, TRUSTED, THRESHOLD
from act.back_end.solver.lp_certificate import rational


def check_result(result, meta):
    statement = meta['statement']; request = statement['request']; routes = statement['routes']
    pairs = route_pairs(request, routes)
    required = {(p, i) for p in pairs for i in range(len(request['properties']))}
    if (result['route_pairs'] != [list(p) for p in pairs] or result['required_obligations'] != len(required) or
            result['trusted_base'] != TRUSTED or result['deployed_float_SAFE'] is not False or
            result['production_gate_changed'] is not False):
        raise ValueError('result theorem/assumption/coverage mismatch')
    rows = result['obligations']
    if not routes['exact'] or routes['unresolved'] or not pairs:
        if result['status'] != 'UNKNOWN_ROUTE_COVERAGE' or rows or result['positive_obligations'] != 0:
            raise ValueError('unresolved routes promoted')
        return
    if len(rows) != len(required) or {(tuple(r['pair']), r['property_index']) for r in rows} != required:
        raise ValueError('result obligations missing/duplicated')
    positive = []; missing = 0; nonpositive = 0
    for row in rows:
        if type(row['property_index']) is not int: raise ValueError('invalid property index')
        state, bound = row['state'], row['lower_bound']
        if state == 'MISSING_EVIDENCE':
            if bound is not None: raise ValueError('bound attached to missing evidence')
            missing += 1
        elif state in ('CHECKED_REUSED_POSITIVE', 'CHECKED_RATIONAL_POSITIVE'):
            value = rational(bound)
            if value <= rational(THRESHOLD): raise ValueError('nonpositive accepted')
            positive.append(value)
        elif state == 'CHECKED_NONPOSITIVE_OR_BELOW_THRESHOLD':
            if rational(bound) > rational(THRESHOLD): raise ValueError('wrong bound state')
            nonpositive += 1
        else: raise ValueError('unsupported bound state')
    if (result['positive_obligations'], result['missing_obligations'], result['nonpositive_obligations']) != (len(positive), missing, nonpositive):
        raise ValueError('result counters differ from obligations')
    # generation_complete is tied to the manifest separately by the runtime.
    if result['status'] == 'CHECKED_CONDITIONAL':
        if len(positive) != len(required) or not required or rational(result['minimum_lower_bound']) != min(positive):
            raise ValueError('incomplete positive aggregate')
    elif result['status'] not in ('UNKNOWN_NONPOSITIVE', 'UNKNOWN_MISSING_EVIDENCE') or result['minimum_lower_bound'] is not None:
        raise ValueError('invalid incomplete result')
