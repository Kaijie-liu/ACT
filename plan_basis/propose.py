"""Original-coordinate assembly copied from frozen sparse engine; new arithmetic only.

No native solve, feasibility acceptance, scaling of original LP or production hook.
"""
from fractions import Fraction as F
import time
from lp_sandwich.check import identity, validate_statement
from sparse_basis.engine import dimensions, key, COORDINATES, parsed_rows, row_identity, Limit, Singular
from plan_basis.engine import POLICY, Budget, statistics, eliminate, Unresolved


def propose(lp, statement, candidate, hint, expected_statement, expected_hint, *, deadline, reuse=True):
    begin = time.monotonic()
    b = Budget(deadline)
    b.where('assembly','validate')
    stats = statistics()
    status, error, bundle, residuals, system_hash = 'ERROR', None, None, None, None
    try:
        b.tick()
        n, ne, na = dimensions(lp)
        validate_statement(statement, lp, expected_statement, b.tick)
        for name in ('c', 'lower', 'upper'):
            for v in lp[name]:
                b.value(v)
        b.value(lp['offset'])
        if any(b.value(l) > b.value(u) for l, u in zip(lp['lower'], lp['upper'])):
            raise ValueError('inverted bounds')
        if (set(candidate) != {'lp_sha256', 'statement_sha256', 'x'} or
                candidate['lp_sha256'] != identity(lp) or candidate['statement_sha256'] != expected_statement or len(candidate['x']) != n):
            raise ValueError('candidate identity')
        point = [b.value(x) for x in candidate['x']]
        fields = {'schema', 'coordinates', 'lp_sha256', 'statement_sha256', 'candidate_sha256', 'rows', 'basic_columns', 'anchors'}
        if (set(hint) != fields or hint['schema'] != 'LARGE_ORIGINAL_BASIS_V1' or hint['coordinates'] != COORDINATES or
                identity(hint) != expected_hint or hint['lp_sha256'] != identity(lp) or
                hint['statement_sha256'] != expected_statement or hint['candidate_sha256'] != identity(candidate)):
            raise ValueError('basis binding/coordinates')
        basic = [key(c, n, ne, na) for c in hint['basic_columns']]
        if len(basic) != ne + na or len(set(basic)) != len(basic):
            raise ValueError('square basis count/uniqueness')
        lookup = {k: j for j, k in enumerate(basic)}
        anchors = {}
        for item in hint['anchors']:
            if set(item) != {'column', 'at'}:
                raise ValueError('anchor schema')
            k = key(item['column'], n, ne, na)
            if k in anchors or k in lookup:
                raise ValueError('duplicate/overlapping coordinate')
            at = item['at']
            if k[0] != 'x':
                if at != 'zero':
                    raise ValueError('nonbasic row residual must be anchored zero')
                value = F(0)
            else:
                if at not in ('lower', 'upper', 'candidate'):
                    raise ValueError('unbound anchor')
                value = point[k[1]] if at == 'candidate' else b.value(lp[at][k[1]])
            anchors[k] = value
        allcols = {('x', i) for i in range(n)} | {('E_residual', i) for i in range(ne)} | {('A_slack', i) for i in range(na)}
        if set(basic) | set(anchors) != allcols:
            raise ValueError('missing original coordinate')
        # All original rows remain present, including redundant equalities.
        expected_rows, system = [], []
        for kind, i, entries, rhs in parsed_rows(lp, b):
            expected_rows.append(row_identity(kind, i, entries, rhs))
            coeff = {('x', j): v for j, v in entries if v}
            coeff[('E_residual' if kind == 'E' else 'A_slack', i)] = F(1)
            reduced = {}
            for k, v in coeff.items():
                if k in lookup:
                    reduced[lookup[k]] = v
                else:
                    rhs = b.value(rhs - v * anchors[k])
            system.append((reduced, rhs))
        if hint['rows'] != expected_rows:
            raise ValueError('complete original row identity/order')
        system_hash = identity([{'row': [[j, str(v)] for j, v in sorted(row.items())], 'rhs': str(rhs)} for row, rhs in system])
        vals = eliminate(system, b, stats, reuse=reuse,
                         scope=identity({'statement':expected_statement,'hint':expected_hint,
                                         'lp':identity(lp)}))
        b.where('candidate','objective')
        values = {**anchors, **dict(zip(basic, vals))}
        x = [values['x', i] for i in range(n)]
        residuals = {'E_fixed_zero_required': [str(values['E_residual', i]) for i in range(ne)],
                     'A_nonnegative_required': [str(values['A_slack', i]) for i in range(na)]}
        objective = b.value(lp['offset'])
        for coeff, v in zip(lp['c'], x):
            objective = b.value(objective + b.value(coeff) * v)
        bundle = {'schema': 'LP_SANDWICH_V1', 'lp': lp, 'statement': statement, 'dual': None,
                  'primal': {'lp_sha256': identity(lp), 'statement_sha256': expected_statement,
                             'x': [str(v) for v in x], 'claimed_objective': str(objective)}}
        b.tick()
        status = 'CANDIDATE_ONLY'
    except Limit as exc:
        status, error = 'LIMIT', str(exc)
    except Unresolved as exc:
        status, error = 'UNRESOLVED_MODULAR_RECONSTRUCTION', str(exc)
    except TimeoutError as exc:
        status, error = 'TIMEOUT', str(exc)
    except Exception as exc:
        error = repr(exc)
    if status != 'CANDIDATE_ONLY':
        bundle, residuals = None, None
    return {'schema': 'PLAN_BASIS_PROPOSAL_V1', 'status': status, 'error': error,
            'bundle': bundle, 'row_residuals': residuals, 'stats': stats, 'operations': b.operations,
            'costs':b.costs(), 'reuse_enabled':reuse, 'policy': dict(POLICY), 'arithmetic': b.arithmetic, 'seconds': time.monotonic() - begin, 'deadline_monotonic': deadline,
            'statement_sha256': expected_statement, 'hint_sha256': expected_hint,
            'assembled_system_sha256': system_hash, 'attempts': 1, 'solver_calls': 0,
            'feasibility_certified': False, 'network_SAFE': False, 'network_UNSAFE': False}
