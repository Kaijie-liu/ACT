"""Solver-free recheck of source controls and an exact MC feasible witness.

No producer imports. The caller supplies expected source identities from a
separate run manifest. This audits algebra, not elapsed time or native execution.
"""
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_source.endpoint_source_check import check, reconstruct_mc
from scoped_source.sparse_ir import index
from source_enclosure.format import identity
from upstream_source.checker import csr


def point_value(lp, point):
    point = list(map(F, point))
    if len(point) != len(lp['c']): raise ValueError('primal dimension')
    if any(not F(lo) <= x <= F(hi) for x, lo, hi in zip(point, lp['lower'], lp['upper'])):
        raise ValueError('primal variable bounds')
    for matrix, rhs in [('A', 'b'), ('E', 'h')]:
        for coefficients, bound in zip(csr(lp[matrix]), lp[rhs]):
            value = sum((a*point[j] for j, a in coefficients.items()), F(0))
            if (value > F(bound) if matrix == 'A' else value != F(bound)):
                raise ValueError('exact primal feasibility')
    return F(lp['offset']) + sum((F(c)*x for c, x in zip(lp['c'], point)), F(0))


def route_witness(doc, point):
    r, nodes, outputs = index(doc, identity(doc), lambda: None)
    values = {}
    if len(point) != sum(node['kind'] == 'input' for node in nodes.values()):
        raise ValueError('route witness dimension')
    for name, node in nodes.items():
        if node['kind'] == 'input':
            value = F(point[int(name.split('/')[1])])
            if not node['bounds'][0] <= value <= node['bounds'][1]: raise ValueError('route witness outside domain')
        elif node['kind'] == 'affine':
            value = node['bias'] + sum((v*values[p] for p, v in node['terms'].items()), F(0))
        else: value = max(F(0), values[node['parent']])
        values[name] = value
    scores = [values[v] for v in outputs['router']]
    pairs = [list(pair) for pair in combinations(range(r['experts']), 2)
             if all(scores[s] >= scores[o] for s in pair for o in range(r['experts']) if o not in pair)]
    return {'point': list(map(str, map(F, point))), 'scores': list(map(str, scores)), 'legal_pairs': pairs}


def audit_case(doc, packages, *, expected_source_sha256, negative_point=None):
    if set(packages) != {'endpoints', 'mccormick'}: raise ValueError('both registered arms required')
    checked = {}
    for mode, package in packages.items():
        result = check(doc, package, expected_source_sha256=expected_source_sha256,
                       expected_mode=mode, deadline=time.monotonic()+300)
        checked[mode] = {'checked': result, 'package_sha256': identity(package),
                         'proposal_stats_recorded': package['proposal_stats'],
                         'proposal_errors_recorded': package['proposal_errors']}
    if (packages['endpoints']['request'] != packages['mccormick']['request'] or
            packages['endpoints']['reuse_requested'] != packages['mccormick']['reuse_requested']):
        raise ValueError('paired source/gate/fact fairness')
    output = {'source_sha256': expected_source_sha256, 'arms': checked}
    if negative_point is not None:
        if set(negative_point) != {'pair', 'competitor', 'lp_sha256', 'point', 'checked_objective',
                                   'network_counterexample', 'lp_optimum_claimed'}:
            raise ValueError('negative point fields')
        # Never trust the point's matrix: reconstruct it from an already source-checked duty.
        duties = packages['mccormick']['request']['duties']
        candidates = [d for d in duties if d['pair'] == negative_point['pair'] and d['competitor'] == negative_point['competitor']]
        if len(candidates) != 1: raise ValueError('negative point duty identity')
        lp = reconstruct_mc(candidates[0]); value = point_value(lp, negative_point['point'])
        if (identity(lp) != negative_point['lp_sha256'] or value >= 0 or
                str(value) != negative_point['checked_objective'] or
                negative_point['network_counterexample'] is not False or negative_point['lp_optimum_claimed'] is not False):
            raise ValueError('negative relaxation witness semantics')
        output['mc_negative_point'] = negative_point
        output['route_witnesses'] = [route_witness(doc, [x]) for x in ('-1', '-1/2', '1')]
    return output
