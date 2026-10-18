"""Static pair-lifetime accounting; never imports a verifier or runs a source."""
import ast
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOC = 'docs/hz_pair_scope_lifetime_20261018.md'
REPORT = 'docs/hz_pair_scope_lifetime_20261018.json'
PARENT = 'scripts/assess_support_amortization.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_bytes())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node, attribute):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == attribute for n in ast.walk(node))


def analyze():
    # The parent is itself a pure static assessor; its 163 frozen hashes and
    # retained-record arithmetic are checked, without importing production.
    spec = importlib.util.spec_from_file_location('static_parent', ROOT / PARENT)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    base = parent.analyze()
    require(base == json.loads((ROOT / parent.REPORT).read_bytes()), 'parent record drift')
    require(calls(method('scoped_source/scaled_endpoints.py', 'Session', '_consume'), 'clear_tile'),
            'current per-tile release boundary')
    require(calls(method('scoped_source/pair_workspace.py', 'PairWorkspace', 'clear_tile'), 'close'),
            'current Scope close, not an implemented reusable lease')
    vertices = []
    for corner in base['vertices']:
        ur, vr = {'positive': (0, 0), 'negative': (0, 128), 'unstable': (128, 0)}[corner['router']]
        nu, nv = corner['unstable_experts'], corner['negative_experts']
        experts = [(384, 0)] * nu + [(0, 384)] * nv + [(0, 0)] * (8 - nu - nv)
        n, f = corner['nodes'], corner['mandatory_factor_lower']
        totals = dict(all_present_component_upper=0, completed_missing_component_upper=0)
        for a, b in combinations(range(8), 2):
            ua, va = experts[a]; ub, vb = experts[b]
            s = 4008 + 4 * ur + vr + 4 * (ua + ub) + va + vb
            t = 3072 + 3 * (ur + ua + ub)
            h = 2 * (n + f) + s + t + 53
            h1, h2 = parent.gains(n, s)
            totals['all_present_component_upper'] += max(10*h1+4*h2-h, 18*h1+9*h2-h)
            totals['completed_missing_component_upper'] += max(
                9*h1+4*max(0, h2)+max(0, h1)-h, 18*h1+9*max(0, h2)-h)
        vertices.append(dict(router=corner['router'], unstable_experts=nu,
                             negative_experts=nv, **totals))
    keys = ('all_present_component_upper', 'completed_missing_component_upper')
    maxima = {key: max(vertices, key=lambda row: row[key]) for key in keys}
    require(all(maxima[key][key] == 2838976 for key in keys), 'component maxima')
    unstable = next(v for v in vertices if v['router'] == 'unstable' and v['unstable_experts'] == 8)
    positive = next(v for v in vertices if v['router'] == 'positive'
                    and v['unstable_experts'] == v['negative_experts'] == 0)
    require(unstable['all_present_component_upper'] == 1602496, 'unstable corner')
    require(positive['all_present_component_upper'] == -747124, 'positive corner')
    require(positive['completed_missing_component_upper'] == -746984, 'positive missing corner')
    paths = (PARENT, parent.REPORT, DOC, 'scripts/assess_pair_scope_lifetime.py',
             'scoped_source/pair_workspace.py', 'scoped_source/bounded_runtime.py',
             'scoped_source/scaled_endpoints.py', 'scoped_source/forest_endpoints.py',
             'scoped_source/direct_candidates.py', 'scoped_source/direct_lifecycle.py',
             'scoped_source/check_shared_source.py', 'scripts/test_support_traversal.py')
    return dict(schema='PAIR_SCOPE_LIFETIME_STATIC_V1',
        base_head='59da49b657f4e817bf2f12c5fa7159aa68762e69',
        anchors={p: sha(p) for p in paths}, vertices_checked=len(vertices),
        maxima=maxima, all_unstable=unstable, all_positive=positive,
        accounting=dict(representation='Shared / tuple nodes only',
            comparison='equal pair lifecycle; original Scope construction cancels',
            scope_construction='K=2*(N+F)+2*(S+T)+4',
            canonical_preparation='H=2*(N+F)+S+T+53, once per pair',
            actual_delta='traversal_and_preparation_delta + J_global - J_scoped',
            lifecycle_costs='J_global and J_scoped are unclosed, nonnegative and separately charged',
            old_baseline='additional (tiles-1)*K is lifetime reuse, not traversal gain',
            claim='positive component upper is opportunity only, not a net performance theorem'),
        decision=dict(status='STATIC_CANDIDATE_NOT_IMPLEMENTED',
            evidence='immutable same-object license, not skipping canonical checks',
            scope_retirement='private Plan final-pair-tile, before next pair or terminal',
            full_shape_admitted=False, source_floor=13922736,
            new_controls=0, new_sources=0, candidates=0, native_calls=0, models=0, gpu=0,
            acceptance_changes=False, quota_changes=False, goals='G1-G6 OPEN'))


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == json.loads((ROOT / REPORT).read_bytes()), 'report drift')
        print('PASS: source-bound static pair-lifetime components at 135 vertices; no verifier execution')
    else:
        require(not sys.argv[1:], 'use no arguments or --check')
        print(json.dumps(result, sort_keys=True, indent=2))
