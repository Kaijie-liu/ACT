"""Source-only traversal audit: AST + integer/Fraction algebra, not a control run.

No production imports, factories, candidates, models, native libraries or GPU.
The report binds claims to source; it is not a full mechanized correctness proof.
"""
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = '2daaaf7d39a5cc285ac9d7094363f45c2d780cbc'
DOC = 'docs/hz_support_traversal_review_20261018.md'
REPORT = 'docs/hz_support_traversal_review_20261018.json'
SHAPE = 'docs/hz_native_full_shape_decision_20261018.json'
PRIOR = 'docs/hz_native_phase_contract_20261018.json'


def require(ok, why):
    if not ok: raise ValueError(why)


def raw(name): return (ROOT/name).read_bytes()
def sha(name): return hashlib.sha256(raw(name)).hexdigest()
def load(name): return json.loads(raw(name))


def method(name, cls, func):
    c = next(n for n in ast.parse(raw(name)).body if isinstance(n,ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name == func)


def gets(node):
    return sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr == 'get' for n in ast.walk(node))


def analyze():
    prior = load(PRIOR)
    for name, expected in prior['anchors'].items():
        require(sha(name) == expected, 'prior source drift '+name)
    shape = load(SHAPE)['shape']
    require(shape['expert_widths'] == [3072,256,128,10] and
            shape['router_widths'] == [3072,128,8], 'registered shape')
    p = 'scoped_source/pair_workspace.py'
    pull = method(p,'PairWorkspace','pullback')
    require(any(isinstance(n,ast.For) and ast.unparse(n.iter) == 'range(len(f.nodes) - 1, -1, -1)'
                for n in ast.walk(pull)), 'current full global scan')
    require(gets(method(p,'Scope','global_id')) == 1 and gets(method(p,'Scope','local')) == 2,
            'existing reverse/forward/reverse getters')
    get = method('scoped_source/bounded_runtime.py','Block','get')
    require(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'visit'
                and len(n.args) == 1 and isinstance(n.args[0],ast.Constant)
                and n.args[0].value == 'coefficient_visits' for n in ast.walk(get)) == 1,
            'one coefficient visit per bounded get')
    assignments = {t.id:ast.literal_eval(n.value) for n in ast.parse(raw(p)).body
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
        for t in n.targets if isinstance(t,ast.Name)}
    require(assignments['CANDIDATE_WIDTH'] == 8 and assignments['CHECK_WIDTH'] == 2,
            'unchanged batch/check widths')
    scans = []
    for x in range(29):
        endpoints,tiles,blocks = 252+9*x,56+x,140+4*x
        explicit = [9]* (28-x)+[18]*x
        actual_tiles = [[min(8,e-i) for i in range(0,e,8)] for e in explicit]
        require(sum(explicit) == endpoints and sum(map(len,actual_tiles)) == tiles and
            sum((k+1)//2 for ts in actual_tiles for k in ts) == blocks, 'strict pair tiles and blocks')
        scans.append(dict(two_endpoint_pairs=x,endpoints=endpoints,tiles_per_checker=tiles,
            blocks_per_checker=blocks,total_scans=2*(28+endpoints+blocks)))
        require(scans[-1]['total_scans'] == 840+26*x, 'scan polynomial')
    # Counting a single call's unmatched operations, not running either algorithm.
    def saving(n,s,b): return (b+1)*(n-s)-3*s
    require(18*saving(19496,7592,1)+9*saving(19496,7592,2) == 135000,
            'double-endpoint pair conditional count')
    require(9*saving(19496,7592,1)+4*saving(19496,7592,2)+saving(19496,7592,1)
            == 32*(19496-7592)-42*7592, 'odd final single-column block')
    # Actual per-role symbolic counts; do not subtract independent N/S upper bounds.
    def summed(ns,experts):
        xe = sum(experts); n = ns+xe+336; result = 0
        for a in range(8):
            for b in range(a+1,8):
                s = ns+experts[a]+experts[b]+12
                result += 2*(18*saving(n,s,1)+9*saving(n,s,2))
        require(result == 1512*xe-4536*ns+1088640, 'same-object pair sum')
        return result
    full = summed(3720,[1930]*8)
    low = summed(3720,[394]*8); high = summed(3208,[1930]*8)
    require((full,low,high) == (7560000,-11019456,9882432), 'conditional shape range')
    s,rhs,y = F(-1,4),F(-1,2),F(-1)
    correct,unsound = s+rhs*y-abs(-y),s+rhs*y
    require((correct,unsound) == (F(-3,4),F(1,4)) and F(-1) <= rhs <= F(1),
            'nonempty guard and false positive if dual dependency omitted')
    files = {DOC,PRIOR,SHAPE,'scripts/assess_support_traversal.py',p,
        'scoped_source/bounded_runtime.py','scoped_source/scaled_endpoints.py',
        'scoped_source/forest_endpoints.py','scoped_source/check_source_dag.py',
        'scoped_source/check_affine_references.py','scoped_source/check_scaled_references.py',
        'scoped_source/check_direct_source.py','scoped_source/check_shared_source.py',
        'scoped_source/shared_source.py','scoped_source/hz_request_store.py',
        'scoped_source/sparse_build.py','scoped_source/sparse_check.py',
        'act/back_end/moe/block_support.py','act/back_end/moe/check_block_support.py'}
    return dict(schema='V3_SUPPORT_TRAVERSAL_STATIC_REVIEW',base_head=BASE,
        anchors={n:sha(n) for n in sorted(files)},
        correctness=dict(seed_rule='complete checked objective or nonzero proposed dual roots, closed under refs',
            property_only_rejected=True,counterexample=dict(correct=str(correct),unsound=str(unsound)),
            same_candidate_arithmetic_equivalence='conditional on complete bindings, descending order and unchanged checks',
            gate_planning_unchanged=True,all_factor_checks_unchanged=True,zero_role_checks_unchanged=True),
        counts=dict(condition='all endpoint candidates present and both checkers complete',
            complete_scans_min=scans[0],complete_scans_max=scans[-1],
            zero_role_extra_pullbacks=0,scope_construction_N_loops_max=336,
            reverse_extra_coefficient_gets_per_selected_node=3,
            same_call_saving='(b+1)*(N-S)-3*S',double_endpoint_pair='63*(N-S)-81*S',
            single_endpoint_pair='32*(N-S)-42*S',full_two_checker_saving='1512*Xe-4536*Ns+1088640',
            conditional_all_unstable_saving=full,allowed_count_range=[low,high],
            counts_are_runtime_observations=False,extra_validation_cost_included=False,
            illustrative_selector_cost_per_call=1,
            illustrative_complete_selector_cost=scans[-1]['total_scans']-56,
            selector_threshold='gain > C_select + C_extra; extra checks/storage/release are not free',
            selector_is_implemented=False),
        decision=dict(property_only='REJECT_UNSOUND',dynamic_cone='STOP_NO_NET_ADVANTAGE_EVIDENCE',
            scope_reverse='STRONG_REFERENCE_WITH_CONDITIONAL_COST_NOT_NEW_DOMAIN',
            next_implementation='separate approval required for default-off counted scope/global choice and bounded controls',
            runtime_changes=0,production_controls=0,native_calls=0,models=0,gpu=0,
            quotas_changed=False,full_shape_admitted=False,source_parameter_floor=2*shape['parameter_values'],
            scientific_upgrade=False,goals='G1-G6 OPEN'))


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == load(REPORT), 'static traversal report drift')
        print('PASS: source/AST and traversal algebra; no algorithm execution or speed claim')
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(result,sort_keys=True,indent=2))
