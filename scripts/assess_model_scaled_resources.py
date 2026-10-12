"""Static resource-policy proposal, not an execution or a policy override.

Reads pinned code and shape metadata only. Structural upper bounds describe
the existing V3 builder, not observed trained sparsity or a runtime forecast.
No source, checkpoint, numerical library, solver or device is imported.
"""
import argparse
import ast
import hashlib
import json
from math import comb
from pathlib import Path

from scripts.analyze_tensor_alias_capacity import analyze as prior_analysis, ROOT

EXTRA_ANCHORS = {
    'scripts/analyze_tensor_alias_capacity.py': 'ecf037bdafb044e3173b8d560a58f7db8686bf0c673da49bd8ecf515c7844e3c',
    'act/back_end/moe/factory.py': 'da472b2bbb3d8f684cb9c9916a735fd7105ceaab925911e1cbc983785a4c7d2d',
    'scoped_source/tensor_rows.py': '5c3ad6d642868e257ac9dd2edbc5e066b4cd793d4c7567a4ea800fc8e8f97f82',
    'scoped_source/verify_scaled_endpoints.py': '67bccc1c02cd50dbd42d35cf6e3a1a98ce65e5476b263ce7fbfdb6ef4aadb220',
    'scoped_source/factored_source.py': 'f7f04c64989a179c36759876edc425ffd3efad4eeb9fbf8118ea2934f4a385cc',
    'scoped_source/factored_io.py': 'd51bcd7eae6734fe6cd676f584a9e9f2220eeea4400d497ccb1dc6950dce120d',
    'scripts/run_scaled_dense_capacity.py': '4b80573abc480c5dde5fc3e9628e95c6284a4ae1d1fa1fa3d4441160227f6181',
}
PHASES = ('source_build', 'source_check', 'gate_and_obligation_plan',
          'native_preparation_and_candidates', 'objective_pullback',
          'dual_check_merge_and_support', 'serialization_receipt_and_terminal')
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')


def _integer(node):
    """Read existing integer constants without importing production modules."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp):
        a, b = _integer(node.left), _integer(node.right)
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.Pow) and 0 <= b <= 32: return a**b
    raise ValueError('unsupported policy integer expression')


def current_policy():
    tree = ast.parse((ROOT/'scoped_source/hz_request_store.py').read_bytes())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'POLICY'
                                              for t in node.targets):
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == 'MappingProxyType' and len(call.args) == 1):
                raise ValueError('policy mapping shape')
            inner = call.args[0]
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == 'dict' and not inner.args):
                raise ValueError('policy literal shape')
            return {k.arg: _integer(k.value) for k in inner.keywords}
    raise ValueError('missing policy')


def shape_inventory(inputs, experts, classes, router_hidden, expert_hidden, width=8):
    dims = (inputs, experts, classes, width, *router_hidden, *expert_hidden)
    if (not all(type(v) is int and v > 0 for v in dims) or
            not 2 <= experts <= 8 or not 2 <= classes <= 10 or
            not 1 <= width <= 8 or not router_hidden or not expert_hidden):
        raise ValueError('bounded positive MLP shape with hidden layers')
    router = [inputs, *router_hidden, experts]
    expert = [inputs, *expert_hidden, classes]
    a = sum(router[1:])+experts*sum(expert[1:])
    u = sum(router_hidden)+experts*sum(expert_hidden)
    w1 = inputs*(router[1]+experts*expert[1])
    w2 = sum(x*y for x, y in zip(router[1:], router[2:]))
    w2 += experts*sum(x*y for x, y in zip(expert[1:], expert[2:]))
    pairs = comb(experts, 2); guards = pairs*2*(experts-2)
    router_ops = 2*len(router_hidden)+2  # Flatten, Linear/ReLU, final Linear.
    expert_ops = 2*len(expert_hidden)+2
    # Every full view has its output roster even when its node IDs alias.
    output_roots = (pairs+experts+3)*inputs+a+u
    per_pair_relu = sum(router_hidden)+2*sum(expert_hidden)
    dual_rows = 3*per_pair_relu+2*(experts-2)
    endpoints = 2*pairs*(classes-1)
    # Loose maxima allow a lift for all four ReLU rows. The exact +/-1
    # inequality rows cannot actually need it; no attainability is claimed.
    return dict(inputs=inputs, experts=experts, classes=classes,
        router_widths=router, expert_widths=expert, affine_outputs=a, relu_units=u,
        first_weight_positions=w1, later_weight_positions=w2,
        parameter_values=w1+w2+a, source_values_with_center=w1+w2+a+inputs,
        raw_binary64_tensor_bytes=8*(w1+w2+a+inputs),
        largest_weight_tensor=max([x*y for x,y in zip(router,router[1:])]+
                                  [x*y for x,y in zip(expert,expert[1:])]),
        pairs=pairs, guards=guards, views=2+router_ops+experts*expert_ops+pairs,
        pair_output_roots=pairs*inputs,
        mandatory_roots=output_roots+guards, roots_upper=output_roots+guards+3*u,
        factors_upper=inputs+2*a+7*u, nodes_upper=inputs+a+4*u+guards,
        local_entries_upper=inputs+w1+2*a+12*u,
        references_upper=w2+u+2*guards,
        property_duties=pairs*(classes-1), endpoints_min=pairs*(classes-1),
        endpoints_upper=endpoints, candidates_upper=2*endpoints,
        pair_dual_rows_upper=dual_rows, dual_scalar_entries_upper=2*endpoints*dual_rows,
        pair_local_tiles_upper=pairs*((2*(classes-1)+width-1)//width), width=width)


def capacity_classification(cap, *, lower=None, upper=None):
    if type(cap) is not int or cap <= 0:
        raise ValueError('positive resource cap')
    for value in (lower, upper):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('nonnegative structural bound or null')
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('reversed structural interval')
    if lower is not None and lower > cap: return 'SHAPE_PROVES_CURRENT_REFUSAL'
    if upper is not None and upper <= cap: return 'STRUCTURAL_UPPER_FITS_ONLY'
    if upper is not None: return 'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN'
    return 'COMPLETE_BOUND_MISSING'


def shared_work_proposal(ledgers, *, ceiling, quantum=1_000_000):
    """Arithmetic rule for a future freeze, NOT a checker of bound derivations.

    Both method ledgers must be audited full-path UPPER bounds before using
    this rule. Unknown phases return no quota. This never admits execution.
    """
    if set(ledgers) != {'reference', 'proposed'}:
        raise ValueError('both internal methods required')
    if type(quantum) is not int or quantum <= 0:
        raise ValueError('positive fixed rounding quantum')
    if ceiling is not None and (type(ceiling) is not int or ceiling < 8_000_000):
        raise ValueError('finite absolute Work ceiling or null')
    unknown = []; totals = {}
    for method, phases in ledgers.items():
        if set(phases) != set(PHASES): raise ValueError('complete phase roster')
        total = 0
        for phase, record in phases.items():
            if record is None:
                unknown.append(method+'/'+phase); continue
            if (type(record) is not dict or set(record) != {'kind', 'counts', 'derivation'} or
                    record['kind'] != 'UPPER' or type(record['derivation']) is not str or
                    not record['derivation'] or type(record['counts']) is not dict or
                    set(record['counts']) != set(COUNTERS) or
                    any(type(v) is not int or v < 0 for v in record['counts'].values())):
                raise ValueError('audited upper-bound record required, not a floor or observation')
            total += sum(record['counts'].values())
        totals[method] = None if any(k.startswith(method+'/') for k in unknown) else total
    candidate = None if unknown else ((max(8_000_000, *totals.values())+quantum-1)//quantum)*quantum
    if unknown: status = 'MISSING_COMPLETE_UPPERS'
    elif ceiling is None: status = 'ABSOLUTE_CEILING_NOT_FROZEN'
    elif candidate > ceiling: status = 'ABOVE_ABSOLUTE_CEILING'
    else: status = 'ARITHMETIC_PROPOSAL_ONLY'
    return dict(status=status, method_upper_totals=totals, missing_phases=unknown,
                candidate_work_cap=candidate, absolute_work_ceiling=ceiling,
                quantum=quantum, bound_derivations_independently_verified=False,
                execution_admitted=False)


def analyze():
    for name, expected in EXTRA_ANCHORS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('reviewed code changed: '+name)
    prior = prior_analysis(); r = prior['recipe']; policy = current_policy()
    inv = shape_inventory(r['inputs'], r['experts'], r['classes'], r['router'][1:-1], r['expert'][1:-1])
    rows = []

    def add(name, cap, lower=None, upper=None):
        rows.append(dict(resource=name, current_cap=cap, structural_lower=lower,
            structural_upper=upper, observed_actual=None,
            status=capacity_classification(cap, lower=lower, upper=upper)))

    add('production_work', policy['visits'], lower=2*inv['first_weight_positions'])
    add('view_root_entries', policy['rows'], inv['mandatory_roots'], inv['roots_upper'])
    add('factors', policy['factors'], upper=inv['factors_upper'])
    add('active_local_factors', policy['active'], upper=inv['factors_upper'])
    add('nodes', policy['rows'], upper=inv['nodes_upper'])
    add('views', policy['views'], inv['views'], inv['views'])
    add('local_entries', policy['nnz'], upper=inv['local_entries_upper'])
    add('references', 65536, upper=inv['references_upper'])
    add('endpoints', 504, inv['endpoints_min'], inv['endpoints_upper'])
    add('tensor_elements', 1_000_000, inv['largest_weight_tensor'], inv['largest_weight_tensor'])
    for name, cap in (('captured_source_bytes',256*2**20), ('forest_bytes',128*2**20),
                      ('candidate_tile_bytes',4*2**20), ('proof_bytes',16*2**20),
                      ('archive_bytes',128*2**20), ('peak_rss_bytes',8*2**30),
                      ('address_space_bytes',16*2**30)):
        add(name, cap)
    structural_caps = {
        'factors': max(policy['factors'], inv['factors_upper']),
        'active_local_factors': max(policy['active'], inv['factors_upper']),
        'nodes': max(policy['rows'], inv['nodes_upper']),
        'view_root_entries': max(policy['rows'], inv['roots_upper']),
        'views': max(policy['views'], inv['views']),
        'local_entries': max(policy['nnz'], inv['local_entries_upper']),
        'references': max(65536, inv['references_upper']),
    }
    return dict(schema='MODEL_SCALED_RESOURCE_STATIC_PROPOSAL_V1',
        reviewed_head='3ae08e36b8f2c4eb3ff4206ba6512b792b770f40',
        assessment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_and_metadata={**prior['code_and_metadata'], **EXTRA_ANCHORS},
        scope='registered seed1/2 MLP topology; no seed0 or live checkpoint correspondence check',
        inventory=inv, current_resource_vector=rows,
        proposed_structural_caps=structural_caps,
        proposed_structural_caps_are_not_runtime_overrides=True,
        proposal=shared_work_proposal({m:dict.fromkeys(PHASES) for m in ('reference','proposed')}, ceiling=None),
        portable_work_upper=None, complete_archive_byte_upper=None,
        peak_live_object_byte_upper=None, completion_seconds_forecast=None,
        invariant_contract=dict(request_seconds=300, stored_bits=policy['stored_bits'],
            working_bits=policy['work_bits'], counter_kinds=list(COUNTERS),
            threshold='strictly greater than 1/10000000', candidate_roles=['proposed','zero'],
            all_tie_legal_obligations=True, offline_check_is_not_late_production_completion=True),
        unknowns=['complete per-method Work upper bounds', 'actual sparsity and activation/error counts',
            'forest/proof/archive encoded byte upper bounds', 'peak live object memory',
            '300-second completion', 'trained source capture and correspondence'],
        decision='DESIGN_ONLY_NOT_ADMISSIBLE_WORK_CHANGE_ALONE_INSUFFICIENT',
        production_policy_changed=False, implementation_admitted=False, real_execution_admitted=False,
        new_source_runs=0, new_bound_queries=0, checkpoint_loads=0, gpu_calls=0,
        scientific_limit='shape bounds and engineering policy only; no safety, speedup or external advantage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args(); result = analyze()
    if args.check_report:
        if result != json.loads(args.check_report.read_bytes()):
            raise ValueError('resource assessment differs')
        print('PASS: static resource proposal; no execution or cap change')
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
