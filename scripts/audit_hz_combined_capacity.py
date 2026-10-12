"""Read-only capacity decision for the sparse-source/direct-block chain.

Only explicit repository code and registered metadata are read. No ACT, model,
checkpoint, data, numerical library, producer, optimizer or device is executed.
The producer count is a structural bound conditional on successful conversion,
not a resource-admission test or a bound for arbitrary accepted certificates.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts import audit_hz_block_capacity as prior

ROOT = prior.ROOT
FILES = tuple(dict.fromkeys(prior.FILES + (
    'docs/hz_block_capacity_20261002_r1.json',
    'scoped_source/hz_sparse_rows.py', 'scoped_source/check_hz_sparse_rows.py',
    'scoped_source/hz_sparse_source.py', 'scoped_source/check_hz_sparse_source.py',
    'act/back_end/moe/batched_support_device.py',
    'scripts/audit_hz_combined_capacity.py', 'scripts/test_hz_combined_capacity.py')))


def assignment(raw, names):
    """Read one literal assignment, without executing the module."""
    key = ast.dump(ast.parse(names, mode='eval').body).replace('Load()', 'Store()')
    found = [n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and ast.dump(n.targets[0]) == key]
    if len(found) != 1:
        raise ValueError('unique constant assignment required: ' + names)
    return ast.literal_eval(found[0])


def producer_bounds(recipe):
    """Count additions of this producer, assuming finite completed conversion.

    Input is already outward binary64. Linear adds <=m error and <=m enclosure
    factors. An unstable ReLU adds 2 continuous+1 binary, then <=2 compensation
    factors (output and new equality). Inherited binary64 rows round exactly in
    THIS producer. Inequality compensation changes RHS, not factor inventory.
    """
    prior.topology_counts(recipe)  # Refuse an incomplete registered inventory.
    n = recipe['input_dimensions']
    rw, ew = recipe['router_widths'], recipe['expert_widths']
    if (type(n) is not int or n < 1 or rw[0] != n or ew[0] != n
            or rw[-1] != recipe['experts'] or ew[-1] != recipe['classes']
            or any(type(v) is not int or v < 1 for v in rw + ew)):
        raise ValueError('registered producer topology')
    result = {}
    for name, copies in (('common', 0), ('template_with_common', 1), ('pair_common_once', 2),
                         ('request_unique_factors_all_templates', recipe['experts'])):
        affine = sum(rw[1:]) + copies * sum(ew[1:])
        unstable = sum(rw[1:-1]) + copies * sum(ew[1:-1])
        continuous = n + 2 * affine + 4 * unstable
        guards = 2 * (recipe['experts'] - 2) if name == 'pair_common_once' else 0
        if name == 'request_unique_factors_all_templates':
            guards = 2 * (recipe['experts'] - 2) * recipe['pairs']
        result[name] = {
            'linear_outputs': affine, 'unstable_relu_upper_bound': unstable,
            'continuous_upper_bound': continuous, 'binary_upper_bound': unstable,
            'total_factor_upper_bound': continuous + unstable,
            'equality_upper_bound': unstable, 'inequality_upper_bound': 2 * unstable + guards,
            'total_constraint_row_upper_bound': 3 * unstable + guards,
        }
    return {
        'scope': 'current producer rules; conditional on complete finite conversion; not actual trained sizes',
        'formula': 'C <= input + 2*A + 4*U; B <= U; total <= input + 2*A + 5*U',
        'input_additional_enclosure_factors': 0,
        'linear_continuous_additions_per_output_at_most': 2,
        'unstable_relu_continuous_additions_at_most': 4,
        'unstable_relu_binary_additions': 1,
        'flatten_common_guard_additional_factors': 0,
        'objects': result,
        'arbitrary_checker_accepted_package_bound': False,
        'reason': 'checker permits sound overcompensation; zero inherited-row compensation is a producer property',
        'actual_factors': None, 'actual_nnz': None, 'actual_bytes': None,
    }


def source_device_contract(raw):
    fn = prior.function(raw, 'build')
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    if 'device' in names or fn.args.kwarg is not None or fn.args.vararg is not None:
        raise ValueError('source device entry changed; reassess')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'propose_request']
    if len(calls) != 1 or any(k.arg in (None, 'device') for k in calls[0].keywords):
        raise ValueError('source device forwarding changed; reassess')
    return {'source_entry_exposes_device': False, 'source_forwards_device': False,
            'source_entry_parameters': names, 'proposal_call_line': calls[0].lineno}


def derive(root=ROOT):
    raw = {n: (root / n).read_bytes() for n in FILES}
    old = prior.derive(root)
    if json.loads(raw['docs/hz_block_capacity_20261002_r1.json']) != old:
        raise ValueError('prior block report/code binding changed')
    recipe = prior.prior.derive(root)['dense_recipe']
    bounds = producer_bounds(recipe)
    row_file = 'scoped_source/check_hz_sparse_rows.py'
    limits = assignment(raw[row_file], '(MAX_FACTORS, MAX_ROWS, MAX_NNZ)')
    bits = assignment(raw[row_file], '(STORED_BITS, WORK_BITS)')
    contracts = [
        (row_file, 'parse_owned', 'len(values) > MAX_ROWS'),
        (row_file, 'parse_owned', 'len(ci) + len(bi) > cap'),
        (row_file, 'parse_owned', 'total > MAX_NNZ'),
        ('act/back_end/moe/block_support.py', 'propose_batch', "device != 'cpu'"),
        ('act/back_end/moe/check_block_support.py', 'check_batch', "candidate['device'] != 'cpu'"),
        ('act/back_end/moe/block_endpoints.py', 'propose_request', "device != 'cpu'"),
    ]
    literal = [{'file': n, **prior.prior.comparison(raw[n].decode(), fn, expr)}
               for n, fn, expr in contracts]
    paths = {}
    for name, fn, called, expected in (
        ('scoped_source/hz_sparse_source.py', 'enclosure', 'parse', 2),
        ('scoped_source/check_hz_sparse_source.py', 'lift', 'parse', 2),
        ('scoped_source/hz_sparse_source.py', 'build', 'live', 4),
        ('scoped_source/hz_sparse_source.py', 'build', 'prepare_request', 1),
        ('scoped_source/hz_sparse_source.py', 'build', 'propose_request', 1),
    ):
        lines = prior.calls(raw[name], fn, called)
        if len(lines) != expected:
            raise ValueError('combined source call path changed')
        paths[name + ':' + fn + ':' + called] = lines
    code = prior.literal(raw['scoped_source/hz_portable_verify.py'], 'CODE')
    if 'scoped_source/check_hz_sparse_source.py' in code:
        raise ValueError('portable integration changed; reassess')
    counts = old['dense_recipe']
    return {
        'schema': 'HZ_COMBINED_SOURCE_BLOCK_CAPACITY_V1',
        'status': 'REAL_COMBINED_INTAKE_NOT_ADMITTED',
        'source_bindings': {n: hashlib.sha256(v).hexdigest() for n, v in raw.items()},
        'source_schema': prior.literal(raw['scoped_source/check_hz_sparse_source.py'], 'SCHEMA'),
        'source_context': prior.literal(raw['scoped_source/check_hz_sparse_source.py'], 'CONTEXT_GUARD'),
        'current_call_sites': paths, 'literal_contracts': literal,
        'registered_recipe': {k: recipe[k] for k in
            ('input_dimensions', 'experts', 'classes', 'router_widths', 'expert_widths', 'registered_recipe_scope')},
        'producer_bound': bounds,
        'old_6684_scenario_is_new_producer_bound': False,
        'complete_obligations': {k: counts[k] for k in
            ('pairs', 'properties_per_pair', 'duties', 'endpoints_min', 'endpoints_max',
             'pair_queries_min', 'pair_queries_max')},
        'intersecting_capacity_contracts': {
            'old_source': {'factors': 128, 'outputs': 128, 'constraints': 256, 'experts': 4, 'classes': 5},
            'sparse_reference': {'factors': limits[0], 'each_entire_vector_rows': limits[1], 'nnz': limits[2],
                'stored_rational_bits': bits[0], 'working_rational_bits': bits[1],
                'target_factor_allowance': limits[0] + 2 * limits[1],
                'chunk_size_is_whole_state_tiling': False},
            'block': {'source_and_global_factors': 128, 'global_constraints': 256, 'experts': 4,
                'properties': 4, 'queries_per_batch': 8},
            'definite_recipe_refusals': old['definite_refusals'],
            'full_producer_bound_exceeds_4096': bounds['objects']['template_with_common']['total_factor_upper_bound'] > limits[0],
            'upper_bound_exceeding_cap_is_not_observed_trained_failure': True,
        },
        'repeated_storage': {
            'support_snapshot_occurrences': counts['support_snapshot_occurrences'],
            'occurrences_per_expert': counts['snapshots_per_expert'],
            'scope': 'stored support references per complete recipe; not unique buffers, bytes, peak RAM or elapsed time',
            'remaining_live_arrays': old['remaining_live_arrays'],
            'hashing': 'whole-object json.dumps in identity/capture, not bounded streaming',
        },
        'device_boundary': {
            'candidate_function': 'act/back_end/moe/block_support.py:_candidate_columns',
            'acceptance_function': 'act/back_end/moe/check_block_support.py:evaluate',
            'invariant': 'sum all local contributions on shared residual coordinates BEFORE one absolute-value box support',
            'updates': 128, 'dtype': 'float64', 'physical_block_kernel_implemented': False,
            'flat_device_kernel_interchangeable': False, 'preproposal_capacity_cleared_by_gpu': False,
            **source_device_contract(raw['scoped_source/hz_sparse_source.py']),
        },
        'integration': {'stdlib_portable_sparse_source': False, 'combined_hard_supervision': False,
                        'conv_operators': old['conv']['unsupported_operators'],
                        'conv_pairs': old['conv']['all_pairs'], 'conv_duties': old['conv']['duties']},
        'new_model_loads': 0, 'new_input_selections': 0, 'new_propagations': 0, 'new_solves': 0, 'cuda_calls': 0,
        'actual_peak_memory': None, 'actual_end_to_end_seconds': None,
        'source_or_output_certificate': False, 'all_six_goal_gates_remain_open': True,
        'decision': 'request-owned checked source/block store plus complete row/endpoint tiling; not cap-only repair; no real/device admission',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, help='compare an existing report; without it print the report')
    args = parser.parse_args()
    report = derive()
    if args.check:
        if json.loads(args.check.read_bytes()) != report:
            raise ValueError('combined capacity report differs')
        print('PASS: static capacity report matches; no model, propagation, solve or CUDA')
    else:
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
