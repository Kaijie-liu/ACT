"""Static block/source capacity assessment; never import or execute ACT.

Only the explicit repository metadata/code inventory is read. Scenarios are
structural counts, not measurements or admission of a real request.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

from scripts import audit_hz_real_intake as prior

ROOT = prior.ROOT
FILES = tuple(dict.fromkeys(prior.FILES + (
    'docs/hz_real_intake_20261001_r1.json',
    'scoped_source/check_hz_binary64.py', 'scoped_source/hz_binary64.py',
    'scoped_source/hz_row_enclosure.py', 'scoped_source/check_hz_row_enclosure.py',
    'scoped_source/hz_lifted_source.py', 'scoped_source/check_hz_lifted_source.py',
    'scoped_source/hz_templates.py', 'scoped_source/check_hz_templates.py',
    'scoped_source/hz_block_source.py', 'scoped_source/check_hz_block_source.py',
    'act/back_end/moe/block_support.py', 'act/back_end/moe/check_block_support.py',
    'act/back_end/moe/block_endpoints.py', 'act/back_end/moe/check_block_endpoints.py',
    'source_enclosure/format.py',
    'scripts/audit_hz_block_capacity.py', 'scripts/test_hz_block_capacity.py')))


def function(raw, name):
    items = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(items) != 1:
        raise ValueError('unique function required: ' + name)
    return items[0]


def calls(raw, name, callee):
    return sorted(n.lineno for n in ast.walk(function(raw, name))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee)


def literal(raw, name):
    values = [n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(values) != 1:
        raise ValueError('unique literal required: ' + name)
    return ast.literal_eval(values[0])


def repeated_parse_contract(raw):
    """Prove the recorded call pattern syntactically, not its measured cost."""
    check = function(raw, 'check')
    initial = [n for n in check.body if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'parse' and ast.unparse(n.value.args[0]) == 'reference']
    loops = [n for n in check.body if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "zip(proof['rows'], roster)"]
    if len(initial) != 1 or len(loops) != 1:
        raise ValueError('row checking structure changed')
    nested = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == 'projected']
    parses = calls(raw, 'projected', 'parse')
    if len(nested) != 1 or len(parses) != 1 or calls(raw, 'parse', 'unpack') == [] or calls(raw, 'parse', 'pack') == []:
        raise ValueError('full reference reparse pattern changed')
    return {'initial_parse_line': initial[0].lineno, 'row_projection_line': nested[0].lineno,
            'projection_parse_line': parses[0], 'whole_reference_parses_for_R_rows': 'R+1',
            'scope': 'syntactic path; excludes final target parses, hashes and local checks; no timing estimate'}


def topology_counts(d):
    e, c, p = d['experts'], d['classes'], d['pairs']
    if p != math.comb(e, 2) or d['duties'] != p * (c - 1):
        raise ValueError('complete route/property roster')
    common = 3 * sum(d['router_widths'][1:-1])
    private = 3 * sum(d['expert_widths'][1:-1])
    guards = 2 * (e - 2)
    return {
        'experts': e, 'classes': c, 'pairs': p, 'properties_per_pair': c - 1,
        'duties': p * (c - 1), 'endpoints_min': p * (c - 1), 'endpoints_max': 2 * p * (c - 1),
        'pair_queries_min': c - 1, 'pair_queries_max': 2 * (c - 1),
        'old_per_pair_expert_propagations': 2 * p, 'template_expert_propagations': e,
        'old_layer_records': d['all_layer_trace_records'],
        'template_layer_records': d['router_layer_records'] + e * (2 * len(d['expert_widths']) - 2),
        'support_snapshot_occurrences': {'common': p, 'entry': p, 'expert_templates': 2 * p, 'total': 4 * p},
        'snapshots_per_expert': e - 1,
        'all_unstable_row_scenario': {
            'common_rows': common, 'entry_rows': common + guards,
            'expert_template_rows_each': common + private,
            'stored_rows_in_four_snapshots': 4 * common + guards + 2 * private,
            'executed_local_block_rows': common + guards + 2 * private,
            'repeated_stored_prefix_rows': 3 * common,
            'scope': 'same all-unstable recipe scenario, not measured rows/nnz/RAM/time; excludes traces and row proofs'},
        'old_factor_scenario': d['new_HZ_pair_scenario']['total_factors'],
        'old_factor_count_is_bound_for_new_lifts': False,
        'new_lift_factors': None,
        'scope': 'registered recipe arithmetic only; all pairs conservatively retained; no actual feasible-pair count',
    }


def derive(root=ROOT):
    raw = {n: (root / n).read_bytes() for n in FILES}
    old = prior.derive(root)
    if json.loads(raw['docs/hz_real_intake_20261001_r1.json']) != old:
        raise ValueError('prior metadata/code binding changed')
    specs = (
        ('scoped_source/check_hz_binary64.py', 'parse', 'len(c)+len(b) > (16 if target else 8)'),
        ('scoped_source/check_hz_row_enclosure.py', 'parse', 'len(c)+len(b) > 128'),
        ('scoped_source/check_hz_row_enclosure.py', 'parse', "1 <= len(h['c']) <= 128"),
        ('scoped_source/check_hz_row_enclosure.py', 'parse', "len(h['b'])+len(h['ub']) > 256"),
        ('act/back_end/moe/block_support.py', 'snapshot', '1 <= hz.n_cont+hz.n_bin <= 128'),
        ('act/back_end/moe/block_support.py', 'snapshot', '1 <= hz.n_out <= 128'),
        ('act/back_end/moe/check_block_support.py', 'structure', '1 <= nc+nb <= 128'),
        ('act/back_end/moe/check_block_support.py', 'structure', "sum(len(p['b'])+len(p['h']) for p in parts) > 256"),
        ('act/back_end/moe/check_block_support.py', 'validate_batch', '1 <= len(queries) <= 8'),
        ('act/back_end/moe/check_block_endpoints.py', 'validate_request', '2 <= e <= 4'),
        ('act/back_end/moe/check_block_endpoints.py', 'validate_request', '1 <= len(props) <= 4'),
    )
    contracts = old['literal_contracts'] + [
        {'file': name, **prior.comparison(raw[name].decode(), fn, expr)} for name, fn, expr in specs]
    repeats = repeated_parse_contract(raw['scoped_source/check_hz_row_enclosure.py'])
    block_schema = literal(raw['scoped_source/check_hz_block_source.py'], 'SCHEMA')
    old_portable_code = literal(raw['scoped_source/hz_portable_verify.py'], 'CODE')
    if 'scoped_source/check_hz_block_source.py' in old_portable_code:
        raise ValueError('portable block support changed; reassess integration')
    # The complete input source has one original factor for every input coordinate,
    # independent of radius. Actual source conversion is never called here.
    counts = topology_counts(old['dense_recipe'])
    return {
        'schema': 'HZ_BLOCK_CAPACITY_METADATA_V1', 'status': 'REAL_BLOCK_INTAKE_NOT_ADMITTED',
        'source_bindings': {n: hashlib.sha256(v).hexdigest() for n, v in raw.items()},
        'prior_metadata_sha256': hashlib.sha256(raw['docs/hz_real_intake_20261001_r1.json']).hexdigest(),
        'literal_contracts': contracts, 'dense_recipe': counts,
        'definite_refusals': {
            'source_expert_class_inventory': [old['dense_recipe']['experts'], old['dense_recipe']['classes']],
            'input_factors_and_output_rows': old['dense_recipe']['input_factor_lower_bound'],
            'full_properties_per_pair': counts['properties_per_pair'],
            'even_degenerate_pair_queries': counts['pair_queries_min'],
            'portable_bytes': old['source_bytes'],
            'conv_operators': old['conv']['unsupported_operators'],
        },
        'wide_rows': {'local_reference_factor_limit': 8, 'mathematical_rule': 'exact coefficient L1 residual, not corner enumeration',
                      'actual_trained_row_widths': None, 'actual_wide_row_failure_observed': False},
        'row_check_repetition': repeats,
        'remaining_live_arrays': ['global objective/residual/argmin N by K', 'local sparse matrices and transposes',
                                  'local duals and best duals M by K', 'all live pairs before request preparation',
                                  'four parsed snapshots per pair plus copied local row dictionaries'],
        'wrapper_integration': {'new_schema': block_schema, 'old_portable_has_block_checker': False,
                                'new_block_hard_supervision_established': False},
        'conv': old['conv'], 'actual_peak_memory': None, 'actual_end_to_end_seconds': None,
        'new_model_loads': 0, 'new_input_selections': 0, 'new_propagations': 0, 'new_solves': 0, 'cuda_calls': 0,
        'source_or_output_certificate': False, 'all_six_goal_gates_remain_open': True,
        'decision': 'separately design one-pass sparse row enclosure/checking with exact residual accumulation; no cap bump or real admission',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path); group.add_argument('--check', type=Path)
    args = parser.parse_args(); report = derive()
    if args.output:
        if not args.output.resolve().is_relative_to(ROOT / 'docs'):
            raise ValueError('repository report only')
        with args.output.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    elif json.loads(args.check.read_bytes()) != report:
        raise ValueError('block capacity report differs')
    print({'status': report['status'], 'model_loads': 0, 'solves': 0})
