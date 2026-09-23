"""Saved-only, separate-process terminal/trace reconstruction; no solver calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from audit_conv_f0_timing import check_trace
from audit_metamoe_csr_paired_r4 import check_receipt, check_candidate, finite
from metamoe_expert_diagnostic import validate, command_for, identity, PROTOCOL
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def diagnose(trace):
    spans = trace['spans']
    by_id = {s['id']: s for s in spans}

    def inside(span, parent):
        while span['parent'] is not None:
            if span['parent'] == parent['id']:
                return True
            span = by_id[span['parent']]
        return False

    evaluations = [s for s in spans if s['name'] == 'HZSolver.evaluate_spec']
    details = []
    for ev in evaluations:
        queries = []
        for s in spans:
            if not s['name'].endswith('._solve_hz_feasibility') or not inside(s, ev):
                continue
            natives = [v for v in spans if v['name'].endswith('.milp') and inside(v, s)]
            checks = [v for v in spans if v['name'].endswith('._valid_milp_point') and inside(v, s)]
            result = (s.get('result') or {}).get('status')
            flags = []
            if s['right_censored']:
                flags.append('OUTER_TERMINATED_QUERY')
            if any(v.get('end_kind') == 'RAISE' for v in natives):
                flags.append('NATIVE_EXCEPTION')
            if any((v.get('result') or {}).get('status') == 1 for v in natives):
                flags.append('NATIVE_LIMIT_REPORTED')
            if any((v.get('result') or {}).get('status') == 4 for v in natives):
                flags.append('NATIVE_OTHER_OR_NUMERICAL_FAILURE')
            if any((v.get('result') or {}).get('value') is False for v in checks):
                flags.append('INCUMBENT_REJECTED_BY_EXISTING_FLOAT_POLICY')
            if not natives and s['arguments'].get('remaining_at_entry_seconds', 1) <= 0:
                flags.append('LOCAL_DEADLINE_EXHAUSTED_WITHOUT_NATIVE_CALL')
            queries.append({'span_id': s['id'], 'arguments': s['arguments'],
                'elapsed_seconds': s.get('seconds'), 'right_censored': s['right_censored'],
                'feasibility_status': result, 'flags': flags,
                'native': [{'span_id': v['id'], 'seconds': v.get('seconds'),
                            'arguments': v['arguments'], 'result': v.get('result'),
                            'exception': v.get('exception')} for v in natives],
                'incumbent_checks': [v.get('result') for v in checks]})
        details.append({'span_id': ev['id'], 'arguments': ev['arguments'],
            'seconds': ev.get('seconds'), 'result': ev.get('result'),
            'queries': queries,
            'property_query_count': sum(q['arguments'].get('extra_rows', 0) > 0 for q in queries)})
    return {'expert_evaluations': details,
        'native_status_counts_all_phases': dict(Counter(
            str((s.get('result') or {}).get('status')) for s in spans if s['name'].endswith('.milp'))),
        'limits': 'Flags are observations, not exclusive causes. A native limit WITH a valid incumbent can still establish feasibility under the existing policy. No LP infeasibility/unsafety claim follows from UNKNOWN. Logging perturbs this new run; historical R4 cause is not reconstructed.'}


def audit(path):
    started = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg)
    root, digest = Path(cfg['output_root']), sha256(path)
    launch = json.loads((root/'launch.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if (launch['config_sha256'] != digest or launch['protocol'] != PROTOCOL or
            launch['automatic_followup'] is not False or summary['config_sha256'] != digest or
            [(r['id'], r['arm']) for r in summary['rows']] != [('mnist_0', 'act')]):
        raise ValueError('launch/roster binding')
    row = summary['rows'][0]
    folder = root/'mnist_0_act'
    receipt = json.loads((folder/'receipt.json').read_text())
    if json.loads((folder/'terminal.json').read_text()) != row or sha256(folder/'receipt.json') != row['receipt_sha256']:
        raise ValueError('terminal/receipt identity')
    check_receipt(receipt, row, cfg, command_for(cfg, path))
    for stream in ('stdout', 'stderr'):
        if sha256(folder/f'{stream}.txt') != receipt[f'{stream}_sha256']:
            raise ValueError('stream identity')
    candidate = folder/'result.json'
    if candidate.exists() != bool(row['result_sha256']):
        raise ValueError('candidate presence')
    result, malformed = None, False
    if candidate.exists():
        if sha256(candidate) != row['result_sha256']:
            raise ValueError('candidate hash')
        try:
            result = json.loads(candidate.read_text())
            keys = {'status', 'config_sha256', 'request_id', 'arm', 'label', 'tensor_file_sha256', 'worker_seconds'}
            if not isinstance(result, dict) or not keys <= result.keys() or not isinstance(result['status'], str):
                raise ValueError('incomplete candidate')
        except ValueError:
            result, malformed = None, True
    if bool(row['result_parse_error']) != malformed:
        raise ValueError('malformed candidate accounting')
    expected = (result['status'] if result else 'ERROR') if receipt['status'] == 'COMPLETED' else receipt['status']
    if row['status'] != expected:
        raise ValueError('outer termination precedence')
    if result is not None:
        request = cfg['requests'][0]
        if (result['config_sha256'] != digest or result['request_id'] != 'mnist_0' or result['arm'] != 'act' or
                result['label'] != request['label'] or result['tensor_file_sha256'] != sha256(request['tensor_file']) or
                result['worker_seconds'] > row['seconds']):
            raise ValueError('candidate request/cost binding')
        check_candidate(result, 'act')
    trace_path = folder/'trace.jsonl'
    if trace_path.exists() != bool(row['trace_sha256']):
        raise ValueError('trace presence')
    trace = None
    if trace_path.exists():
        if sha256(trace_path) != row['trace_sha256']:
            raise ValueError('trace identity')
        trace = check_trace(trace_path, row['seconds'], identity(path), killed=receipt['status'] != 'COMPLETED')
    elif receipt['status'] == 'COMPLETED':
        raise ValueError('completed without trace')
    cost = json.loads((root/'batch_cost.json').read_text())
    if (cost['config_sha256'] != digest or cost['charged_request_seconds'] != row['seconds'] or
            not finite(cost['batch_wall_through_summary_seconds']) or
            cost['batch_wall_through_summary_seconds'] < receipt['total_with_postflight_seconds']):
        raise ValueError('batch cost')
    return {'audit': 'PASS', 'config_sha256': digest, 'row': row, 'receipt': receipt,
        'result': result, 'batch_cost': cost, 'trace': trace,
        'diagnosis': diagnose(trace) if trace else None,
        'files': {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                  for p in sorted(root.rglob('*')) if p.is_file()},
        'opens_formal_cohort': False, 'independent_safe_reproof': False,
        'separate_audit_seconds': time.monotonic()-started}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    write(a.output, value)
    print(value['audit'], value['row']['status'])
