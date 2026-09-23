"""Saved-only deadline, query-coverage and matrix consistency audit.

No solver calls. Float point validation repeats the original policy, not an
independent exact certificate or source-to-HZ proof.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from act.back_end.solver import solver_hz as sh
from audit_conv_f0_timing import check_trace
from audit_metamoe_csr_paired_r4 import check_receipt, check_candidate, finite
from metamoe_protected_smoke_r1 import validate, command_for, identity, PROTOCOL
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def csr(z, prefix=''):
    return sp.csr_matrix((z[prefix+'data'], z[prefix+'indices'], z[prefix+'indptr']), shape=tuple(z[prefix+'shape']))


def evaluation(folder, expected_properties, partial=False):
    plan = read(folder/'plan.json')
    start, end = plan['started_monotonic'], plan['deadline_monotonic']
    if (plan['schema'] != 'protected-hz-v1' or plan['base_fraction'] != .1 or
            not finite(start) or not finite(end) or not 0 < plan['total_seconds'] <= 30 or
            end != start+plan['total_seconds'] or
            plan['base_deadline_monotonic'] != start+.1*plan['total_seconds'] or
            plan['numerical_policy'] != sh.hz_numerical_policy_manifest() or plan['tolerance'] != 1e-7):
        raise ValueError('evaluation budget/numerical contract')
    if not (folder/'result.json').exists():
        if not partial:
            raise ValueError('completed request missing evaluation terminal')
        return {'status': 'PARTIAL', 'queries': len(list(folder.glob('query_*'))), 'accepted': False}
    terminal = read(folder/'result.json')
    meta = terminal['metadata']
    status = 'unknown' if (folder/'late_result_rejected.json').exists() else terminal['status']
    if terminal['finished_monotonic'] >= end and status != 'unknown':
        raise ValueError('late expert positive/violation accepted')
    props = meta['properties']
    M = meta['required_properties']
    if M not in (0, expected_properties) or [p['row'] for p in props] != list(range(M)):
        raise ValueError('property coverage/identity')
    if status == 'certified' and (M != expected_properties or meta['base_status'] != 'feasible' or
            any(p['status'] != 'infeasible' for p in props)):
        raise ValueError('incomplete positive obligations/nonvacuity')
    model, coeff, const, thresholds = None, None, None, None
    if (folder/'base_model.npz').exists():
        z = arrays(folder/'base_model.npz')
        integer = z['integrality']
        n_cont, n_bin = int(np.count_nonzero(integer == 0)), int(np.count_nonzero(integer == 1))
        if not np.array_equal(integer, np.array([0]*n_cont+[1]*n_bin)):
            raise ValueError('integer factors relaxed or reordered')
        model = sh._HZMILP(z['value_center'], csr(z, 'value_'), csr(z), z['row_lb'], z['row_ub'],
            z['var_lb'], z['var_ub'], integer, n_cont, n_bin)
        p = arrays(folder/'properties.npz')
        pid = read(folder/'property_identity.json')
        if (pid['properties_sha256'] != sha256(folder/'properties.npz') or pid['count'] != M or
                meta['matrix_sha256'] != sha256(folder/'base_model.npz')):
            raise ValueError('matrix/property identity')
        coeff = (sp.csr_matrix(p['C']) @ model.value_matrix).tocsr()
        const, thresholds = p['C'] @ model.value_center, p['thresholds']
        if not np.array_equal(const, p['constants']) or coeff.shape[0] != M:
            raise ValueError('property projection mismatch')
    folders = sorted(folder.glob('query_*'))
    if len(folders) != len(meta['queries']):
        raise ValueError('missing/extra query')
    query_info = []
    previous_row, base_count = -1, 0
    for i, qdir in enumerate(folders):
        ret = read(qdir/'return.json')
        if (qdir/'late_return_rejected.json').exists():
            ret.update(returned_status='unknown', accepted_after_receipt=False, terminal='RETURN_PUBLICATION_DEADLINE')
        if qdir.name != f'query_{i:03d}' or meta['queries'][i] != ret:
            raise ValueError('query return/ledger mismatch')
        receipt = read(qdir/'receipt.json')
        for key in ('token', 'scope', 'deadline_monotonic', 'artifact_hashes'):
            if receipt[key] != ret[key]:
                raise ValueError('query receipt binding')
        observed = {p.name: sha256(p) for p in qdir.iterdir()
                    if p.is_file() and p.name not in ('receipt.json', 'return.json', 'late_return_rejected.json')}
        if ret['artifact_hashes'] != observed:
            raise ValueError('query artifact integrity')
        phase = ret['scope']['phase']
        if (not finite(ret['deadline_monotonic']) or ret['deadline_monotonic'] > end or
                not all(finite(ret[k]) for k in ('return_elapsed_seconds', 'elapsed_through_cleanup_seconds', 'cleanup_seconds', 'started_monotonic'))):
            raise ValueError('query outside expert budget')
        if phase == 'base':
            base_count += 1
            if i != 0 or ret['deadline_monotonic'] != plan['base_deadline_monotonic']:
                raise ValueError('base monopolized property allocation')
        elif phase in ('expanded', 'contracted'):
            row = ret['scope']['row']
            if not 0 <= row < M or (phase == 'expanded' and row != previous_row+1) or (phase == 'contracted' and row != previous_row):
                raise ValueError('changed/duplicate property order')
            previous_row = row
        else:
            raise ValueError('unknown query phase')
        if model is not None:
            row = ret['scope'].get('row')
            E = sp.csr_matrix((0, model.n_var)) if phase == 'base' else coeff[row]
            lo = np.zeros(0) if phase == 'base' else np.array([
                thresholds[row]+(-1 if phase == 'expanded' else 1)*plan['tolerance']-const[row]])
            hi = np.zeros(0) if phase == 'base' else np.array([np.inf])
            if (qdir/'extra.npz').exists():
                e = arrays(qdir/'extra.npz')
                if csr(e).shape != E.shape or (csr(e) != E).nnz or not np.array_equal(e['lb'], lo) or not np.array_equal(e['ub'], hi):
                    raise ValueError('query differs from original expanded/contracted property')
                req = read(qdir/'request.json')
                if (req['model_sha256'] != meta['matrix_sha256'] or req['query_sha256'] != sha256(qdir/'extra.npz') or
                        req['deadline_monotonic'] != ret['deadline_monotonic'] or req['token'] != ret['token'] or req['scope'] != ret['scope']):
                    raise ValueError('request matrix/scope/deadline binding')
            if ret['returned_status'] != 'unknown':
                A, lb, ub = sh._combined_constraints(model, E, lo, hi)
                if ret['terminal'] == 'CONSTANT':
                    if model.n_var != 0:
                        raise ValueError('false constant query')
                    valid = sh._valid_milp_point(model, np.zeros(0), A, lb, ub, plan['tolerance'])
                    inferred = 'feasible' if valid else 'infeasible'
                else:
                    raw = read(qdir/'native_result.json')
                    if raw != ret['native_result'] or raw['finished_monotonic'] >= ret['deadline_monotonic']:
                        raise ValueError('native return binding/deadline')
                    valid = False
                    if raw['candidate_sha256'] is not None:
                        if raw['candidate_sha256'] != sha256(qdir/'candidate.npz'):
                            raise ValueError('candidate hash')
                        valid = sh._valid_milp_point(model, arrays(qdir/'candidate.npz')['x'], A, lb, ub, plan['tolerance'])
                    inferred = 'feasible' if valid else ('infeasible' if raw['status'] == 2 else 'unknown')
                if inferred != ret['returned_status']:
                    raise ValueError('original feasibility gate differs')
        elif ret['returned_status'] != 'unknown':
            raise ValueError('conclusion without bound matrix')
        if ret['native_started'] != (qdir/'native_started.json').exists():
            raise ValueError('native count')
        query_info.append({k: ret[k] for k in ('scope', 'terminal', 'returned_status', 'native_started',
            'elapsed_through_cleanup_seconds', 'cleanup_seconds', 'return_elapsed_seconds', 'incumbent_valid')})
    if base_count != (1 if folders else 0):
        raise ValueError('base query multiplicity')
    for prop in props:
        if 'expanded_query' in prop:
            q = meta['queries'][prop['expanded_query']]
            if q['scope'] != {'phase': 'expanded', 'row': prop['row']} or q['returned_status'] != prop['status']:
                raise ValueError('property result misbound')
        elif prop['status'] != 'NOT_STARTED':
            raise ValueError('property conclusion without query')
    if folders and meta['base_status'] != meta['queries'][0]['returned_status']:
        raise ValueError('base result misbound')
    return {'status': status, 'reason': meta['reason'], 'required_properties': M, 'base_status': meta['base_status'],
        'property_status_counts': dict(Counter(p['status'] for p in props)), 'properties': props,
        'queries': query_info, 'native_property_calls': sum(q['native_started'] for q in query_info if q['scope']['phase'] != 'base'),
        'completed_property_native_returns': sum(q['scope']['phase'] != 'base' and q['native_result'] is not None for q in meta['queries']),
        'expert_elapsed_before_publication': meta['elapsed_before_publication'],
        'allocation_seconds': plan['total_seconds'], 'matrix_sha256': meta['matrix_sha256']}


def audit(path):
    began = time.monotonic()
    cfg = read(path)
    validate(cfg)
    root, h = Path(cfg['output_root']), sha256(path)
    launch, summary = read(root/'launch.json'), read(root/'summary.json')
    if (launch['config_sha256'] != h or launch['protocol'] != PROTOCOL or launch['automatic_followup'] is not False or
            summary['config_sha256'] != h or [(r['id'], r['arm']) for r in summary['rows']] != [('mnist_0', 'act')]):
        raise ValueError('launch/denominator binding')
    row = summary['rows'][0]
    folder = root/'mnist_0_act'
    receipt = read(folder/'receipt.json')
    if read(folder/'terminal.json') != row or sha256(folder/'receipt.json') != row['receipt_sha256']:
        raise ValueError('outer receipt/terminal binding')
    check_receipt(receipt, row, cfg, command_for(cfg, path))
    for stream in ('stdout', 'stderr'):
        if sha256(folder/f'{stream}.txt') != receipt[stream+'_sha256']:
            raise ValueError('outer stream hash')
    from metamoe_csr_paired_r4 import collect_terminal
    expected = collect_terminal(folder, receipt)
    if any(row[k] != v for k, v in expected.items()):
        raise ValueError('outer precedence/candidate hash')
    result = read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
    if result:
        req = cfg['requests'][0]
        if (result['config_sha256'] != h or result['request_id'] != 'mnist_0' or result['arm'] != 'act' or
                result['label'] != req['label'] or result['tensor_file_sha256'] != sha256(req['tensor_file']) or
                result['worker_seconds'] > row['seconds']):
            raise ValueError('request identity/cost')
        check_candidate(result, 'act')
    artifacts = {str(p.relative_to(folder)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted((folder/'protected').rglob('*')) if p.is_file()}
    if artifacts != row['protected_artifacts']:
        raise ValueError('protected evidence inventory')
    killed = receipt['status'] != 'COMPLETED'
    trace_path = folder/'trace.jsonl'
    trace = None
    if trace_path.exists():
        if sha256(trace_path) != row['trace_sha256']:
            raise ValueError('trace hash')
        trace = check_trace(trace_path, row['seconds'], identity(path), killed=killed)
    elif not killed:
        raise ValueError('missing completed trace')
    reviews = [evaluation(p, 19, partial=killed) for p in sorted((folder/'protected').glob('evaluation_*'))]
    if result and receipt['status'] == 'COMPLETED' and 'expert_statuses' in result:
        if [r['status'] for r in reviews] != list(result['expert_statuses'].values()):
            raise ValueError('expert terminal/aggregate mismatch')
    cost = read(root/'batch_cost.json')
    if (cost['config_sha256'] != h or cost['charged_request_seconds'] != row['seconds'] or
            not finite(row['postflight_inventory_seconds']) or
            cost['batch_wall_through_summary_seconds'] < receipt['total_with_postflight_seconds']+row['postflight_inventory_seconds']):
        raise ValueError('cost accounting')
    return {'audit': 'PASS', 'config_sha256': h, 'row': row, 'receipt': receipt,
        'result': {k:v for k,v in (result or {}).items() if k != 'sparse_resource_events'},
        'evaluations': reviews, 'batch_cost': cost,
        'trace': {k:v for k,v in (trace or {}).items() if k != 'spans'},
        'files': {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                  for p in sorted(root.rglob('*')) if p.is_file()},
        'opens_formal_cohort': False, 'independent_safe_reproof': False,
        'limits': 'Matrix/row consistency and original float feasibility checks, not exact reproof. UNKNOWN does not establish LP weakness. Partial evidence cannot upgrade outer failure.',
        'separate_audit_seconds': time.monotonic()-began}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    write(a.output, value)
    print(value['audit'], value['row']['status'])
