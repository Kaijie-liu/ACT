"""Saved-only current-assignment compatibility + identical-query accounting.

No checkpoint/forward/LP/MILP calls. Does NOT rewrite the sealed R1 result.
Only one fixed free-factor seed; checked existence is not output safety.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.current_assignment import (
    AssignmentScope, ProposalUnavailable, propose_current_assignment, check_current_assignment,
)
from act.back_end.solver.isolated_feasibility import save_npz
from audit_metamoe_protected_smoke_r1 import audit, arrays, csr
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/recent_moe/metamoe_protected_smoke_r1.json'
ARCHIVE = ROOT/'docs/metamoe_protected_archive_20260923_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_r1')


def query_groups(rows, model_identity):
    """Exact stored-row identity only; never tolerance-based grouping.

    This groups obligations for ANALYSIS; it does not copy UNKNOWN into a proof
    or change the protected scheduler. Explicit byte comparison supplements hash.
    """
    if [r['row'] for r in rows] != list(range(len(rows))) or not model_identity:
        raise ValueError('ordered complete rows and model identity required')
    groups, objects = [], []
    for row in rows:
        z = row['arrays']
        if set(z) != {'data', 'indices', 'indptr', 'shape', 'lb', 'ub'}:
            raise ValueError('query fields')
        A = csr(z)
        if (A.shape[0] != 1 or not A.has_canonical_format or not np.isfinite(A.data).all() or
                z['lb'].shape != (1,) or z['ub'].shape != (1,) or
                np.isnan(z['lb']).any() or np.isnan(z['ub']).any()):
            raise ValueError('unsupported query row')
        payload = []
        for name in sorted(z):
            a = np.ascontiguousarray(z[name])
            payload.append((name, a.dtype.str, a.shape, a.tobytes()))
        key = hashlib.sha256(model_identity.encode()+repr(payload).encode()).hexdigest()
        found = None
        for k, previous in enumerate(objects):
            if previous == payload:
                found = k
                break
        if found is None:
            found = len(groups)
            groups.append({'identity': key, 'rows': []})
            objects.append(payload)
        groups[found]['rows'].append(row['row'])
    return groups


def run():
    started = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError('saved-only diagnostic is not retryable/overwritable')
    cfg = json.loads(CONFIG.read_text())
    archived = json.loads(ARCHIVE.read_text())
    with patch.object(sh, 'milp', side_effect=AssertionError('NO new native query authorized')):
        reviewed = audit(CONFIG)
    if (reviewed['files'] != archived['files'] or reviewed['row'] != archived['row'] or
            reviewed['evaluations'] != archived['evaluations']):
        raise ValueError('sealed source/review mismatch')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT/'launch.json', {'mode': 'SAVED_ONLY_NO_NEW_SOLVING', 'config_sha256': sha256(CONFIG),
        'archive_sha256': sha256(ARCHIVE), 'fixed_seed': 'all free factors zero',
        'proposal_and_check_budget_seconds': 3., 'core_acceptance_changed': False,
        'integration_or_full_request_rerun': False, 'source_sha256': sha256(Path(__file__))})
    folder = Path(cfg['output_root'])/'mnist_0_act/protected/evaluation_000'
    z = arrays(folder/'base_model.npz')
    nc = int(np.count_nonzero(z['integrality'] == 0))
    nb = int(np.count_nonzero(z['integrality'] == 1))
    model = sh._HZMILP(z['value_center'], csr(z, 'value_'), csr(z), z['row_lb'], z['row_ub'],
                      z['var_lb'], z['var_ub'], z['integrality'], nc, nb)
    # Fresh diagnostic invocation, never a reused historical solver incumbent.
    scope = AssignmentScope(sha256(CONFIG), archived['result']['tensor_file_sha256'], uuid.uuid4().hex)
    candidate_start = time.monotonic()
    deadline = candidate_start+3.
    proposal, point, report = None, None, {'accepted': False, 'reason': 'not_proposed'}
    with patch.object(sh, 'milp', side_effect=AssertionError('NO new native query authorized')):
        try:
            proposal = propose_current_assignment(model, scope, deadline)
            point, report = check_current_assignment(model, proposal, scope, deadline)
        except ProposalUnavailable as exc:
            report['reason'] = str(exc)
    charged = time.monotonic()-candidate_start
    if proposal is not None:
        save_npz(OUTPUT/'proposal.npz', point=proposal.point)
    write(OUTPUT/'assignment.json', {'scope': asdict(scope), 'check': report,
        'model_sha256': proposal.model_sha256 if proposal else None,
        'base_npz_sha256': sha256(folder/'base_model.npz'),
        'proposal_npz_sha256': sha256(OUTPUT/'proposal.npz') if proposal else None,
        'free_prefix': proposal.free_prefix if proposal else None,
        'relu_rows': proposal.relu_rows if proposal else None,
        'construction_seconds': proposal.construction_seconds if proposal else None,
        'proposal_and_check_seconds': charged, 'deadline_seconds': 3.,
        'point_is_full_model_witness': False, 'old_result_upgraded': False,
        'postcheck_publication_outside_candidate_clock': True})
    records = []
    for qdir in sorted(folder.glob('query_*')):
        ret = json.loads((qdir/'return.json').read_text())
        if ret['scope']['phase'] != 'expanded':
            continue
        entry = json.loads((qdir/'native_started.json').read_text())
        raw = ret['native_result']
        records.append({'row': ret['scope']['row'], 'arrays': arrays(qdir/'extra.npz'),
            'query_sha256': sha256(qdir/'extra.npz'), 'returned_status': ret['returned_status'],
            'local_terminal': ret['terminal'], 'return_seconds': ret['return_elapsed_seconds'],
            'pre_entry_seconds': entry['started_monotonic']-ret['started_monotonic'],
            'entry_to_deadline_window_seconds': ret['deadline_monotonic']-entry['started_monotonic'],
            'completed_native_seconds': raw['native_seconds'] if raw else None,
            'native_objective_or_bound_record': raw is not None})
    groups = query_groups(records, reviewed['evaluations'][0]['matrix_sha256'])
    for g in groups:
        members = [records[i] for i in g['rows']]
        g['statuses'] = dict(Counter(r['returned_status'] for r in members))
        g['observed_return_seconds'] = sum(r['return_seconds'] for r in members)
        g['repeat_return_seconds_after_first'] = sum(r['return_seconds'] for r in members[1:])
    checked_point_values = None
    if point is not None:
        p = arrays(folder/'properties.npz')
        values = p['C'] @ (model.value_center+model.value_matrix @ point)
        checked_point_values = {'property_values': values.tolist(), 'thresholds': p['thresholds'].tolist(),
            'violating_expanded_rows_at_one_point': np.flatnonzero(values >= p['thresholds']-1e-7).tolist(),
            'not_a_robust_bound_or_source_replay': True}
    summary = {'mode': 'SAVED_ONLY_NO_NEW_SOLVING', 'original_status': reviewed['row']['status'],
        'solver_calls': 0, 'full_request_executed': False, 'scope': asdict(scope),
        'assignment': json.loads((OUTPUT/'assignment.json').read_text()),
        'source_config_sha256': sha256(CONFIG), 'source_archive_sha256': sha256(ARCHIVE),
        'source_raw_files': reviewed['files'],
        'output_properties': len(records), 'distinct_stored_violation_queries': len(groups),
        'groups': groups, 'rows': [{k:v for k,v in r.items() if k != 'arrays'} for r in records],
        'one_point_observations': checked_point_values,
        'partial_native_phase_known': False, 'weak_relaxation_established': False,
        'historical_unknowns_relabelled': False,
        'analysis_through_summary_construction_seconds': time.monotonic()-started,
        'candidate_cost_boundary': 'Model load/audit and candidate publication separate; not a new end-to-end request speed measurement.'}
    write(OUTPUT/'summary.json', summary)
    print('BASE_PROPOSAL_CHECK', report['accepted'], report['reason'])
    print('properties', len(records), 'distinct_queries', len(groups), 'new_solver_calls', 0)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
