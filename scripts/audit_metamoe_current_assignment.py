"""Saved-only independent point/identity audit; no construction or model run.

The checker uses scalar fsum over stored CSR rows, not the proposal algorithm.
This is still float feasibility policy, not rational/source-complete proof.
Original forward arrays are hash-bound records, not independently re-executed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from act.back_end.solver import solver_hz as sh
from audit_metamoe_protected_smoke_r1 import audit as audit_parent, arrays, csr
from metamoe_assignment_replay import CONFIG, DIAG, OUTPUT, PARENT, ROOT, validate
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def scalar_rows(matrix, point):
    require(matrix.shape[1] == point.size and np.isfinite(point).all()
            and np.isfinite(matrix.data).all(), 'nonfinite/incorrect factor map')
    return np.array([math.fsum(float(matrix.data[j])*float(point[matrix.indices[j]])
        for j in range(matrix.indptr[i], matrix.indptr[i+1])) for i in range(matrix.shape[0])])


def check_point(z, point, tol=1e-7):
    n = z['integrality'].size
    require(point.shape == (n,) and np.isfinite(point).all(), 'incomplete/nonfinite point')
    require(np.all((z['integrality'] == 0) | (z['integrality'] == 1)), 'integrality schema')
    require(np.all(point >= z['var_lb']-tol) and np.all(point <= z['var_ub']+tol), 'factor bounds')
    integers = point[z['integrality'] == 1]
    require(np.all(np.abs(integers-np.rint(integers)) <= tol), 'binary residual')
    values = scalar_rows(csr(z), point)
    require(np.all(values >= z['row_lb']-tol) and np.all(values <= z['row_ub']+tol), 'base row residual')
    eq = np.isfinite(z['row_lb']) & (z['row_lb'] == z['row_ub'])
    return {'variables': n, 'rows': len(values), 'tolerance': tol,
            'max_equality_residual_fsum': float(np.max(np.abs(values[eq]-z['row_lb'][eq]), initial=0.))}


def recover_and_compare(z, point, mapping, replay):
    nc = int(np.count_nonzero(z['integrality'] == 0))
    gc, gb = csr(mapping, 'Gc_'), csr(mapping, 'Gb_')
    require(gc.shape[1] <= nc and gb.shape[1] <= point.size-nc, 'input map dimensions')
    recovered = (mapping['center'].reshape(-1) + scalar_rows(gc, point[:gc.shape[1]])
                 + scalar_rows(gb, 2*point[nc:nc+gb.shape[1]]-1))
    require(np.array_equal(recovered, replay['point'].reshape(-1)), 'input recovery mismatch')
    require(np.array_equal(replay['point'], replay['center']), 'fixed seed not saved center')
    require(np.all(replay['point'] >= replay['lower']) and np.all(replay['point'] <= replay['upper']),
            'outside request')
    represented = z['value_center'] + scalar_rows(csr(z, 'value_'), point)
    require(np.allclose(represented, replay['represented'], atol=1e-12, rtol=0), 'output map mismatch')
    require(np.array_equal(replay['source'], replay['padded']), 'source/padded forward differs')
    require(all(np.isfinite(a).all() for a in replay.values()), 'nonfinite replay')
    margins = replay['rows'] @ replay['source']
    abstract = replay['rows'] @ represented
    source_bad = np.flatnonzero(margins < replay['thresholds']).tolist()
    represented_bad = np.flatnonzero(abstract < replay['thresholds']).tolist()
    return {'source_violated_rows': source_bad, 'represented_violated_rows': represented_bad,
            'source_minimum_margin': float(margins.min()),
            'max_abs_output_difference': float(np.max(np.abs(replay['source']-represented))),
            'input_n_cont': gc.shape[1], 'input_n_bin': gb.shape[1],
            'status': ('REPLAYED_VIOLATION' if source_bad else
                       'HZ_SOURCE_POINT_MISMATCH' if represented_bad else 'NO_VIOLATION_AT_POINT'),
            'full_model_violation': bool(source_bad)}


def check_terminal(cfg, launch, terminal, receipt, result):
    require(launch['config_sha256'] == terminal['config_sha256'] == sha256(CONFIG), 'config identity')
    require(receipt == terminal['receipt'] and result == terminal['result'], 'terminal/worker mismatch')
    require(receipt['status'] == terminal['outer_status'] == 'COMPLETED' and receipt['exit_code'] == 0
            and receipt['error'] is None, 'outer execution not complete')
    require(terminal['status'] == result['status'], 'late/contradictory result')
    require(receipt['command'] == [cfg['python'], str(ROOT/'scripts/metamoe_assignment_replay.py'), '--worker'],
            'worker invocation')
    elapsed = receipt['execution_including_preflight_seconds']
    require(receipt['deadline_seconds'] == cfg['seconds'] and 0 < elapsed < cfg['seconds']
            and 0 < result['worker_through_result_seconds'] <= elapsed
            and elapsed <= receipt['total_with_postflight_seconds'] <= terminal['elapsed_through_terminal_seconds'],
            'deadline/cost accounting')
    require(receipt['group_rss_limit_bytes'] == cfg['group_rss_limit_bytes'] and
            receipt['peak_sampled_group_rss_bytes'] <= cfg['group_rss_limit_bytes'], 'resource receipt')
    require(not terminal['historical_result_relabelled'] and not terminal['opens_formal_cohort'], 'sealed boundary')


def inventory(root):
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def audit():
    start = time.monotonic()
    cfg = read(CONFIG)
    parent = validate(cfg)
    with patch.object(sh, 'milp', side_effect=AssertionError('no audit solving')):
        original = audit_parent(PARENT)
    old = read(ROOT/'docs/metamoe_protected_archive_20260923_r1.json')
    require(all(original[k] == old[k] for k in ('files','row','evaluations')), 'sealed source changed')
    folder = Path(parent['output_root'])/'mnist_0_act/protected/evaluation_000'
    z, props = arrays(folder/'base_model.npz'), arrays(folder/'properties.npz')
    proposal = arrays(DIAG/'proposal.npz')['point']
    meta, summary = read(DIAG/'assignment.json'), read(DIAG/'summary.json')
    require(meta == summary['assignment'] and meta['proposal_npz_sha256'] == sha256(DIAG/'proposal.npz')
            and meta['base_npz_sha256'] == sha256(folder/'base_model.npz'), 'assignment identity')
    require(meta['scope']['request_sha256'] == sha256(PARENT) and
            meta['scope']['input_sha256'] == original['result']['tensor_file_sha256'] and
            bool(meta['scope']['evaluation_nonce']) and summary['scope'] == meta['scope'], 'request scope')
    require(meta['check']['accepted'] and summary['solver_calls'] == 0 and
            not summary['full_request_executed'] and 0 < meta['proposal_and_check_seconds'] < 3., 'proposal contract')
    base_check = check_point(z, proposal)
    row_records, groups = [], []
    for qdir in sorted(folder.glob('query_*')):
        ret = read(qdir/'return.json')
        if ret['scope']['phase'] != 'expanded':
            continue
        idx = ret['scope']['row']
        require(idx == len(row_records), 'property row omitted/reordered')
        q = arrays(qdir/'extra.npz')
        value = float(scalar_rows(csr(q), proposal)[0])
        feasible = bool(q['lb'][0]-1e-7 <= value <= q['ub'][0]+1e-7)
        row_records.append({'row': idx, 'historical_status': ret['returned_status'],
            'extra_row_feasible_at_checked_point': feasible, 'extra_row_value': value,
            'extra_lower_bound': float(q['lb'][0]), 'return_seconds': ret['return_elapsed_seconds']})
        # Independent byte-equality grouping; raw CSR order and types are bound.
        key = tuple((k, q[k].dtype.str, q[k].shape, q[k].tobytes()) for k in sorted(q))
        for prior_key, members in groups:
            if prior_key == key:
                members.append(idx)
                break
        else:
            groups.append((key, [idx]))
    require(len(row_records) == 19 and [g[1] for g in groups] == [g['rows'] for g in summary['groups']],
            'query coverage/grouping')
    wr = OUTPUT/'worker'
    result, terminal, receipt = read(wr/'result.json'), read(OUTPUT/'terminal.json'), read(wr/'receipt.json')
    check_terminal(cfg, read(OUTPUT/'launch.json'), terminal, receipt, result)
    for stream in ('stdout','stderr'):
        require(sha256(wr/f'{stream}.txt') == receipt[stream+'_sha256'], 'stream identity')
    for name in ('replay','input_map'):
        require(sha256(wr/f'{name}.npz') == result[name+'_sha256'], 'replay artifact identity')
    require(result['model_sha256'] == result['saved_model_sha256'] == meta['model_sha256']
            and result['fresh_full_matrix_check']['accepted'], 'fresh model check')
    replay = arrays(wr/'replay.npz')
    require(np.array_equal(-replay['rows'], props['C']) and
            np.array_equal(-replay['thresholds'], props['thresholds']), 'property identity')
    tensors = arrays(Path(parent['requests'][0]['tensor_file']))
    require(all(np.array_equal(replay[k], tensors[k]) for k in ('center','lower','upper')), 'physical request')
    comparison = recover_and_compare(z, proposal, arrays(wr/'input_map.npz'), replay)
    require(comparison['status'] == result['status'] and comparison['full_model_violation'] == result['full_model_violation']
            and comparison['source_violated_rows'] == result['source_violated_rows']
            and comparison['represented_violated_rows'] == result['hz_violated_rows'], 'point conclusion')
    require(np.allclose(replay['rows'] @ replay['source'], result['original_margins'], atol=1e-12, rtol=0)
            and np.allclose(replay['rows'] @ replay['represented'], result['represented_margins'], atol=1e-12, rtol=0),
            'recorded margins')
    r1 = DIAG.with_name('metamoe_current_assignment_20260923_r1')
    require(not (r1/'summary.json').exists() and not (r1/'proposal.npz').exists()
            and read(r1/'assignment.json')['check']['accepted'] is False, 'R1 failed attempt lost')
    unknown = [r for r in row_records if r['historical_status'] == 'unknown']
    return {'audit':'PASS', 'issues':0, 'status':comparison['status'], 'config_sha256':sha256(CONFIG),
        'base_check':base_check, 'point_comparison':comparison, 'properties':row_records,
        'duplicate_groups':[g[1] for g in groups], 'distinct_queries':len(groups),
        'unknown_properties':len(unknown), 'unknown_point_feasible':sum(r['extra_row_feasible_at_checked_point'] for r in unknown),
        'identical_first10_seconds':sum(r['return_seconds'] for r in row_records[:10]),
        'identical_first10_repeat_seconds':sum(r['return_seconds'] for r in row_records[1:10]),
        'assignment_timing':{k:meta[k] for k in ('construction_seconds','proposal_and_check_seconds','deadline_seconds')},
        'replay_receipt':receipt, 'replay_total_seconds':terminal['elapsed_through_terminal_seconds'],
        'fresh_model_sha256':result['model_sha256'], 'native_queries':0, 'historical_relabelled':False,
        'original_forward_reexecuted_by_auditor':False, 'production_integration':False,
        'trusted_boundary':'Stored HZ float feasibility and hash-bound single original forward. No rational feasibility, source-to-HZ proof, SAFE, original UNSAFE, or global LP-incompleteness claim.',
        'R1_failed_files':inventory(r1), 'R2_files':inventory(DIAG), 'replay_files':inventory(OUTPUT),
        'sealed_parent_inventory_sha256':sha256(ROOT/'docs/metamoe_protected_archive_20260923_r1.json'),
        'audit_source_sha256':sha256(Path(__file__)), 'audit_seconds':time.monotonic()-start}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output already exists')
    value = audit()
    write(args.output, value)
    print(value['audit'], value['status'], value['unknown_point_feasible'])
