"""Saved-only same-object/obligation/terminal/cost audit; no optimization.

Original-model counterexamples have a separate replay. HZ and author positive
statuses retain distinct numerical trust; no source-complete proof is inferred.
"""
import argparse
from collections import Counter
from decimal import Decimal
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit_metamoe_current_assignment import read, require, inventory
from audit_metamoe_protected_smoke_r1 import arrays
from audit_metamoe_checked_base_control import evaluate
from audit_metamoe_checked_routing_control import routing_query
from audit_metamoe_nonzero_precheck_control import nonzero_query
from audit_metamoe_csr_paired_r4 import check_receipt, check_candidate, replay_binding
from audit_conv_f0_timing import check_trace
from metamoe_csr_paired_r4 import collect_terminal, ARTIFACTS
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import metamoe_checked_paired as control


def check_author(folder, cfg, request, result):
    if result['status'] == 'UNSAFE_REPLAYED':
        return {'kind': 'original_model_witness'}
    if not (folder/'obligation_identity.json').exists():
        require(result['status'] in ('TIMEOUT', 'ERROR'), 'missing author obligations')
        return None
    v = read(folder/'obligation_identity.json')
    require(v['route'] == result['clean_route'] and v['sign'] in (-1, 1) and
            v['output_rows'] == 19 and v['router_rows'] == 2 and
            v['checkpoint_sha256'] == cfg['files'][cfg['checkpoint']], 'author joint identity')
    require(np.asarray(v['probe']).shape == (1, 22) and np.isfinite(v['probe']).all(), 'author joint shape')
    # Recreate exact real endpoints and every violation disjunct independently
    # of the production spec_text function, including router/nonzero obligations.
    z = arrays(Path(request['tensor_file']))
    lines = [f'(declare-const X_{i} Real)' for i in range(z['lower'].size)]
    lines += [f'(declare-const Y_{i} Real)' for i in range(22)]
    for i, (a, b) in enumerate(zip(z['lower'].ravel(), z['upper'].ravel())):
        lines += [f'(assert (>= X_{i} {Decimal.from_float(float(a)):f}))',
                  f'(assert (<= X_{i} {Decimal.from_float(float(b)):f}))']
    lines += ['(assert (or '+' '.join(f'(and (>= Y_0 Y_{i}))' for i in range(1, 22))+'))']
    require((folder/'request.vnnlib').read_text() == '\n'.join(lines)+'\n', 'author box/property changed')
    if 'backend_status' in result:
        from metamoe_component_control import parse_backend_result
        parsed = parse_backend_result((folder/'backend.stdout').read_text(), 0, 0., 300.)
        require(parsed.get('backend_status') == result['backend_status'], 'author log/result mismatch')
    return v


def check_act(folder, cfg, path, request, result, row, killed):
    h, ih = sha256(path), cfg['files'][request['tensor_file']]
    if not (folder/'trace.jsonl').exists():
        require(killed, 'missing completed trace')
        return {'partial': True}
    runtime = read(folder/'runtime.json')
    require(runtime['identity'] == control.identity(cfg, path, request['id'], 'act') and
            runtime['act_options'] == cfg['act_options'], 'runtime scope/options')
    clock = runtime['worker_started_monotonic']
    trace = check_trace(folder/'trace.jsonl', row['seconds'], runtime['identity'], killed=killed)
    routes = []
    for i, p in enumerate(sorted((folder/'routing').glob('query_*'))):
        require(p.name == f'query_{i:03d}' and i < 2, 'route coverage')
        routes.append(routing_query(p, i, True, h, ih, clock, trace, killed))
    expert_dirs = sorted((folder/'protected').glob('evaluation_*'))
    experts = [evaluate(p, 19, h, ih, killed) for p in expert_dirs]
    for p in expert_dirs:
        if not (p/'properties.npz').exists():
            require(killed or result['status'] != 'POSITIVE', 'missing properties')
            continue
        prop = arrays(p/'properties.npz')
        C = np.zeros((19, 20))
        for i, other in enumerate(j for j in range(20) if j != request['label']):
            C[i, request['label']], C[i, other] = -1, 1
        require(np.array_equal(prop['C'], C) and np.array_equal(prop['thresholds'],
                np.full(19, -cfg['margin'])), 'wrong requested global properties')
    nz = []
    for i, p in enumerate(sorted((folder/'nonzero').glob('query_*'))):
        require(p.name == f'query_{i:03d}' and i < 2, 'score query coverage')
        if not (p/'begin.json').exists():
            require(killed, 'missing score begin')
            continue
        score = read(p/'begin.json')['row']
        require(type(score) is int and 0 <= score < 2, 'score index')
        f = folder/f'routing/query_{score:03d}/model.npz'
        require(f.exists() or killed, 'missing guarded matrix')
        nz.append(nonzero_query(p, score, True, h, ih, clock, trace, killed, arrays(f) if f.exists() else None))
    nonces = [q['scope']['evaluation_nonce'] for q in routes+nz if q.get('scope')]
    require(len(nonces) == len(set(nonces)), 'reused query nonce')
    if result and not killed and 'candidates' in result:
        require(len(routes) == 2 and result['candidates'] == [q['index'] for q in routes if q['status'] != 'infeasible'] and
                result['excluded'] == [q['index'] for q in routes if q['status'] == 'infeasible'] and
                result['unresolved'] == [q['index'] for q in routes if q['status'] == 'unknown'], 'route aggregate')
        require([e['status'] for e in experts] == list(result['expert_statuses'].values()), 'expert aggregate')
        obligations = result.get('nonzero_obligations', [])
        require(len(nz) == len(obligations) and [q['row'] for q in nz] == [o['expert'] for o in obligations], 'nonzero coverage')
        for q, o in zip(nz, obligations):
            require(q['accepted'] == o['accepted'] and all(q['result'][k] == o[k] for k in
                    ('lower', 'upper', 'lower_status', 'upper_status')), 'nonzero aggregate')
    return {'routing': routes, 'experts': experts, 'nonzero': nz,
            'timing': trace['aggregates'], 'right_censored_spans': trace['open_span_ids'], 'last_trace_hash': trace['last_hash']}


def audit(path, replay_path=None):
    began = time.monotonic()
    cfg = read(path)
    control.validate(cfg)
    root, h = Path(cfg['output_root']), sha256(path)
    summary, launch = read(root/'summary.json'), read(root/'launch.json')
    require(summary['config_sha256'] == launch['config_sha256'] == h and
            launch['protocol'] == cfg['protocol'] and launch['roster'] == cfg['roster'] and
            not launch['automatic_followup'] and [[r['id'], r['arm']] for r in summary['rows']] == cfg['roster'], 'batch identity/denominator')
    rows, blocked, accounted = [], False, 0.
    for row in summary['rows']:
        if row['status'] == 'NOT_STARTED_AFTER_ERROR':
            require(blocked and set(row) == {'id', 'arm', 'status'}, 'unexplained omission')
            rows.append(row)
            continue
        require(not blocked, 'execution after error')
        rid, arm = row['id'], row['arm']
        req = next(r for r in cfg['requests'] if r['id'] == rid)
        folder = root/f'{rid}_{arm}'
        receipt = read(folder/'receipt.json')
        killed = receipt['status'] != 'COMPLETED'
        require(row == read(folder/'terminal.json') and row['receipt_sha256'] == sha256(folder/'receipt.json'), 'terminal identity')
        check_receipt(receipt, row, cfg, control.command(cfg, path, rid, arm))
        require(all(row[k] == v for k, v in collect_terminal(folder, receipt).items()), 'outer precedence')
        for s in ('stdout', 'stderr'):
            require(receipt[s+'_sha256'] == sha256(folder/f'{s}.txt'), 'stream changed')
        require(row['artifacts'] == {n:sha256(folder/n) for n in (*ARTIFACTS, 'runtime.json', 'trace.jsonl')
                if (folder/n).is_file()}, 'artifact bindings')
        require(row['evidence_artifacts'] == {area+'/'+k:v for area in ('protected','routing','nonzero')
                for k,v in inventory(folder/area).items()}, 'evidence inventory')
        post = row['postflight_inventory_seconds']
        require(math.isfinite(post) and post >= 0, 'postflight cost')
        accounted += receipt['total_with_postflight_seconds']+post
        result = read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
        if result:
            require(result['config_sha256'] == h and result['request_id'] == rid and result['arm'] == arm and
                    result['label'] == req['label'] and result['tensor_file_sha256'] == cfg['files'][req['tensor_file']] and
                    0 <= result['worker_seconds'] <= row['seconds'], 'candidate identity/cost')
            if not killed:
                check_candidate(result, arm)
        details = check_act(folder, cfg, path, req, result, row, killed) if arm == 'act' else (
            check_author(folder, cfg, req, result) if result and not killed else None)
        rows.append({**row, 'grade': result.get('evidence_grade','NONE') if result and not killed else 'NONE',
                     'result': result, 'details': details, 'receipt': receipt})
        blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
    cost = read(root/'batch_cost.json')
    require(cost['config_sha256'] == h and cost['charged_request_seconds'] == sum(r.get('seconds',0) for r in rows) and
            math.isfinite(cost['batch_wall_through_summary_seconds']) and cost['batch_wall_through_summary_seconds'] >= accounted, 'batch cost')
    review = {'audit':'PASS', 'issues':0, 'config_sha256':h, 'summary_sha256':sha256(root/'summary.json'),
              'rows':rows, 'cost':cost, 'numerical_guarantees_equated':False}
    if replay_path:
        replay_binding(review, read(replay_path), cfg)
    # This small smoke is specifically an UNSAFE center and a completed positive
    # path. An audited timeout is preserved but does not open the next gate.
    gate = (cfg['protocol'] == 'metamoe_checked_paired_smoke_r1' and replay_path is not None and
            [r['status'] for r in rows] == ['UNSAFE_REPLAYED','UNSAFE_REPLAYED','BACKEND_POSITIVE','POSITIVE'])
    comparisons = {}
    for arm in ('act', 'author'):
        rr = [r for r in rows if r['arm'] == arm]
        comparisons[arm] = {'counts':dict(Counter(r['status'] for r in rr)),
            'positive_ids':[r['id'] for r in rr if r['status'] in ('POSITIVE','BACKEND_POSITIVE')],
            'unsafe_ids':[r['id'] for r in rr if r['status'] == 'UNSAFE_REPLAYED'],
            'charged_seconds':sum(r.get('seconds',0) for r in rr)}
    a, b = [set(comparisons[arm]['positive_ids']) for arm in ('act','author')]
    return {**review, 'comparisons':comparisons, 'positive_intersection':sorted(a&b),
            'act_only_positive':sorted(a-b), 'author_only_positive':sorted(b-a),
            'smoke_gate_pass':gate, 'automatic_followup':False,
            'replay_sha256':sha256(replay_path) if replay_path else None,
            'files':inventory(root), 'audit_seconds':time.monotonic()-began, 'new_audit_solves':0,
            'trust':'Stored point/sign checks plus native HZ policy and author numerical filter; source/guard lowering still trusted.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    v = audit(a.config, a.replay)
    write(a.output, v)
    print(v['audit'], 'smoke_gate', v['smoke_gate_pass'], v['comparisons'])
