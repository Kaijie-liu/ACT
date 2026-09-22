"""R4 independent terminal/cost audit, with separately bound original replay.

Not independent reproof of HZ or author positive bounds. Never opens a cohort.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_paired_r4 import validate, command_for, PROTOCOL, ARTIFACTS, ROSTER


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_receipt(receipt, row, cfg, command):
    seconds = receipt['execution_including_preflight_seconds']
    post = receipt['total_with_postflight_seconds']
    peak = receipt['peak_sampled_group_rss_bytes']
    if (not finite(seconds) or not finite(post) or post < seconds or
            type(peak) is not int or peak < 0 or row['seconds'] != seconds or
            receipt['deadline_seconds'] != cfg['seconds'] or
            receipt['group_rss_limit_bytes'] != cfg['group_rss_limit_bytes'] or
            receipt['command'] != command or receipt['rss_poll_seconds'] != .05 or
            receipt['postflight_in_execution_budget'] is not False or
            receipt['rss_is_sampled_not_instantaneous_cap'] is not True or
            receipt['receipt_own_write_excluded_from_this_clock'] is not True):
        raise ValueError('receipt command/resource/cost contract')
    if receipt['status'] not in ('COMPLETED', 'TIMEOUT', 'RESOURCE_LIMIT', 'ERROR'):
        raise ValueError('unknown outer terminal')
    if receipt['status'] == 'COMPLETED' and (
            seconds >= cfg['seconds'] or peak > cfg['group_rss_limit_bytes'] or
            receipt['exit_code'] != 0 or receipt['error'] is not None):
        raise ValueError('completed outside resource/success contract')


def check_candidate(result, arm):
    status = result['status']
    allowed = {'act': {'POSITIVE', 'UNKNOWN', 'TIMEOUT', 'UNSAFE_REPLAYED'},
               'author': {'BACKEND_POSITIVE', 'UNKNOWN', 'TIMEOUT', 'ERROR', 'UNSAFE_REPLAYED'}}
    if status not in allowed[arm] or not finite(result['worker_seconds']):
        raise ValueError('invalid arm candidate')
    if status == 'UNSAFE_REPLAYED':
        if result.get('evidence_grade') != 'FULL_MODEL_REPLAY' or not result.get('witness'):
            raise ValueError('missing full-model witness evidence')
    if status == 'BACKEND_POSITIVE' and result.get('evidence_grade') != 'AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER':
        raise ValueError('author numerical grade')
    if arm == 'author' and status in ('BACKEND_POSITIVE', 'UNKNOWN'):
        raw = result.get('backend_status')
        safe = raw in ('safe', 'safe-complete', 'safe-incomplete')
        if not isinstance(raw, str) or (status == 'BACKEND_POSITIVE') != safe:
            raise ValueError('author raw terminal/status mismatch')
        if status == 'UNKNOWN':
            if raw.startswith('unsafe'):
                if result.get('reason') != 'component counterexample is not a full-model witness':
                    raise ValueError('unreplayed author counterexample promoted')
            elif not (raw in ('unknown', 'timeout') or 'unknown' in raw):
                raise ValueError('unknown author raw terminal')
    if status == 'POSITIVE':
        candidates, excluded, unresolved = [result[k] for k in ('candidates', 'excluded', 'unresolved')]
        nonzero = result['nonzero_obligations']
        if (result.get('evidence_grade') != 'HZ_POLICY_ACCEPTED' or result.get('source_complete') is not False or
                result.get('class_counts') != [10, 10] or result.get('property_rows') != 19 or
                result.get('semantics') != 'class_separated_raw_top1_zero_fill_any_legal_ties' or
                not candidates or len(set(candidates)) != len(candidates) or
                any(type(i) is not int for i in candidates+excluded) or
                len(set(excluded)) != len(excluded) or unresolved or
                set(candidates) & set(excluded) or set(candidates) | set(excluded) != {0, 1} or
                len(nonzero) != len(candidates) or {n['expert'] for n in nonzero} != set(candidates) or
                result['expert_statuses'] != {str(i): 'certified' for i in candidates}):
            raise ValueError('incomplete ACT positive obligations')
        for item in nonzero:
            lo, hi = item['lower'], item['upper']
            if (type(item['expert']) is not int or item['accepted'] is not True or
                    type(lo) not in (int, float) or type(hi) not in (int, float) or
                    not math.isfinite(lo) or not math.isfinite(hi) or
                    lo > hi or not (lo > 0 or hi < 0)):
                raise ValueError('undefined selected-score branch')


def complete_numerical(row, result):
    """A raw backend timeout is NOT a completed numerical unknown."""
    if row['status'] not in ('POSITIVE', 'BACKEND_POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN') or not result:
        return False
    raw = str(result.get('backend_status', '')).lower()
    if any(s in raw for s in ('timeout', 'timed out', 'resource', 'memory')):
        return False
    if row['status'] == 'UNKNOWN':
        allowed = {'act': {'incomplete_route_coverage', 'nonzero_or_global_output_obligation_unproved'},
            'author': {'backend_unresolved', 'component counterexample is not a full-model witness'}}
        return result.get('reason') in allowed[row['arm']]
    return True


def replay_binding(review, replay, cfg):
    if (replay['audit'] != 'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS' or
            replay['config_sha256'] != review['config_sha256'] or
            replay['summary_sha256'] != review['summary_sha256'] or
            not finite(replay['separate_audit_seconds'])):
        raise ValueError('independent replay binding/cost')
    expected = {(r['id'], r['arm'], r['result_sha256']) for r in review['rows'] if r['status'] == 'UNSAFE_REPLAYED'}
    observed = [(r['id'], r['arm'], r['result_sha256']) for r in replay['rows']]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('missing/extra/duplicate witness replay')
    labels = {r['id']: r['label'] for r in cfg['requests']}
    for row in replay['rows']:
        margin, prediction = row.get('minimum_margin'), row.get('prediction')
        if (type(row.get('label')) is not int or row['label'] != labels[row['id']] or
                type(prediction) is not int or not 0 <= prediction < 20 or
                type(margin) not in (int, float) or not math.isfinite(margin) or margin >= cfg['margin']):
            raise ValueError('invalid original replay label/prediction/margin')


def audit(path, replay_path=None):
    started = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg)
    root = Path(cfg['output_root'])
    summary = json.loads((root/'summary.json').read_text())
    if (summary['config_sha256'] != sha256(path) or
            [(r['id'], r['arm']) for r in summary['rows']] != ROSTER):
        raise ValueError('summary binding/ordered denominator')
    launch = json.loads((root/'launch.json').read_text())
    if (launch['config_sha256'] != sha256(path) or launch['protocol'] != PROTOCOL or
            launch['automatic_followup'] is not False):
        raise ValueError('launch binding')
    overhead, complete, blocked, rows = 0., True, False, []
    for row in summary['rows']:
        if row['status'] == 'NOT_STARTED_AFTER_ERROR':
            if not blocked or set(row) != {'id', 'arm', 'status'}:
                raise ValueError('unexplained/misaccounted unstarted row')
            complete = False
            rows.append(row)
            continue
        if blocked:
            raise ValueError('ran after stop-on-error')
        folder = root/f"{row['id']}_{row['arm']}"
        receipt = json.loads((folder/'receipt.json').read_text())
        if (json.loads((folder/'terminal.json').read_text()) != row or
                sha256(folder/'receipt.json') != row['receipt_sha256']):
            raise ValueError('terminal/receipt binding')
        check_receipt(receipt, row, cfg, command_for(cfg, path, row['id'], row['arm']))
        for stream in ('stdout', 'stderr'):
            if sha256(folder/f'{stream}.txt') != receipt[f'{stream}_sha256']:
                raise ValueError('outer log identity')
        overhead += receipt['total_with_postflight_seconds']-row['seconds']
        observed = {name: sha256(folder/name) for name in ARTIFACTS if (folder/name).is_file()}
        if row['artifacts'] != observed:
            raise ValueError('auxiliary artifact identity/coverage changed')
        candidate = folder/'result.json'
        if candidate.exists() != bool(row['result_sha256']):
            raise ValueError('unaccounted candidate file')
        result, malformed = None, False
        if candidate.exists():
            if sha256(candidate) != row['result_sha256']:
                raise ValueError('candidate hash')
            try:
                result = json.loads(candidate.read_text())
                required = {'status', 'config_sha256', 'request_id', 'arm', 'label',
                            'tensor_file_sha256', 'worker_seconds'}
                if not isinstance(result, dict) or not required <= result.keys() or not isinstance(result['status'], str):
                    raise ValueError('candidate incomplete schema')
            except ValueError:
                malformed, result = True, None
        if malformed != bool(row['result_parse_error']):
            raise ValueError('candidate parse-failure accounting')
        expected_status = ((result['status'] if result else 'ERROR')
                           if receipt['status'] == 'COMPLETED' else receipt['status'])
        if row['status'] != expected_status:
            raise ValueError('outer termination/candidate precedence')
        if result is not None:
            req = next(r for r in cfg['requests'] if r['id'] == row['id'])
            if (result['config_sha256'] != sha256(path) or result['request_id'] != row['id'] or
                    result['arm'] != row['arm'] or result['label'] != req['label'] or
                    result['tensor_file_sha256'] != sha256(req['tensor_file'])):
                raise ValueError('same-object result identity')
            check_candidate(result, row['arm'])
            if result['worker_seconds'] > row['seconds']:
                raise ValueError('worker cost exceeds outer receipt')
            if 'backend_status' in result:
                from metamoe_component_control import parse_backend_result
                # Only reconstruct the parser's raw token here. Original timing
                # and exit status stay in the frozen worker/outer receipts.
                parsed = parse_backend_result((folder/'backend.stdout').read_text(), 0, 0., 300.)
                if parsed.get('backend_status') != result['backend_status']:
                    raise ValueError('backend log/result disagreement')
        complete = complete and complete_numerical(row, result)
        blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
        rows.append({**row, 'grade': result.get('evidence_grade', 'NONE') if result else 'NONE',
            'reason': result.get('reason') if result else None,
            'backend_status': result.get('backend_status') if result else None,
            'peak_sampled_group_rss_bytes': receipt['peak_sampled_group_rss_bytes'],
            'receipt_total_with_postflight_seconds': receipt['total_with_postflight_seconds']})
    cost = json.loads((root/'batch_cost.json').read_text())
    if (cost['config_sha256'] != sha256(path) or cost['charged_request_seconds'] !=
            sum(r.get('seconds', 0.) for r in rows)):
        raise ValueError('batch accounting/binding')
    if any(not finite(cost[k]) for k in ('charged_request_seconds', 'batch_wall_through_summary_seconds')):
        raise ValueError('nonfinite batch cost')
    if cost['batch_wall_through_summary_seconds'] < cost['charged_request_seconds']+overhead:
        raise ValueError('postflight not included in batch cost')
    review = {'config_sha256': sha256(path), 'summary_sha256': sha256(root/'summary.json'),
        'rows': rows, 'batch_cost': cost,
        'counts': {arm: dict(Counter(r['status'] for r in rows if r['arm'] == arm)) for arm in ('act', 'author')},
        'numerical_guarantees_equated': False}
    replay_seconds = None
    if replay_path:
        replay = json.loads(replay_path.read_text())
        replay_binding(review, replay, cfg)
        replay_seconds = replay['separate_audit_seconds']
    return {**review, 'audit': 'PASS',
        'execution_control_pass': complete,
        'smoke_gate_pass': complete and replay_path is not None,
        'independent_original_replay_sha256': sha256(replay_path) if replay_path else None,
        'independent_original_replay_seconds': replay_seconds,
        'receipt_postflight_seconds_included_in_batch': overhead,
        'separate_audit_seconds': time.monotonic()-started,
        'opens_formal_cohort': False,
        'scope': 'four old-input controls only; passing requires separate original replay, not independent SAFE proof',
        'execution_and_freeze_are_distinct': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.replay)
    write(args.output, result)
    print('smoke_gate_pass', result['smoke_gate_pass'])
