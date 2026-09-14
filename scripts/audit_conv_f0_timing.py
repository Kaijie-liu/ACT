"""Separate-process trace/terminal audit; timing, not independent SAFE proof."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def check_trace(path, cutoff, identity=None, killed=False):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    partial_bytes = 0
    if lines and not lines[-1].endswith(b'\n'):
        if not killed:
            raise ValueError('non-terminated trace tail without outer kill')
        partial_bytes = len(lines.pop())
    stack, spans, events = [], {}, []
    previous = '0' * 64
    elapsed = -1
    for seq, line in enumerate(lines):
        event = json.loads(line)
        digest = event.pop('sha256')
        encoded = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if (event['seq'] != seq or event['previous'] != previous
                or hashlib.sha256(encoded).hexdigest() != digest):
            raise ValueError('broken trace sequence/hash')
        if not math.isfinite(event['elapsed']) or not elapsed <= event['elapsed'] <= cutoff:
            raise ValueError('trace clock outside request')
        if not 0 <= event['logging_seconds_before'] <= event['elapsed']:
            raise ValueError('invalid logging accounting')
        previous, elapsed = digest, event['elapsed']
        kind = event['kind']
        if seq == 0 and (kind != 'IDENTITY' or (identity is not None and event['identity'] != identity)):
            raise ValueError('wrong trace identity')
        if kind == 'BEGIN':
            if event['parent'] != (stack[-1] if stack else None):
                raise ValueError('invalid span parent')
            spans[seq] = {'id':seq, 'name':event['name'], 'parent':event['parent'],
                          'start':elapsed, 'arguments':event['arguments'], 'end':None}
            stack.append(seq)
        elif kind in ('END', 'RAISE'):
            if not stack or event['span'] != stack.pop():
                raise ValueError('unmatched span end')
            spans[event['span']].update(end=elapsed, end_kind=kind,
                                       result=event.get('result'), exception=event.get('exception'))
        elif kind not in ('IDENTITY', 'INSTALL_BEGIN', 'INSTALLED', 'WORKER_COMPLETE'):
            raise ValueError('unknown trace event')
        events.append({**event, 'sha256':digest})
    if not events or (stack and not killed):
        raise ValueError('missing trace or unclosed spans without kill')
    if not killed and not any(e['kind']=='WORKER_COMPLETE' for e in events):
        raise ValueError('missing worker completion')
    aggregates = defaultdict(lambda: {'closed_calls':0, 'inclusive_closed_seconds':0.0,
                                     'exclusive_closed_seconds':0.0, 'open_calls':0})
    for span in spans.values():
        group = aggregates[span['name']]
        if span['end'] is None:
            group['open_calls'] += 1
            span['right_censored'] = True
            span['observed_lower_seconds'] = elapsed - span['start']
            span['exposure_to_cutoff_seconds'] = cutoff - span['start']
        else:
            seconds = span['end'] - span['start']
            children = [s for s in spans.values() if s['parent'] == span['id']]
            if any(s['end'] is None for s in children):
                raise ValueError('closed parent with open child')
            exclusive = seconds - sum(s['end'] - s['start'] for s in children)
            if exclusive < -1e-9:
                raise ValueError('negative exclusive duration')
            span.update(seconds=seconds, exclusive_seconds=max(0, exclusive), right_censored=False)
            group['closed_calls'] += 1
            group['inclusive_closed_seconds'] += seconds
            group['exclusive_closed_seconds'] += max(0, exclusive)
    return {'event_count':len(events), 'partial_tail_bytes':partial_bytes,
            'last_hash':previous, 'last_event_elapsed':elapsed,
            'logging_seconds_observed':events[-1]['logging_seconds_before'],
            'aggregates':dict(aggregates), 'spans':list(spans.values()),
            'open_span_ids':stack,
            'limits':'Nested inclusive times overlap; open exposure is censored, not exact native solve time. Trace overhead is charged; last-event logging cost excludes its own write.'}


def audit(root):
    from scripts.conv_three_arm_contract import ROOT, read, selection, wrapper_hashes, request_for
    from scripts.run_conv_f0_timing import FILES, PROTOCOL, DEFAULT
    from scripts.audit_conv_three_arm import terminal_contract
    from act.pipeline.moe.experiment1 import _sha256
    from act.pipeline.moe.schedule_confirmation import inspect_row
    from act.pipeline.moe.external_pair_worker import load
    if root.resolve() != DEFAULT:
        raise ValueError('wrong diagnostic directory')
    runtime = read(root/'runtime.json'); value = selection()
    if (runtime['selection'] != value or runtime['protocol'] != read(PROTOCOL)
            or runtime['sources'] != {p:_sha256(ROOT/p) for p in FILES}
            or runtime['old_wrappers'] != wrapper_hashes() or runtime['full_started'] is not False):
        raise ValueError('runtime/source drift')
    job = next(j for j in value['smoke_jobs'] if j['job_id']==runtime['protocol']['job_id'])
    directory = root/job['job_id']; parent = ROOT/runtime['protocol']['parent_root']/job['job_id']
    request = request_for(value, job, runtime['git_head'])
    row = read(directory/'terminal.json'); end = read(root/'run_terminal.json')
    if (read(directory/'request.json') != request or row['request_sha256'] != _sha256(directory/'request.json')
            or [json.loads(l) for l in (root/'rows.jsonl').read_text().splitlines()] != [row]
            or end['row'] != row or end['full_started'] is not False
            or any(row[k] != v for k,v in job.items())):
        raise ValueError('request/terminal roster differs')
    if (runtime['parent_request_sha256'] != _sha256(parent/'request.json')
            or runtime['parent_terminal_sha256'] != _sha256(parent/'terminal.json')
            or {k:v for k,v in read(parent/'request.json').items() if k!='head'}
               != {k:v for k,v in request.items() if k!='head'}):
        raise ValueError('parent request changed')
    terminal_contract(row)
    state = row['resource_wait']['at_launch']
    if state['available_ram_gib'] < 16 or state['free_disk_gib'] < 5 or not 0 <= state['load_per_core'] <= .5:
        raise ValueError('resource gate mismatch')
    load(request)  # independently reload exact checkpoint/materialized tensors
    configs = {a:read(v['path']) for a,v in value['identities']['method_configs'].items()}
    generic = {**value, 'models':{'conv':value['subject']}, 'request':{'epsilon':2/255}}
    detail = inspect_row(root, {**row, 'model':'conv'},
        {**runtime, 'smoke':True, 'config':{'methods':value['identities']['method_configs']}}, generic, configs)
    if end['trace_sha256'] != _sha256(directory/'trace.jsonl'):
        raise ValueError('terminal trace hash changed')
    trace = check_trace(directory/'trace.jsonl', row['wall_seconds'],
        {'request_sha256':row['request_sha256'], 'runtime_sha256':_sha256(root/'runtime.json')},
        killed=row['outer_timeout'])
    return {'status':'PASS', 'issues':[], 'row':row, 'detail':detail, 'trace':trace,
            'full_started':False, 'old_smoke_gate':'FAIL_UNCHANGED',
            'scope':'Diagnostic conformance only; no independent SAFE bound reproof or speed comparison'}


if __name__ == '__main__':
    from act.pipeline.moe.conv_training import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except Exception as exc:
        result = {'status':'FAIL', 'issues':[repr(exc)]}
    atomic_json(args.output, result)
    print(json.dumps({'status':result['status'], 'issues':result['issues']}), flush=True)
    raise SystemExit(0 if result['status']=='PASS' else 1)
