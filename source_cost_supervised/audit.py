"""Saved-only reception/cost audit. No producer, checkpoint or solver required.

Online receipt validation is watched under the inherited work deadline.
Optional offline source recheck is separately charged, never repairs a timeout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from scoped_proof.io import load, save, sha, tick

PHASES = ['identity', 'prefix', 'proposal', 'construct', 'publish_R1', 'source_check']


def finite(v): return type(v) in (int, float) and math.isfinite(v) and v >= 0


def validate_spec(spec):
    if set(spec) != {'id', 'fixture', 'source_sha256'} or not re.fullmatch(r'[a-z0-9_]{1,80}', spec['id']):
        raise ValueError('synthetic spec schema')
    f = spec['fixture']
    if (set(f) != {'experts', 'classes', 'width', 'depth', 'seed'}
            or any(type(v) is not int for v in f.values())
            or not 2 <= f['experts'] <= 8 or not 2 <= f['classes'] <= 10
            or not 1 <= f['width'] <= 8 or not 0 <= f['depth'] <= 2
            or not 0 <= f['seed'] <= 10000
            or not re.fullmatch('[0-9a-f]{64}', spec['source_sha256'])):
        raise ValueError('finite synthetic-only source')


def file_check(path, record, deadline):
    path = Path(path)
    if path.is_symlink() or set(record) != {'sha256', 'bytes'} or path.stat().st_size != record['bytes']:
        raise ValueError('artifact type/size')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(65536), b''): tick(deadline); h.update(block)
    tick(deadline)
    if h.hexdigest() != record['sha256']: raise ValueError('artifact identity')


def journal_check(path, inv, *, complete, deadline):
    stack = []; finished = []; previous = inv['started_monotonic']; count = 0; truncated = False; late = 0
    if not Path(path).exists():
        if complete: raise ValueError('missing journal')
        return {'finished': [], 'open': [], 'truncated_tail': False, 'missing': True, 'late_events': 0}
    if Path(path).is_symlink(): raise ValueError('journal symlink')
    with Path(path).open('rb') as f:
        while True:
            tick(deadline); raw = f.readline(16385)
            if not raw: break
            if len(raw) > 16384: raise ValueError('oversized event')
            if not raw.endswith(b'\n'):
                if complete: raise ValueError('incomplete journal')
                truncated = True; break
            def pairs(items):
                d = {}
                for k, v in items:
                    if k in d: raise ValueError('duplicate event field')
                    d[k] = v
                return d
            row = json.loads(raw, object_pairs_hook=pairs)
            if (row['invocation'] != inv['invocation'] or row['spec_sha256'] != inv['spec_sha256']
                    or row['index'] != count or not finite(row['monotonic'])
                    or row['monotonic'] < previous
                    or row['kind'] not in ('worker', 'phase', 'component')):
                raise ValueError('event binding/order/deadline')
            if row['monotonic'] >= inv['work_deadline_monotonic']:
                if complete: raise ValueError('late complete event')
                late += 1  # Cooperative deadline exceptions may themselves be logged late.
            previous = row['monotonic']; count += 1
            key = (row['kind'], row['name'])
            if row['event'] == 'ENTER':
                if row['kind'] == 'component' and (not stack or stack[-1]['kind'] != 'phase'):
                    raise ValueError('overlapping/nonphase component')
                stack.append(row)
            elif row['event'] in ('EXIT', 'ERROR'):
                if not stack or (stack[-1]['kind'], stack[-1]['name']) != key:
                    raise ValueError('unmatched event')
                enter = stack.pop()
                finished.append({'kind': row['kind'], 'name': row['name'], 'event': row['event'],
                                 'seconds': row['monotonic'] - enter['monotonic']})
            else: raise ValueError('event status')
    if complete and (stack or truncated or any(r['event'] != 'EXIT' for r in finished)):
        raise ValueError('unfinished journal claimed complete')
    opened = [dict(kind=r['kind'], name=r['name'], seconds=None,
                   observed_lower_seconds=previous - r['monotonic'], status='RIGHT_CENSORED') for r in stack]
    return {'finished': finished, 'open': opened, 'truncated_tail': truncated, 'missing': False, 'late_events': late}


def receive(root, inv, spec, *, deadline):
    tick(deadline); validate_spec(spec)
    candidate = load(root / 'candidate.json', limit=65536)
    if (candidate['schema'] != 'SOURCE_COST_CANDIDATE_V1'
            or candidate['invocation'] != inv['invocation'] or candidate['spec_sha256'] != inv['spec_sha256']
            or candidate['source_sha256'] != spec['source_sha256']
            or candidate['complete_output_positive_proof'] is not False or candidate['native_solver_calls'] != 0
            or set(candidate['records']) != {'source.json', 'construction.json', 'report.json', 'journal.jsonl'}):
        raise ValueError('candidate binding/scope/inventory')
    for name, record in candidate['records'].items():
        file_check(root / ('profile/' + name if name != 'journal.jsonl' else name), record, deadline)
    if candidate['records']['source.json']['sha256'] != spec['source_sha256']:
        raise ValueError('source spec not bound to canonical source')
    report = load(root / 'profile/report.json', limit=16 * 2**20)
    if (report['schema'] != 'SOURCE_COST_DIAGNOSTIC_V2' or report['source_sha256'] != spec['source_sha256']
            or report['complete_output_positive_proof'] is not False or report['native_solver_calls'] != 0
            or report['files'] != {n: candidate['records'][n + '.json'] for n in ('source', 'construction')}
            or [p['name'] for p in report['phases']] != PHASES):
        raise ValueError('profile identity/inventory')
    for p in report['phases']:
        ops = [v for v in report['operations'] if v['phase'] == p['name']]
        if (any(not finite(v['seconds']) or v['status'] != 'COMPLETED' for v in ops)
                or not all(finite(p[k]) for k in ('seconds', 'component_seconds', 'other_seconds'))
                or abs(sum(v['seconds'] for v in ops) - p['component_seconds']) > 1e-8
                or abs(p['seconds'] - p['component_seconds'] - p['other_seconds']) > 1e-8):
            raise ValueError('nonoverlapping phase cost')
    if (any(o['phase'] not in PHASES for o in report['operations'])
            or not finite(report['seconds_before_report'])
            or sum(p['seconds'] for p in report['phases']) > report['seconds_before_report']):
        raise ValueError('unaccounted phase cost')
    checked = report['source_check']; f = spec['fixture']
    duties = f['experts'] * (f['experts'] - 1) // 2 * (f['classes'] - 1)
    if (checked['status'] != 'CHECKED_ROUTE_COVER_AND_RETAINED_OUTPUT_CONSTRUCTIONS'
            or checked['source_sha256'] != spec['source_sha256']
            or checked['original_output_obligations'] != duties
            or checked['excluded_output_obligations'] + checked['output_obligations'] != duties
            or checked['lower_bounds_checked'] != 0
            or any(checked[k] is not False for k in ('complete_output_positive_proof',
                                                   'route_changing_established', 'native_float_proof'))):
        raise ValueError('all original construction obligations/no output bounds')
    traces = journal_check(root / 'journal.jsonl', inv, complete=True, deadline=deadline)
    if [r['name'] for r in traces['finished'] if r['kind'] == 'phase'] != PHASES:
        raise ValueError('journal phase inventory')
    tick(deadline)
    return {'status': 'PROFILE_COMPLETE_NOT_OUTPUT_PROOF', 'invocation': inv['invocation'],
            'spec_sha256': inv['spec_sha256'], 'candidate_sha256': sha(root / 'candidate.json'),
            'source_sha256': spec['source_sha256'], 'original_output_obligations': duties,
            'complete_output_positive_proof': False}


def cost_check(inv, cost, terminal, returned):
    for k, v in inv.items():
        if cost[k] != v: raise ValueError('invocation ledger mismatch')
    budget = inv['budget_seconds']; work = inv['work_deadline_monotonic']; start = inv['started_monotonic']
    if (not finite(budget) or not 0 < budget <= 300 or
            abs(inv['deadline_monotonic'] - start - budget) > 1e-7 or
            abs(inv['deadline_monotonic'] - work - min(2., budget / 10)) > 1e-7):
        raise ValueError('absolute budget changed')
    vals = [cost['seconds_before_ledger'], cost['overhead_seconds'], terminal['seconds_before_terminal'],
            returned['seconds_including_terminal']]
    if (not all(finite(v) for v in vals) or not vals[0] <= vals[2] <= vals[3]
            or abs(vals[0] - vals[1] - sum(s['seconds'] for s in cost['stages'])) > 1e-7):
        raise ValueError('whole wall cost')
    previous = 0.
    for i, s in enumerate(cost['stages']):
        if (i >= 2 or s['phase'] != ('profile', 'receive')[i] or not finite(s['seconds'])
                or not all(finite(s[k]) for k in ('start_seconds', 'end_seconds'))
                or not previous <= s['start_seconds'] <= s['end_seconds'] <= vals[0]
                or s['seconds'] > s['end_seconds'] - s['start_seconds'] + 1e-7
                or s['deadline_monotonic'] != work or s['cleanup_included'] is not True
                or s['status'] not in ('COMPLETED', 'ERROR', 'TIMEOUT', 'RESOURCE_LIMIT')
                or (s['status'] == 'COMPLETED' and start + s['end_seconds'] >= work)):
            raise ValueError('stage deadline/cleanup/cost')
        if i and cost['stages'][i-1]['status'] != 'COMPLETED': raise ValueError('late rescue after failure')
        previous = s['end_seconds']
    status = cost['status_before_publication']
    if status not in ('COMPLETED', 'ERROR', 'TIMEOUT', 'RESOURCE_LIMIT'): raise ValueError('unknown terminal')
    if status == 'COMPLETED' and (len(cost['stages']) != 2 or cost['received_sha256'] is None
            or any(s['status'] != 'COMPLETED' for s in cost['stages'])):
        raise ValueError('incomplete execution promoted')
    if (terminal['invocation'] != inv['invocation'] or terminal['spec_sha256'] != inv['spec_sha256']
            or terminal['complete_output_positive_proof'] is not False
            or terminal['status'] != ('TIMEOUT' if vals[2] >= budget else status)
            or returned['status'] != ('TIMEOUT' if vals[3] >= budget else terminal['status'])):
        raise ValueError('late terminal/status drift')


def review(root, returned, *, recheck=False):
    began = time.monotonic(); root = Path(root)
    inv = load(root / 'invocation.json'); spec = load(root / 'spec.json', inv['spec_sha256'])
    terminal = load(root / 'terminal.json', returned['terminal_sha256'])
    cost = load(root / 'cost.json', terminal['cost_sha256'])
    validate_spec(spec); cost_check(inv, cost, terminal, returned)
    result = None
    traces = journal_check(root / 'journal.jsonl', inv, complete=returned['status'] == 'COMPLETED',
                           deadline=time.monotonic() + 300)
    if returned['status'] == 'COMPLETED':
        result = receive(root, inv, spec, deadline=time.monotonic() + 300)
        if result != load(root / 'received.json', cost['received_sha256']): raise ValueError('changed reception')
        report = load(root / 'profile/report.json')
        if report['seconds_before_report'] > cost['stages'][0]['seconds']: raise ValueError('profile cost outside worker')
        if recheck:
            from residual_proof.check import check
            checked = check(load(root / 'profile/source.json'), load(root / 'profile/construction.json'),
                invocation='profile_v2', expected_source_sha256=spec['source_sha256'], deadline=time.monotonic()+300)
            if checked != report['source_check']: raise ValueError('independent source recheck')
    return {'audit': 'PASS', 'status': returned['status'], 'receipt': result, 'journal': traces,
            'source_rechecked': recheck and result is not None,
            'separate_audit_seconds': time.monotonic() - began,
            'complete_output_positive_proof': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    a = p.parse_args(); inv = load(a.root / 'invocation.json', limit=65536)
    spec = load(a.root / 'spec.json', inv['spec_sha256'], limit=65536)
    out = receive(a.root, inv, spec, deadline=inv['work_deadline_monotonic'])
    save(a.root / 'received.json', out); tick(inv['work_deadline_monotonic'])
