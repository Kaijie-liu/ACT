"""Optional evidence-tail execution; no cohort selection or model execution.

Caller supplies the original monotonic request start: earlier capture/proposal
cost is NOT refunded. All tail work shares 300s, with an owned 298s work cutoff.
Cooperative bookkeeping may finish late, but a late publication is TIMEOUT.
run() produces candidate records. supervise() is the whole-driver admission
entry, charging all candidate publication and driver cleanup to the same clock.
"""
import math
import os
from pathlib import Path
import subprocess
import time

from portable_proof.runtime import strict_json, original_bytes, compact, digest

ROOT = Path(__file__).resolve().parents[1]
ACT = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
STAGES = ('precheck', 'package', 'check')
VALID = ('CHECKED_CONDITIONAL', 'UNKNOWN_MISSING_EVIDENCE', 'UNKNOWN_NONPOSITIVE', 'UNKNOWN_ROUTE_COVERAGE')


def read(p): return strict_json(p.read_bytes())


def save_new(p, value):
    with p.open('xb') as f: f.write(original_bytes(value))


def left(started, clock=time.monotonic):
    now = clock()
    if not math.isfinite(started) or started > now:
        raise ValueError('original request start required')
    value = started+298-now
    if value <= 0: raise TimeoutError('no request work budget remains')
    return value


def verdict(stages, output, elapsed, packing):
    if elapsed >= 300 or any(s['state'] == 'TIMEOUT' for s in stages):
        return 'TIMEOUT', False
    if [s['name'] for s in stages] != list(STAGES) or any(s['state'] != 'COMPLETED' for s in stages):
        return ('ERROR' if any(s['state'] == 'ERROR' for s in stages) else 'TIMEOUT'), False
    if (not output['isolated'] or not output['site_disabled'] or output['solver_imported'] or
            output['bundle_sha256'] != packing['bundle_sha256'] or
            output['statement_sha256'] != packing['statement_sha256'] or
            output['cache']['enabled'] != packing['cache_enabled']):
        raise ValueError('isolated checker or bundle identity mismatch')
    if any(output['cache'][k] for k in ('live_entries', 'live_cells', 'live_payload_bytes')):
        raise ValueError('cache escaped request')
    result = output['result']; status = result['status']
    if status not in VALID: raise ValueError('unsupported evidence grade')
    if status == 'CHECKED_CONDITIONAL' and (not result['required_obligations'] or
            result['positive_obligations'] != result['required_obligations']):
        raise ValueError('incomplete proof promoted')
    return status, True


def publish(root, terminal, started, *, clock=time.monotonic, write=save_new):
    # Candidate file alone is NEVER an admission record.
    write(root/'candidate.json', terminal)
    observed = clock()-started
    receipt = {'schema': 'CACHED_PORTABLE_ADMISSION_V1',
               'candidate_sha256': digest((root/'candidate.json').read_bytes()),
               'observed_submission_seconds': observed,
               'status': 'TIMEOUT' if observed >= 300 else terminal['status'],
               'complete_independent_check': observed < 300 and terminal['complete_independent_check']}
    write(root/'admission.json', receipt)
    end = clock()-started
    if end >= 300:
        # Invalidate even a good candidate if terminal serialization consumed the reserve.
        late = {'status': 'TIMEOUT', 'complete_independent_check': False,
                'observed_seconds': end, 'admission_sha256': digest((root/'admission.json').read_bytes())}
        write(root/'publication_timeout.json', late)
        receipt.update(status='TIMEOUT', complete_independent_check=False)
    return receipt


def run(source, request, destination, *, started, enabled=False):
    # No default fresh start: this is a tail of an existing request, not free extra time.
    from evidence_cohort.run import wait_owned, environment
    if type(enabled) is not bool: raise ValueError('explicit boolean cache option')
    now = time.monotonic()
    if not math.isfinite(started) or started > now: raise ValueError('invalid original request start')
    root = Path(destination).resolve(); source = Path(source).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('write outside authorized workspace')
    root.mkdir(exist_ok=False)
    stages = []; status = 'ERROR'; complete = False; error = None
    job = {'schema': 'CACHED_PORTABLE_TAIL_V1', 'source': str(source), 'request': request,
           'cache_enabled': enabled, 'started_monotonic': started, 'budget_seconds': 300,
           'work_seconds': 298, 'upstream_elapsed_at_entry': now-started}
    try:
        left(started); job['manifest_sha256'] = digest((source/'manifest.json').read_bytes())
        save_new(root/'job.json', job)
        for name in STAGES:
            allowance = left(started); begin = time.monotonic()-started
            if name == 'check':
                packing = read(root/'packing.json')
                cmd = [ACT, '-I', '-S', str(root/'portable/verify.py'),
                       '--bundle-hash', packing['bundle_sha256'], '--statement-hash', packing['statement_sha256'],
                       '--deadline-monotonic', repr(started+298)]
            else:
                cmd = [ACT, '-m', 'cached_portable.worker', name, str(root)]
            with (root/(name+'.log')).open('xb') as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                outcome = wait_owned(p, started+298)
            end = time.monotonic()-started
            state = 'TIMEOUT' if outcome['killed'] or outcome['return_code'] == 3 or end >= 298 else (
                'COMPLETED' if outcome['return_code'] == 0 else 'ERROR')
            stages.append({'name': name, 'start_seconds': begin, 'end_seconds': end,
                           'elapsed_seconds': end-begin, 'allowed_seconds': allowance,
                           'state': state, 'process': outcome})
            save_new(root/(name+'_stage.json'), stages[-1])
            if state != 'COMPLETED': break
        output = read(root/'check.log') if len(stages) == 3 and stages[-1]['state'] == 'COMPLETED' else None
        packing = read(root/'packing.json') if (root/'packing.json').exists() else None
        if output is not None and output['deadline_monotonic'] != started+298:
            raise ValueError('checker clock reset')
        status, complete = verdict(stages, output, time.monotonic()-started, packing)
    except TimeoutError:
        status = 'TIMEOUT'
    except Exception as exc:
        status = 'ERROR'; error = repr(exc)
    # Inventory and serialization are charged; no reads of checkpoint/history needed.
    inventory = {p.name: digest(p.read_bytes()) for p in root.iterdir() if p.is_file()}
    elapsed = time.monotonic()-started
    if elapsed >= 300: status, complete = 'TIMEOUT', False
    terminal = {'schema': 'CACHED_PORTABLE_TAIL_TERMINAL_V1', 'status': status,
                'complete_independent_check': complete, 'stages': stages, 'artifact_sha256': inventory,
                'budget_seconds': 300, 'work_seconds': 298, 'wall_seconds': elapsed,
                'upstream_elapsed_at_entry': now-started,
                'stage_seconds': sum(s['elapsed_seconds'] for s in stages),
                'tail_overhead_seconds': elapsed-(now-started)-sum(s['elapsed_seconds'] for s in stages),
                'evidence_level': 'CHECKED_RATIONAL_CONDITIONAL', 'error': error,
                'production_gate_changed': False, 'deployed_float_SAFE': False}
    return publish(root, terminal, started)


def audit(root):
    """Read-only terminal/budget/binding review; not a fresh bound proof."""
    root = Path(root); t = read(root/'candidate.json'); r = read(root/'admission.json')
    if r['candidate_sha256'] != digest((root/'candidate.json').read_bytes()):
        raise ValueError('terminal changed')
    for name, sha in t['artifact_sha256'].items():
        if Path(name).name != name or digest((root/name).read_bytes()) != sha:
            raise ValueError('terminal artifact changed')
    if t['budget_seconds'] != 300 or t['work_seconds'] != 298:
        raise ValueError('budget modified')
    stages = t['stages']; prev = t['upstream_elapsed_at_entry']
    if not 0 <= prev <= t['wall_seconds']: raise ValueError('invalid upstream accounting')
    for i, s in enumerate(stages):
        if s['name'] != STAGES[i] or s['start_seconds'] < prev or s['end_seconds'] < s['start_seconds']:
            raise ValueError('stage ordering/accounting mismatch')
        if abs(s['elapsed_seconds']-(s['end_seconds']-s['start_seconds'])) > 1e-8 or s['allowed_seconds'] > 298-s['start_seconds']+.01:
            raise ValueError('stage budget reset')
        proc = s['process']
        expected_state = 'TIMEOUT' if proc['killed'] or proc['return_code'] == 3 or s['end_seconds'] >= 298 else (
            'COMPLETED' if proc['return_code'] == 0 else 'ERROR')
        if s['state'] != expected_state: raise ValueError('process state inconsistent with stage')
        prev = s['end_seconds']
    if prev > t['wall_seconds'] or abs(t['wall_seconds']-t['upstream_elapsed_at_entry']-
            sum(s['elapsed_seconds'] for s in stages)-t['tail_overhead_seconds']) > 1e-8:
        raise ValueError('unaccounted cost')
    if t['tail_overhead_seconds'] < 0 or r['observed_submission_seconds'] < t['wall_seconds']:
        raise ValueError('negative/omitted publication cost')
    if (root/'publication_timeout.json').exists():
        late = read(root/'publication_timeout.json')
        if late['status'] != 'TIMEOUT' or late['observed_seconds'] < 300 or late['admission_sha256'] != digest((root/'admission.json').read_bytes()):
            raise ValueError('invalid late-publication record')
        return {'status': 'TIMEOUT', 'complete_independent_check': False}
    if r['observed_submission_seconds'] >= 300:
        if r['status'] != 'TIMEOUT' or r['complete_independent_check']: raise ValueError('late admission')
        return r
    output = read(root/'check.log') if len(stages) == 3 and stages[-1]['state'] == 'COMPLETED' else None
    packing = read(root/'packing.json') if (root/'packing.json').exists() else None
    expected, complete = verdict(stages, output, t['wall_seconds'], packing)
    if t['error'] is not None: expected, complete = 'ERROR', False
    if (t['status'], t['complete_independent_check']) != (expected, complete) or (r['status'], r['complete_independent_check']) != (expected, complete):
        raise ValueError('terminal result mismatch')
    if output is not None:
        job = read(root/'job.json'); meta = read(root/'portable/bundle.json')
        from moe_evidence.schema import validate_request
        if (output['deadline_monotonic'] != job['started_monotonic']+298 or
                digest((root/'portable/bundle.json').read_bytes()) != packing['bundle_sha256'] or
                digest(compact(meta['statement'])) != packing['statement_sha256'] or
                meta['parser']['enabled'] != job['cache_enabled'] or
                meta['statement']['request'] != job['request'] or output['scope'] != validate_request(job['request']) or
                output['result'] != read(root/'precheck.json')['result']):
            raise ValueError('clock/request/result binding mismatch')
        portable = (root/'portable').resolve()
        for name, sha in meta['files'].items():
            path = portable/name
            if path.is_symlink() or not path.resolve().is_relative_to(portable) or digest(path.read_bytes()) != sha:
                raise ValueError('portable bytes changed after check')
    return r


def supervise(source, request, destination, *, started, enabled=False):
    """Own the entire tail, not only native checker calls; never start a new clock."""
    from evidence_cohort.run import wait_owned, environment
    if type(enabled) is not bool or not math.isfinite(started) or started > time.monotonic():
        raise ValueError('original clock and explicit cache option required')
    root = Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    plan = {'source': str(Path(source).resolve()), 'request': request,
            'destination': str(root/'tail'), 'started': started, 'enabled': enabled}
    save_new(root/'plan.json', plan)
    owner = None; error = None; admitted = None
    try:
        left(started)
        with (root/'driver.log').open('xb') as log:
            p = subprocess.Popen([ACT, '-m', 'cached_portable.execution', '--driver', str(root/'plan.json')],
                                 cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owner = wait_owned(p, started+298)
        exited = time.monotonic()-started
        # Never examine positive-looking files produced by killed/late/failed drivers.
        if owner['killed'] or exited >= 298: status = 'TIMEOUT'
        elif owner['return_code'] != 0: status = 'ERROR'
        else:
            admitted = audit(root/'tail'); status = admitted['status']
    except TimeoutError:
        exited = time.monotonic()-started; status = 'TIMEOUT'
    except Exception as exc:
        exited = time.monotonic()-started; status = 'ERROR'; error = repr(exc)
    observed = time.monotonic()-started
    if observed >= 300: status = 'TIMEOUT'
    value = {'schema': 'CACHED_PORTABLE_OUTER_V1', 'status': status, 'budget_seconds': 300,
             'work_seconds': 298, 'driver_exit_seconds': exited, 'observed_seconds': observed,
             'complete_independent_check': status in VALID and bool(admitted and admitted['complete_independent_check']),
             'outer_process': owner, 'plan_sha256': digest((root/'plan.json').read_bytes()),
             'error': error, 'deployed_float_SAFE': False}
    save_new(root/'outer.json', value)
    if time.monotonic()-started >= 300:
        save_new(root/'outer_publication_timeout.json', {'status': 'TIMEOUT',
                 'outer_sha256': digest((root/'outer.json').read_bytes()), 'observed_seconds': time.monotonic()-started})
        value.update(status='TIMEOUT', complete_independent_check=False)
    return value


def audit_outer(root):
    root = Path(root); value = read(root/'outer.json'); plan = read(root/'plan.json')
    if (value['plan_sha256'] != digest((root/'plan.json').read_bytes()) or
            value['budget_seconds'] != 300 or value['work_seconds'] != 298 or
            not 0 <= value['driver_exit_seconds'] <= value['observed_seconds']):
        raise ValueError('outer identity/budget mismatch')
    late = root/'outer_publication_timeout.json'
    if late.exists():
        event = read(late)
        if event['outer_sha256'] != digest((root/'outer.json').read_bytes()) or event['observed_seconds'] < 300 or event['status'] != 'TIMEOUT':
            raise ValueError('outer publication event mismatch')
        return {**value, 'status': 'TIMEOUT', 'complete_independent_check': False}
    owner = value['outer_process']
    if owner is None or owner['killed'] or value['driver_exit_seconds'] >= 298 or value['observed_seconds'] >= 300:
        expected, complete = 'TIMEOUT', False
        if value['error'] and owner is None and value['observed_seconds'] < 298: expected = 'ERROR'
    elif owner['return_code'] != 0 or value['error']:
        expected, complete = 'ERROR', False
    else:
        inner = audit(root/'tail'); expected, complete = inner['status'], inner['complete_independent_check']
        if (root/'tail/job.json').exists():
            job = read(root/'tail/job.json')
            if job['started_monotonic'] != plan['started'] or job['request'] != plan['request'] or job['cache_enabled'] != plan['enabled']:
                raise ValueError('whole-request clock/identity changed')
    if (value['status'], value['complete_independent_check']) != (expected, complete):
        raise ValueError('outer admission mismatch')
    return value


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--driver', type=Path, required=True)
    a = p.parse_args(); run(**read(a.driver))
