"""Opt-in full capture/proposal/tail flow. Original methods remain frozen."""
import importlib
import math
from pathlib import Path
import subprocess
import time

from single_check_portable.execution import ROOT, ACT, read, save_new, left
from portable_proof.runtime import digest
from evidence_cohort.run import environment, wait_owned

ARMS = ('double_check', 'single_check')
VALID = ('CHECKED_CONDITIONAL', 'UNKNOWN_MISSING_EVIDENCE',
         'UNKNOWN_NONPOSITIVE', 'UNKNOWN_ROUTE_COVERAGE')


def tail(arm):
    if arm not in ARMS: raise ValueError('unregistered arm')
    return importlib.import_module(('cached_portable' if arm == 'double_check'
                                    else 'single_check_portable') + '.execution')


def phase_state(owner, elapsed):
    if owner['killed'] or owner['return_code'] == 3 or elapsed >= 298: return 'TIMEOUT'
    return 'COMPLETED' if owner['return_code'] == 0 else 'ERROR'


def drive(root):
    root = Path(root); plan = read(root/'plan.json'); started = plan['started']
    module = tail(plan['arm']); source = root/'source'; stages = []
    status = 'ERROR'; complete = False; error = None
    try:
        left(started)
        source.mkdir(exist_ok=False)
        save_new(source/'request.json', plan['request'])
        for name in ('capture', 'propose'):
            left(started); begin = time.monotonic()-started
            save_new(root/(name+'_entered.json'), {'phase': name, 'seconds': begin})
            with (root/(name+'.log')).open('xb') as log:
                proc = subprocess.Popen([ACT, '-m', 'evidence_handoff.worker', name,
                       str(source), '--started', repr(started)], cwd=ROOT, env=environment(),
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       start_new_session=True)
                owner = wait_owned(proc, started+298)
            end = time.monotonic()-started
            row = {'name': name, 'start_seconds': begin, 'end_seconds': end,
                   'elapsed_seconds': end-begin, 'process': owner, 'state': phase_state(owner, end)}
            stages.append(row); save_new(root/(name+'_stage.json'), row)
            if row['state'] != 'COMPLETED':
                status = row['state']; break
        else:
            left(started)
            save_new(root/'tail_entered.json', {'phase': 'tail', 'seconds': time.monotonic()-started})
            # Same original clock. The whole driver owns upstream AND tail at +298.
            module.run(source, plan['request']['evidence_request'], root/'tail',
                       started=started, enabled=True)
            admission = module.audit(root/'tail')
            status, complete = admission['status'], admission['complete_independent_check']
    except TimeoutError:
        status = 'TIMEOUT'
    except Exception as exc:
        error = repr(exc); status = 'ERROR'
    elapsed = time.monotonic()-started
    if elapsed >= 300: status, complete = 'TIMEOUT', False
    files = [root/(n+suffix) for n in ('capture', 'propose')
             for suffix in ('_entered.json', '_stage.json', '.log')]
    files += [source/'request.json', source/'manifest.json', source/'handoff.json',
              source/'capture.json', source/'query_log.json', root/'tail_entered.json']
    inventory = {str(p.relative_to(root)): digest(p.read_bytes()) for p in files if p.exists()}
    save_new(root/'candidate.json', {'status': status, 'complete_independent_check': complete,
             'stages': stages, 'error': error, 'artifact_sha256': inventory,
             'plan_sha256': digest((root/'plan.json').read_bytes()),
             'wall_seconds': time.monotonic()-started})


def review_candidate(root):
    """Independent execution/binding/accounting review; LP math remains in checker."""
    root = Path(root); plan = read(root/'plan.json'); c = read(root/'candidate.json')
    if c['plan_sha256'] != digest((root/'plan.json').read_bytes()): raise ValueError('plan changed')
    for name, sha in c['artifact_sha256'].items():
        p = root/name
        if not p.resolve().is_relative_to(root.resolve()) or digest(p.read_bytes()) != sha:
            raise ValueError('upstream artifact changed')
    request_file=root/'source/request.json'
    if request_file.exists():
        if read(request_file) != plan['request']: raise ValueError('request changed')
    elif c['wall_seconds'] < 298 and not c['error']: raise ValueError('missing request')
    previous = 0.
    for i, row in enumerate(c['stages']):
        if (i >= 2 or row['name'] != ('capture', 'propose')[i] or
                not previous <= row['start_seconds'] <= row['end_seconds'] <= c['wall_seconds'] or
                abs(row['elapsed_seconds']-row['end_seconds']+row['start_seconds']) > 1e-8 or
                row['state'] != phase_state(row['process'], row['end_seconds']) or
                row != read(root/(row['name']+'_stage.json'))): raise ValueError('upstream stage accounting')
        previous = row['end_seconds']
    done = len(c['stages']) == 2 and all(s['state'] == 'COMPLETED' for s in c['stages'])
    if c['wall_seconds'] >= 298 and not (root/'tail/admission.json').exists():
        # Deadline at the upstream→tail boundary: no checker output exists to inspect.
        expected, complete = 'TIMEOUT', False
    elif c['wall_seconds'] >= 300: expected, complete = 'TIMEOUT', False
    elif c['error']: expected, complete = 'ERROR', False
    elif not done:
        expected, complete = (c['stages'][-1]['state'] if c['stages'] else 'TIMEOUT'), False
    else:
        result = tail(plan['arm']).audit(root/'tail')
        job_path=root/'tail/job.json'
        if not job_path.exists():
            if result['status']!='TIMEOUT' or result['complete_independent_check']:
                raise ValueError('missing tail job')
            if (c['status'],c['complete_independent_check']) != ('TIMEOUT',False):
                raise ValueError('missing job promoted')
            return c
        job = read(job_path)
        if (job['started_monotonic'] != plan['started'] or job['cache_enabled'] is not True or
                job['request'] != plan['request']['evidence_request'] or
                Path(job['source']) != (root/'source').resolve() or
                job['manifest_sha256'] != digest((root/'source/manifest.json').read_bytes()) or
                job['upstream_elapsed_at_entry'] < previous): raise ValueError('tail clock/source binding')
        handoff = read(root/'source/handoff.json')
        if (handoff['total_seconds'], handoff['proposal_cap_seconds'], handoff['check_reserve_seconds']) != (300,60,80):
            raise ValueError('proposal policy changed')
        # The old diagnostic says next_stage=UNCHANGED_PRECHECK; it has no authority here.
        expected, complete = result['status'], result['complete_independent_check']
    if (c['status'], c['complete_independent_check']) != (expected, complete):
        raise ValueError('candidate acceptance mismatch')
    return c


def supervise(request, arm, destination, *, started):
    if arm not in ARMS or not math.isfinite(started) or started > time.monotonic():
        raise ValueError('registered arm/original clock required')
    root = Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    save_new(root/'plan.json', {'request': request, 'arm': arm, 'started': started,
                              'total_seconds': 300, 'work_seconds': 298})
    owner = None; candidate = None; error = None; status = 'TIMEOUT'
    try:
        left(started)
        with (root/'driver.log').open('xb') as log:
            proc = subprocess.Popen([ACT, '-m', 'upstream_portable.flow', str(root)],
                   cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL, stdout=log,
                   stderr=subprocess.STDOUT, start_new_session=True)
            owner = wait_owned(proc, started+298)
        exited = time.monotonic()-started
        if owner['killed'] or exited >= 298: status = 'TIMEOUT'
        elif owner['return_code'] != 0: status = 'ERROR'
        else:
            candidate = review_candidate(root); status = candidate['status']
    except TimeoutError:
        exited = time.monotonic()-started
    except Exception as exc:
        exited = time.monotonic()-started; error = repr(exc); status = 'ERROR'
    observed = time.monotonic()-started
    if observed >= 300: status = 'TIMEOUT'
    value = {'schema': 'UPSTREAM_PORTABLE_V1', 'status': status,
             'complete_independent_check': status in VALID and bool(candidate and candidate['complete_independent_check']),
             'arm': arm, 'plan_sha256': digest((root/'plan.json').read_bytes()),
             'candidate_sha256': digest((root/'candidate.json').read_bytes()) if candidate else None,
             'outer_process': owner, 'driver_exit_seconds': exited, 'observed_seconds': observed,
             'error': error, 'budget_seconds': 300, 'work_seconds': 298,
             'evidence_level': 'CHECKED_RATIONAL_CONDITIONAL', 'production_gate_changed': False,
             'deployed_float_SAFE': False}
    save_new(root/'outer.json', value)
    end = time.monotonic()-started
    save_new(root/'publication.json', {'outer_sha256': digest((root/'outer.json').read_bytes()),
                                     'observed_seconds': end})
    if time.monotonic()-started >= 300:
        save_new(root/'publication_timeout.json', {'status': 'TIMEOUT', 'complete_independent_check': False})
        value.update(status='TIMEOUT', complete_independent_check=False)
    return value


def audit(root):
    root = Path(root); v = read(root/'outer.json'); p = read(root/'plan.json'); pub = read(root/'publication.json')
    tail(p['arm'])
    if (v['plan_sha256'] != digest((root/'plan.json').read_bytes()) or
            pub['outer_sha256'] != digest((root/'outer.json').read_bytes()) or
            v['arm'] != p['arm'] or (p['total_seconds'],p['work_seconds']) != (300,298) or
            (v['budget_seconds'],v['work_seconds']) != (300,298) or
            not 0 <= v['driver_exit_seconds'] <= v['observed_seconds'] <= pub['observed_seconds']):
        raise ValueError('outer identity/clock mismatch')
    owner = v['outer_process']; expected = 'TIMEOUT'; complete = False
    if owner and not owner['killed'] and v['driver_exit_seconds'] < 298 and v['observed_seconds'] < 300:
        if owner['return_code'] != 0 or v['error']: expected = 'ERROR'
        else:
            if v['candidate_sha256'] != digest((root/'candidate.json').read_bytes()): raise ValueError('candidate drift')
            c = review_candidate(root); expected, complete = c['status'], c['complete_independent_check']
    elif owner is None and v['error'] and v['observed_seconds'] < 298: expected = 'ERROR'
    if (v['status'], v['complete_independent_check']) != (expected, complete): raise ValueError('outer admission mismatch')
    if pub['observed_seconds'] >= 300 or (root/'publication_timeout.json').exists():
        return {**v, 'status': 'TIMEOUT', 'complete_independent_check': False}
    return v


def costs(root):
    """Saved clocks only. Missing/cut-off phases are never charged as zero."""
    root=Path(root);v=audit(root);parts={}
    for name in ('capture','propose'):
        path=root/(name+'_stage.json');entered=root/(name+'_entered.json')
        if path.exists():
            row=read(path);parts[name]={'seconds':row['elapsed_seconds'],'state':row['state'],
                                     'censored':row['state']=='TIMEOUT'}
        elif entered.exists():
            parts[name]={'seconds':None,'observed_window_seconds':max(0,v['driver_exit_seconds']-read(entered)['seconds']),
                         'state':'INTERRUPTED','censored':True}
        else:parts[name]={'seconds':None,'state':'NOT_REACHED','censored':False}
    for name in ('precheck','package','check'):
        path=root/'tail'/(name+'_stage.json')
        if path.exists():
            row=read(path);parts[name]={'seconds':row['elapsed_seconds'],'state':row['state'],
                                      'censored':row['state']=='TIMEOUT'}
        else:parts[name]={'seconds':None,'state':'NOT_COMPLETED_OR_NOT_APPLICABLE','censored':None}
    query=root/'source/query_log.json'
    proposals=read(query) if query.exists() else []
    # Query time includes proposed solve/serialization and overlaps propose phase.
    return {'whole_request_seconds':read(root/'publication.json')['observed_seconds'],
            'phases':parts,'proposal_queries':proposals,
            'query_cost_scope':'nested in propose; do not add twice; PENDING is censored',
            'capture_scope':'loading, routing, support, propagation and export combined; see budget_journal for subevents',
            'unassigned_seconds':'startup, source I/O, packing inventories, audits and terminal publication included in whole clock'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    drive(parser.parse_args().root)
