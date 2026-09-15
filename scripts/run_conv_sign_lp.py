"""Two separately frozen sign-evidence controls; never changes a SAFE gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.conv_sign_lp_contract import (ROOT, DEFAULT, FREEZE, FILES, read, save, sha,
                                           git, parents, jobs, source_identity, verify_freeze, validate_job)


class CapturedProperty(BaseException):
    """Intentional capture stop, never a verification result."""


def capture_worker(directory):
    from scripts.budget_contract_v2 import verify_v2
    from act.pipeline.moe.external_pair_worker import load
    from act.back_end.solver.hz_lp_export import export
    import act.back_end.moe.monolithic_f0 as mono
    job = validate_job(directory); req = job['parent_request']
    if (directory / 'export.json').exists() or (directory / 'budget_journal.jsonl').exists():
        raise FileExistsError('capture is one-shot')
    started = time.monotonic(); model, tensors = load(req)
    original = mono._build_disjunction
    captured = {}
    def stop(encodings):
        scope = {'pairs': [list(e.pair) for e in encodings], 'row': list(encodings[0].property_row),
                 'constant': encodings[0].property_constant}
        if len(encodings) != 1 or scope != job['expected_scope']:
            raise ValueError('first property differs; do not seek a replacement')
        record = export(encodings[0].output_hz, [1], sparse=True)
        save(directory / 'export.json', record)
        captured.update(scope=scope, source_sha256=record['source_sha256'], export_sha256=sha(directory / 'export.json'),
                        parent_request_sha256=job['parent_request_sha256'], capture_seconds=time.monotonic()-started,
                        property_milp_called=False)
        raise CapturedProperty()
    mono._build_disjunction = stop
    try:
        try:
            verify_v2(model, tensors['center'], req['epsilon'], read(req['config']['path']),
                journal_path=directory / 'budget_journal.jsonl', started=started,
                identity={'proof_control_job_sha256': sha(directory / 'job.json')},
                expected_clean_prediction=req['sample']['label'],
                checkpoint_identity={'path': req['subject']['checkpoint'], 'sha256': req['subject']['checkpoint_sha256']},
                common_fact_callback=lambda v: save(directory / 'common_facts.json', v))
        except CapturedProperty:
            pass
        else:
            raise ValueError('verification returned before fixed property was captured')
    finally:
        mono._build_disjunction = original
    captured['journal_sha256'] = sha(directory / 'budget_journal.jsonl')
    save(directory / 'capture.json', captured)


def proposal_worker(directory):
    from act.back_end.solver.lp_certificate import propose
    job = validate_job(directory); started = time.monotonic()
    if (directory / 'proposal.json').exists() or (directory / 'certificate.json').exists():
        raise FileExistsError('proposal is one-shot')
    rec = read(directory / 'export.json')
    try:
        cert = propose(rec['lp'], time_limit=job['protocol']['proposal_solver_seconds'])
    except ValueError as exc:
        if str(exc) != 'proposal solver did not complete': raise
        result = {'status': 'UNAVAILABLE', 'reason': str(exc)}
    else:
        save(directory / 'certificate.json', cert)
        result = {'status': 'PROPOSED', 'certificate_sha256': sha(directory / 'certificate.json')}
    save(directory / 'proposal.json', {**result, 'export_sha256': sha(directory / 'export.json'),
                                       'proposal_and_local_check_seconds': time.monotonic()-started})


def subprocess_stage(args, directory, name, seconds, env):
    started = time.monotonic()
    with (directory / (name + '.log')).open('x') as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
            state = 'COMPLETED' if code == 0 else 'ERROR'
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(); code = process.returncode; state = 'TIMEOUT'
    result = {'state': state, 'return_code': code, 'wall_seconds': time.monotonic()-started, 'cap_seconds': seconds}
    save(directory / (name + '.terminal.json'), result)
    return result


def resources():
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    free = shutil.disk_usage(ROOT).free
    value = {'available_ram_gib': available / 2**30, 'free_disk_gib': free / 2**30,
             'load_per_core': os.getloadavg()[0] / os.cpu_count()}
    if value['available_ram_gib'] < 16 or value['free_disk_gib'] < 5 or value['load_per_core'] > .5:
        raise RuntimeError('resource gate closed; no automatic rerun')
    return value


def run():
    if git('branch','--show-current') != 'feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    freeze = verify_freeze(); resource = resources(); DEFAULT.mkdir(exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':''}
    rt = {'schema':'CONV_SIGN_LP_R1_RUN', 'execution_head':git('rev-parse','HEAD'), 'freeze_sha256':sha(FREEZE),
          'resource_at_start':resource, 'cases':[], 'unattempted':[j['case'] for j in freeze['jobs']],
          'state':'RUNNING', 'extra_queries_queued':False}
    save(DEFAULT / 'runtime.json', rt)
    for job in freeze['jobs']:
        if source_identity() != freeze['sources'] or git('status','--porcelain'): raise ValueError('execution source drift')
        directory = DEFAULT / job['case']['job_id']; directory.mkdir()
        save(directory / 'job.json', {**job, 'protocol':freeze['protocol'], 'freeze_sha256':sha(FREEZE)})
        row = {'case':job['case'], 'stages':{}, 'result':None}
        rt['unattempted'].pop(0); rt['cases'].append(row); save(DEFAULT / 'runtime.json', rt)
        for name, seconds in [('capture',freeze['protocol']['capture_seconds']), ('proposal',freeze['protocol']['proposal_outer_seconds']),
                              ('check',freeze['protocol']['checker_seconds']), ('independent',freeze['protocol']['checker_seconds'])]:
            args = ([sys.executable,'-m','scripts.run_conv_sign_lp','--'+name,str(directory)] if name in ('capture','proposal') else
                    [sys.executable,'-S','-m','scripts.check_conv_sign_lp',str(directory),str(directory / (name+'.json'))])
            row['stages'][name] = subprocess_stage(args,directory,name,seconds,env); save(DEFAULT / 'runtime.json', rt)
            if row['stages'][name]['state'] != 'COMPLETED':
                rt['state'] = row['stages'][name]['state']; save(DEFAULT / 'runtime.json', rt); return
        if read(directory / 'check.json') != read(directory / 'independent.json'): raise ValueError('checks disagree')
        row['result'] = read(directory / 'check.json'); save(DEFAULT / 'runtime.json', rt)
    parents(); rt['state']='COMPLETED_CHECKED'; save(DEFAULT / 'runtime.json', rt)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true'); group.add_argument('--run',action='store_true')
    group.add_argument('--capture',type=Path); group.add_argument('--proposal',type=Path); a=parser.parse_args()
    if a.freeze:
        if FREEZE.exists(): raise FileExistsError('no overwrite of freeze')
        save(FREEZE,{'protocol':parents(),'jobs':jobs(),'sources':source_identity(),'scope':'Frozen before any sign-control model query'})
    elif a.capture: capture_worker(a.capture)
    elif a.proposal: proposal_worker(a.proposal)
    else:
        with (ROOT / 'data/moe/results/conv_sign_lp_r1.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Do not mutate an old execution if invocation targets an existing root.
            if DEFAULT.exists(): raise FileExistsError('no resume/retry')
            try: run()
            except BaseException as exc:
                if (DEFAULT / 'runtime.json').exists():
                    rt = read(DEFAULT / 'runtime.json'); rt.update(state='ERROR', error=repr(exc))
                    save(DEFAULT / 'runtime.json', rt)
                raise
