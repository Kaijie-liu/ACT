"""Frozen two-arm observed-input smoke. Every evidence stage shares 300s."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from portable_proof.runtime import digest
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired, terminal_status
from scripts.optional_evidence_dev_contract import ACT, FREEZE, OUTPUT, ROOT, git, make_freeze, read, save, verify_freeze


def stage(args, directory, name, budget, env):
    allowance = budget.remaining(2)
    started = time.monotonic()
    with (directory / (name + '.log')).open('x') as log:
        process = subprocess.Popen(args, cwd=directory if name=='check' else ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=allowance)
            state = 'COMPLETED' if code == 0 else ('BUDGET_EXHAUSTED' if code == 3 else 'ERROR')
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, state = process.returncode, 'OUTER_TIMEOUT'
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    return {'state': state, 'return_code': code, 'start_seconds': started-budget.started,
            'elapsed_seconds': time.monotonic()-started, 'allowed_seconds': allowance}


def run_request(arm, freeze, env):
    directory = OUTPUT / arm
    directory.mkdir()
    started = time.monotonic()
    budget = EvidenceBudget(started)
    save(directory/'job.json', freeze['job'])
    save(directory/'execution.json', {'started_monotonic': started, 'total_seconds': 300, 'arm': arm})
    stages = {}
    verdict, complete_check, error = 'UNKNOWN_INCOMPLETE_EVIDENCE', False, None
    try:
        names = ['production'] if arm=='production_matched_v2' else ['capture','propose','precheck','package','check']
        for name in names:
            if name == 'check':
                info = read(directory/'packing.json')
                args = [ACT, '-I', '-S', str(directory/'portable/verify.py'),
                        '--bundle-hash', info['bundle_sha256'], '--statement-hash', info['statement_sha256']]
            else:
                args = [ACT, '-m', 'scripts.optional_evidence_dev_worker', name, str(directory), '--started', repr(started)]
            stages[name] = stage(args, directory, name, budget, env)
            save(directory/'stage_progress.json', stages)
            if stages[name]['state'] != 'COMPLETED':
                verdict = 'ERROR' if stages[name]['state']=='ERROR' else 'TIMEOUT'
                break
        else:
            if arm=='production_matched_v2':
                verdict = read(directory/'package/manifest.json')['status']
            else:
                result = read(directory/'check.log')['result']
                complete_check = True
                verdict = ('CHECKED_CONDITIONAL' if result['positive_obligations']==result['required_obligations']
                           else 'UNKNOWN_NONPOSITIVE_OR_MISSING_EVIDENCE')
    except EvidenceBudgetExpired:
        verdict = 'TIMEOUT'
    except BaseException as exc:
        verdict, error = 'ERROR', repr(exc)
    inventory = {str(p.relative_to(directory)): digest(p.read_bytes()) for p in directory.rglob('*') if p.is_file()}
    terminal = {'arm': arm, 'dataset_index': 98, 'stages': stages, 'error': error,
                'complete_independent_check': complete_check, 'artifact_sha256': inventory,
                'evidence_level': 'HZ_POLICY_ACCEPTED' if arm=='production_matched_v2' else 'CHECKED_RATIONAL_CONDITIONAL',
                'production_gate_changed': False, 'deployed_float_SAFE': False, 'budget_seconds':300,
                'wall_seconds': time.monotonic()-started, 'status': verdict}
    terminal['status'] = terminal_status(verdict, terminal['wall_seconds'], 300, complete_check)
    save(directory/'terminal.json', terminal)
    elapsed = time.monotonic()-started
    if elapsed > 300:
        terminal['status']='TIMEOUT'
    terminal['wall_seconds']=elapsed
    save(directory/'terminal.json', terminal)
    print(arm, terminal['status'], f'{elapsed:.3f}s', flush=True)
    return terminal


def run():
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    if Path(sys.executable).resolve()!=Path(ACT).resolve():
        raise ValueError('ACT environment required')
    freeze = verify_freeze()
    head = git('rev-parse','HEAD')
    remote = git('ls-remote','origin','refs/heads/feat/moe-route-verification').split()[0]
    if remote != head:
        raise ValueError('live remote publication required')
    from scripts.run_conv_sign_lp import resources
    resource = resources()
    OUTPUT.mkdir(exist_ok=False)
    runtime = {'execution_head':head, 'freeze_sha256':digest(FREEZE.read_bytes()),
               'resource':resource, 'remote_before_launch':remote, 'rows':[], 'unattempted':freeze['arms'],
               'classification':freeze['classification'], 'state':'RUNNING', 'no_follow_on_run':True}
    save(OUTPUT/'runtime.json', runtime)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
           'CUDA_VISIBLE_DEVICES':'', 'PYTHONHASHSEED':'0'}
    for arm in freeze['arms']:
        verify_freeze()
        row = run_request(arm, freeze, env)
        runtime['rows'].append(row)
        runtime['unattempted']=freeze['arms'][len(runtime['rows']):]
        save(OUTPUT/'runtime.json', runtime)
        if row['status']=='ERROR':
            runtime['state']='STOPPED_ERROR';break
    else:
        runtime['state']='COMPLETED'
    save(OUTPUT/'runtime.json',runtime)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',action='store_true')
    args=p.parse_args()
    if args.freeze:
        if FREEZE.exists():raise FileExistsError('freeze immutable')
        save(FREEZE,make_freeze())
    else:
        with (ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
