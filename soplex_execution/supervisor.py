"""Sequential fixed roster, resource gate, bounded workers and fail-closed terminals."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from lp_sandwich.check import strict_json
from soplex_fidelity.io import save,sha
from soplex_execution.runtime import ROOT,PYTHON,env,limits,wait,costs,output_sizes

FREEZE=ROOT/'docs/soplex_finite_comparison_v1_freeze.json'
ADDENDUM=ROOT/'docs/soplex_execution_v1_freeze.json'


def verify_freeze():
    add=strict_json(ADDENDUM.read_bytes());f=strict_json(FREEZE.read_bytes())
    if add['protocol_sha256']!=sha(FREEZE) or not add['execution_ready']:raise ValueError('execution identity')
    for path,digest in add['bindings'].items():
        if sha(path)!=digest:raise ValueError('frozen execution dependency drift: '+path)
    return f,add


def resources():
    mem={a.split(':')[0]:int(a.split()[1]) for a in Path('/proc/meminfo').read_text().splitlines()}
    return dict(available_ram_gib=mem['MemAvailable']/1024**2,
                free_disk_gib=shutil.disk_usage(ROOT).free/1024**3,
                load_per_core=os.getloadavg()[0]/os.cpu_count())


def resource_gate(root,job_id,policy):
    start=time.monotonic();observations=[]
    while True:
        state=resources();elapsed=time.monotonic()-start
        okay=(state['available_ram_gib']>=policy['minimum_ram_gib'] and
              state['free_disk_gib']>=policy['minimum_disk_gib'] and
              state['load_per_core']<=policy['maximum_load_per_core'])
        observations.append(dict(seconds=elapsed,**state,passed=okay))
        with (root/'resource_wait.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(job_id=job_id,**observations[-1]))+'\n');stream.flush()
        if okay:return dict(seconds=time.monotonic()-start,polls=len(observations),last=state)
        if elapsed>=policy['wait_limit_seconds']:raise TimeoutError('resource wait limit, no LP launched')
        print(json.dumps(dict(state='WAITING_RESOURCES',job_id=job_id,**state)),flush=True)
        time.sleep(min(policy['poll_seconds'],policy['wait_limit_seconds']-elapsed))


def publication_valid(terminal,receipt,total):
    return (receipt['terminal_sha256']==terminal['sha256'] and receipt['published_offset']<=total
            and receipt['published_offset']>=receipt['serialization_started_offset']>=0)


def supervise(job,root,runtime,policy,command=None):
    """Test commands exercise the same watchdog; production always uses the frozen worker."""
    root=Path(root);root.mkdir();start=time.monotonic()
    spec=dict(root=str(root),start=start,job=job,runtime=runtime,
              proposal=policy['proposal_seconds'],work=policy['work_seconds'])
    save(root/'spec.json',spec)
    if command is None:command=[str(PYTHON),'-m','soplex_execution.worker',str(root/'spec.json')]
    def deadline():
        return start+(policy['work_seconds'] if (root/'package.entered.json').exists()
                      else policy['proposal_seconds'])
    with (root/'worker.stdout').open('xb') as out,(root/'worker.stderr').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,env=env(),
                                 preexec_fn=limits,start_new_session=True)
        watched=wait(process,deadline,root)
    observed=time.monotonic()-start
    candidate=root/'worker_result.json'
    result=dict(status='ERROR',error='worker exited without terminal',upper_bound=None,
                network_SAFE=False,network_UNSAFE=False)
    if watched['termination']:
        result.update(status='LIMIT' if watched['termination'].startswith('LimitError') else 'TIMEOUT',error=watched['termination'])
    elif candidate.exists():
        try:
            result=strict_json(candidate.read_bytes())
            if (result['job_id']!=job['job_id'] or result['statement_sha256']!=job['statement_sha256']
                or result['network_SAFE'] or result['network_UNSAFE']):raise ValueError('worker result identity')
            if result['completed_offset']>policy['work_seconds']:raise TimeoutError('late worker result')
            if result['status']=='CHECKED':
                for name in ('load','export','import','readback','solve','capture','package','check','review'):
                    row=strict_json((root/f'{name}.done.json').read_bytes())
                    if row['end_offset']>(policy['proposal_seconds'] if name in ('load','export','import','readback','solve','capture') else policy['work_seconds']):
                        raise TimeoutError('late completed phase')
                if sha(root/'bundle.json')!=result['bundle_sha256'] or sha(root/'checker.stdout')!=result['checker_stdout_sha256']:
                    raise ValueError('worker evidence identity')
        except TimeoutError as exc:result=dict(status='TIMEOUT',error=str(exc),upper_bound=None,network_SAFE=False,network_UNSAFE=False)
        except Exception as exc:result=dict(status='ERROR',error=str(exc),upper_bound=None,network_SAFE=False,network_UNSAFE=False)
    # Review and hash verification are also charged; never retain an upper bound after work expiry.
    if time.monotonic()-start>policy['work_seconds'] and result['status']=='CHECKED':
        result.update(status='TIMEOUT',error='outer review deadline',upper_bound=None)
    stages=costs(root,observed)
    result.update(schema='SOPLEX_SUPERVISED_TERMINAL_V1',job_id=job['job_id'],
        statement_sha256=job['statement_sha256'],watchdog=watched,phase_costs=stages,
        observed_worker_seconds=observed,prepublication_seconds=time.monotonic()-start,
        accounted_phase_seconds=sum(v['observed_seconds'] or 0 for v in stages.values()),
        files_before_terminal=output_sizes(root,False),proposal_seconds=policy['proposal_seconds'],
        work_seconds=policy['work_seconds'],total_seconds=policy['total_seconds'])
    result['startup_gaps_cleanup_review_seconds']=result['prepublication_seconds']-result['accounted_phase_seconds']
    serial_start=time.monotonic()-start
    save(root/'terminal.json',result)
    receipt=dict(serialization_started_offset=serial_start,published_offset=time.monotonic()-start,
                 terminal_sha256=sha(root/'terminal.json'))
    # The tiny receipt is the accepting boundary; a late receipt invalidates the whole candidate terminal.
    receipt['published_offset']=time.monotonic()-start
    save(root/'publication.json',receipt)
    actual=time.monotonic()-start
    save(root/'publication_observed.json',dict(completed_offset=actual,
        accepted_before_deadline=actual<=policy['total_seconds'],seconds=actual-serial_start))
    actual=time.monotonic()-start
    if actual>policy['total_seconds']:
        result.update(status='TIMEOUT',upper_bound=None,error='late publication')
    result['whole_request_seconds']=actual
    print(json.dumps(dict(job_id=job['job_id'],status=result['status'],upper_bound=result['upper_bound'],seconds=actual)),flush=True)
    return result


def batch():
    f,add=verify_freeze()
    readiness=strict_json((ROOT/'docs/soplex_execution_v1_readiness_review.json').read_bytes())
    if readiness['status']!='PASS' or readiness['issues'] or readiness['addendum_sha256']!=sha(ADDENDUM):
        raise ValueError('fresh readiness review required')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':
        raise RuntimeError('feature branch required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('clean committed execution tree required')
    root=Path(f['output']);root.mkdir()
    save(root/'execution_identity.json',dict(protocol_sha256=sha(FREEZE),addendum_sha256=sha(ADDENDUM),
         git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    runtime={k:f['runtime'][k] for k in ('soplex','reader','settings')};runtime['checker']=f['independent_checker']
    return run_jobs(f['jobs'],root,runtime,f['policy'],verify_freeze)


def run_jobs(jobs,root,runtime,policy,revalidate):
    rows=[];stopped=False;started=time.monotonic()
    for job in jobs:
        if stopped:
            rows.append(dict(job_id=job['job_id'],status='NOT_RUN_AFTER_ERROR'));continue
        try:gate=resource_gate(root,job['job_id'],policy['resource'])
        except TimeoutError as exc:
            rows.append(dict(job_id=job['job_id'],status='NOT_RUN_RESOURCE_WAIT_LIMIT',error=str(exc)))
            continue
        revalidate()
        row=supervise(job,root/job['job_id'],runtime,policy);row['resource_wait']=gate
        rows.append(row);save(root/(job['job_id']+'.summary.json'),row)
        stopped=row['status']=='ERROR'
    summary=dict(schema='SOPLEX_FINITE_BATCH_V1',rows=rows,denominator=len(jobs),
                 wall_seconds=time.monotonic()-started,optimization_calls=sum((root/j['job_id']/'solver.command.json').exists() for j in jobs),
                 network_SAFE=False,network_UNSAFE=False)
    save(root/'batch.json',summary)
    return summary


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();batch()
