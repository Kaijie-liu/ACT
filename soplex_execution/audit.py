"""Saved evidence/cost audit and fresh isolated original-LP checks; no solver calls."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import time

from lp_sandwich.check import identity,strict_json
from soplex_fidelity.io import sha,save
from soplex_fidelity.review import compare
from soplex_execution.runtime import PYTHON,env,limits


def audit_job(job,root,review_root,policy,runtime):
    start=time.monotonic();t=strict_json((root/'terminal.json').read_bytes())
    p=strict_json((root/'publication.json').read_bytes());o=strict_json((root/'publication_observed.json').read_bytes())
    spec=strict_json((root/'spec.json').read_bytes())
    if spec['job']!=job or spec['runtime']!=runtime:raise ValueError('job/runtime binding')
    if (spec['proposal']!=policy['proposal_seconds'] or spec['work']!=policy['work_seconds'] or
        t['total_seconds']!=policy['total_seconds']):raise ValueError('clock policy drift')
    if p['terminal_sha256']!=sha(root/'terminal.json'):raise ValueError('terminal changed')
    timely=o['completed_offset']<=policy['total_seconds']
    if o['accepted_before_deadline']!=timely:raise ValueError('publication acceptance')
    if not 0<=t['observed_worker_seconds']<=t['prepublication_seconds']<=p['serialization_started_offset']<=p['published_offset']<=o['completed_offset']:
        raise ValueError('whole cost ordering')
    total_phase=0;previous=0
    for name,row in t['phase_costs'].items():
        # Dict sorting is not temporal sorting; use explicit interval overlap check below.
        entered=root/f'{name}.entered.json';done=root/f'{name}.done.json'
        if not entered.exists():
            if row!={'seconds':None,'observed_seconds':None,'state':'NOT_ENTERED'}:raise ValueError('missing cost is not null')
            continue
        a=strict_json(entered.read_bytes());b=strict_json(done.read_bytes()) if done.exists() else None
        expected=(t['observed_worker_seconds'] if b is None else b['end_offset'])-a['start_offset']
        if abs(row['observed_seconds']-expected)>1e-7 or expected<0:raise ValueError('phase elapsed accounting')
        if b is None:
            if row['seconds'] is not None or row['state']!='CENSORED':raise ValueError('censored cost')
        elif row['seconds']!=b['seconds'] or row['state']!='COMPLETE':raise ValueError('complete cost')
        total_phase+=expected
    intervals=sorted((v['start_offset'],v['end_offset'] or t['observed_worker_seconds']) for v in t['phase_costs'].values() if 'start_offset'in v)
    for a,b in intervals:
        if a<previous or b<a:raise ValueError('overlapping phase costs')
        previous=b
    if abs(total_phase-t['accounted_phase_seconds'])>1e-6 or abs(total_phase+t['startup_gaps_cleanup_review_seconds']-t['prepublication_seconds'])>1e-6:
        raise ValueError('cost total closure')
    if t['network_SAFE'] or t['network_UNSAFE']:raise ValueError('network overclaim')
    if t['job_id']!=job['job_id'] or t['statement_sha256']!=job['statement_sha256']:raise ValueError('statement drift')
    result=dict(job_id=job['job_id'],status=t['status'] if timely else 'TIMEOUT',
                upper_bound=None,whole_request_seconds=o['completed_offset'],phase_costs=t['phase_costs'],
                publication_seconds=o['seconds'],startup_gaps_cleanup_review_seconds=t['startup_gaps_cleanup_review_seconds'],
                solver_attempts=int((root/'solver.command.json').exists()),network_SAFE=False,network_UNSAFE=False)
    if sha(job['export']['path'])!=job['export']['sha256']:raise ValueError('original source bytes')
    original=strict_json(Path(job['export']['path']).read_bytes());lp=original['lp']
    if identity(lp)!=job['statement']['lp_sha256']:raise ValueError('original LP')
    if (root/'readback.done.json').exists():result['input_comparison']=compare(lp,root/'readback.txt')
    for name in ('reader','solver','checker'):
        path=root/f'{name}.command.json'
        if path.exists():
            r=strict_json(path.read_bytes())
            boundary=policy['work_seconds'] if name=='checker' else policy['proposal_seconds']
            if r['deadline_offset']!=boundary or not 0<=r['start_offset']<boundary:raise ValueError('native clock reset')
            cmd=r['command']
            if name=='solver':
                if len(cmd)!=5 or cmd[0]!=runtime['soplex']['path'] or cmd[1]!='--loadset='+runtime['settings']['path'] or cmd[3]!='-X='+str(root/'point.txt') or cmd[4]!=str(root/'input.lp'):
                    raise ValueError('solver options changed')
                if not 0<float(cmd[2][2:])<=boundary-r['start_offset']+.01:raise ValueError('extra native time')
        resource_path=root/f'{name}.resources'
        result[name+'_gnu_time']=resource_path.read_text() if resource_path.exists() else None
    if t['status']=='CHECKED' and timely:
        for name in ('load','export','import','readback','solve','capture','package','check','review'):
            b=strict_json((root/f'{name}.done.json').read_bytes())
            if b['end_offset']>(policy['proposal_seconds'] if name in ('load','export','import','readback','solve','capture') else policy['work_seconds']):raise ValueError('late proof phase')
        if sha(root/'check.py')!=runtime['checker']['sha256']:raise ValueError('checker drift')
        b=strict_json((root/'bundle.json').read_bytes())
        if b['lp']!=lp or b['statement']!=job['statement'] or b['dual'] is not None:raise ValueError('original LP replaced')
        if sha(root/'bundle.json')!=t['bundle_sha256'] or sha(root/'checker.stdout')!=t['checker_stdout_sha256']:raise ValueError('evidence changed')
        review_root.mkdir()
        command=[str(PYTHON),'-I','-S',str(root/'check.py'),str(root/'bundle.json'),
                 '--bundle-sha256',sha(root/'bundle.json'),'--statement-sha256',job['statement_sha256'],'--timeout-seconds','60']
        cp=subprocess.run(command,capture_output=True,timeout=62,env=env(),preexec_fn=limits)
        save(review_root/'process.json',dict(command=command,returncode=cp.returncode,stdout=cp.stdout.decode(),stderr=cp.stderr.decode()))
        if cp.returncode!=0:raise ValueError('fresh isolated checker rejected')
        fresh=strict_json(cp.stdout);saved=strict_json((root/'checker.stdout').read_bytes())
        if {k:v for k,v in fresh.items() if k!='seconds'}!={k:v for k,v in saved.items() if k!='seconds'}:raise ValueError('fresh checker disagreement')
        if fresh['upper_bound']!=t['upper_bound']:raise ValueError('terminal upper differs')
        result.update(upper_bound=fresh['upper_bound'],primal_status=fresh['primal_status'],classification=fresh['classification'])
        if fresh['upper_bound'] is not None:
            result['interpretation']='LP relaxation obstruction, NOT network UNSAFE' if F(fresh['upper_bound'])<=0 else 'Feasible upper bound only, safety unresolved'
    result['output_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    result['artifacts']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    result['archival_review_seconds']=time.monotonic()-start
    return result


def audit_batch(freeze,addendum,output):
    f=strict_json(freeze.read_bytes());root=Path(f['output']);a=strict_json(addendum.read_bytes())
    for path,digest in a['bindings'].items():
        if sha(path)!=digest:raise ValueError('execution source/runtime drift')
    execution=strict_json((root/'execution_identity.json').read_bytes())
    if execution['protocol_sha256']!=sha(freeze) or execution['addendum_sha256']!=sha(addendum):raise ValueError('execution identity')
    saved=strict_json((root/'batch.json').read_bytes());rows=[];start=time.monotonic()
    if [v['job_id'] for v in saved['rows']]!=[j['job_id'] for j in f['jobs']] or saved['denominator']!=4:raise ValueError('roster/denominator')
    runtime={k:f['runtime'][k] for k in ('soplex','reader','settings')};runtime['checker']=f['independent_checker']
    review=root/'archival_review';review.mkdir();stopped=False
    for job,row in zip(f['jobs'],saved['rows']):
        if stopped:
            if row['status']!='NOT_RUN_AFTER_ERROR':raise ValueError('fail-stop violated')
            rows.append(row);continue
        if row['status']=='NOT_RUN_RESOURCE_WAIT_LIMIT':rows.append(row);continue
        fresh=audit_job(job,root/job['job_id'],review/job['job_id'],f['policy'],runtime)
        if row['whole_request_seconds']<fresh['whole_request_seconds']:raise ValueError('publication receipt cost missing')
        fresh['whole_request_seconds']=row['whole_request_seconds']
        if row['whole_request_seconds']>f['policy']['total_seconds']:
            fresh.update(status='TIMEOUT',upper_bound=None)
        if fresh['status']!=row['status'] or fresh['upper_bound']!=row['upper_bound']:raise ValueError('batch terminal disagreement')
        fresh['resource_wait']=row['resource_wait'];rows.append(fresh);stopped=fresh['status']=='ERROR'
    result=dict(status='PASS',issues=[],schema='SOPLEX_FINITE_EXECUTION_REVIEW_V1',denominator=4,
         rows=rows,raw_root=str(root),protocol_sha256=sha(freeze),addendum_sha256=sha(addendum),
         execution=execution,optimization_calls=saved['optimization_calls'],fresh_review_solver_calls=0,
         batch_wall_seconds=saved['wall_seconds'],
         seconds=time.monotonic()-start,network_SAFE=False,network_UNSAFE=False,
         batch_sha256=sha(root/'batch.json'),scope='Original supplied rational LP feasible points and full execution accounting; no network proof.')
    save(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    from soplex_execution.supervisor import FREEZE,ADDENDUM
    print(json.dumps(audit_batch(FREEZE,ADDENDUM,a.output),indent=2))
