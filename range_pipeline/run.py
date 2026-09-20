"""Two separately budgeted, frozen stored-source arms; no old results overwritten."""
import json
import os
from pathlib import Path
import subprocess
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save

FREEZE=ROOT/'docs/range_pipeline_v1_freeze.json'


def supervise(root,command,env,budget=300):
    start=time.monotonic();deadline=start+budget-min(2.,budget/10);stages=[];result=None;error=None;status='ERROR'
    try:
        for phase,cap in [('build',100),('propose',120),('seal',5),('check',300)]:
            begin=time.monotonic()-start;end=min(deadline,time.monotonic()+cap)
            save(root/(phase+'_entered.json'),{'seconds':begin})
            if phase=='build':save(root/'build_window.json',{'deadline':end,'request_deadline':deadline})
            row=execute(command(phase),root/(phase+'.log'),end,env)
            row.update(phase=phase,cap_seconds=cap,start_seconds=begin,end_seconds=time.monotonic()-start)
            stages.append(row);save(root/(phase+'_stage.json'),row)
            if row['state']!='COMPLETED' and phase!='propose':status=row['state'];break
        else:
            result=json.loads((root/'check.log').read_bytes());status=result['status']
            if status not in ('CHECKED_POSITIVE_DECLARED_REAL_MOE','UNKNOWN_MISSING_BOUND_EVIDENCE','UNKNOWN_NONPOSITIVE_BOUNDS'):
                raise ValueError('unregistered endpoint')
            if result['production_verdict_changed'] or result['deployed_floating_point_proof']:
                raise ValueError('unregistered claim upgrade')
            if result['complete_declared_real_output_proof']!=(status=='CHECKED_POSITIVE_DECLARED_REAL_MOE'):
                raise ValueError('incomplete positivity')
            if any(s['state']=='ERROR' for s in stages):status='ERROR'
    except Exception as exc:error=repr(exc);result=None;status='ERROR'
    inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    elapsed=time.monotonic()-start
    if elapsed>=budget:status='TIMEOUT'
    terminal={'schema':'RANGED_SOURCE_TERMINAL_V1','status':status,'error':error,'stages':stages,'check':result,
        'budget_seconds':budget,'elapsed_before_publication':elapsed,'inventory':inventory,
        'complete_declared_real_output_proof':status=='CHECKED_POSITIVE_DECLARED_REAL_MOE',
        'production_verdict_changed':False,'deployed_floating_point_proof':False}
    save(root/'terminal.json',terminal)
    save(root/'publication.json',{'seconds':time.monotonic()-start,'terminal_sha256':sha(root/'terminal.json')})
    if time.monotonic()-start>=budget:save(root/'publication_timeout.json',{'status':'TIMEOUT','complete_declared_real_output_proof':False})
    return terminal


def commands(root):
    def command(phase):
        if phase=='build':return [ACT,'-m','range_pipeline.build',str(root)]
        if phase in ('propose','seal'):return [ACT,'-m','full_bounds.worker',phase,str(root)]
        sealed=json.loads((root/'sealed.json').read_bytes())
        return [ACT,'-I','-S',str(root/'relocated/verify_bounds.py'),'--manifest-hash',sealed['manifest_sha256']]
    return command


def run():
    f=json.loads(FREEZE.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('freeze drift: '+name)
    if f['arms']!=['range_off','range_on'] or f['budget_seconds_per_arm']!=300:raise ValueError('protocol drift')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':raise ValueError('branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('clean worktree required')
    if os.getloadavg()[0]/os.cpu_count()>.5:raise ValueError('resource gate: not started')
    root=ROOT/f['output'];root.mkdir(exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(root/'execution.json',{'head':head,'freeze_sha256':sha(FREEZE),'arms':f['arms']})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    for arm in f['arms']:
        if os.getloadavg()[0]/os.cpu_count()>.5:
            save(root/(arm+'_not_started.json'),{'state':'RESOURCE_WAIT_NOT_STARTED'});break
        dst=root/arm;dst.mkdir()
        job={'arm':arm,'prefix_source':str(ROOT/f['prefix_source']),'prefix_source_hash':f['prefix_source_hash'],
            'expert_document':str(ROOT/f['expert_document']),
            'input_files':{str(ROOT/k):v for k,v in f['artifacts'].items()}}
        save(dst/'job.json',job);result=supervise(dst,commands(dst),env)
        results.append({'arm':arm,'status':result['status'],'seconds':json.loads((dst/'publication.json').read_bytes())['seconds']})
    save(root/'batch_terminal.json',{'arms':results,'required_arms':f['arms'],'complete':len(results)==2,
        'scope':'Each arm has its own300s budget; no speedup or native floating claim.'})
    print(json.dumps(results))


if __name__=='__main__':run()
