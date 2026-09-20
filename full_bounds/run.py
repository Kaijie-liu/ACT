"""300s stored-source candidate/check protocol, with partial evidence preservation."""
import json
import os
from pathlib import Path
import subprocess
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save

FREEZE=ROOT/'docs/full_bounds_v1_freeze.json'
DEST=ROOT/'data/moe/results/full_bounds_conv98_20260920_v1'


def supervise(root,command,env,budget=300,propose_cap=180):
    start=time.monotonic();stages=[];result=None;error=None;status='ERROR'
    reserve=min(2.,budget/10);deadline=start+budget-reserve
    try:
        for phase,cap in [('prepare',15),('propose',propose_cap),('seal',10),('check',budget)]:
            begin=time.monotonic()-start;save(root/(phase+'_entered.json'),{'seconds':begin})
            row=execute(command(phase),root/(phase+'.log'),min(deadline,time.monotonic()+cap),env)
            row.update(phase=phase,cap_seconds=cap,start_seconds=begin,end_seconds=time.monotonic()-start)
            stages.append(row);save(root/(phase+'_stage.json'),row)
            if row['state']!='COMPLETED' and phase!='propose':status=row['state'];break
        else:
            result=json.loads((root/'check.log').read_bytes());status=result['status']
            if status not in ('CHECKED_POSITIVE_DECLARED_REAL_MOE','UNKNOWN_MISSING_BOUND_EVIDENCE','UNKNOWN_NONPOSITIVE_BOUNDS'):
                raise ValueError('unregistered checker endpoint')
            if (result['production_verdict_changed'] or result['deployed_floating_point_proof'] or
                    result['complete_declared_real_output_proof']!=(status=='CHECKED_POSITIVE_DECLARED_REAL_MOE')):
                raise ValueError('claim/endpoint mismatch')
            if any(s['state']=='ERROR' for s in stages):status='ERROR'
    except Exception as exc:error=repr(exc);status='ERROR';result=None
    inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    elapsed=time.monotonic()-start
    if elapsed>=budget:status='TIMEOUT'
    positive=status=='CHECKED_POSITIVE_DECLARED_REAL_MOE'
    terminal={'status':status,'error':error,'check':result,'stages':stages,'inventory':inventory,
        'budget_seconds':budget,'elapsed_before_publication':elapsed,'complete_declared_real_output_proof':positive,
        'production_verdict_changed':False,'deployed_floating_point_proof':False}
    save(root/'terminal.json',terminal)
    save(root/'publication.json',{'seconds':time.monotonic()-start,'terminal_sha256':sha(root/'terminal.json')})
    if time.monotonic()-start>=budget:save(root/'publication_timeout.json',{'status':'TIMEOUT','complete_declared_real_output_proof':False})
    return terminal


def run():
    f=json.loads(FREEZE.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('frozen dependency changed: '+name)
    if f['output']!=str(DEST.relative_to(ROOT)) or f['budget_seconds']!=300:raise ValueError('protocol drift')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':raise ValueError('branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('clean worktree required')
    if os.getloadavg()[0]/os.cpu_count()>.5:raise ValueError('resource gate, not launched')
    DEST.mkdir(exist_ok=False)
    save(DEST/'execution.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'freeze_sha256':sha(FREEZE),'budget_seconds':300,'cost_scope':f['cost_scope']})
    def command(phase):
        if phase!='check':return [ACT,'-m','full_bounds.worker',phase,str(DEST)]
        sealed=json.loads((DEST/'sealed.json').read_bytes())
        return [ACT,'-I','-S',str(DEST/'relocated/verify_bounds.py'),'--manifest-hash',sealed['manifest_sha256']]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=supervise(DEST,command,env)
    print(json.dumps({'status':result['status'],'seconds':json.loads((DEST/'publication.json').read_bytes())['seconds'],'root':str(DEST)}))


if __name__=='__main__':run()
