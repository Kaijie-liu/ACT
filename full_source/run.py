"""One new 300s source-to-output-construction run, no optimizer or retries."""
import json
import os
from pathlib import Path
import subprocess
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save

FREEZE=ROOT/'docs/full_source_v1_freeze.json'
DEST=ROOT/'data/moe/results/full_source_conv98_20260920_v1'


def supervise(root, command, env, budget=300):
    """Owned child groups; partial output never substitutes for a completed check."""
    start=time.monotonic();stages=[];result=None;status='ERROR';error=None
    reserve=min(2.,budget/10)
    try:
        for phase in ('build','check'):
            begin=time.monotonic()-start;save(root/(phase+'_entered.json'),{'seconds':begin})
            row=execute(command(phase),root/(phase+'.log'),start+budget-reserve,env)
            row.update(phase=phase,start_seconds=begin,end_seconds=time.monotonic()-start)
            stages.append(row);save(root/(phase+'_stage.json'),row)
            if row['state']!='COMPLETED':status=row['state'];break
        else:
            result=json.loads((root/'check.log').read_bytes())
            if (result['status']!='CHECKED_FULL_EXPERTS_AND_NEW_LP_CONSTRUCTIONS' or
                    result['complete_output_positive_proof'] or result['complete_strict_network_certificate'] or
                    result['production_verdict_changed'] or result['old_LP_certificates_used']!=0):
                raise ValueError('unexpected claimed endpoint')
            status=result['status']
    except Exception as exc:error=repr(exc);status='ERROR';result=None
    inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    elapsed=time.monotonic()-start
    if elapsed>=budget:status='TIMEOUT';result=None
    terminal={'status':status,'error':error,'stages':stages,'check':result,'budget_seconds':budget,
              'elapsed_before_publication':elapsed,'inventory':inventory,'complete_output_positive_proof':False,
              'complete_strict_network_certificate':False,'production_verdict_changed':False}
    save(root/'terminal.json',terminal)
    save(root/'publication.json',{'seconds':time.monotonic()-start,'terminal_sha256':sha(root/'terminal.json')})
    if time.monotonic()-start>=budget:save(root/'publication_timeout.json',{'status':'TIMEOUT'})
    return terminal


def run():
    f=json.loads(FREEZE.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('freeze drift: '+name)
    if f['output']!=str(DEST.relative_to(ROOT)) or f['budget_seconds']!=300:raise ValueError('protocol scope drift')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':raise ValueError('branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('clean worktree required')
    if os.getloadavg()[0]/os.cpu_count()>.5:raise ValueError('resource gate, not launched')
    DEST.mkdir(exist_ok=False)
    save(DEST/'execution.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'freeze_sha256':sha(FREEZE),'budget_seconds':300,'cost_scope':f['cost_scope']})
    def cmd(phase):
        if phase=='build':return [ACT,'-m','full_source.build',str(DEST)]
        g=json.loads((DEST/'generation.json').read_bytes())
        return [ACT,'-I','-S',str(DEST/'relocated/verify_full.py'),'--manifest-hash',g['manifest_sha256']]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    t=supervise(DEST,cmd,env)
    print(json.dumps({'status':t['status'],'seconds':json.loads((DEST/'publication.json').read_bytes())['seconds'],'root':str(DEST)}))


if __name__=='__main__':run()
