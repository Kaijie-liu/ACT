"""Two separately budgeted, frozen stored-source arms; no old results overwritten."""
import json
import os
from pathlib import Path
import subprocess
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save

from range_pipeline.run import supervise

FREEZE=ROOT/'docs/property_ranges_v1_freeze.json'


def commands(root):
    def command(phase):
        if phase=='build':return [ACT,'-m','property_ranges.build',str(root)]
        if phase in ('propose','seal'):return [ACT,'-m','full_bounds.worker',phase,str(root)]
        sealed=json.loads((root/'sealed.json').read_bytes())
        return [ACT,'-I','-S',str(root/'relocated/verify_bounds.py'),'--manifest-hash',sealed['manifest_sha256']]
    return command


def run():
    f=json.loads(FREEZE.read_bytes())
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('freeze drift: '+name)
    if (f['arms']!=['prefix','property'] or f['budget_seconds_per_arm']!=300 or
        f['rows_per_expert']!=2 or f['max_range_calls']!=8 or f['native_range_seconds']!=3 or
        f['max_output_calls']!=9 or f['native_output_seconds']!=16 or
        f['phase_caps_seconds']!={'build':100,'propose':120,'seal':5,'check':'actual remainder'} or
        f['publication_reserve_seconds']!=2):raise ValueError('protocol drift')
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
