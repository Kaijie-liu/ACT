"""One frozen source-proof extension, timed as a stored-evidence study."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from checked_gate.candidate_run import execute,ACT
from router_source.capture import ROOT,sha
from router_source.build import save

FREEZE=ROOT/'docs/router_source_v1_freeze.json'
DEST=ROOT/'data/moe/results/router_source_conv98_20260920_v1'


def run():
    freeze=json.loads(FREEZE.read_bytes())
    for name,h in freeze['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('source freeze drift: '+name)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':
        raise ValueError('wrong branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('dirty checkout')
    if os.getloadavg()[0]/os.cpu_count()>.5:raise ValueError('resource gate; not launched')
    start=time.monotonic();DEST.mkdir(exist_ok=False)
    save(DEST/'execution.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'freeze_sha256':sha(FREEZE),'budget_seconds':300,'started_monotonic':start})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    status='ERROR';stages=[];error=None;result=None
    try:
        for name,cap in [('build',60),('check',180)]:
            begin=time.monotonic()-start
            save(DEST/(name+'_entered.json'),{'seconds':begin})
            if name=='build':cmd=[ACT,'-m','router_source.build',str(DEST)]
            else:
                generated=json.loads((DEST/'generation.json').read_bytes())
                cmd=[ACT,'-I','-S',str(DEST/'relocated/verify_with_router.py'),
                     '--extension-hash',generated['extension_sha256']]
            row=execute(cmd,DEST/(name+'.log'),min(start+298,time.monotonic()+cap),env)
            row.update(phase=name,start_seconds=begin,end_seconds=time.monotonic()-start)
            stages.append(row);save(DEST/(name+'_stage.json'),row)
            if row['state']!='COMPLETED':status=row['state'];break
        else:
            result=json.loads((DEST/'check.log').read_bytes());status=result['status']
    except Exception as exc:error=repr(exc);status='ERROR'
    elapsed=time.monotonic()-start
    if elapsed>=300:status='TIMEOUT'
    save(DEST/'terminal.json',{'status':status,'error':error,'stages':stages,'check':result,
        'budget_seconds':300,'elapsed_before_publication':elapsed,
        'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'inventory':{str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file()}})
    save(DEST/'publication.json',{'seconds':time.monotonic()-start,'terminal_sha256':sha(DEST/'terminal.json')})
    if time.monotonic()-start>=300:save(DEST/'publication_timeout.json',{'status':'TIMEOUT'})
    print(json.dumps(json.loads((DEST/'terminal.json').read_bytes()),indent=2))


if __name__=='__main__':argparse.ArgumentParser(description=__doc__).parse_args();run()
