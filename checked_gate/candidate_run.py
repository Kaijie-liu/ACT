"""Bounded one-call diagnostic. Old model propagation is explicitly NOT timed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from checked_gate.bootstrap import ROOT
from checked_gate.candidate_worker import load, save, PARENT_HASH, REQUEST_ID

ACT='/data1/Kane/miniconda3/envs/act-py312/bin/python'
PROTOCOL=ROOT/'docs/checked_gate_candidate_v2_freeze.json'
DEST=ROOT/'data/moe/results/checked_gate_candidate_20260920_v2'


def execute(command, log_path, deadline, env):
    start=time.monotonic()
    if start>=deadline:
        return {'state':'TIMEOUT','seconds':0.,'return_code':None,'started':False}
    with Path(log_path).open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        killed=False
        try:
            while p.poll() is None:
                if time.monotonic()>=deadline:
                    killed=True; break
                time.sleep(min(.02,max(0,deadline-time.monotonic())))
        finally:
            # Only this child's owned session. No other processes are touched.
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGKILL)
            p.wait()
    end=time.monotonic()
    return {'state':('TIMEOUT' if killed or end>=deadline else 'COMPLETED' if p.returncode==0 else 'ERROR'),
            'seconds':end-start,'return_code':p.returncode,'started':True}


def run():
    freeze=load(PROTOCOL)
    for name,digest in freeze['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('execution source changed: '+name)
    if (freeze['request_id']!=REQUEST_ID or freeze['parent_result_sha256']!=PARENT_HASH or
            freeze['output']!=str(DEST.relative_to(ROOT)) or freeze['total_seconds']!=300):
        raise ValueError('frozen scientific identity changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='feat/moe-route-verification' or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):
        raise ValueError('clean registered checkout required')
    if os.getloadavg()[0]/os.cpu_count()>.5:
        raise ValueError('resource gate: CPU load/core exceeds .5; not launched')
    start=time.monotonic(); DEST.mkdir(exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(DEST/'execution.json',{'head':head,'freeze_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        'request_id':REQUEST_ID,'started_monotonic':start,'total_seconds':300,
        'scope':'Stored-source proof diagnostic; original propagation and old support proposal costs excluded.'})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    stages=[]; status='ERROR'; error=None; checked=None
    try:
        for phase,cap in (('validate',80),('propose',120),('check',80)):
            begin=time.monotonic()-start
            save(DEST/(phase+'_entered.json'),{'start_seconds':begin})
            command=[ACT]+([] if phase=='propose' else ['-S'])+[
                '-m','checked_gate.candidate_worker',phase,str(DEST)]
            result=execute(command,DEST/(phase+'.log'),min(start+298,time.monotonic()+cap),env)
            result.update(phase=phase,start_seconds=begin,end_seconds=time.monotonic()-start,cap_seconds=cap)
            stages.append(result); save(DEST/(phase+'_stage.json'),result)
            if result['state']!='COMPLETED':
                status=result['state']; break
        else:
            checked=load(DEST/'checked.json')
            status=checked['status']
    except Exception as exc:
        error=repr(exc); status='ERROR'
    elapsed=time.monotonic()-start
    if elapsed>=300: status='TIMEOUT'
    inventory={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
               for p in sorted(DEST.iterdir()) if p.is_file()}
    terminal={'schema':'CHECKED_GATE_CANDIDATE_TERMINAL_V1','status':status,'stages':stages,
        'error':error,'elapsed_seconds_before_terminal':time.monotonic()-start,
        'total_seconds':300,'artifacts':inventory,
        'complete_conditional_request':status=='CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING',
        'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'cost_scope':'Stored-source validation, one proposal, serialization, fresh complete check; not production end-to-end cost.'}
    save(DEST/'terminal.json',terminal)
    save(DEST/'publication.json',{'elapsed_seconds':time.monotonic()-start,
                                'terminal_sha256':hashlib.sha256((DEST/'terminal.json').read_bytes()).hexdigest()})
    if time.monotonic()-start>=300:
        save(DEST/'publication_timeout.json',{'status':'TIMEOUT','complete_conditional_request':False})
    print(json.dumps(terminal,indent=2))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); run()
