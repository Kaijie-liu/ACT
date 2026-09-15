"""One fixed 300s clock, owned process groups, terminal-only acceptance."""
import os
from pathlib import Path
import signal
import subprocess
import time
from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ROOT,ACT,read,save
from scripts.optional_evidence_budget import EvidenceBudget,EvidenceBudgetExpired,terminal_status

PHASES={'matched':('matched',),'evidence':('capture','propose','precheck','package','check'),'crown':('crown',)}
LEVELS={'matched':'HZ_POLICY_ACCEPTED','evidence':'CHECKED_RATIONAL_CONDITIONAL','crown':'CROWN_NUMERICAL_FILTER'}


def phase(args,root,name,budget,env):
    start=time.monotonic();allowance=budget.remaining(2)
    with (root/(name+'.log')).open('x') as log:
        proc=subprocess.Popen(args,cwd=root if name=='check' else ROOT,env=env,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=allowance)
            state='COMPLETED' if code==0 else 'BUDGET_EXHAUSTED' if code==3 else 'ERROR'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=proc.returncode;state='OUTER_TIMEOUT'
        except BaseException:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();raise
    return {'state':state,'return_code':code,'start_seconds':start-budget.started,
            'elapsed_seconds':time.monotonic()-start,'allowed_seconds':allowance}


def accept(arm,stages,read_result,elapsed):
    """No files from a failed/late process can promote its terminal."""
    if arm not in PHASES:raise ValueError('unknown arm')
    if elapsed>300:return 'TIMEOUT',False
    if list(stages)!=list(PHASES[arm]) or any(s['state']!='COMPLETED' for s in stages.values()):
        return ('ERROR' if any(s['state']=='ERROR' for s in stages.values()) else 'TIMEOUT'),False
    value=read_result()
    if arm=='evidence':
        if not value['isolated'] or not value['site_disabled'] or value['solver_imported']:raise ValueError('not isolated')
        result=value['result'];status=result['status']
        if status not in ('CHECKED_CONDITIONAL','UNKNOWN_MISSING_EVIDENCE','UNKNOWN_NONPOSITIVE','UNKNOWN_ROUTE_COVERAGE'):
            raise ValueError('unknown checker result')
        if status=='CHECKED_CONDITIONAL' and (not result['required_obligations'] or result['positive_obligations']!=result['required_obligations']):
            raise ValueError('incomplete result called positive')
        return status,True
    status=value['status']
    allowed=('SAFE','UNSAFE','UNKNOWN','TIMEOUT') if arm=='matched' else ('POSITIVE','UNSAFE','UNKNOWN')
    if status not in allowed:raise ValueError('unsupported production status')
    return status,False


def run_request(arm,request,directory,env):
    # This callable does not select inputs or authorize an experiment.
    if arm not in PHASES:raise ValueError('unknown arm')
    root=Path(directory).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside authorized workspace')
    started=time.monotonic();budget=EvidenceBudget(started);root.mkdir(exist_ok=False)
    save(root/'request.json',request)
    save(root/'execution.json',{'arm':arm,'started_monotonic':started,'total_seconds':300})
    stages={};error=None;complete=False;verdict='UNKNOWN_MISSING_EVIDENCE'
    try:
        for name in PHASES[arm]:
            if name=='check':
                info=read(root/'packing.json')
                args=[ACT,'-I','-S',str(root/'portable/verify.py'),'--bundle-hash',info['bundle_sha256'],
                      '--statement-hash',info['statement_sha256']]
            else:args=[ACT,'-m','moe_evidence.worker',name,str(root),'--started',repr(started)]
            stages[name]=phase(args,root,name,budget,env);save(root/'stage_progress.json',stages)
            if stages[name]['state']!='COMPLETED':break
        result_file={'matched':'package/manifest.json','crown':'external.json','evidence':'check.log'}[arm]
        verdict,complete=accept(arm,stages,lambda:read(root/result_file),time.monotonic()-started)
    except EvidenceBudgetExpired:verdict='TIMEOUT'
    except Exception as exc:verdict='ERROR';error=repr(exc)
    # Hashing/serialization remain charged; no late result promotion.
    inventory={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    elapsed=time.monotonic()-started
    terminal={'arm':arm,'dataset_index':request['sample']['dataset_index'],
        'request_sha256':digest((root/'request.json').read_bytes()),'stages':stages,
        'artifact_sha256':inventory,'error':error,'complete_independent_check':complete,
        'evidence_level':LEVELS[arm],'production_gate_changed':False,'deployed_float_SAFE':False,
        'budget_seconds':300,'wall_seconds':elapsed,'status':terminal_status(verdict,elapsed,300,complete),
        'outer_timeout':any(v['state']=='OUTER_TIMEOUT' for v in stages.values())}
    save(root/'terminal.json',terminal)
    terminal['wall_seconds']=time.monotonic()-started
    terminal['status']=terminal_status(terminal['status'],terminal['wall_seconds'],300,complete)
    save(root/'terminal.json',terminal)
    return terminal
