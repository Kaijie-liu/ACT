"""Source-defined conformance controls only: no trained-model endpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.conv_three_arm_contract import ROOT, ACT, read, selection
from scripts.conv_budget_worker_v2 import execution_identity
from scripts.budget_contract_v2 import verify_v2
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256, _git_value

RAW=ROOT/'data/moe/results/budget_contract_v2_controls_20260915_r1'


def run():
    if (Path(sys.executable).resolve()!=Path(ACT).resolve()
            or _git_value('status','--porcelain')
            or _git_value('branch','--show-current')!='feat/moe-route-verification'
            or _git_value('rev-parse','HEAD')!=_git_value('rev-parse','@{upstream}')):
        raise ValueError('clean committed/pushed feature branch and ACT env required')
    selection()  # frozen ACT/model/env identities, no trained forward/query
    RAW.mkdir(exist_ok=False)
    runtime={'head':_git_value('rev-parse','HEAD'), 'execution':execution_identity(),
        'runner_sha256':_sha256(Path(__file__)), 'scope':'SOURCE_DEFINED_TOY_ONLY',
        'real_model_smoke_started':False, 'full_started':False}
    atomic_json(RAW/'runtime.json',runtime)
    import torch
    from act.pipeline.moe.test_route_complexity_schedule import model,config
    from act.pipeline.moe.staged_verifier import verify_staged_linf,write_evidence_package
    from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
    net=model(((0.,1.,-2.),(3.,0.,-2.),(4.,0.,-2.)))
    center=torch.full((1,2),.5,dtype=torch.float64)
    rows=[]
    for arm in ('staged','monolithic_f0'):
        directory=RAW/arm;directory.mkdir()
        cfg=config(arm)
        baseline=verify_staged_linf(net,center,.1,cfg)
        write_evidence_package(baseline,directory/'baseline')
        report=verify_v2(net,center,.1,cfg,journal_path=directory/'journal.jsonl',
            started=time.monotonic(),identity={'arm':arm,'runtime_sha256':_sha256(RAW/'runtime.json')})
        write_evidence_package(report,directory/'v2')
        # Separate process; checker has no site-packages, Torch or solver import.
        completed=subprocess.run([ACT,'-S','-m','scripts.check_budget_contract_v2',
            str(directory/'journal.jsonl')],cwd=ROOT,text=True,capture_output=True,check=True)
        checked=json.loads(completed.stdout)
        atomic_json(directory/'journal.audit.json',checked)
        audits={kind:audit_evidence_package(directory/kind,replay_unsafe=True) for kind in ('baseline','v2')}
        if baseline.status!=report.status or any(v['issues'] for v in audits.values()):
            raise ValueError('toy differential or structural audit failed')
        rows.append({'arm':arm,'baseline_status':baseline.status,'v2_status':report.status,
            'route_pairs':report.evidence['route_coverage']['feasible_route_sets'],
            'journal_check':checked,'package_audits':audits,
            'budget':report.evidence['route_complexity_schedule']['budget'],
            'numerical_safety_unchanged':baseline.evidence['numerical_safety']==report.evidence['numerical_safety']})
    result={'status':'PASS','issues':[], 'runtime':runtime,'rows':rows,
        'scope':'Toy conformance, not conv performance or new certificates',
        'real_model_smoke_started':False,'full_started':False}
    atomic_json(RAW/'review.json',result)
    print(json.dumps({'status':result['status'],'arms':[r['arm'] for r in rows]}))


if __name__=='__main__':run()
