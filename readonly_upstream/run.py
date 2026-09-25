"""Controls -> committed freeze -> four complete upstream integration calls."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import time

from scoped_proof.io import ROOT,PYTHON,load,save,sha
from source_cost_supervised.run import clean,admitted
from source_cost_supervised.supervisor import ENV

PROTOCOL=ROOT/'docs/readonly_upstream_protocol_20260925_r1.md'
GATE=ROOT/'docs/readonly_upstream_controls_20260925_r1.json'
CONFIG=ROOT/'configs/backend_controls/readonly_upstream_r1.json'
DEST=ROOT/'data/moe/results/readonly_upstream_20260925_r1'
PARENT=ROOT/'docs/readonly_source_audit_20260925_r1.json'


def sources():
    old=load(ROOT/'configs/backend_controls/readonly_source_r1.json')['sources']
    for path,digest in old.items():
        if sha(ROOT/path)!=digest: raise ValueError('sealed source drift: '+path)
    paths=[PROTOCOL,ROOT/'scripts/audit_readonly_upstream.py',*sorted((ROOT/'readonly_upstream').glob('*.py'))]
    return {**old,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}


def roster():
    # Only metadata from the prior two fixed synthetic recipes. Workers must
    # freshly generate/propagate/construct; no saved source input is accepted.
    old=load(ROOT/'configs/backend_controls/source_cost_supervised_r1.json')['calls']
    out=[]
    for index,spec in enumerate(old):
        for enabled in ((False,True) if index==0 else (True,False)):
            name=spec['id']+('_readonly' if enabled else '_direct')
            out.append({'id':name,'fixture_id':spec['id'],'spec':dict(spec,id=name),
                'method':{'schema':'READONLY_UPSTREAM_R1','readonly':enabled}})
    return out


def controls(root):
    from bounded_evidence.controls import MODULES
    root=Path(root).resolve()
    if GATE.exists() or not root.is_relative_to(ROOT/'data/moe/results') or not admitted():
        raise ValueError('new control directory/resource admission required')
    root.mkdir(parents=True,exist_ok=False); bound=sources(); start=time.monotonic(); rows=[]
    save(root/'launch.json',{'sources':bound,'real_requests':0})
    groups=[('new',['readonly_upstream.tests'],24),
        ('readonly',['readonly_source.tests','readonly_source.supervision_tests'],33),
        ('parse',['parsed_source_reuse.tests','parsed_source_reuse.supervision_tests'],27),
        ('supervision',['source_cost_supervised.tests','source_cost_controls.tests'],22),
        ('batch',['batched_evidence.tests'],21),('previous',MODULES,137)]
    for name,modules,expected in groups:
        began=time.monotonic(); code=None; error=None
        env=dict(ENV,SOURCE_COST_CONTROL_ROOT=str(root/(name+'_upstream')),
            PARSED_SOURCE_CONTROL_ROOT=str(root/(name+'_parser')),BOUNDED_CONTROL_EVIDENCE_ROOT=str(root/(name+'_pipeline')))
        with (root/(name+'.log')).open('xb') as log:
            try: code=subprocess.run([PYTHON,'-m','unittest',*modules,'-v'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired as exc: error=str(exc)
        m=re.search(r'Ran (\d+) tests',(root/(name+'.log')).read_text()); count=int(m[1]) if m else 0
        row={'name':name,'status':'PASS' if code==0 and count==expected else 'FAIL','tests':count,
            'expected':expected,'returncode':code,'error':error,'seconds':time.monotonic()-began,
            'log_sha256':sha(root/(name+'.log'))}; rows.append(row); print(row,flush=True)
    report={'status':'PASS' if all(r['status']=='PASS' for r in rows) and bound==sources() else 'FAIL',
        'tests':sum(r['tests'] for r in rows),'suites':rows,'sources':bound,'root':str(root),
        'seconds':time.monotonic()-start,'real_requests':0,'native_solver_queries':0,'new_real_certificates':0}
    save(GATE,report)
    if report['status']!='PASS': raise SystemExit(1)


def freeze():
    clean(); gate=load(GATE)
    if gate['status']!='PASS' or gate['tests']!=264 or gate['sources']!=sources(): raise ValueError('controls gate')
    if CONFIG.exists() or DEST.exists(): raise FileExistsError('new freeze and output required')
    save(CONFIG,{'schema':'READONLY_UPSTREAM_STUDY_R1','sources':sources(),'gate_sha256':sha(GATE),
        'parent_audit_sha256':sha(PARENT),'calls':roster(),'output':str(DEST),'budget_seconds':300,
        'rss_limit':8*2**30,'threads':2,'reserve_seconds':2,'python':PYTHON,'python_version':sys.version,
        'no_retry_or_expansion':True,'real_requests':0,'native_solver_queries':0,
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'fresh synthetic FULL UPSTREAM; no positive output bound or full MoE SAFE'})
    print({'calls':4,'config_sha256':sha(CONFIG),'executed':False})


def verify():
    c=load(CONFIG); gate=load(GATE,c['gate_sha256'])
    if (c['sources']!=sources() or gate['sources']!=c['sources'] or gate['tests']!=264 or gate['status']!='PASS'
            or c['parent_audit_sha256']!=sha(PARENT) or c['calls']!=roster() or c['output']!=str(DEST)
            or c['budget_seconds']!=300 or c['rss_limit']!=8*2**30 or c['threads']!=2 or c['reserve_seconds']!=2
            or c['python']!=PYTHON or c['python_version']!=sys.version or not c['no_retry_or_expansion']
            or c['real_requests']!=0 or c['native_solver_queries']!=0): raise ValueError('frozen scope drift')
    return c


def execute():
    from readonly_upstream.execution import supervise
    clean(); cfg=verify()
    if DEST.exists() or not admitted(): raise ValueError('new destination/resource admission')
    DEST.mkdir(); start=time.monotonic(); rows=[]
    save(DEST/'launch.json',{'config':cfg,'config_sha256':sha(CONFIG),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    for call in cfg['calls']:
        if not admitted(): row={'id':call['id'],'launched':False,'status':'NOT_STARTED_RESOURCE','seconds':None}
        else:
            began=time.monotonic(); r=supervise(DEST/call['id'],call['spec'],call['method'])
            row={'id':call['id'],'launched':True,'status':r['status'],'returned':r,'caller_seconds':time.monotonic()-began}
        save(DEST/(call['id']+'_terminal.json'),row); rows.append(row); print(row,flush=True)
    save(DEST/'execution.json',{'config_sha256':sha(CONFIG),'rows':rows,'no_retries':True,
        'real_requests':0,'native_solver_queries':0,'batch_seconds_before_summary':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('controls','freeze','verify','execute'))
    p.add_argument('--root',type=Path); a=p.parse_args()
    if a.action=='controls':
        if a.root is None:p.error('--root required')
        controls(a.root)
    elif a.action=='freeze':freeze()
    elif a.action=='verify':
        c=verify();print({'freeze':'PASS','calls':len(c['calls']),'sources':len(c['sources']),'config_sha256':sha(CONFIG)})
    else:execute()
