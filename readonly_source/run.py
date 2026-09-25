"""Controls, independent freeze and one finite saved-evidence cost study."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from source_cost_supervised.run import admitted, clean
from source_cost_supervised.supervisor import ENV
from parsed_source_reuse.run import inputs

PROTOCOL = ROOT/'docs/readonly_source_protocol_20260925_r1.md'
GATE = ROOT/'docs/readonly_source_controls_20260925_r1.json'
CONFIG = ROOT/'configs/backend_controls/readonly_source_r1.json'
DEST = ROOT/'data/moe/results/readonly_source_20260925_r1'
PARENT = ROOT/'docs/parsed_source_reuse_audit_20260925_r1.json'
AUDITOR = ROOT/'scripts/audit_readonly_source.py'


def sources():
    prior = load(ROOT/'configs/backend_controls/parsed_source_reuse_r1.json')['sources']
    for path, digest in prior.items():
        if sha(ROOT/path) != digest: raise ValueError('sealed source changed: '+path)
    added = [PROTOCOL, AUDITOR, *sorted((ROOT/'readonly_source').glob('*.py'))]
    return {**prior, **{str(p.relative_to(ROOT)):sha(p) for p in added}}


def roster():
    out = []; modes = ('none','copy','readonly')
    for item in inputs():
        for repeat in range(3):
            for mode in modes[repeat:] + modes[:repeat]:
                name = item['spec']['id']+'_'+str(repeat)+'_'+mode
                out.append({'id':name,'spec':dict(item['spec'],id=name),'repeat':repeat,
                    'fixture_id':item['spec']['id'],
                    'method':dict(item['method'], representation=mode, enabled=mode!='none')})
    return out


def controls(root):
    from bounded_evidence.controls import MODULES
    root = Path(root).resolve()
    if GATE.exists() or not root.is_relative_to(ROOT/'data/moe/results') or not admitted():
        raise ValueError('new controls path/resource gate')
    root.mkdir(parents=True, exist_ok=False); bound=sources(); started=time.monotonic(); rows=[]
    save(root/'launch.json', {'sources':bound,'real_requests':0})
    groups = [('new',['readonly_source.tests','readonly_source.supervision_tests'],33),
              ('parse_r1',['parsed_source_reuse.tests','parsed_source_reuse.supervision_tests'],27),
              ('supervision',['source_cost_supervised.tests','source_cost_controls.tests'],22),
              ('batch',['batched_evidence.tests'],21), ('previous',MODULES,137)]
    for name, modules, expected in groups:
        begin=time.monotonic(); code=None; error=None
        env=dict(ENV, PARSED_SOURCE_CONTROL_ROOT=str(root/(name+'_artifacts')),
                 SOURCE_COST_CONTROL_ROOT=str(root/(name+'_supervision')),
                 BOUNDED_CONTROL_EVIDENCE_ROOT=str(root/(name+'_pipeline')))
        with (root/(name+'.log')).open('xb') as log:
            try: code=subprocess.run([PYTHON,'-m','unittest',*modules,'-v'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired as exc: error=str(exc)
        match=re.search(r'Ran (\d+) tests',(root/(name+'.log')).read_text()); count=int(match[1]) if match else 0
        row={'name':name,'status':'PASS' if code==0 and count==expected else 'FAIL','tests':count,
             'expected':expected,'returncode':code,'error':error,'seconds':time.monotonic()-begin,
             'log_sha256':sha(root/(name+'.log'))}; rows.append(row); print(row,flush=True)
    report={'status':'PASS' if all(r['status']=='PASS' for r in rows) and bound==sources() else 'FAIL',
            'tests':sum(r['tests'] for r in rows),'suites':rows,'sources':bound,'root':str(root),
            'seconds':time.monotonic()-started,'real_requests':0,'native_solver_queries':0,'new_real_certificates':0}
    save(GATE,report)
    if report['status']!='PASS': raise SystemExit(1)


def freeze():
    clean(); gate=load(GATE)
    if gate['status']!='PASS' or gate['tests']!=240 or gate['sources']!=sources(): raise ValueError('control gate')
    if CONFIG.exists() or DEST.exists(): raise FileExistsError('new freeze/output required')
    if load(PARENT)['audit']!='PASS': raise ValueError('archived negative comparator')
    save(CONFIG, {'schema':'READONLY_SOURCE_STUDY_R1','sources':sources(),'gate_sha256':sha(GATE),
        'parent_audit_sha256':sha(PARENT),'calls':roster(),'output':str(DEST),'budget_seconds':300,
        'rss_limit':8*2**30,'threads':2,'reserve_seconds':2,'no_retry_or_expansion':True,
        'python':PYTHON,'python_version':sys.version,'real_requests':0,'native_solver_queries':0,
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'saved-source checking only, NOT full MoE latency or new output certificates'})
    print({'frozen_calls':18,'config_sha256':sha(CONFIG),'executed':False})


def verify():
    c=load(CONFIG); gate=load(GATE,c['gate_sha256'])
    if (c['sources']!=sources() or gate['sources']!=c['sources'] or gate['status']!='PASS' or gate['tests']!=240
            or c['parent_audit_sha256']!=sha(PARENT) or c['calls']!=roster() or c['output']!=str(DEST)
            or c['budget_seconds']!=300 or c['rss_limit']!=8*2**30 or c['threads']!=2 or c['reserve_seconds']!=2
            or c['python']!=PYTHON or c['python_version']!=sys.version or c['no_retry_or_expansion'] is not True
            or c['real_requests']!=0 or c['native_solver_queries']!=0): raise ValueError('freeze/environment drift')
    return c


def execute():
    from readonly_source.execution import supervise
    clean(); cfg=verify()
    if DEST.exists() or not admitted(): raise ValueError('new directory/resource admission')
    DEST.mkdir(); start=time.monotonic()
    save(DEST/'launch.json', {'config':cfg,'config_sha256':sha(CONFIG),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    rows=[]
    for call in cfg['calls']:
        if not admitted(): row={'id':call['id'],'launched':False,'status':'NOT_STARTED_RESOURCE','seconds':None}
        else:
            began=time.monotonic(); r=supervise(DEST/call['id'],call['spec'],call['method'])
            row={'id':call['id'],'launched':True,'status':r['status'],'returned':r,'caller_seconds':time.monotonic()-began}
        save(DEST/(call['id']+'_terminal.json'),row); rows.append(row); print(row,flush=True)
    save(DEST/'execution.json', {'config_sha256':sha(CONFIG),'rows':rows,'no_retries':True,
        'real_requests':0,'native_solver_queries':0,'batch_seconds_before_summary':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('controls','freeze','verify','execute'))
    p.add_argument('--root',type=Path); a=p.parse_args()
    if a.action=='controls':
        if a.root is None: p.error('--root required')
        controls(a.root)
    elif a.action=='freeze': freeze()
    elif a.action=='verify':
        c=verify(); print({'freeze':'PASS','calls':len(c['calls']),'sources':len(c['sources']),'config_sha256':sha(CONFIG)})
    else: execute()
