"""Controls -> committed freeze -> exactly two synthetic repair diagnostics.

No real loader, tuning loop, resume, retry, extra sample or output solve.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from source_cost_supervised.supervisor import ENV, supervise

PROTOCOL = ROOT / 'docs/source_cost_supervised_protocol_20260925_r1.md'
GATE = ROOT / 'docs/source_cost_supervised_controls_20260925_r1.json'
CONFIG = ROOT / 'configs/backend_controls/source_cost_supervised_r1.json'
DEST = ROOT / 'data/moe/results/source_cost_supervised_20260925_r1'


def sources():
    prior = load(ROOT / 'configs/backend_controls/batched_evidence_study_r1.json')['sources']
    repaired = load(ROOT / 'docs/source_cost_interface_controls_20260925_r1.json')['new_sources']
    for name, digest in {**prior, **repaired}.items():
        if sha(ROOT / name) != digest: raise ValueError('sealed source drift: '+name)
    added = [PROTOCOL, *sorted((ROOT/'source_cost_supervised').glob('*.py'))]
    return {**prior, **repaired, **{str(p.relative_to(ROOT)): sha(p) for p in added}}


def admitted():
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    return available >= 10 * 2**30 and os.getloadavg()[0] / os.cpu_count() <= .5


def clean():
    if subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('research branch required')
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT): raise ValueError('clean checkout required')


def roster():
    # Regeneration here binds declared inputs, but performs NO source/HZ/profile work.
    from source_enclosure.format import identity
    from source_construction_lab.fixtures import document
    out = []
    for name, e, c, w, d in [('small',4,3,4,1), ('medium',8,10,8,2)]:
        f = dict(experts=e, classes=c, width=w, depth=d, seed=724)
        out.append({'id': 'repair_' + name, 'fixture': f, 'source_sha256': identity(document(**f))})
    return out


def controls(root):
    from bounded_evidence.controls import MODULES
    root = Path(root).resolve()
    if GATE.exists() or not root.is_relative_to(ROOT/'data/moe/results') or not admitted():
        raise ValueError('new controls directory and resource gate required')
    root.mkdir(parents=True, exist_ok=False); bound = sources(); began = time.monotonic()
    save(root/'launch.json', {'sources': bound, 'real_requests': 0})
    suites=[]
    for name, modules, expected in [('new',['source_cost_supervised.tests'],18),
                                   ('interface',['source_cost_controls.tests'],4),
                                   ('batch',['batched_evidence.tests'],21), ('previous',MODULES,137)]:
        started=time.monotonic(); code=None; error=None
        env=dict(ENV,SOURCE_COST_CONTROL_ROOT=str(root/(name+'_artifacts')),
                 BOUNDED_CONTROL_EVIDENCE_ROOT=str(root/(name+'_pipeline')))
        with (root/(name+'.log')).open('xb') as log:
            try:
                code=subprocess.run([PYTHON,'-m','unittest',*modules,'-v'],cwd=ROOT,env=env,
                                    stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired as exc:error=str(exc)
        match=re.search(r'Ran (\d+) tests',(root/(name+'.log')).read_text())
        count=int(match[1]) if match else 0
        suites.append({'name':name,'status':'PASS' if code==0 and count==expected else 'FAIL',
                       'tests':count,'expected':expected,'returncode':code,'error':error,
                       'seconds':time.monotonic()-started,'log_sha256':sha(root/(name+'.log'))})
        print(suites[-1],flush=True)
    report={'status':'PASS' if all(s['status']=='PASS' for s in suites) and bound==sources() else 'FAIL',
            'tests':sum(s['tests'] for s in suites),'suites':suites,'sources':bound,'root':str(root),
            'seconds':time.monotonic()-began,'real_requests':0,'native_solver_queries':0,
            'new_real_certificates':0,'original_two_profiles_repeated':False}
    save(GATE,report)
    if report['status']!='PASS':raise SystemExit(1)


def freeze():
    clean(); gate=load(GATE)
    if gate['status']!='PASS' or gate['tests']!=180 or gate['sources']!=sources():
        raise ValueError('test/source gate')
    if CONFIG.exists() or DEST.exists(): raise FileExistsError('new freeze/output required')
    save(CONFIG, {'schema':'SOURCE_COST_SUPERVISED_REPAIR_R1','sources':sources(),
        'gate_sha256':sha(GATE),'calls':roster(),'output':str(DEST),'budget_seconds':300,
        'terminal_reserve_seconds':2,'rss_limit':8*2**30,'threads':2,
        'serializer':'bounded_evidence.stream (unchanged R1)',
        'profiler':'source_cost_controls.profile (unchanged repaired V2)',
        'python':PYTHON,'python_version':sys.version,'no_retry_or_expansion':True,
        'real_requests':0,'native_solver_queries':0,'complete_output_positive_proof':False,
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    print({'frozen_calls':2,'config_sha256':sha(CONFIG),'executed':False})


def verify_freeze():
    cfg=load(CONFIG);gate=load(GATE,cfg['gate_sha256'])
    if (cfg['sources']!=sources() or cfg['sources']!=gate['sources'] or gate['status']!='PASS'
            or gate['tests']!=180 or cfg['calls']!=roster() or cfg['output']!=str(DEST)
            or cfg['budget_seconds']!=300 or cfg['terminal_reserve_seconds']!=2
            or cfg['rss_limit']!=8*2**30 or cfg['threads']!=2 or cfg['python']!=PYTHON
            or cfg['python_version']!=sys.version or cfg['no_retry_or_expansion'] is not True
            or cfg['real_requests']!=0 or cfg['native_solver_queries']!=0
            or cfg['complete_output_positive_proof'] is not False):
        raise ValueError('freeze/environment/denominator drift')
    return cfg


def execute():
    clean();cfg=verify_freeze()
    if DEST.exists() or not admitted():raise ValueError('new directory/resource gate')
    began=time.monotonic();DEST.mkdir()
    save(DEST/'launch.json',{'config_sha256':sha(CONFIG),'config':cfg,
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    rows=[]
    for spec in cfg['calls']:
        if not admitted():row={'id':spec['id'],'status':'NOT_STARTED_RESOURCE','launched':False,'seconds':None}
        else:
            started=time.monotonic()
            result=supervise(DEST/spec['id'],spec,budget=300,rss_limit=cfg['rss_limit'])
            row={'id':spec['id'],'launched':True,'returned':result,
                 'caller_seconds':time.monotonic()-started,'status':result['status']}
        save(DEST/(spec['id']+'_terminal.json'),row);rows.append(row)
        print(row,flush=True)
    save(DEST/'execution.json',{'config_sha256':sha(CONFIG),'rows':rows,
        'batch_seconds_before_summary':time.monotonic()-began,
        'real_requests':0,'native_solver_queries':0,'no_retries':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controls','freeze','verify','execute'))
    p.add_argument('--root',type=Path);a=p.parse_args()
    if a.action=='controls':
        if a.root is None:p.error('--root required')
        controls(a.root)
    elif a.action=='freeze':freeze()
    elif a.action=='verify':
        cfg=verify_freeze();print({'freeze':'PASS','calls':len(cfg['calls']),'sources':len(cfg['sources']),'config_sha256':sha(CONFIG)})
    else:execute()
