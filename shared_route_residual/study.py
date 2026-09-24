"""One bounded router-proof segment, not the full MoE proof pipeline."""
import argparse
from itertools import combinations
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.supervisor import execute
from source_enclosure.format import identity

CONFIG=ROOT/'configs/backend_controls/shared_route_residual_synthetic_r1.json'
OUTPUT=ROOT/'data/moe/results/shared_route_residual_synthetic_20260924_r1'


def accept(root,plan):
    doc,pre,cand,check=(load(root/name) for name in ('source.json','prefix.json','candidates.json','check.json'))
    r=check['result'];e=doc['request']['experts']
    expected={'schema':'ROUTER_RESIDUAL_RECEIPT_V1','invocation':plan['invocation'],
        'mode':plan['mode'],'source_sha256':identity(doc),'prefix_sha256':identity(pre),
        'candidate_sha256':identity(cand),'result':r,'complete_output_positive_proof':False,
        'native_float_proof':False,'new_real_positive':False}
    if check!=expected or identity(doc)!=plan['expected_source_sha256']:
        raise ValueError('unbound/overclaimed receipt')
    if (r['source_sha256']!=identity(doc) or r['complete_output_positive_proof'] is not False or
            r['route_changing_established'] is not False or
            [(b['higher'],b['lower']) for b in r['bounds']]!=[(j,i) for j in range(e) for i in range(e) if j!=i] or
            [p['pair'] for p in r['pairs']]!=[list(p) for p in combinations(range(e),2)]):
        raise ValueError('complete router inventory required')
    kept=[p['pair'] for p in r['pairs'] if p['status']=='RETAINED']
    if (any(p['status'] not in ('RETAINED','EXCLUDED_BY_CHECKED_STRICT_MARGIN') for p in r['pairs']) or
            not kept or r['total_pairs']!=e*(e-1)//2 or r['retained_pairs']!=len(kept) or
            r['excluded_pairs']!=len(r['pairs'])-len(kept) or r['needed_experts']!=sorted({i for p in kept for i in p})):
        raise ValueError('router aggregate mismatch')
    return check


def run_one(root,kind,mode,*,budget=30.,commands=None,expected_source_sha256=None):
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    if (kind not in ('prunable','tied','random') or mode not in ('pairwise','shared') or
            type(budget) not in (int,float) or not math.isfinite(budget) or not 0<budget<=300):
        raise ValueError('fixed synthetic policy')
    started=time.monotonic();deadline=started+budget;work_deadline=deadline-min(2.,budget/10)
    root.mkdir(parents=True,exist_ok=False)
    if expected_source_sha256 is None:
        from shared_route_residual.fixtures import fixture
        expected_source_sha256=identity(fixture(kind))
    plan={'fixture':kind,'mode':mode,'invocation':uuid.uuid4().hex,'budget_seconds':budget,
        'started_monotonic':started,'deadline':deadline,'work_deadline':work_deadline,
        'expected_source_sha256':expected_source_sha256,'real_request':False}
    plan_info=save(root/'plan.json',plan)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    phases=[];status='ERROR';error=None
    try:
        for phase in ('build','check'):
            command=([PYTHON,'-S','-m','shared_route_residual.worker',phase,str(root),'--deadline',str(work_deadline)]
                     if commands is None else commands(phase,root,work_deadline))
            begin=time.monotonic()-started
            row=execute(command,root/(phase+'.log'),work_deadline,env,8*2**30)
            row.update(phase=phase,start_seconds=begin,end_seconds=time.monotonic()-started)
            phases.append(row);save(root/(phase+'_stage.json'),row)
            if row['status']!='COMPLETED':status=row['status'];break
        else:
            accept(root,plan);status='COMPLETED_ROUTER_SEGMENT'
    except Exception as exc:status='ERROR';error=repr(exc)
    if time.monotonic()>=work_deadline:status='TIMEOUT'
    terminal={'status':status,'plan_sha256':plan_info['sha256'],'phases':phases,'error':error,
        'seconds_before_publication':time.monotonic()-started,'complete_output_positive_proof':False,
        'new_real_positive':False,'check_sha256':sha(root/'check.json') if (root/'check.json').exists() else None}
    term_info=save(root/'terminal.json',terminal);elapsed=time.monotonic()-started
    if elapsed>=budget:status='TIMEOUT'
    stage=sum(r['seconds'] for r in phases)
    cost={'status':status,'terminal_sha256':term_info['sha256'],'end_to_end_seconds':elapsed,
        'stage_seconds':stage,'overhead_seconds':elapsed-stage,'budget_seconds':budget,
        'sampled_peak_rss':max((r['sampled_peak_rss'] for r in phases),default=0),
        'includes':'source generation, router construction, candidates, source check, all margins, serialization, imports, cleanup, terminal',
        'excludes':'output expert/guard/LP work (router-only), final ledger write, later independent audit'}
    save(root/'cost.json',cost)
    if time.monotonic()>=deadline:
        save(root/'publication_timeout.json',{'status':'TIMEOUT','seconds':time.monotonic()-started});status='TIMEOUT'
    return {'status':status,'seconds':time.monotonic()-started,'root':str(root)}


def run():
    cfg=load(CONFIG)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':
        raise ValueError('branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('clean committed freeze')
    for name,digest in cfg['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source drift: '+name)
    gate=load(ROOT/cfg['controls']['path'],cfg['controls']['sha256'])
    if gate['status']!='PASS':raise ValueError('control gate')
    if OUTPUT.exists() or cfg['output']!=str(OUTPUT):raise ValueError('new output only')
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<10*2**30 or os.getloadavg()[0]/os.cpu_count()>.5:raise ValueError('resource gate')
    OUTPUT.mkdir();save(OUTPUT/'launch.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                         'config_sha256':sha(CONFIG),'config':cfg})
    rows=[]
    for call in cfg['calls']:
        row={**call,**run_one(OUTPUT/call['id'],call['fixture'],call['mode'],budget=cfg['budget_seconds'],
                             expected_source_sha256=cfg['fixture_sha256'][call['fixture']])}
        rows.append(row);print(row,flush=True)
    save(OUTPUT/'execution.json',{'rows':rows,'new_real_requests':0,'native_solver_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-frozen-synthetic',action='store_true');a=p.parse_args()
    if not a.execute_frozen_synthetic:p.error('explicit frozen synthetic execution required')
    run()
