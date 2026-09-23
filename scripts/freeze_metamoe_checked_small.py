"""Plan -> forward-only selection -> audited small comparison freeze; NO run.

Raw order, five clean-correct per dataset, first 1000 only, previously used
indices excluded. No router scores, feasibility, support or verification in
selection. Failed selections cannot extend the scan or replace the protocol.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise
from metamoe_paired_execution_r2 import validate as validate_environment
import metamoe_checked_paired as paired

ROOT = paired.ROOT
PLAN = ROOT/'configs/recent_moe/metamoe_checked_small_plan_r1.json'
CONFIG = ROOT/'configs/recent_moe/metamoe_checked_paired_small_r1.json'
GATE = ROOT/'docs/metamoe_checked_paired_smoke_archive_20260923_r1.json'
SELECTION = Path('/data1/Kane/MOE/baseline_data/metamoe_checked_small_20260923_r1')
OUTPUT = '/data1/Kane/MOE/baseline_runs/metamoe_checked_paired_20260923_r1_small'
ADDED = ('scripts/freeze_metamoe_checked_small.py', 'tests/test_metamoe_checked_selection.py',
         'scripts/review_metamoe_checked_selection.py', 'docs/metamoe_checked_small_protocol_20260923_r1.md')


def historical_indices(configs):
    excluded = {'CIFAR10':set(), 'MNIST':set()}
    for cfg in configs:
        for r in cfg.get('requests', []):
            ds = r.get('dataset')
            if ds is None:
                ds = next((k for k in excluded if r.get('id','').lower().startswith(k.lower())), None)
            index = r.get('index', r.get('dataset_index'))
            if ds in excluded:
                if type(index) is not int or index < 0:
                    raise ValueError('unresolved historical request index')
                excluded[ds].add(index)
    return {k:sorted(v) for k,v in excluded.items()}


def choose_prefix(records, excluded, count):
    """Only ordered original logits/labels decide eligibility, no route fields."""
    selected = []
    for i, r in enumerate(records):
        if r['index'] != i:
            raise ValueError('nonconsecutive raw scan')
        if not r['finite']:
            raise ValueError('undefined original clean output')
        if i not in excluded and r['label'] == r['prediction']:
            selected.append(i)
        if len(selected) == count:
            if i != len(records)-1:
                raise ValueError('scan extended after sufficient count')
            return selected
    raise ValueError('frozen prefix insufficient; no extension')


def build_plan():
    smoke = json.loads(paired.SMOKE.read_text())
    paired.validate(smoke)
    gate = json.loads(GATE.read_text())
    if not gate['smoke_gate_pass'] or gate['audit'] != 'PASS' or gate['config_sha256'] != sha256(paired.SMOKE):
        raise ValueError('independent smoke/replay gate closed')
    histories = sorted((ROOT/'configs/recent_moe').glob('metamoe*.json'))
    histories = [p for p in histories if p not in (PLAN, CONFIG)]
    data = Path(json.loads((ROOT/'configs/recent_moe/metamoe_full_intake_r2.json').read_text())['data_root'])
    raw = sorted(p for p in data.rglob('*') if p.is_file())
    return {'protocol':'metamoe_checked_small_selection_r1', 'per_dataset':5, 'scan_cap':1000,
        'datasets':['CIFAR10','MNIST'], 'global_offsets':{'CIFAR10':0,'MNIST':10},
        'excluded':historical_indices([json.loads(p.read_text()) for p in histories]),
        'history_files':{str(p):sha256(p) for p in histories}, 'data_root':str(data),
        'raw_data_files':{str(p):sha256(p) for p in raw}, 'smoke_sha256':sha256(paired.SMOKE),
        'gate_sha256':sha256(GATE), 'python':smoke['python']['act'],
        'sources':{p:sha256(ROOT/p) for p in ADDED}, 'selection_root':str(SELECTION),
        'seconds':300, 'group_rss_limit_bytes':8*2**30, 'automatic_execution':False,
        'rule':'first5 each raw order clean-correct, exclude all declared previous MetaMoE requests; normalized2/255, no route/bound selection'}


def validate_plan(plan):
    if {k:v for k,v in plan.items() if k!='execution_commit'} != build_plan():
        raise ValueError('selection plan/source/input/gate drift')


def load_data(plan, repo, name):
    sys.path.insert(0,str(Path(repo)/'src/Formal_Neural_Network_Verification/alpha-beta-crown'))
    from create_vnnlib_specs import load_dataset
    # Files are hash checked first; author download=True performs its existing
    # local integrity check. No missing-data download is needed or accepted.
    return load_dataset(name, plan['data_root'], plan['scan_cap'])


def select_worker():
    began=time.monotonic();plan=json.loads(PLAN.read_text());validate_plan(plan)
    cfg=json.loads(paired.SMOKE.read_text());paired.validate(cfg)
    import numpy as np
    import torch
    from metamoe_paired_model import load_full
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(100)
    model=load_full(cfg['repo'],cfg['checkpoint'],cfg['files'][cfg['checkpoint']])
    scans={};requests=[]
    for name in plan['datasets']:
        images,labels,_=load_data(plan,cfg['repo'],name)
        rec=[];selected=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            x=image.unsqueeze(0).double()
            with torch.no_grad(): out=model(x)[0]
            finite=bool(torch.isfinite(out).all())
            row={'index':i,'label':int(label)+plan['global_offsets'][name],
                 'prediction':int(out.argmax(1)), 'finite':finite}
            rec.append(row)
            if not finite: raise ValueError('undefined source output during selection')
            if i not in plan['excluded'][name] and row['prediction']==row['label']:
                rid=f'{name.lower()}_{i}';file=SELECTION/f'{rid}.npz'
                with file.open('xb') as f:
                    np.savez(f,center=x.numpy(),lower=(x-cfg['epsilon']).clamp(-10,10).numpy(),
                             upper=(x+cfg['epsilon']).clamp(-10,10).numpy())
                requests.append({'id':rid,'dataset':name,'index':i,'label':row['label'],
                    'clean_prediction':row['prediction'],'tensor_file':str(file)})
                selected.append(i)
            if len(selected)==plan['per_dataset']: break
        if selected!=choose_prefix(rec,plan['excluded'][name],plan['per_dataset']):
            raise ValueError('selection mismatch')
        scans[name]=rec
        write(SELECTION/f'{name}_scan.json',{'records':rec,'selected_indices':selected})
    write(SELECTION/'selection.json',{'plan_sha256':sha256(PLAN),'requests':requests,'scans':scans,
        'tensor_hashes':{r['tensor_file']:sha256(r['tensor_file']) for r in requests},
        'verification_calls':0,'worker_seconds':time.monotonic()-began})


def check_selection(plan, selection):
    import numpy as np
    if selection['plan_sha256']!=sha256(PLAN) or selection['verification_calls']!=0:
        raise ValueError('selection identity')
    requests=selection['requests']
    if len(requests)!=10 or len({r['id'] for r in requests})!=10:
        raise ValueError('selection count/duplicates')
    for name in plan['datasets']:
        indices=choose_prefix(selection['scans'][name],plan['excluded'][name],plan['per_dataset'])
        rr=[r for r in requests if r['dataset']==name]
        if [r['index'] for r in rr]!=indices: raise ValueError('not first eligible prefix')
    for r in requests:
        if r['id']!=f"{r['dataset'].lower()}_{r['index']}" or r['clean_prediction']!=r['label']:
            raise ValueError('selected identity/clean label')
        rec=selection['scans'][r['dataset']][r['index']]
        if rec['label']!=r['label'] or rec['prediction']!=r['clean_prediction']: raise ValueError('label binding')
        f=Path(r['tensor_file'])
        if f != SELECTION/f"{r['id']}.npz" or sha256(f)!=selection['tensor_hashes'][str(f)]:
            raise ValueError('tensor identity')
        with np.load(f,allow_pickle=False) as z:
            x=z['center'];lo=z['lower'];hi=z['upper']
            if (x.dtype!=np.float64 or x.shape!=(1,3,32,32) or not np.isfinite(x).all() or
                not np.array_equal(lo,np.clip(x-2/255,-10,10)) or
                not np.array_equal(hi,np.clip(x+2/255,-10,10))):
                raise ValueError('materialized numerical box')


def build_config(plan,selection):
    cfg=json.loads(paired.SMOKE.read_text())
    cfg.update(protocol='metamoe_checked_paired_small_r1',output_root=OUTPUT,requests=selection['requests'],
        roster=paired.roster(selection['requests']),scope='10 clean-correct inputs,20 calls; small descriptive paired followup',
        selection=plan['rule'],selection_plan_sha256=sha256(PLAN),
        smoke_gate_sha256=sha256(GATE),automatic_followup=False)
    paths=[PLAN,GATE,SELECTION/'selection.json',SELECTION/'receipt.json',SELECTION/'selection_cost.json',
           SELECTION/'selection_review.json',SELECTION/'review/result.json',SELECTION/'review/receipt.json',paired.SMOKE]
    cfg['files'].update({str(p):sha256(p) for p in paths})
    cfg['files'].update(plan['sources']);cfg['files'].update(plan['raw_data_files'])
    cfg['files'].update(selection['tensor_hashes'])
    return cfg


def contract(cfg):
    plan=json.loads(PLAN.read_text());validate_plan(plan)
    selection=json.loads((SELECTION/'selection.json').read_text());check_selection(plan,selection)
    reviewed=json.loads((SELECTION/'review/result.json').read_text())
    receipt=json.loads((SELECTION/'review/receipt.json').read_text())
    if (reviewed['status']!='INDEPENDENT_SOURCE_SELECTION_PASS' or receipt['status']!='COMPLETED' or
        reviewed['plan_sha256']!=sha256(PLAN) or reviewed['selection_sha256']!=sha256(SELECTION/'selection.json') or
        reviewed['reviewed']!=[r['id'] for r in selection['requests']]):
        raise ValueError('independent selection review gate')
    expected=build_config(plan,selection)
    if {k:v for k,v in cfg.items() if k!='execution_commit'}!={k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('small comparison frozen contract')


def select_and_freeze():
    require_clean();start=time.monotonic()
    if CONFIG.exists() or SELECTION.exists() or Path(OUTPUT).exists(): raise FileExistsError('new identity required')
    plan=json.loads(PLAN.read_text());validate_plan(plan)
    subprocess.run(['git','merge-base','--is-ancestor',plan['execution_commit'],'HEAD'],cwd=ROOT,check=True)
    receipt=supervise([plan['python'],str(Path(__file__).resolve()),'--worker'],str(ROOT),SELECTION,
                      plan['seconds'],plan['group_rss_limit_bytes'])
    write(SELECTION/'selection_cost.json',{'plan_sha256':sha256(PLAN),'receipt':receipt,
        'through_receipt_seconds':time.monotonic()-start,'verification_calls':0,
        'excludes':'independent selection review and config publication; not request solving cost'})
    if receipt['status']!='COMPLETED': raise RuntimeError('selection failed; preserve files, no retry')
    selection=json.loads((SELECTION/'selection.json').read_text());check_selection(plan,selection)
    validate_plan(plan)
    review_receipt=supervise([plan['python'],str(ROOT/'scripts/review_metamoe_checked_selection.py'),str(PLAN),str(SELECTION)],
                             str(ROOT),SELECTION/'review',300,8*2**30)
    if review_receipt['status']!='COMPLETED':raise RuntimeError('independent selection failed; no comparison freeze')
    write(SELECTION/'selection_review.json',{'status':'SAVED_SELECTION_RULE_PASS','plan_sha256':sha256(PLAN),
        'selection_sha256':sha256(SELECTION/'selection.json'), 'requests':len(selection['requests']),
        'selection_is_forward_only':True, 'independent_replay_sha256':sha256(SELECTION/'review/result.json'),
        'independent_review_receipt':review_receipt,'through_independent_review_seconds':time.monotonic()-start})
    cfg=build_config(plan,selection)
    cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract(cfg);validate_environment(cfg);write(CONFIG,cfg)
    print('FROZEN_NOT_EXECUTED',[r['id'] for r in cfg['requests']],sha256(CONFIG))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--plan',action='store_true');g.add_argument('--select-freeze',action='store_true');g.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.plan:
        require_clean();v=build_plan();v['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();write(PLAN,v)
    elif a.worker:select_worker()
    else:select_and_freeze()
