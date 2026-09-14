"""Frozen single-request supplied-HZ/LP control; never a production SAFE gate."""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from act.pipeline.moe.check_request_lp import TRUSTED, check_directory, property_row
from act.back_end.solver.lp_certificate import identity
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value

SELECTION = PROJECT_ROOT/'act/pipeline/moe/configs/schedule_confirmation_selection_r2.json'
SELECTION_HASH = 'e001940bf28f64c997dc1f947dd2fcda515f4051985836756c25843b89dcd987'


def frozen_request():
    if _sha256(SELECTION) != SELECTION_HASH: raise ValueError('selection changed')
    selection = json.loads(SELECTION.read_text())
    sample = selection['smoke_samples'][0]; subject = selection['models']['seed0']
    if sample['dataset_index'] != 3000: raise ValueError('request changed')
    return {'selection_sha256':SELECTION_HASH, 'dataset_index':3000,
            'checkpoint_sha256':subject['checkpoint_sha256'], 'epsilon':2/255,
            'center':sample['center'], 'lower':sample['lower'], 'upper':sample['upper'],
            'clean_prediction':sample['clean_predictions']['seed0'], 'classes':10,
            'experts':8, 'top_k':2, 'tie_policy':'ANY_LEGAL_TOPK'}


def outward(value, upper=False):
    result = float(value)
    if not math.isfinite(result): raise ValueError('nonfinite bound')
    if (upper and Fraction.from_float(result) < value) or (not upper and Fraction.from_float(result) > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def worker(root):
    import torch
    from dataclasses import asdict
    from act.back_end.moe import load_output_moe_checkpoint, build_act_moe_program, condition_topk_set
    from act.back_end.moe.hz_routing import analyze_topk_sets, guarded_input_domain, guarded_input_topk_set
    from act.back_end.moe.weighted_top2 import WeightedTop2GateRange, WeightedTop2DifferenceRange, build_weighted_top2_f0
    from act.back_end.solver.hz_lp_export import export
    from act.back_end.solver.lp_certificate import propose
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.config.config import HybridZConfig
    from act.front_end.specs import OutputSpec, OutKind
    from act.pipeline.moe.experiment1 import _propagate_component, shared_input_pair_propagation
    from act.pipeline.moe.train import _load_dataset
    from act.pipeline.moe.staged_verifier import _tensor_identity
    from act.util.device_manager import initialize_device

    request = frozen_request(); rid = identity(request)
    selection = json.loads(SELECTION.read_text()); subject = selection['models']['seed0']
    checkpoint = Path(subject['checkpoint'])
    if _sha256(checkpoint) != request['checkpoint_sha256']: raise ValueError('checkpoint drift')
    if _sha256(Path(selection['dataset']['raw_test_batch'])) != selection['dataset']['raw_test_batch_sha256']:
        raise ValueError('dataset drift')
    torch.set_num_threads(1); initialize_device('cpu','float64')
    os.environ['ACT_TORCHVISION_DATA_ROOT'] = str(PROJECT_ROOT/'data/torchvision')
    model,payload = load_output_moe_checkpoint(checkpoint,map_location='cpu'); model.cpu().double().eval()
    image,label = _load_dataset(payload['dataset'],False,download=False)[3000]
    center = image.unsqueeze(0).double(); lower=(center-2/255).clamp(0,1); upper=(center+2/255).clamp(0,1)
    for k,v in [('center',center),('lower',lower),('upper',upper)]:
        if _tensor_identity(v) != request[k]: raise ValueError('input identity drift')
    with torch.no_grad(): pred=int(model(center).argmax())
    if pred != request['clean_prediction'] or pred != label: raise ValueError('clean prediction drift')
    torch.save({'center':center,'lower':lower,'upper':upper},root/'request.pt')
    program=build_act_moe_program(model,center=center,lower=lower,upper=upper,
        output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=[pred]))
    config=HybridZConfig(max_input_dim=1024,guarded_support_enabled=False,expert_property_solver_backend='scipy')
    router=_propagate_component(program.router,hybridz_config=config)
    manifest={'schema':'request_lp_v1','request':request,'request_id':rid,'trusted_base':TRUSTED,
              'positive_threshold':1e-7,'proofs':{},'obligations':[],
              'routes':{'feasible':[],'infeasible':[],'unresolved':list(itertools.combinations(range(8),2)),'exact':False}}
    flush=lambda:save(root/'manifest.json',manifest)
    flush()
    routes=analyze_topk_sets(router.output_hz,2,time_limit_per_set=10,router_exact=router.output_hz.exact)
    manifest['routes']=asdict(routes)
    manifest['obligations']=[{'pair':list(p),'property_index':i,'kind':'unknown','reason':'NOT_YET_CHECKED'}
                            for p in routes.feasible for i in range(9)]
    flush()  # Full inventory precedes any output solve.
    if not routes.exact or routes.unresolved: return
    values={}
    def proof(key,hz,q,kind,scope,prop):
        record=export(hz,q,sparse=True); path=root/(key+'.export.json'); save(path,record)
        ref=lambda p:{'file':p.name,'sha256':_sha256(p)}
        item={'request_id':rid,'kind':kind,'scope':scope,'property_index':prop,'export':ref(path),
              'hz_sha256':record['source_sha256'],'status':'PENDING','certificate':None}
        manifest['proofs'][key]=item; flush(); value=None; started=time.monotonic()
        try:
            cert=propose(record['lp'],time_limit=10); cp=root/(key+'.certificate.json'); save(cp,cert)
            item['certificate']=ref(cp)
            checked=check_export(record,cert,expected_source_sha256=item['hz_sha256'])
            value=Fraction(checked['bound']['checked_lower_bound']); item['status']='CHECKED'
            item['checked_lower_bound']=str(value)
        except Exception as exc:
            item['status']='UNKNOWN'; item['error']=type(exc).__name__+': '+str(exc)
        item['seconds']=time.monotonic()-started; values[key]=value; flush()
        print(key,item['status'],str(value),flush=True)
        return value
    for expert in sorted({i for p in routes.feasible for i in p}):
        entry=guarded_input_domain(router.input_hz,router.output_hz,expert,2).hz
        hz=_propagate_component(program.experts[expert],entry_hz=entry,hybridz_config=config).output_hz
        for i in range(9): proof(f'e{expert}_p{i}',hz,property_row(10,pred,i),'expert',{'membership':expert},i)
    for p in routes.feasible:
        pair=list(p); pending=[]
        for row in [r for r in manifest['obligations'] if r['pair']==pair]:
            i=row['property_index']; keys=[f'e{e}_p{i}' for e in p]
            if all(values[k] is not None and values[k]>Fraction.from_float(1e-7) for k in keys):
                row.update(kind='reused',sources=keys)
            else: pending.append(row)
        flush()
        if not pending: continue
        entry=guarded_input_topk_set(router.input_hz,router.output_hz,p).hz
        joint=shared_input_pair_propagation(program.experts[p[0]],program.experts[p[1]],entry_hz=entry,hybridz_config=config).joint
        conditioned=condition_topk_set(router.output_hz,p).hz
        gate=WeightedTop2GateRange(tuple(p),conditioned,conditioned.frame_id,conditioned.n_out,(-math.inf,math.inf),(0.,1.),None)
        for row in pending:
            i=row['property_index']; q=property_row(10,pred,i); qdiff=q+[-v for v in q]; prefix=f's{p[0]}_{p[1]}_p{i}'
            lo_key,hi_key=prefix+'_lo',prefix+'_hi'
            lo=proof(lo_key,joint.output_hz,qdiff,'difference',{'pair':pair},i)
            nhi=proof(hi_key,joint.output_hz,[-v for v in qdiff],'difference',{'pair':pair},i)
            if lo is None or nhi is None:
                row['reason']='DIFFERENCE_LP_UNCHECKED'; flush(); continue
            bounds=(outward(lo),outward(-nhi,True))
            difference=WeightedTop2DifferenceRange(tuple(p),joint.output_hz,joint.output_hz.frame_id,joint.output_hz.n_out,tuple(float(v) for v in q),bounds,None)
            encoding=build_weighted_top2_f0(joint,conditioned,p,q,0.,difference_time_limit=10,gate_range=gate,difference_range=difference)
            key=prefix+'_output'; proof(key,encoding.output_hz,[1],'weighted',{'pair':pair},i)
            row.update(kind='residual',difference_lower=lo_key,difference_upper=hi_key,
                       difference_bounds=list(bounds),lambda_bounds=[0,1],source=key)
            flush()


def run(root):
    if _git_value('branch','--show-current')!='feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise RuntimeError('clean feature branch required')
    root=_inside(root,WRITE_ROOT); root.mkdir(exist_ok=False)
    request=frozen_request()
    save(root/'launch.json',{'head':_git_value('rev-parse','HEAD'),'source_sha256':source_identity(),
                            'request':request,'request_id':identity(request),'outer_seconds':3600})
    start=time.monotonic()
    with (root/'worker.log').open('w') as log:
        try:
            done=subprocess.run([sys.executable,'-m',__name__ if __name__!='__main__' else 'act.pipeline.moe.request_lp_control',
                                 '--worker',str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=False)
            terminal={'returncode':done.returncode}
        except subprocess.TimeoutExpired: terminal={'status':'TIMEOUT'}
    terminal['seconds']=time.monotonic()-start; save(root/'terminal.json',terminal)
    if (root/'manifest.json').exists():
        result=check_directory(root,expected_request_id=identity(request)); save(root/'check.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path); group.add_argument('--worker',type=Path); group.add_argument('--check',type=Path)
    args=parser.parse_args()
    if args.run: run(args.run)
    elif args.worker: worker(args.worker)
    else: print(json.dumps(check_directory(args.check,expected_request_id=identity(frozen_request())),indent=2))
