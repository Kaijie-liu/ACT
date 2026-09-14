"""Five-query maximum order-only refinement of the sealed request LP control."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from fractions import Fraction

from act.pipeline.moe.request_lp_control import frozen_request, outward, SELECTION
from act.pipeline.moe.check_request_lp import check_directory, order_envelope, property_row
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value
from act.back_end.solver.lp_certificate import identity

PARENT=PROJECT_ROOT/'data/moe/results/request_lp_20260914_r1'
PARENT_HASH='90aad930c9f8b1f7018c3a7ef30cee80ab4487e5fae16b231bc6432c7b3cf082'


def validated_parent():
    if _sha256(PARENT/'manifest.json')!=PARENT_HASH: raise ValueError('frozen parent manifest drift')
    review=json.loads((PROJECT_ROOT/'act/pipeline/moe/results/request_lp_review_20260914_r1.json').read_text())
    for name,h in review['raw_hashes'].items():
        path=(PARENT/name).resolve()
        if not path.is_relative_to(PARENT) or _sha256(path)!=h: raise ValueError('parent artifact drift')
    check_directory(PARENT,expected_request_id=identity(frozen_request()))
    return json.loads((PARENT/'manifest.json').read_text())


def worker(root):
    import torch
    from act.back_end.moe import load_output_moe_checkpoint, build_act_moe_program, condition_topk_set
    from act.back_end.moe.hz_routing import guarded_input_topk_set
    from act.back_end.moe.weighted_top2 import WeightedTop2GateRange, WeightedTop2DifferenceRange, build_weighted_top2_f0
    from act.back_end.solver.hz_lp_export import export, snapshot
    from act.back_end.solver.lp_certificate import propose
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.config.config import HybridZConfig
    from act.front_end.specs import OutputSpec, OutKind
    from act.pipeline.moe.experiment1 import _propagate_component, shared_input_pair_propagation
    from act.pipeline.moe.staged_verifier import _tensor_identity
    from act.util.device_manager import initialize_device

    parent=validated_parent(); (root/'base').mkdir()
    # Immutable copies, no re-solve of R1 or rewriting its verdict.
    for path in PARENT.iterdir():
        if path.is_file(): shutil.copy2(path,root/'base'/path.name)
    manifest=copy.deepcopy(parent);manifest['schema']='request_lp_order_v2'
    manifest['parent']={'manifest':'base/manifest.json','sha256':PARENT_HASH}
    for item in manifest['proofs'].values():
        for key in ('export','certificate'):
            if item[key]:item[key]['file']='base/'+item[key]['file']
    for row in manifest['obligations']:
        if row['kind']=='residual':row.update(kind='unknown',reason='ORDER_REFINEMENT_PENDING')
    flush=lambda:save(root/'manifest.json',manifest)
    flush()
    torch.set_num_threads(1);initialize_device('cpu','float64')
    subject=json.loads(SELECTION.read_text())['models']['seed0'];checkpoint=Path(subject['checkpoint'])
    if _sha256(checkpoint)!=parent['request']['checkpoint_sha256']:raise ValueError('checkpoint drift')
    model,_=load_output_moe_checkpoint(checkpoint,map_location='cpu');model.cpu().double().eval()
    tensors=torch.load(root/'base/request.pt',map_location='cpu',weights_only=True)
    for key,tensor in tensors.items():
        if _tensor_identity(tensor)!=parent['request'][key]:raise ValueError('input drift')
    program=build_act_moe_program(model,**tensors,output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=[parent['request']['clean_prediction']]))
    config=HybridZConfig(max_input_dim=1024,guarded_support_enabled=False,expert_property_solver_backend='scipy')
    router=_propagate_component(program.router,hybridz_config=config)
    def proof(key,hz,q,kind,pair,prop):
        record=export(hz,q,sparse=True);p=root/(key+'.export.json');save(p,record)
        ref=lambda p:{'file':p.name,'sha256':_sha256(p)}
        item={'request_id':parent['request_id'],'kind':kind,'scope':{'pair':pair},'property_index':prop,
              'export':ref(p),'certificate':None,'hz_sha256':record['source_sha256'],'status':'PENDING'}
        manifest['proofs'][key]=item;flush();value=None;start=time.monotonic()
        try:
            cert=propose(record['lp'],time_limit=10);cp=root/(key+'.certificate.json');save(cp,cert)
            item['certificate']=ref(cp)
            checked=check_export(record,cert,expected_source_sha256=item['hz_sha256'])
            value=Fraction(checked['bound']['checked_lower_bound']);item.update(status='CHECKED',checked_lower_bound=str(value))
        except Exception as exc:item.update(status='UNKNOWN',error=type(exc).__name__+': '+str(exc))
        item['seconds']=time.monotonic()-start;flush();print(key,item['status'],str(value),flush=True)
        return value
    for pair in parent['routes']['feasible']:
        pending=[r for r in parent['obligations'] if r['pair']==pair and r['kind']=='residual']
        if not pending:continue
        conditioned=condition_topk_set(router.output_hz,pair).hz
        qm=[0]*8;qm[pair[0]]=1;qm[pair[1]]=-1
        prefix=f'order_{pair[0]}_{pair[1]}'
        kl,ku=prefix+'_lo',prefix+'_hi'
        low=proof(kl,conditioned,qm,'router_order',pair,None)
        neg=proof(ku,conditioned,[-v for v in qm],'router_order',pair,None)
        envelope=order_envelope(low,neg)
        if envelope==[0,1]:
            for original in pending:
                row=next(r for r in manifest['obligations'] if r['pair']==pair and r['property_index']==original['property_index'])
                row.update(original,gate_lower=kl,gate_upper=ku)
            flush();continue
        entry=guarded_input_topk_set(router.input_hz,router.output_hz,pair).hz
        joint=shared_input_pair_propagation(program.experts[pair[0]],program.experts[pair[1]],entry_hz=entry,hybridz_config=config).joint
        gate=WeightedTop2GateRange(tuple(pair),conditioned,conditioned.frame_id,conditioned.n_out,(-math.inf,math.inf),tuple(envelope),None)
        for original in pending:
            prop=original['property_index'];q=property_row(10,parent['request']['clean_prediction'],prop)
            lower_proof=parent['proofs'][original['difference_lower']]
            upper_proof=parent['proofs'][original['difference_upper']]
            if identity(snapshot(joint.output_hz))!=lower_proof['hz_sha256'] or lower_proof['hz_sha256']!=upper_proof['hz_sha256']:
                raise ValueError('reconstructed shared HZ differs from frozen disagreement source')
            bounds=(outward(Fraction(lower_proof['checked_lower_bound'])),outward(-Fraction(upper_proof['checked_lower_bound']),True))
            difference=WeightedTop2DifferenceRange(tuple(pair),joint.output_hz,joint.output_hz.frame_id,joint.output_hz.n_out,tuple(float(v) for v in q),bounds,None)
            encoding=build_weighted_top2_f0(joint,conditioned,pair,q,0.,difference_time_limit=10,gate_range=gate,difference_range=difference)
            key=f'{prefix}_p{prop}_output';proof(key,encoding.output_hz,[1],'weighted',pair,prop)
            row=next(r for r in manifest['obligations'] if r['pair']==pair and r['property_index']==prop)
            row.update(original,source=key,lambda_bounds=envelope,gate_lower=kl,gate_upper=ku)
            flush()


def run(root):
    if _git_value('branch','--show-current')!='feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise RuntimeError('clean feature branch required')
    root=_inside(root,WRITE_ROOT);root.mkdir(exist_ok=False)
    save(root/'launch.json',{'head':_git_value('rev-parse','HEAD'),'source_sha256':source_identity(),
                            'parent_manifest_sha256':PARENT_HASH,'request_id':identity(frozen_request()),'outer_seconds':600})
    start=time.monotonic()
    with (root/'worker.log').open('w') as log:
        try:
            done=subprocess.run([sys.executable,'-m','act.pipeline.moe.request_lp_order','--worker',str(root)],
                stdout=log,stderr=subprocess.STDOUT,timeout=600,check=False)
            terminal={'returncode':done.returncode}
        except subprocess.TimeoutExpired:terminal={'status':'TIMEOUT'}
    terminal['seconds']=time.monotonic()-start;save(root/'terminal.json',terminal)
    if (root/'manifest.json').exists():
        result=check_directory(root,expected_request_id=identity(frozen_request()));save(root/'check.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run',type=Path);g.add_argument('--worker',type=Path)
    args=p.parse_args();run(args.run) if args.run else worker(args.worker)
