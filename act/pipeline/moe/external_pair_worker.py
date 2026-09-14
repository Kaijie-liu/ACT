"""Complete-request workers; data loading and cross-env handoff are charged."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from act.pipeline.moe.external_compatibility import TOOL,ENV,dump,sha


def load(request):
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    import torch
    from act.util.device_manager import initialize_device
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.pipeline.moe.staged_verifier import _tensor_identity,_model_state_identity
    torch.set_num_threads(1);initialize_device('cpu','float64')
    subject=request['subject'];sample=request['sample']
    if sha(Path(subject['checkpoint']))!=subject['checkpoint_sha256']:raise ValueError('checkpoint changed')
    model,_=load_output_moe_checkpoint(subject['checkpoint'],map_location='cpu');model.cpu().double().eval()
    if _model_state_identity(model)!=subject['model_state']:raise ValueError('model-state mismatch')
    path=Path(request['tensors']['path'])
    if sha(path)!=request['tensors']['sha256']:raise ValueError('materialized tensor file changed')
    tensors=torch.load(path,map_location='cpu',weights_only=True)
    for name,value in tensors.items():
        if value.dtype!=torch.float64 or value.device.type!='cpu' or _tensor_identity(value)!=sample[name]:
            raise ValueError('represented input mismatch')
    for name,sign in [('lower',-1),('upper',1)]:
        if not torch.equal(tensors[name],(tensors['center']+sign*request['epsilon']).clamp(0,1)):
            raise ValueError('box differs from production contract')
    with torch.no_grad():
        if int(model(tensors['center']).argmax())!=sample['label']:raise ValueError('clean property mismatch')
    return model,tensors


def frontend(root,started):
    request=json.loads((root/'request.json').read_text());model,tensors=load(request)
    if request['method']=='adaptive':
        from act.pipeline.moe.staged_verifier import verify_staged_linf,write_evidence_package
        from act.pipeline.moe.common_fact_snapshot import publish_snapshot
        cfgpath=Path(request['config']['path'])
        if sha(cfgpath)!=request['config']['sha256']:raise ValueError('adaptive config changed')
        report=verify_staged_linf(model,tensors['center'],request['epsilon'],json.loads(cfgpath.read_text()),
            expected_clean_prediction=request['sample']['label'],budget_started_at=started,
            checkpoint_identity={'path':request['subject']['checkpoint'],'sha256':request['subject']['checkpoint_sha256']},
            progress_callback=lambda v:dump(root/'progress.json',v),
            common_fact_callback=lambda v:publish_snapshot(root/'common_facts.json',v))
        report.evidence['execution']={'git_head':request['head'],'config_path':str(cfgpath),
            'config_sha256':request['config']['sha256'],'dataset_index':request['sample']['dataset_index']}
        write_evidence_package(report,root/'package')
        return
    from act.back_end.moe import build_act_moe_program
    from act.back_end.moe.hz_routing import analyze_topk_sets
    from act.config.config import HybridZConfig
    from act.front_end.specs import OutputSpec,OutKind
    from act.pipeline.moe.experiment1 import _propagate_component
    program=build_act_moe_program(model,**tensors,output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST,y_true=[request['sample']['label']]))
    cfg=HybridZConfig(max_input_dim=1024,guarded_support_enabled=False,expert_property_solver_backend='scipy')
    router=_propagate_component(program.router,hybridz_config=cfg)
    def remaining():return max(.001,min(10.,300-(time.monotonic()-started)))
    routes=asdict(analyze_topk_sets(router.output_hz,2,time_limit_per_set=remaining,router_exact=router.output_hz.exact))
    dump(root/'routes.json',{'request':request,'routes':routes,'elapsed_seconds':time.monotonic()-started})
    if not routes['exact'] or routes['unresolved'] or not routes['feasible']:
        dump(root/'external.json',{'status':'UNKNOWN','reason':'INCOMPLETE_ROUTE_COVERAGE','pairs':[]});return
    # Child shares the request process group, killed by the outer watchdog.
    subprocess.run([ENV,'-m','act.pipeline.moe.external_pair_worker','--root',str(root),'--external',
                    '--started',repr(started)],check=True)


def external(root,started):
    sys.path[:0]=[str(TOOL/'auto_LiRPA'),str(TOOL/'complete_verifier')]
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    import torch
    import auto_LiRPA
    from auto_LiRPA import BoundedModule,BoundedTensor,PerturbationLpNorm
    from act.back_end.moe.static_pair import StaticSelectedSoftmaxPair
    from act.pipeline.moe.check_request_lp import property_row
    from act.pipeline.moe.staged_verifier import _tensor_identity,_model_state_identity
    if not Path(auto_LiRPA.__file__).resolve().is_relative_to(TOOL/'auto_LiRPA'):raise ValueError('wrong external source')
    if sys.version_info[:3]!=(3,11,16) or torch.__version__!='2.11.0+cu130':raise ValueError('frozen external environment drift')
    request=json.loads((root/'request.json').read_text());model,tensors=load(request)
    routes=json.loads((root/'routes.json').read_text())['routes']
    center,lo,hi=[tensors[k] for k in ('center','lower','upper')];pred=request['sample']['label']
    C=torch.tensor([property_row(10,pred,i) for i in range(9)],dtype=torch.float64).unsqueeze(0)
    mask=torch.arange(center.numel()).reshape(center.shape)%2==0
    points=[center,lo,hi,torch.where(mask,lo,hi),torch.where(mask,hi,lo)]
    result={'status':'UNKNOWN','reason':'PAIRS_PENDING','pairs':[], 'C':C.tolist(),
            'tensor_identity':{k:_tensor_identity(t) for k,t in tensors.items()},'model_state':_model_state_identity(model),
            'environment':{'python':sys.version,'torch':torch.__version__,'auto_lirpa_file':auto_LiRPA.__file__,
                           'device':'cpu','dtype':'float64','threads':torch.get_num_threads()},
            'backend':{'method':'CROWN','bound_opts':{'conv_mode':'matrix'}},'formal_SAFE':False}
    dump(root/'external.json',result)
    # These five conformance probes are not a strong attack. Any discovered
    # violation must be replayed again by the ACT auditor on the dynamic model.
    with torch.no_grad():
        for point in points:
            if int(model(point).argmax())!=pred:
                torch.save(point,root/'witness.pt')
                result.update(status='UNSAFE',reason='DYNAMIC_CONFORMANCE_PROBE',witness_sha256=sha(root/'witness.pt'))
                dump(root/'external.json',result);return
    for pair in routes['feasible']:
        start=time.monotonic();adapter=StaticSelectedSoftmaxPair(model,pair)
        with torch.no_grad():
            refs=[];errors=[]
            for point in points:
                weights=torch.softmax(model.router(point)[:,pair],dim=1)
                expected=sum(weights[:,i:i+1]*model.experts[e](point) for i,e in enumerate(pair))
                refs.append(expected);errors.append(float((adapter(point)-expected).abs().max()))
        if max(errors)>1e-10:raise ValueError('forced-branch mismatch')
        bounded=BoundedModule(adapter,center,device='cpu',bound_opts={'conv_mode':'matrix'})
        with torch.no_grad():lowered=max(float((bounded(p)-v).abs().max()) for p,v in zip(points,refs))
        if lowered>1e-10:raise ValueError('lowered graph mismatch')
        bound_start=time.monotonic()
        lower,upper=bounded.compute_bounds(x=(BoundedTensor(center,PerturbationLpNorm(norm=float('inf'),x_L=lo,x_U=hi)),),C=C,method='CROWN')
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():raise ValueError('nonfinite bound')
        result['pairs'].append({'pair':pair,'lower':lower.detach().tolist()[0],'upper':upper.detach().tolist()[0],
            'concrete_max_error':max(errors),'lowered_max_error':lowered,'nodes':len(bounded._modules),
            'graph_and_probes_seconds':bound_start-start,'bound_seconds':time.monotonic()-bound_start})
        result['elapsed_seconds']=time.monotonic()-started;dump(root/'external.json',result)
    positive=all(all(v>1e-7 for v in p['lower']) for p in result['pairs'])
    result.update(status='POSITIVE' if positive else 'UNKNOWN',reason='ALL_PAIRS_POSITIVE' if positive else 'NONPOSITIVE_NUMERICAL_BOUND')
    dump(root/'external.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--started',type=float,required=True);p.add_argument('--external',action='store_true');a=p.parse_args()
    external(a.root,a.started) if a.external else frontend(a.root,a.started)
