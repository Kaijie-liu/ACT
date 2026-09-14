"""Generate a complete small MoE request and independently check its LP proof.

No downloads, private checkpoints, dataset files or private absolute paths.
This constructed correctness example is not empirical accuracy evidence.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.output.resolve(); root.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo))
    started=time.monotonic()
    import torch
    from act.util.device_manager import initialize_device
    from act.back_end.moe import OutputMoEFactoryConfig,GateKind,build_output_moe
    from act.pipeline.moe.staged_verifier import verify_staged_linf,write_evidence_package,_tensor_identity,_model_state_identity
    from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
    from act.pipeline.moe.request_lp_cases import generate,save,sha
    initialize_device('cpu','float64');torch.set_num_threads(1)
    net=build_output_moe(OutputMoEFactoryConfig(input_shape=(2,),num_classes=2,
        num_experts=3,top_k=2,gate=GateKind.SELECTED_SOFTMAX,router_hidden=(),expert_hidden=(),seed=7)).cpu().double().eval()
    with torch.no_grad():
        net.router[1].weight.zero_(); net.router[1].bias.zero_()
        for expert,margin in zip(net.experts,[-.2,1.,2.]):
            expert[1].weight.zero_(); expert[1].bias.copy_(torch.tensor([margin,0.]))
    center=torch.full((1,2),.5,dtype=torch.float64);eps=.1
    tensors={'center':center,'lower':center-eps,'upper':center+eps}
    request={'classes':2,'experts':3,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK',
        'clean_prediction':0,'epsilon':eps,'model_state':_model_state_identity(net),
        **{k:_tensor_identity(v) for k,v in tensors.items()},
        'source_model':'fully specified affine all-tie control; not a trained model'}
    cfg_path=repo/'act/pipeline/moe/configs/route_complexity_reuse_v1.json'
    cfg=json.loads(cfg_path.read_text())
    report=verify_staged_linf(net,center,eps,cfg,expected_clean_prediction=0)
    write_evidence_package(report,root/'staged')
    audit=audit_evidence_package(root/'staged')
    if report.status!='SAFE' or audit['issues']:raise RuntimeError('staged control failed')
    proof=root/'proof';proof.mkdir(); generate(net,tensors,request,proof)
    generated=time.monotonic()-started
    check_start=time.monotonic()
    done=subprocess.run([sys.executable,'-S',str(repo/'scripts/check_moe_request_lp.py'),str(proof)],
                        check=True,capture_output=True,text=True,cwd=root)
    check=json.loads(done.stdout)
    if check['status']!='CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING':
        raise RuntimeError('rational control not complete')
    result={'model_and_request':request,'staged_status':report.status,'staged_audit':audit,
        'rational_check':check,'generation_including_staged_seconds':generated,
        'fresh_process_check_seconds':time.monotonic()-check_start,
        'config_sha256':sha(cfg_path),'script_sha256':sha(Path(__file__)),
        'environment':{'python':sys.version,**{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','torchvision')}},
        'scope':'Source-defined small all-tie control, not trained-model accuracy or a deployed floating-point proof.'}
    save(root/'summary.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__':main()
