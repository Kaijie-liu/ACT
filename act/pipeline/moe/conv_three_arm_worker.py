"""Four-expert request adapter; no dataset field inferred from checkpoint payload.

This is a worker, not an authorized launch pipeline. The future outer runner
must enforce the frozen whole-request deadline, smoke audit and artifact lock.
"""
import argparse
import json
from pathlib import Path

from act.pipeline.moe.external_compatibility import dump, sha
from act.pipeline.moe.external_pair_worker import load, frontend


def validate_request(request):
    if request['protocol']!='CONV_FAMILY_THREE_ARM_R1' or request['method'] not in ('adaptive','monolithic','crown'):
        raise ValueError('wrong request protocol/method')
    if request['epsilon']!=2/255 or request['topology']!={'num_experts':4,'top_k':2,'classes':10}:
        raise ValueError('wrong frozen convolutional request')


def worker(root,started):
    request=json.loads((root/'request.json').read_text());validate_request(request)
    if request['method']=='crown':
        # Existing loader binds the complete model state and materialized inputs;
        # it enumerates actual router outputs, not a hard-coded E=8 universe.
        return frontend(root,started)
    model,tensors=load(request)
    if len(model.experts)!=4 or model.spec.top_k!=2:raise ValueError('checkpoint topology mismatch')
    config_path=Path(request['config']['path'])
    if sha(config_path)!=request['config']['sha256']:raise ValueError('method config drift')
    config=json.loads(config_path.read_text())
    expected='staged' if request['method']=='adaptive' else 'monolithic_f0'
    if config['comparison_method']!=expected:raise ValueError('wrong ACT dispatch')
    from act.pipeline.moe.staged_verifier import verify_staged_linf,write_evidence_package
    from act.pipeline.moe.common_fact_snapshot import publish_snapshot
    report=verify_staged_linf(model,tensors['center'],request['epsilon'],config,
        expected_clean_prediction=request['sample']['label'],budget_started_at=started,
        checkpoint_identity={'path':request['subject']['checkpoint'],'sha256':request['subject']['checkpoint_sha256']},
        progress_callback=lambda v:dump(root/'progress.json',v),
        common_fact_callback=lambda v:publish_snapshot(root/'common_facts.json',v))
    report.evidence['execution']={'git_head':request['head'],'config_path':str(config_path),
        'config_sha256':request['config']['sha256'],'dataset_index':request['sample']['dataset_index']}
    write_evidence_package(report,root/'package')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--started',type=float,required=True);a=p.parse_args();worker(a.root.resolve(),a.started)
