"""V2 worker adapter, NOT a supervisor or permission to launch a cohort.

Requires a new runtime-bound request with execution_budget_contract identity.
An outer supervisor must provide the original monotonic start and300s kill.
"""
import argparse
from pathlib import Path

from scripts.conv_three_arm_contract import ROOT, read, selection, request_for
from scripts.budget_contract_v2 import verify_v2
from act.pipeline.moe.conv_three_arm_worker import validate_request
from act.pipeline.moe.external_pair_worker import load
from act.pipeline.moe.external_compatibility import dump
from act.pipeline.moe.experiment1 import _sha256

POLICY=ROOT/'scripts/conv_budget_contract_v2.json'
SOURCES=('scripts/budget_contract_v2.py','scripts/check_budget_contract_v2.py',
         'scripts/conv_budget_worker_v2.py','scripts/conv_budget_contract_v2.json',
         'scripts/f0_timing_trace.py')


def execution_identity():
    policy=read(POLICY)
    expected={'version':2,'whole_request_seconds':300,'terminal_reserve_seconds':5,
              'minimum_native_seconds':.001,'arms':['adaptive','monolithic'],
              'multi_pair_tier1_fraction':.25,'partial_journal_can_establish_SAFE':False,
              'full_90_authorized':False}
    if any(policy.get(k)!=v for k,v in expected.items()):
        raise ValueError('policy does not describe implemented V2 contract')
    return {'policy':policy, 'sources':{p:_sha256(ROOT/p) for p in SOURCES}}


def validate(root):
    if not root.resolve().is_relative_to(ROOT/'data/moe/results'):
        raise ValueError('outside local result tree')
    request=read(root/'request.json');validate_request(request)
    if request['method'] not in ('adaptive','monolithic'):
        raise ValueError('V2 has no CROWN/backend fallback')
    if request.get('execution_budget_contract') != execution_identity():
        raise ValueError('missing or changed V2 execution identity')
    if (root/'package').exists() or (root/'budget_journal.jsonl').exists():
        raise ValueError('refuse resumed/replaced request')
    value=selection()
    jobs=[j for j in value['smoke_jobs'] if j['method']==request['method']
          and j['dataset_index']==request['sample']['dataset_index']]
    if len(jobs)!=1:raise ValueError('V2 worker only accepts old frozen smoke inputs')
    expected=request_for(value,jobs[0],request['head'])
    if {k:v for k,v in request.items() if k!='execution_budget_contract'} != expected:
        raise ValueError('base request/config/input/model differs')
    return request


def worker(root,started):
    request=validate(root)
    model,tensors=load(request)
    from act.pipeline.moe.staged_verifier import write_evidence_package
    from act.pipeline.moe.common_fact_snapshot import publish_snapshot
    config=read(Path(request['config']['path']))
    report=verify_v2(model,tensors['center'],request['epsilon'],config,
        journal_path=root/'budget_journal.jsonl',started=started,
        identity={'request_sha256':_sha256(root/'request.json'),
                  'execution_budget_contract':request['execution_budget_contract']},
        expected_clean_prediction=request['sample']['label'],
        checkpoint_identity={'path':request['subject']['checkpoint'],'sha256':request['subject']['checkpoint_sha256']},
        progress_callback=lambda record:dump(root/'progress.json',record),
        common_fact_callback=lambda record:publish_snapshot(root/'common_facts.json',record))
    report.evidence['execution']={'git_head':request['head'],'config_path':request['config']['path'],
        'config_sha256':request['config']['sha256'],'dataset_index':request['sample']['dataset_index']}
    report.evidence['execution_budget_contract']['journal_sha256']=_sha256(root/'budget_journal.jsonl')
    write_evidence_package(report,root/'package')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--started',type=float,required=True)
    args=parser.parse_args();worker(args.root,args.started)
