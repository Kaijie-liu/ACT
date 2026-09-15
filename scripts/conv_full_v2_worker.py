"""Full cohort wrapper: V2 ACT, byte-unchanged external frontend."""
import argparse
from pathlib import Path

from scripts.conv_full_v2_contract import validate_worker, read
from scripts.budget_contract_v2 import verify_v2
from act.pipeline.moe.external_pair_worker import load, frontend
from act.pipeline.moe.external_compatibility import dump
from act.pipeline.moe.experiment1 import _sha256


def worker(root,started):
    request=validate_worker(root)
    if request['method']=='crown':return frontend(root,started)
    model,tensors=load(request)
    from act.pipeline.moe.staged_verifier import write_evidence_package
    from act.pipeline.moe.common_fact_snapshot import publish_snapshot
    config=read(request['config']['path'])
    report=verify_v2(model,tensors['center'],request['epsilon'],config,
        journal_path=root/'budget_journal.jsonl',started=started,
        identity={'request_sha256':_sha256(root/'request.json'),
                  'execution_budget_contract':request['execution_budget_contract']},
        expected_clean_prediction=request['sample']['label'],
        checkpoint_identity={'path':request['subject']['checkpoint'],'sha256':request['subject']['checkpoint_sha256']},
        progress_callback=lambda v:dump(root/'progress.json',v),
        common_fact_callback=lambda v:publish_snapshot(root/'common_facts.json',v))
    report.evidence['execution']={'git_head':request['head'],'config_path':request['config']['path'],
        'config_sha256':request['config']['sha256'],'dataset_index':request['sample']['dataset_index']}
    report.evidence['execution_budget_contract']['journal_sha256']=_sha256(root/'budget_journal.jsonl')
    write_evidence_package(report,root/'package')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--started',type=float,required=True);args=p.parse_args();worker(args.root,args.started)
