"""Process-isolated generic evidence phases; no frozen example dependency."""
import argparse
from pathlib import Path
import sys
from scripts.optional_evidence_dev_contract import read,save
from scripts.optional_evidence_budget import EvidenceBudget,EvidenceBudgetExpired
from portable_proof.runtime import digest


def validate_transport(req):
    from moe_evidence.schema import validate_request
    r=req['evidence_request'];validate_request(r)
    if (r['model_state']!=req['subject']['model_state'] or r['epsilon']!=req['epsilon']
            or r['clean_prediction']!=req['sample']['label']
            or any(r[k]!=req['sample'][k] for k in ('center','lower','upper'))):
        raise ValueError('transport/evidence request mismatch')
    p=Path(req['config']['path'])
    if digest(p.read_bytes())!=req['config']['sha256']:raise ValueError('configuration changed')
    config=read(p)
    if config['comparison_method']!='monolithic_f0':raise ValueError('matched V2 configuration required')
    return r,config


def execute(stage,root,budget):
    req=read(root/'request.json');r,config=validate_transport(req);budget.remaining(2)
    if stage in ('capture','matched'):
        from act.pipeline.moe.external_pair_worker import load
        model,tensors=load(req)
        if stage=='capture':
            from moe_evidence.generate import capture
            capture(model,tensors,r,config,root,budget)
        else:
            from scripts.budget_contract_v2 import verify_v2
            from act.pipeline.moe.staged_verifier import write_evidence_package
            from act.pipeline.moe.common_fact_snapshot import publish_snapshot
            from moe_evidence.schema import classification_properties
            if r['properties']!=classification_properties(r['classes'],r['clean_prediction']):
                raise ValueError('unchanged production comparison supports classification only')
            report=verify_v2(model,tensors['center'],req['epsilon'],config,started=budget.started,
                journal_path=root/'budget_journal.jsonl',identity={'request_sha256':digest((root/'request.json').read_bytes())},
                expected_clean_prediction=req['sample']['label'],
                checkpoint_identity={'path':req['subject']['checkpoint'],'sha256':req['subject']['checkpoint_sha256']},
                common_fact_callback=lambda v:publish_snapshot(root/'common_facts.json',v))
            report.evidence['execution']={'git_head':req['head'],'config_path':req['config']['path'],
                'config_sha256':req['config']['sha256'],'dataset_index':req['sample']['dataset_index']}
            report.evidence['execution_budget_contract']['journal_sha256']=digest((root/'budget_journal.jsonl').read_bytes())
            write_evidence_package(report,root/'package')
    elif stage=='propose':
        from moe_evidence.generate import propose_all
        propose_all(root,budget)
    elif stage=='precheck':
        from moe_evidence.checker import check_manifest
        from moe_evidence.storage import loader
        tick=lambda:budget.remaining(2)
        result=check_manifest(read(root/'manifest.json'),r,loader(root,tick),tick=tick)
        save(root/'independent.json',result)
    elif stage=='package':
        from moe_evidence.bundle import pack
        result=pack(root,root/'portable',read(root/'independent.json'),tick=lambda:budget.remaining(2))
        save(root/'packing.json',result)
    elif stage=='crown':
        from moe_evidence.schema import classification_properties
        if (r['classes']!=10 or r['properties']!=classification_properties(10,r['clean_prediction'])
                or req['method']!='crown'):
            raise ValueError('unchanged external control is frozen C10 classification')
        from act.pipeline.moe.external_pair_worker import frontend
        frontend(root,budget.started)
    else:raise ValueError('unknown phase')
    budget.remaining(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');p.add_argument('directory',type=Path)
    p.add_argument('--started',type=float,required=True);a=p.parse_args()
    try:execute(a.stage,a.directory,EvidenceBudget(a.started))
    except EvidenceBudgetExpired:sys.exit(3)
