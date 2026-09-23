"""Saved identity only: no checkpoint load, inference, HZ or solver call."""
import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

ROOT=Path(__file__).resolve().parents[1]
SELECTION=ROOT/'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
LEDGER=ROOT/'docs/main_table_source_applicability_20260921.json'
DOC=ROOT/'docs/source_output_closure_scope_20260923_r1.md'
OUTPUT=ROOT/'configs/backend_controls/source_output_closure_scope_r1.json'


def build():
    selection=json.loads(SELECTION.read_text());ledger=json.loads(LEDGER.read_text())
    name=sorted(selection['models'])[0];model=selection['models'][name]
    sample=min(selection['samples'],key=lambda r:r['sample_rank'])
    assert (name,sample['sample_rank'],sample['dataset_index'],sample['label'])==('seed0',0,4088,7)
    src=next(r['request'] for r in ledger['artifact_inventory'] if r['job_id']=='rank0_seed0_adaptive')
    request=ROOT/src['path'];checkpoint=Path(model['checkpoint'])
    if sha256(request)!=src['sha256'] or sha256(checkpoint)!=model['checkpoint_sha256']:
        raise ValueError('bound stored input/checkpoint drift')
    pairs=list(combinations(range(8),2));classes=[k for k in range(10) if k!=sample['label']]
    margin=Fraction.from_float(1e-7)
    return {'protocol':'same_object_source_output_scope_r1','status':'PROTOCOL_ONLY_IMPLEMENTATION_GATE_PENDING',
        'execution_authorized':False,'real_requests_executed':0,
        'selection_rule':'first model key and first rank of bound old selection; no outcome/route predicate',
        'model':model,'sample':{k:sample[k] for k in ('sample_rank','dataset_index','label','center')},
        'stored_request':{'path':str(request),'sha256':src['sha256'],'read_for_center_only':True},
        'requested_domain':{'center':'exact binary rationals of bound stored center','radius':'2/255','clip':['0','1'],
            'old_materialized_endpoints_are_not_authoritative':True},
        'margin':str(margin),'semantics':'eval real graph; selected-softmax top2; any legal ties',
        'pairs':[list(p) for p in pairs],'no_pair_exclusion':True,
        'output_obligations':[{'pair':list(p),'label':7,'competitor':k} for p in pairs for k in classes],
        'required_output_obligations':252,'gate_range':['0','1'],'historical_bounds_or_facts_allowed':False,
        'limits':{'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30},
        'route_changing_requires_separate_checked_witnesses':True,'no_sample_substitution':True,'no_input98':True,
        'trusted_boundary_to_disclose':['stored center preprocessing','declared graph/program correspondence if unproved',
            'checker implementation/runtime','native floating-point deployment is outside real-graph theorem'],
        'files':{str(p.relative_to(ROOT)):sha256(p) for p in (SELECTION,LEDGER,DOC,Path(__file__).resolve())}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    cfg=build()
    if a.check:
        if json.loads(OUTPUT.read_text())!=cfg:raise ValueError('scope/identity/obligation drift')
        print('PASS: identities and 252 planned obligations; NO new proof or solver result')
    else:
        if OUTPUT.exists():raise FileExistsError(OUTPUT)
        write(OUTPUT,cfg)
        print('Prepared protocol only; implementation controls and execution freeze remain pending')
