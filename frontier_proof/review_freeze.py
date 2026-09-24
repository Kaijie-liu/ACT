"""Independent stdlib freeze review, no producer/model/solver imports or execution."""
import argparse
from fractions import Fraction
from itertools import combinations
import json
import sys
from scoped_proof.io import ROOT, PYTHON, load, save, sha

CONFIG=ROOT/'configs/backend_controls/frontier_proof_compare_r1.json'
RECORD=ROOT/'docs/frontier_proof_freeze_review_20260924_r1.json'


def review():
    cfg=load(CONFIG);common=cfg['common'];scope=common['scope'];expected={}
    for name in ('scoped_parse_proof_compare_r1.json','checked_route_frontier_synthetic_r1.json'):
        old=load(ROOT/'configs/backend_controls'/name)
        values=old['common']['sources'] if 'common' in old else old['sources']
        for path,digest in values.items():
            if path in expected and expected[path]!=digest:raise ValueError('inconsistent old binding')
            expected[path]=digest
    paths=[ROOT/'docs/frontier_proof_protocol_20260924_r1.md',*sorted((ROOT/'frontier_proof').glob('*.py'))]
    expected.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    if common['sources']!=expected:raise ValueError('tested source inventory')
    for path,digest in expected.items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen source drift')
    gate=load(ROOT/'docs/frontier_proof_controls_20260924_r1.json',cfg['controls_sha256'])
    if gate['status']!='PASS' or gate['tests_passed']!=94 or gate['source_hashes']!=expected or gate['real_requests_executed']!=0:
        raise ValueError('failed/unbound control gate')
    names=['act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json','docs/main_table_source_applicability_20260921.json']
    if set(cfg['selection_files'])!=set(names):raise ValueError('selection provenance')
    selection,ledger=[load(ROOT/n,cfg['selection_files'][n]) for n in names]
    sample=next(s for s in selection['samples'] if s['sample_rank']==2);model=selection['models']['seed0']
    if sample['dataset_index']!=4098 or sample['label']!=9 or cfg['sample']!={k:sample[k] for k in ('sample_rank','dataset_index','label','center')}:
        raise ValueError('sequential fixed engineering object')
    want={'experts':8,'classes':10,'label':9,'radius':'2/255','clip':['0','1'],
        'margin':str(Fraction.from_float(1e-7)),'model_state':model['model_state'],'center':sample['center']}
    if scope!=want:raise ValueError('domain/property/model drift')
    required=[{'pair':list(p),'label':9,'competitor':k} for p in combinations(range(8),2) for k in range(10) if k!=9]
    if cfg['output_obligations']!=required or cfg['required_per_call']!=252:raise ValueError('missing tie-legal duty')
    if (cfg['automatic_execution'] is not False or cfg['no_resume_retry_expansion'] is not True or cfg['calls_per_arm']!=1 or
        cfg['proposal_stage_fraction']!='1/2' or cfg['publication_reserve_seconds']!=2 or cfg['route_candidate_count']!=56 or
        common['limits']!={'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30} or
        common['no_historical_proofs'] is not True or common['route_changing_claim'] is not False or common['native_float_claim'] is not False):
        raise ValueError('budget/acceptance/guarantee drift')
    calls=[{'id':m,'proof_policy':{'schema':'FRONTIER_PROOF_POLICY_V1','mode':m,
        'router_proposal':'FINAL_AFFINE_DUAL_ONLY','parse_cache':False,'checker_cache':False}}
        for m in ('exhaustive','checked_frontier')]
    if cfg['calls']!=calls:raise ValueError('fixed two-arm order/identity')
    asset=next(r['request'] for r in ledger['artifact_inventory'] if r['job_id']=='rank2_seed0_adaptive')
    assets={'checkpoint':{'path':model['checkpoint'],'sha256':model['checkpoint_sha256']},
        'input':{'path':str(ROOT/asset['path']),'sha256':asset['sha256']}}
    for name,info in assets.items():
        if common[name]!=info or sha(info['path'])!=info['sha256']:raise ValueError('bound asset identity')
    if common['source_protocol']!={'path':'docs/frontier_proof_protocol_20260924_r1.md',
            'sha256':expected['docs/frontier_proof_protocol_20260924_r1.md']}:raise ValueError('protocol binding')
    if common['environment']['python']!=sys.version or common['environment']['executable_sha256']!=sha(PYTHON):
        raise ValueError('Python binding')
    output=ROOT/'data/moe/results/frontier_proof_source4098_compare_20260924_r1'
    if cfg['output']!=str(output) or output.exists():raise ValueError('fresh absent output required')
    return {'audit':'PASS','issues':0,'config_sha256':sha(CONFIG),'implementation_commit':cfg['implementation_commit'],
        'source_files_checked':len(expected),'tests_passed':94,'dataset_index':4098,'model':'seed0',
        'required_per_call':252,'requests_frozen':2,'real_requests_executed':0,'output_absent':True,
        'real_asset_reads':'raw SHA-256 only; no checkpoint/tensor decoding or inference',
        'execution_authorized_by_freeze':False,'native_float_claim':False,'route_changing_claim':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();result=review()
    if a.check:
        if result!=load(RECORD):raise ValueError('saved review differs')
        print('PASS: two real requests frozen, not executed')
    else:save(RECORD,result);print(json.dumps(result))
