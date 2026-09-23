"""Independent stdlib freeze review: no model/array load or optimizer."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from scoped_proof.io import ROOT, load, sha, save

CONFIG = ROOT/'configs/backend_controls/scoped_proof_execution_r1.json'
OUTPUT = ROOT/'docs/scoped_proof_freeze_review_20260924_r1.json'


def review():
    cfg = load(CONFIG); parent = ROOT/cfg['source_protocol']['path']
    p = load(parent,cfg['source_protocol']['sha256']); gate_path = ROOT/'docs/scoped_proof_supervision_controls_20260924_r1.json'
    gate = load(gate_path,cfg['controls_sha256'])
    if gate['status'] != 'PASS' or gate['tests_passed'] != 53 or gate['real_requests_executed'] != 0:
        raise ValueError('unpassed control gate')
    expected_sources = {'docs/scoped_proof_supervision_protocol_20260924_r1.md'}
    for folder in ('scoped_proof','scoped_source','full_source','source_enclosure','upstream_source','router_source','act'):
        expected_sources.update(str(path.relative_to(ROOT)) for path in (ROOT/folder).rglob('*.py'))
    if set(cfg['sources']) != expected_sources or cfg['sources'] != gate['source_hashes']:
        raise ValueError('tested implementation inventory changed')
    for name, value in cfg['sources'].items():
        if sha(ROOT/name) != value: raise ValueError('tested source drift: '+name)
    for name,value in p['files'].items():
        if sha(ROOT/name) != value: raise ValueError('original protocol changed')
    expected_scope = {'experts':8,'classes':10,'label':7,'center':p['sample']['center'],
        'model_state':p['model']['model_state'],'radius':'2/255','clip':['0','1'],'margin':p['margin']}
    if cfg['scope'] != expected_scope or p['sample']['dataset_index'] != 4088 or p['sample']['sample_rank'] != 0:
        raise ValueError('same-object scope changed')
    obligations = [{'pair':list(pair),'label':7,'competitor':k} for pair in itertools.combinations(range(8),2) for k in range(10) if k != 7]
    if p['output_obligations'] != obligations or len(obligations) != 252 or not p['no_pair_exclusion']:
        raise ValueError('complete tie-legal output coverage')
    if (cfg['limits'] != {'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30} or
        cfg['proposal_stage_fraction_of_remaining_work_time'] != '1/2' or cfg['publication_reserve_seconds'] != 2 or
        not cfg['no_historical_proofs'] or cfg['automatic_execution'] or not cfg['no_resume_or_expansion'] or
        cfg['route_changing_claim'] or cfg['native_float_claim']): raise ValueError('execution/guarantee boundary')
    targets = {'checkpoint':(p['model']['checkpoint'],p['model']['checkpoint_sha256']),
        'input':(p['stored_request']['path'],p['stored_request']['sha256'])}
    for key,(path,digest) in targets.items():
        if cfg[key] != {'path':path,'sha256':digest} or sha(path) != digest: raise ValueError('bound asset drift')
    dest = ROOT/'data/moe/results/scoped_proof_source4088_20260924_r1'
    if cfg['output'] != str(dest) or dest.exists(): raise ValueError('real request already launched/output not fresh')
    return {'audit':'PASS','issues':0,'config_sha256':sha(CONFIG),'controls_sha256':sha(gate_path),
        'implementation_commit':cfg['implementation_commit'],'source_files_checked':len(cfg['sources']),
        'request_count':1,'required_output_obligations':252,'real_requests_executed':0,
        'real_assets_read':'raw SHA-256 only; no torch/pickle/data decoding',
        'output_absent':True,'execution_authorized_by_freeze':False,
        'scope':'protocol/source/asset identity and controls gate, not mathematical proof or real execution result'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); a = parser.parse_args()
    result = review()
    if a.check:
        if result != load(OUTPUT): raise ValueError('freeze review drift')
        print('PASS: freeze intact; real request not run')
    else:
        save(OUTPUT,result); print(json.dumps(result))
