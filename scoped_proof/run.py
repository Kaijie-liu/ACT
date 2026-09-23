"""Separate controls/freeze/explicit execution; freezing NEVER launches a request."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.evidence import roster
from scoped_proof.supervisor import supervise

PROTOCOL = ROOT/'configs/backend_controls/source_output_closure_scope_r1.json'
GATE = ROOT/'docs/scoped_proof_supervision_controls_20260924_r1.json'
FREEZE = ROOT/'configs/backend_controls/scoped_proof_execution_r1.json'
DEST = ROOT/'data/moe/results/scoped_proof_source4088_20260924_r1'
DOC = ROOT/'docs/scoped_proof_supervision_protocol_20260924_r1.md'


def sources():
    paths = [DOC]
    for folder in ('scoped_proof','scoped_source','full_source','source_enclosure','upstream_source','router_source','act'):
        paths.extend(sorted((ROOT/folder).rglob('*.py')))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def make_spec():
    p = load(PROTOCOL)
    for name,digest in p['files'].items():
        if sha(ROOT/name) != digest: raise ValueError('old scope protocol changed')
    if (p['sample']['dataset_index'] != 4088 or p['sample']['sample_rank'] != 0 or p['sample']['label'] != 7 or
        p['required_output_obligations'] != 252 or not p['no_pair_exclusion'] or p['historical_bounds_or_facts_allowed'] or
        p['limits'] != {'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30}):
        raise ValueError('registered one-object scope drift')
    scope = {'experts':8,'classes':10,'label':p['sample']['label'], 'center':p['sample']['center'],
        'model_state':p['model']['model_state'],'radius':p['requested_domain']['radius'],
        'margin':p['margin'],'clip':p['requested_domain']['clip']}
    if roster(scope) != p['output_obligations']: raise ValueError('all252 obligation identity')
    spec = {'schema':'SCOPED_PROOF_EXECUTION_SPEC_V1','scope':scope,'sources':sources(),
        'source_protocol':{'path':str(PROTOCOL.relative_to(ROOT)),'sha256':sha(PROTOCOL)},
        'checkpoint':{'path':p['model']['checkpoint'],'sha256':p['model']['checkpoint_sha256']},
        'input':{'path':p['stored_request']['path'],'sha256':p['stored_request']['sha256']},
        'environment':{'python':sys.version,'executable_sha256':sha(PYTHON),
            'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy')}},
        'limits':p['limits'], 'output':str(DEST), 'selection':'bound first model/first rank, not outcome selected',
        'proposal_policy':'one HiGHS LP attempt per lexicographic pair/property, equal remaining proposal-time share; no retry',
        'proposal_stage_fraction_of_remaining_work_time':'1/2', 'publication_reserve_seconds':2,
        'acceptance':'all252 exact residual-corrected LP lower bounds strictly positive on new checked source',
        'no_historical_proofs':True,'route_changing_claim':False,'native_float_claim':False,
        'automatic_execution':False,'no_resume_or_expansion':True}
    for key in ('checkpoint','input'):
        if sha(spec[key]['path']) != spec[key]['sha256']: raise ValueError('bound asset changed')
    return spec


def clean():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('registered branch only')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT): raise ValueError('clean worktree required')


def freeze():
    clean()
    if FREEZE.exists() or DEST.exists(): raise FileExistsError('freeze or output already exists')
    gate = load(GATE); spec = make_spec()
    if gate['status'] != 'PASS' or gate['source_hashes'] != spec['sources'] or gate['real_requests_executed'] != 0:
        raise ValueError('control gate closed/changed')
    spec.update(controls_sha256=sha(GATE), implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(FREEZE,spec)
    print('FROZEN ONLY: 1 request / 252 obligations; no execution or output directory created')


def execute():
    clean(); frozen = load(FREEZE)
    review = load(ROOT/'docs/scoped_proof_freeze_review_20260924_r1.json')
    if review['audit'] != 'PASS' or review['config_sha256'] != sha(FREEZE): raise ValueError('freeze review identity')
    if sha(GATE) != frozen['controls_sha256'] or load(GATE)['status'] != 'PASS': raise ValueError('control gate drift')
    if frozen['output'] != str(DEST) or frozen['limits'] != {'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30}:
        raise ValueError('output/resource policy drift')
    if DEST.exists(): raise FileExistsError(DEST)
    # This resource wait is admission, not a free model/source/LP preprocessing phase.
    mem = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 10*2**30 or os.getloadavg()[0]/os.cpu_count() > .5:
        raise ValueError('resource admission closed; no request launched')
    result = supervise(DEST,frozen,budget=300,rss_limit=8*2**30)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('freeze','execute'))
    parser.add_argument('--acknowledge-frozen-execution',action='store_true'); a = parser.parse_args()
    if a.action == 'freeze': freeze()
    elif not a.acknowledge_frozen_execution: parser.error('explicit frozen execution acknowledgement required')
    else: execute()
