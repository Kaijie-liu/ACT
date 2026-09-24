"""Controls, freeze and opt-in batch execution are separate operations."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.evidence import roster
from scoped_parse_proof.contract import SCHEMA, LIMITS, MODES
from scoped_parse_proof.supervisor import supervise

DOC=ROOT/'docs/scoped_parse_proof_protocol_20260924_r1.md'
GATE=ROOT/'docs/scoped_parse_proof_controls_20260924_r1.json'
CONFIG=ROOT/'configs/backend_controls/scoped_parse_proof_compare_r1.json'
REVIEW=ROOT/'docs/scoped_parse_proof_freeze_review_20260924_r1.json'
DEST=ROOT/'data/moe/results/scoped_parse_proof_source4096_compare_20260924_r1'
SELECTION=ROOT/'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
LEDGER=ROOT/'docs/main_table_source_applicability_20260921.json'
RESOURCES={'whole_pipeline_seconds':300,'cpu_threads':2,'sampled_group_rss_bytes':8*2**30}


def sources():
    previous=load(ROOT/'configs/backend_controls/source_construction_parse_r1.json')['sources']
    for name,digest in previous.items():
        if sha(ROOT/name)!=digest:raise ValueError('sealed implementation changed: '+name)
    paths=[DOC,*sorted((ROOT/'scoped_parse_proof').glob('*.py'))]
    return {**previous,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}


def make_config():
    selection=load(SELECTION);ledger=load(LEDGER)
    model=selection['models']['seed0']
    sample=next(s for s in selection['samples'] if s['sample_rank']==1)
    if (sample['dataset_index'],sample['label'])!=(4096,8):raise ValueError('fixed next-rank identity')
    old=next(r['request'] for r in ledger['artifact_inventory'] if r['job_id']=='rank1_seed0_adaptive')
    prior=load(ROOT/'configs/backend_controls/scoped_proof_execution_r1.json')
    scope={'experts':8,'classes':10,'label':8,'center':sample['center'],'model_state':model['model_state'],
        'radius':'2/255','clip':['0','1'],'margin':prior['scope']['margin']}
    common={'scope':scope,'sources':sources(), 'limits':RESOURCES,
        'source_protocol':{'path':str(DOC.relative_to(ROOT)),'sha256':sha(DOC)},
        'checkpoint':{'path':model['checkpoint'],'sha256':model['checkpoint_sha256']},
        'input':{'path':str(ROOT/old['path']),'sha256':old['sha256']},
        'environment':{'python':sys.version,'executable_sha256':sha(PYTHON),
            'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy')}},
        'no_historical_proofs':True,'route_changing_claim':False,'native_float_claim':False}
    for key in ('checkpoint','input'):
        if sha(common[key]['path'])!=common[key]['sha256']:raise ValueError('bound asset drift')
    return {'schema':'SCOPED_PARSE_PROOF_COMPARISON_R1','output':str(DEST),'common':common,
        'selection_files':{str(p.relative_to(ROOT)):sha(p) for p in (SELECTION,LEDGER)},
        'selection':'seed0, next manifest rank1 after sealed rank0; no outcome/route predicate; observed development only',
        'sample':{k:sample[k] for k in ('sample_rank','dataset_index','label','center')},
        'calls':[{'id':m,'construction_policy':{'schema':SCHEMA,'mode':m,'cache_limits':dict(LIMITS),'checker_cache':False}} for m in MODES],
        'output_obligations':roster(scope),'required_per_call':252,'calls_per_arm':1,
        'proposal_stage_fraction':'1/2','proposal_rule':'one lexicographic LP per obligation, equal actual remaining time share',
        'publication_reserve_seconds':2,'automatic_execution':False,'no_resume_retry_expansion':True,
        'read_historical_request':'center only; not endpoints, source, matrices, bounds, exclusions or witness',
        'acceptance':'all252 freshly checked strictly positive exact residual-corrected bounds; unchanged original aggregate',
        'comparison':'two adapter modes, not original production versus cached; one pair cannot establish stable speedup'}


def clean():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='feat/moe-route-verification':
        raise ValueError('registered branch required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('clean checkout required')


def freeze():
    clean()
    if CONFIG.exists() or DEST.exists():raise FileExistsError('new config and absent output required')
    cfg=make_config();gate=load(GATE)
    if gate['status']!='PASS' or gate['source_hashes']!=cfg['common']['sources'] or gate['tests_passed']!=85:
        raise ValueError('control gate not closed')
    cfg.update(controls_sha256=sha(GATE),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(CONFIG,cfg);print('FROZEN ONLY: CIFAR4096, 2 calls, 300s each, all252 per arm. ZERO real execution.')


def admitted():
    mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    return mem['MemAvailable']>=10*2**30 and os.getloadavg()[0]/os.cpu_count()<=.5


def execute():
    clean();cfg=load(CONFIG);review=load(REVIEW)
    if review['audit']!='PASS' or review['config_sha256']!=sha(CONFIG):raise ValueError('freeze review required')
    if sha(GATE)!=cfg['controls_sha256'] or load(GATE)['status']!='PASS' or sources()!=cfg['common']['sources']:
        raise ValueError('tested implementation drift')
    if cfg['output']!=str(DEST) or DEST.exists():raise ValueError('exclusive fixed output required')
    if not admitted():raise ValueError('resource gate closed; no batch launched')
    DEST.mkdir()
    save(DEST/'launch.json',{'config_sha256':sha(CONFIG),'config':cfg,
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    rows=[]
    for call in cfg['calls']:
        spec={**copy.deepcopy(cfg['common']),'construction_policy':call['construction_policy']}
        if not admitted():
            row={'id':call['id'],'status':'NOT_STARTED_RESOURCE','seconds':None,'launched':False}
        else:
            try:
                result=supervise(DEST/call['id'],spec,budget=300,rss_limit=8*2**30)
                row={'id':call['id'],**result,'launched':True}
            except Exception as exc:
                row={'id':call['id'],'status':'SUPERVISOR_ERROR','seconds':None,'launched':True,'error':repr(exc)}
        save(DEST/(call['id']+'_batch_terminal.json'),row);rows.append(row);print(json.dumps(row),flush=True)
    save(DEST/'execution.json',{'config_sha256':sha(CONFIG),'rows':rows,'no_retries':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','execute'))
    p.add_argument('--acknowledge-new-frozen-comparison',action='store_true');a=p.parse_args()
    if a.action=='freeze':freeze()
    elif not a.acknowledge_new_frozen_comparison:p.error('later explicit execution authorization required')
    else:execute()
