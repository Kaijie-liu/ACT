"""Read-only, stdlib-only V4 finite freeze builder/checker. --emit never runs controls.

Preserves the old NOT_READY V2 candidate; no factory, runtime or registry writes.
Applying the emitted canonical documents is an explicitly authorized repo edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scripts.check_endpoint_freeze_candidate import (constant, closure, PREFIXES,
    file_hash, read, compact, ident, need, sha, syntax)

FREEZE='docs/hz_endpoint_midpoint_freeze_v4_20261019.json'
REVIEW='docs/hz_endpoint_midpoint_freeze_v4_review_20261019.json'
NATIVE='docs/hz_native_execution_approvals.json'
BATCH='docs/hz_endpoint_execution_approvals.json'
OLD='docs/hz_endpoint_midpoint_freeze_candidate_20261019.json'
OLD_REVIEW='docs/hz_endpoint_midpoint_freeze_review_20261019.json'


def build():
    start=time.monotonic(); deadline=start+60
    raw=read(ROOT/OLD)
    need(sha(raw)=='bb7bdadda356202f8dec7132782ac6e4e73dd445571094eb6179eac2e3faf107','old candidate unchanged')
    old=json.loads(raw); review_raw=read(ROOT/OLD_REVIEW)
    need(sha(review_raw)=='b67e8ba8df9f69c9dbd0c50c9a664f4eda2abc007f727f3b93492277d27cccc7','old review unchanged')
    prior=json.loads(review_raw)
    for case,record in prior['sources'].items():
        root=Path(record['directory']); raw=read(root/'manifest.json')
        need(sha(raw)==record['sha256'],'same saved source')
        manifest=json.loads(raw); names={'manifest.json'}; total=len(raw)
        for tensor in manifest['tensors']:
            for chunk in tensor['chunks']:
                n=chunk['file']; need(n not in names,'source unique member')
                file_hash(root/n,chunk['bytes'],chunk['sha256'],deadline)
                names.add(n); total+=chunk['bytes']
        need(total==record['bytes'] and len(names)==record['members']
            and {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==names,
            'same complete source inventory')
    env=old['records'][0]['profile']['environment']
    for r in env['files']: file_hash(r['file'],r['bytes'],r['sha256'],deadline)
    native_code=closure(old['records'][0]['profile']['implementation'],PREFIXES,deadline)
    batch_code=closure(list(native_code)+constant('scoped_source/endpoint_batch.py','SEEDS'),('scripts',)+PREFIXES,deadline)
    records=[]
    for record in old['records']:
        p=dict(record['profile'],implementation=native_code)
        records.append(dict(id=record['id'],profile=p,sha256=ident(p)))
    old_native=prior['authority_projection']['native']
    old_added=set(prior['added_profiles'])
    preserved=set(old_native['profiles'])-old_added
    need(len(preserved)==26,'preserve historical native profiles')
    new_added={r['sha256'] for r in records}; need(len(new_added)==6,'six fixed new profiles')
    native=dict(schema='NATIVE_EXECUTION_APPROVAL_REGISTRY_V1',profiles=sorted(preserved|new_added))
    active=json.loads(read(ROOT/NATIVE)); need(set(active['profiles']) in (preserved,preserved|new_added),
        'only old authority or this exact approved addition')
    component=dict(old['component'],profile=records[0]['sha256'],code=dict(native_code))
    for n in ('scripts/endpoint_midpoint_components.py','scoped_source/scaled_endpoint_controls.py'):
        component['code'][n]=batch_code[n]
    issuer=ident(dict(schema='ENDPOINT_RETAINED_HOST_WIRING_V1',
        path='scripts/endpoint_midpoint_launcher.py',sha256=batch_code['scripts/endpoint_midpoint_launcher.py'],
        clock='CLOCK_MONOTONIC',origin='AFTER_HOST_PROVISIONING_BEFORE_REQUEST',
        as_bytes=8*2**30,file_bytes=128*2**20,mode='IN_PROCESS_FROM_EXTERNALLY_RETAINED_ORIGIN',attempts=1))
    startup=dict(schema='ENDPOINT_PROVISIONED_HOST_CONTRACT_V1',clock='CLOCK_MONOTONIC',
        origin='AFTER_HOST_PROVISIONING_BEFORE_REQUEST',issuer=issuer,caller_limits_before_request=True,
        measurement_host='scripts/endpoint_measurement_host.py',
        provisioning='SEPARATE_OBSERVED_PREFIX_COLD_START_UNKNOWN',request_work_after_origin=True)
    freeze=dict(schema='ENDPOINT_FINITE_FREEZE_V4',limits=constant('scoped_source/endpoint_batch.py','LIMITS',{'REPORT_CAP':1048576}),
        admission=old['admission'],policy=old['policy'],records=records,component=component,
        code=batch_code,native_registry=ident(native),startup=startup)
    need(freeze['policy']==constant('scoped_source/hz_request_store.py','POLICY'),'same policy')
    need({k:v for k,v in freeze['limits'].items() if k!='observation'}==
         {k:v for k,v in old['limits'].items() if k!='observation'},'numeric caps unchanged')
    need(len(compact(freeze))<=131072,'original freeze capacity')
    authority=dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[ident(freeze)])
    active_batch=json.loads(read(ROOT/BATCH))
    need(active_batch in (dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[]),authority),'only exact new batch')
    resource_sha=sha(read(ROOT/'docs/hz_native_resource_execution_approvals.json'))
    need(resource_sha=='4ed38318d04d0b48b34fffe6359b1dc53bb01984a44b712c7138745bd5fb7af4','resource registry unchanged')
    review=dict(schema='ENDPOINT_V4_STATIC_FREEZE_REVIEW_V1',freeze=ident(freeze),
        freeze_bytes=len(compact(freeze)),base_head='e81339949242fb4442f169c6ac859eeea772fd9e',
        scope='USER_APPROVED_ONE_FINITE_CPU_BATCH_NOT_REAL_MODEL_OR_GPU',
        sources=prior['sources'],source_factory_calls=0,environment=env,
        code_counts=dict(native=len(native_code),component=len(component['code']),batch=len(batch_code)),
        preserved_native_profiles=26,added_native_profiles=sorted(new_added),
        resource_registry_sha256=resource_sha,host_code_sha256=batch_code[startup['measurement_host']],
        directory=constant('scripts/endpoint_midpoint_caller.py','RUN_DIRECTORY'),
        timing='PROVISIONED_STANDARD_LIBRARY_HOST_REQUEST_WORK_FULLY_CHARGED',
        old_candidate_sha256=sha(read(ROOT/OLD)),old_review_sha256=sha(review_raw),
        controls_executed=0,models=0,gpu=0)
    return {FREEZE:freeze,REVIEW:review,NATIVE:native,BATCH:authority}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--emit',action='store_true'); args=parser.parse_args()
    result=build()
    if args.emit:
        print(compact({n:compact(v).decode() for n,v in result.items()}).decode())
    else:
        for n,v in result.items(): need(read(ROOT/n)==compact(v),'canonical frozen identity: '+n)
        print(json.dumps(dict(status='PASS_STATIC_APPROVED_V4',freeze=ident(result[FREEZE]),
            code_counts=result[REVIEW]['code_counts'],positions=7,components=22,controls_executed=0)))
