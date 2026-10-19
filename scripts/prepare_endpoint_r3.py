"""Read-only R3 freeze builder. Emits edits; never creates sources or runs controls."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scripts.check_endpoint_freeze_candidate import (
    constant, closure, PREFIXES, file_hash, read, compact, ident, need, sha)

FREEZE='docs/hz_endpoint_midpoint_freeze_r3_20261020.json'
REVIEW='docs/hz_endpoint_midpoint_freeze_r3_review_20261020.json'
BATCH='docs/hz_endpoint_execution_approvals.json'


def build():
    deadline=time.monotonic()+60
    old_raw=read(ROOT/'docs/hz_endpoint_midpoint_freeze_v4_20261019.json')
    need(sha(old_raw)=='f42a8d2a563764de4a2790c762b222ac4cca9600eb47ef00dffc4ff5233106c8','retained V4')
    old=json.loads(old_raw)
    prior=json.loads(read(ROOT/'docs/hz_endpoint_midpoint_freeze_v4_review_20261019.json'))
    delta=json.loads(read(ROOT/'docs/hz_endpoint_binding_delta_20261020.json'))
    for case,record in prior['sources'].items():
        root=Path(record['directory']); raw=read(root/'manifest.json')
        need(sha(raw)==record['sha256']==delta['sources'][case],'same source')
        manifest=json.loads(raw); names={'manifest.json'}; total=len(raw)
        for tensor in manifest['tensors']:
            for chunk in tensor['chunks']:
                n=chunk['file']; need(n not in names,'source unique member')
                file_hash(root/n,chunk['bytes'],chunk['sha256'],deadline)
                names.add(n); total+=chunk['bytes']
        need(total==record['bytes'] and len(names)==record['members']
             and {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==names,
             'unchanged complete source inventory')
    env=old['records'][0]['profile']['environment']
    for r in env['files']: file_hash(r['file'],r['bytes'],r['sha256'],deadline)
    native=closure(old['records'][0]['profile']['implementation'],PREFIXES,deadline)
    need(native==old['records'][0]['profile']['implementation'],'native60 unchanged')
    code=closure(list(native)+constant('scoped_source/endpoint_batch.py','SEEDS'),('scripts',)+PREFIXES,deadline)
    host=[]
    for row in delta['host_environment_proposed']:
        path=Path(row['path']); target=str(path.resolve(strict=True))
        need(target==(str(path)+'3.12' if path.name=='python' else str(path)),
             'fixed existing environment targets')
        file_hash(path,row['bytes'],row['sha256'],deadline)
        host.append(dict(file=str(path),resolved=target,bytes=row['bytes'],sha256=row['sha256']))
    need([r['file'] for r in host]==constant('scoped_source/endpoint_batch.py','HOST_FILES'),'exact host roster')
    host_environment=dict(schema='ENDPOINT_SELECTED_HOST_FILES_V1',
        scope='SELECTED_FILES_NOT_LINKMAP_OR_OS_PROOF',files=host)
    launcher='scripts/endpoint_midpoint_launcher.py'
    issuer=ident(dict(schema='ENDPOINT_RETAINED_HOST_WIRING_V1',path=launcher,sha256=code[launcher],
        clock='CLOCK_MONOTONIC',origin='AFTER_HOST_PROVISIONING_BEFORE_REQUEST',as_bytes=8*2**30,
        file_bytes=128*2**20,mode='IN_PROCESS_FROM_EXTERNALLY_RETAINED_ORIGIN',attempts=1))
    freeze=dict(old,schema='ENDPOINT_FINITE_FREEZE_V5',code=code,
        startup=dict(old['startup'],issuer=issuer),host_environment=host_environment)
    need(freeze['limits']==constant('scoped_source/endpoint_batch.py','LIMITS',{'REPORT_CAP':1048576})
         and freeze['policy']==constant('scoped_source/hz_request_store.py','POLICY'),'all caps and policy unchanged')
    native_raw=read(ROOT/'docs/hz_native_execution_approvals.json')
    resource_raw=read(ROOT/'docs/hz_native_resource_execution_approvals.json')
    need(sha(native_raw)=='da5260bcf34f4fb4fd1c1bb6592402d67e2f8658b59aa000a9006aa352a9858c'
         and ident(json.loads(native_raw))==old['native_registry'],'native32 byte identical')
    need(sha(resource_raw)=='4ed38318d04d0b48b34fffe6359b1dc53bb01984a44b712c7138745bd5fb7af4',
         'resource authority unchanged')
    need(len(compact(freeze))<=131072,'old freeze cap')
    authority=dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[ident(freeze)])
    need(json.loads(read(ROOT/BATCH)) in (authority,dict(schema=authority['schema'],batches=[])),
         'no unrelated batch approval')
    need(constant(launcher,'FREEZE_FILE')==str(ROOT/FREEZE)
         and constant(launcher,'RUN_DIRECTORY')==constant('scripts/endpoint_midpoint_caller.py','RUN_DIRECTORY'),
         'one new launch identity')
    review=dict(schema='ENDPOINT_R3_STATIC_REVIEW_V1',freeze=ident(freeze),freeze_bytes=len(compact(freeze)),
        base_head='7cd7476fefb4c3cc4f260f9e7c20943dba02dfa6',sources=prior['sources'],
        environment=env,host_environment=host_environment,
        code_counts=dict(native=len(native),component=len(freeze['component']['code']),batch=len(code)),
        unchanged=['records','component','limits','policy','admission','native_registry'],
        native_registry_sha256=sha(native_raw),resource_registry_sha256=sha(resource_raw),
        old_freeze_sha256=sha(old_raw),directory=constant(launcher,'RUN_DIRECTORY'),
        scope='ONE_APPROVED_SEVEN_POSITION_CPU_BATCH_STOP_NO_RETRY',source_factory_calls=0,
        controls_executed=0,models=0,gpu=0)
    return {FREEZE:freeze,REVIEW:review,BATCH:authority}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--emit',action='store_true'); args=p.parse_args()
    result=build()
    if args.emit: print(compact({n:compact(v).decode() for n,v in result.items()}).decode())
    else:
        for name,value in result.items(): need(read(ROOT/name)==compact(value),'canonical frozen identity: '+name)
        print(json.dumps(dict(status='PASS_STATIC_R3',freeze=ident(result[FREEZE]),
            code_counts=result[REVIEW]['code_counts'],positions=7,components=22,controls_executed=0)))
