"""Continue only unexecuted S2a requests after the dead-stat parsing fix.

Original terminals (including the cleanup error) remain authoritative. This
changes neither the mathematical engine nor the shared128MiB artifact limit.
"""
import argparse
import copy
import os
from pathlib import Path
import resource
import sys
import time

ENTRY=time.monotonic()
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from architecture_s1_execution import Artifacts,digest,read_record,require,finite_queue
from architecture_s1_continuation import tree_identity,remaining_requests,locate_requests
from architecture_s2_study import all_requests

CHANGED_SOURCES={'scoped_proof/owned_session.py','scripts/architecture_s2_audit.py'}


def validate(spec):
    c=spec['continuation'];old=load(c['manifest'],c['manifest_sha256'])
    require(old['study']==spec['study']=='S2A' and 'continuation' not in old,
            'one S2a cleanup continuation')
    excluded={'request_catalog_sha256','source_files','continuation'}
    require({k:v for k,v in spec.items() if k not in excluded}==
            {k:v for k,v in old.items() if k not in excluded},
            'same methods,models,inputs,limits and statistical contract')
    require(spec['artifact_bytes']==128*2**20,'same aggregate128MiB')
    require(set(spec['source_files'])==set(old['source_files']),'same source roster')
    changed={p for p,h in old['source_files'].items() if spec['source_files'][p]!=h}
    require(changed==CHANGED_SOURCES,'only dead-stat parser and aggregate auditor change')
    prior=Path(c['root']);require(tree_identity(prior)==c['tree'],'unchanged prior evidence')
    account=load(prior/'artifact_account.json')
    require(account['used']==c['prior_charged_bytes'] and account['cap']==spec['artifact_bytes'],
            'carry every prior charge without new allowance')
    full=all_requests(old['models']);require(digest(full)==old['request_catalog_sha256'],
                                           'original full roster')
    remaining=remaining_requests(full,prior);queue=read_record(prior/'queue_complete.json.gz')
    require(queue['status']=='PREFIX_STOP' and queue['settled']==len(full)-len(remaining)
            and queue['not_executed']==len(remaining) and queue['missing_request_cost']==0,
            'settled failed prefix')
    require(queue['source_manifest_sha256']==c['manifest_sha256']
            and queue['request_catalog_sha256']==digest(full),'prior queue binding')
    require(digest(remaining)==spec['request_catalog_sha256'] and len(remaining)==c['remaining'],
            'only never-executed remainder in original order')
    require(queue['summed_request_seconds']==c['prior_request_seconds'],'prior cost retained')
    return old,full,remaining


def freeze(path,prior,original_execution,control):
    prior=Path(prior);manifest=prior/'manifest.json';old=load(manifest)
    require(load(control)['status']=='PASS','cleanup/continuation controls passed')
    require(all(sha(Path(original_execution)/p)==h for p,h in old['source_files'].items()),
            'original frozen executable retained')
    new=copy.deepcopy(old)
    for p in CHANGED_SOURCES:new['source_files'][p]=sha(ROOT/p)
    remaining=remaining_requests(all_requests(old['models']),prior)
    new['request_catalog_sha256']=digest(remaining)
    new['continuation']=dict(root=str(prior.resolve()),manifest=str(manifest.resolve()),
        manifest_sha256=sha(manifest),tree=tree_identity(prior),
        prior_charged_bytes=load(prior/'artifact_account.json')['used'],
        prior_request_seconds=read_record(prior/'queue_complete.json.gz')['summed_request_seconds'],
        remaining=len(remaining),original_execution=str(Path(original_execution).resolve()),
        control=str(Path(control).resolve()),control_sha256=sha(control),
        controller_sources={p:sha(ROOT/p) for p in ('scripts/architecture_s2_continuation.py',
            'scripts/architecture_s1_continuation.py')},
        authorization='S2a: tool errors repaired; continue unexecuted requests; no retry or quota change')
    validate(new);save(path,new)
    print('FROZEN',len(remaining),sha(path),flush=True)


def run(root,manifest):
    from architecture_s1_study import resources
    import genbab_cpu_ready as backend
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),
            'immutable committed execution copy')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    spec=load(manifest);old,full,remaining=validate(spec);c=spec['continuation']
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'continued source binding')
    require(all(sha(ROOT/p)==h for p,h in c['controller_sources'].items()),'continuation source binding')
    require(all(sha(Path(c['original_execution'])/p)==h for p,h in old['source_files'].items()),
            'old executable unchanged')
    require(sha(c['control'])==c['control_sha256'],'control receipt')
    require(digest(backend.source_identity())==spec['backend_sources_sha256'],'unchanged numerical engine')
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'same physical inputs')
    for m in spec['models']:require(sha(m['checkpoint'])==m['checkpoint_sha256'],'same trained weights')
    observed=resources(spec['workers'])
    require(observed['ordinary_available_disk_bytes']>=2*2**30,'disk headroom')
    require(observed['load'][0]+spec['workers']<=observed['allowed_cpus'],'CPU headroom')
    require(observed['memory']['MemAvailable']>=(8*spec['workers']+4)*2**30,'memory headroom')
    store=Artifacts.create(root,initial_bytes=c['prior_charged_bytes'],cap=spec['artifact_bytes'])
    bound=store.root/'manifest.json';store.write(bound,spec)
    store.write(store.root/'start.json.gz',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),
        manifest_sha256=sha(bound),pid=os.getpid(),clock=time.time(),resources=observed,
        prior_charged_bytes=c['prior_charged_bytes'],startup_and_admission_seconds=time.monotonic()-ENTRY))
    summary=finite_queue(store.root,remaining,bound,workers=spec['workers'],seconds=300,
                         timing_reserve=tuple(spec['timing_reserve']))
    print(summary,flush=True)
    from architecture_s2_audit import analyze
    report=analyze(store.root)
    store.write(store.root/'evaluation_summary.json.gz',report,terminal=True)
    print(report['status'],report['registered_decision'],report['online_states'],flush=True)
    return 0 if summary['status']=='COMPLETE' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path)
    p.add_argument('--prior',type=Path);p.add_argument('--original-execution',type=Path)
    p.add_argument('--control',type=Path);a=p.parse_args()
    if a.command=='freeze':freeze(a.manifest,a.prior,a.original_execution,a.control)
    else:raise SystemExit(run(a.root,a.manifest))
