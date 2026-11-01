"""Fixed final reproduction. Uses the original request supervisor and algorithms.

Two IBP batches cover 100 model-inputs; 100 complete calls cover the same inputs.
The ten-request smoke is a prefix of this queue, never an additional rerun.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from architecture_s1_execution import Artifacts,digest,read_record,require,finite_queue
import genbab_cpu_ready as backend

ORIGINAL=Path('/data1/Kane/MOE/results/architecture_s2a_evaluation_20261101_r1')
CONTINUATION=Path('/data1/Kane/MOE/results/architecture_s2a_evaluation_20261101_r2')
MODELS=('D-act_seed0','M_seed0')
CHANGED={'scripts/genbab_cpu_ready.py','scripts/architecture_s2_primary.py',
         'scripts/architecture_s1_execution.py'}


def catalog():
    out=[dict(id='primary_'+m,model=m,mode='primary',radius=2) for m in MODELS]
    for i in range(50):
        for m in (MODELS if i%2==0 else MODELS[::-1]):
            out.append(dict(id=f'complete_{m}_{i:04d}',model=m,index=i,mode='complete',radius=2))
    return out


def prepare(base):
    from architecture_s2_study import source_files
    began=time.monotonic();base=base.resolve();assets=base/'assets';assets.mkdir()
    old=load(ORIGINAL/'manifest.json'); prior=load(CONTINUATION/'manifest.json')
    for p,h in prior['source_files'].items():
        if p not in CHANGED:require(sha(ROOT/p)==h,'unchanged numerical source: '+p)
    models=[dict(m) for m in old['models'] if m['id'] in MODELS]
    require([m['id'] for m in models]==list(MODELS),'fixed selected model identities')
    for m in models:
        target=assets/(m['id']+'.pt');shutil.copy2(m['checkpoint'],target)
        require(sha(target)==m['checkpoint_sha256'],'copied frozen weights')
        m['checkpoint']=str(target)
    target=assets/'inputs.json.gz';shutil.copy2(old['inputs']['path'],target)
    require(sha(target)==old['inputs']['sha256'],'identical uint8 materialized inputs')
    # Copy historic install receipts as provenance, not as evidence of this install.
    receipts=assets/'historic_install_receipts';receipts.mkdir()
    for n in ('packages.json','converter.json'):shutil.copy2(backend.OLD/n,receipts/n)
    shutil.copytree(backend.COMPAT,base/'engine-source',symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    locations=dict(schema='S2A_RELOCATION_V1',primary_python=str(base/'env-primary/bin/python'),
        engine_environment=str(base/'env-engine'),author_source=str(base/'author-source'),
        compat_source=str(base/'engine-source'),install_receipts=str(receipts))
    save(base/'locations.json',locations)
    source=source_files();source['scripts/architecture_s2_reproduce.py']=sha(__file__)
    spec=dict(old,models=models,source_files=source,workers=4,primary_models=2,
        incomplete_calls=0,complete_calls=100,queue_budget_seconds=4*3600,
        request_catalog_sha256=digest(catalog()),
        inputs=dict(old['inputs'],path=str(target),indices=list(range(50))),
        reproduction=dict(indices=list(range(50)),models=list(MODELS),
            locations_sha256=sha(base/'locations.json'),
            original_manifest_sha256=sha(ORIGINAL/'manifest.json'),
            original_complete_manifest_sha256=sha(CONTINUATION/'manifest.json'),
            original_execution='8087806b695ed55cb5a8b4b300181648ba91ebdb',
            changed_transport_files=sorted(CHANGED),original_source_files=prior['source_files'],
            scope='same host and base interpreters; fresh venv installs; same numerical sources',
            primary_batch_note='50 versus original 256; exact differences reported, no tolerance-based reclassification',
            smoke_prefix=10,max_calls=200,planned_calls=102,model_input_mode_evaluations=200),
        preparation_seconds=time.monotonic()-began)
    # New backend identity is populated by the separately installed environment.
    save(base/'manifest.draft.json',spec)


def freeze(base):
    require(backend.RELOCATION_SHA256==sha(base/'locations.json'),'active relocation binding')
    spec=load(base/'manifest.draft.json');spec['backend_sources']=backend.source_identity()
    for p,h in spec['reproduction']['original_source_files'].items():
        if p not in CHANGED:require(sha(ROOT/p)==h,'unchanged numerical source: '+p)
    spec['source_files']={p:sha(ROOT/p) for p in spec['source_files']}
    spec['inputs']['indices']=list(range(50))
    spec['backend_sources_sha256']=digest(spec['backend_sources'])
    spec['reproduction']['environment_locks']={n:sha(base/n) for n in
        ('primary-lock.txt','engine-lock.txt','environment_inventory.json')}
    save(base/'manifest.json',spec)
    return spec


def run(base,smoke=False):
    from architecture_s1_study import resources
    spec=load(base/'manifest.json');requests=catalog()
    require(spec['request_catalog_sha256']==digest(requests),'fixed reproduction queue')
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen execution sources')
    require(sha(base/'locations.json')==spec['reproduction']['locations_sha256'],'environment binding')
    observation=resources(spec['workers'])
    require(observation['memory']['MemAvailable']>=(8*spec['workers']+4)*2**30,'memory headroom')
    require(observation['load'][0]+spec['workers']<=observation['allowed_cpus'],'CPU headroom')
    # --smoke selects the first ten calls; --run after it executes only the suffix.
    chosen=requests[:10] if smoke else requests
    runroot=base/('smoke' if smoke else 'results')
    previous_bytes=Path(spec['inputs']['path']).stat().st_size
    elapsed=0.
    if not smoke and (base/'smoke').exists():
        finished=read_record(base/'smoke/queue_complete.json.gz')
        require(finished['status']=='COMPLETE' and finished['settled']==10,'smoke transport completion')
        chosen=requests[10:]
        previous_bytes=load(base/'smoke/artifact_account.json')['used']
        elapsed=finished['queue_wall_seconds']
    part=dict(spec,request_catalog_sha256=digest(chosen),queue_budget_seconds=4*3600-elapsed)
    store=Artifacts.create(runroot,initial_bytes=previous_bytes)
    manifest=runroot/'manifest.json';store.write(manifest,part)
    store.write(runroot/'start.json.gz',dict(clock=time.time(),pid=os.getpid(),
        immutable_commit=(ROOT/'.frozen_commit').read_text().strip(),manifest_sha256=sha(manifest),
        resources=observation))
    result=finite_queue(runroot,chosen,manifest,workers=spec['workers'])
    print(json.dumps(result),flush=True)
    if result['status']!='COMPLETE':raise SystemExit(1)


def compare(base):
    import numpy as np
    from architecture_s1_audit import audit_request
    from act.pipeline.moe.architecture_s2_evaluation import unpack_arrays,check_primary_arrays
    rows=[];primary={};began=time.monotonic()
    for q in catalog():
        part=base/('smoke' if (base/'smoke'/q['id']).exists() else 'results')
        row,val=audit_request(part,q,sha(part/'manifest.json'))
        require(row['status']!='NOT_EXECUTED','all fixed reproduction requests settled')
        if q['mode']=='primary' and val is not None:
            a=unpack_arrays(read_record(part/q['id']/'arrays.json.gz'))
            old=unpack_arrays(read_record(ORIGINAL/q['id']/'arrays.json.gz'))
            model=next(m for m in load(base/'manifest.json')['models'] if m['id']==q['model'])
            check_primary_arrays(a,family=model['family'],count=50)
            changes={}
            for k,v in a.items():
                ref=old[k][:50];require(v.shape==ref.shape and v.dtype==ref.dtype,'same array contract')
                changed=np.flatnonzero(np.any(v!=ref,axis=tuple(range(1,v.ndim))) if v.ndim>1 else v!=ref)
                changes[k]=dict(changed_indices=changed.tolist(),
                    max_abs_difference=float(np.max(np.abs(v.astype(float)-ref.astype(float)))))
            primary[q['model']]=dict(changes=changes,new_certified=int(a['certified'].sum()),
                old_certified=int(old['certified'][:50].sum()),seconds=row['seconds'],
                array_sha256=sha(part/q['id']/'arrays.json.gz'))
        elif q['mode']=='complete':
            original,value=audit_request(CONTINUATION,q,sha(CONTINUATION/'manifest.json'))
            rows.append(dict(model=q['model'],index=q['index'],original=original['status'],
                reproduced=row['status'],same=original['status']==row['status'],
                original_seconds=original['seconds'],seconds=row['seconds']))
    result=dict(schema='S2A_REPRODUCTION_COMPARISON_V1',models=list(MODELS),indices=list(range(50)),
        primary=primary,complete=rows,complete_matches=sum(r['same'] for r in rows),
        primary_batches=len(primary),complete_calls=len(rows),analysis_seconds=time.monotonic()-began,
        independent_human_replication=False,host_isolation=False,
        sources={str(p.relative_to(base)):sha(p) for p in base.glob('*/queue_complete.json.gz')})
    save(base/'comparison.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','smoke','run','compare'])
    p.add_argument('base',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare(a.base)
    elif a.command=='freeze':freeze(a.base)
    elif a.command=='compare':compare(a.base)
    else:run(a.base,smoke=a.command=='smoke')
