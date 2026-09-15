"""General request bundle, retaining the original checker isolation contract."""
import json
from pathlib import Path
import time
import zipfile
from portable_proof.runtime import compact,digest,original_bytes
from scripts.build_portable_conv_proof import extract

ROOT=Path(__file__).resolve().parents[1]


def pack(source,destination,result,tick=lambda:None):
    source=Path(source);d=Path(destination);started=time.monotonic();tick();d.mkdir(exist_ok=False)
    def put(name,data):
        tick();p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data if isinstance(data,bytes) else data.encode())
    for name in ('act','act/back_end','act/back_end/solver','moe_evidence'):put('code/'+name+'/__init__.py','')
    sources={}
    for name in ('lp_certificate','sparse_lp_certificate','check_hz_lp_export','check_rational_mccormick'):
        rel='act/back_end/solver/'+name+'.py';sources[rel]=digest((ROOT/rel).read_bytes());put('code/'+rel,extract(ROOT/rel))
    for src,target in [('moe_evidence/schema.py','code/moe_evidence/schema.py'),
                       ('moe_evidence/checker.py','code/moe_evidence/checker.py'),
                       ('moe_evidence/bundle_runtime.py','code/runtime.py'),
                       ('portable_proof/runtime.py','code/transport.py'),
                       ('portable_proof/launcher.py','verify.py'),('LICENSE','LICENSE')]:
        if not (ROOT/src).exists():continue
        raw=(ROOT/src).read_bytes();sources[src]=digest(raw);put(target,raw)
    manifest=json.loads((source/'manifest.json').read_bytes())
    refs={'manifest.json':digest((source/'manifest.json').read_bytes())}
    def discover(v):
        if isinstance(v,dict):
            if 'file' in v and 'sha256' in v:
                if Path(v['file']).name!=v['file']:raise ValueError('nonlocal reference')
                if refs.setdefault(v['file'],v['sha256'])!=v['sha256']:raise ValueError('conflicting ref')
            for value in v.values():discover(value)
        elif isinstance(v,list):
            for value in v:discover(value)
    discover(manifest);logical={};seen=set();original=0;unique_bytes=0
    with zipfile.ZipFile(d/'evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def intern(obj):
            nonlocal unique_bytes
            raw=compact(obj);sha=digest(raw)
            if sha not in seen:z.writestr(sha,raw);seen.add(sha);unique_bytes+=len(raw)
            return sha
        def encode(obj):
            if isinstance(obj,list):return {'$array':intern(obj)} if len(obj)>=32 else [encode(v) for v in obj]
            if isinstance(obj,dict):
                if '$array' in obj:raise ValueError('reserved transport key')
                return {k:encode(v) for k,v in obj.items()}
            return obj
        for name,sha in refs.items():
            tick();raw=(source/name).read_bytes();obj=json.loads(raw)
            if digest(raw)!=sha or original_bytes(obj)!=raw:raise ValueError('logical serialization/hash mismatch')
            original+=len(raw);logical[name]={'root':intern(encode(obj)),'original_sha256':sha}
    statement={k:manifest[k] for k in ('request','routes','common_facts','contexts')}
    meta={'schema':'PORTABLE_WEIGHTED_TOP2_V1','statement':statement,'checker_sources':sources,
          'files':{str(p.relative_to(d)):digest(p.read_bytes()) for p in d.rglob('*') if p.is_file()},
          'logical_files':logical,'manifest':{'file':'manifest.json','sha256':refs['manifest.json']},'expected_result':result}
    put('bundle.json',original_bytes(meta));tick()
    return {'bundle_sha256':digest((d/'bundle.json').read_bytes()),'statement_sha256':digest(compact(statement)),
            'pack_seconds':time.monotonic()-started,'original_bytes':original,'deduplicated_bytes':unique_bytes,
            'bundle_bytes':sum(p.stat().st_size for p in d.rglob('*') if p.is_file())}
