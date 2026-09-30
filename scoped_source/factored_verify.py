"""Stdlib bootstrap for relocated factored H2. No model, repository or solver.

This is an audited read policy, not a sandbox against malicious Python code.
Envelope hashes are raw-byte SHA; inner identities are canonical JSON SHA.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)
CODE=('scoped_source/factored_check.py','scoped_source/factored_source.py',
      'scoped_source/factored_ir.py','scoped_source/factored_io.py',
      'scoped_source/graph.py','scoped_source/sparse_ir.py','scoped_source/sparse_check.py',
      'scoped_source/endpoint_check.py','scoped_source/endpoint_source_check.py',
      'source_enclosure/format.py','upstream_source/checker.py','router_source/checker.py')
MEMBER_LIMIT=16*2**20
ENVELOPE_LIMIT=4*2**20
TOTAL_LIMIT=2*2**30


def sha_value(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('mandatory SHA identity')
    return value


def decode(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('duplicate JSON field')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def member(root,name,limit):
    p=Path(name)
    if p.is_absolute() or not p.parts or str(p)!=name or '..' in p.parts: raise ValueError('member path')
    current=Path(root)
    if current.is_symlink(): raise ValueError('symlink root')
    for part in p.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('symlink member')
    if not current.is_file() or current.stat().st_size>limit: raise ValueError('missing/large member')
    return current


def envelope(root,expected_manifest,source,proof,mode,deadline):
    def tick():
        if time.monotonic()>=deadline: raise TimeoutError('shared envelope deadline')
    tick(); path=member(root,'manifest.json',ENVELOPE_LIMIT)
    with path.open('rb') as f: raw=f.read(ENVELOPE_LIMIT+1)
    if len(raw)>ENVELOPE_LIMIT or hashlib.sha256(raw).hexdigest()!=sha_value(expected_manifest):
        raise ValueError('external envelope identity')
    m=decode(raw)
    if (set(m)!={'schema','source_manifest_sha256','proof_manifest_sha256','mode','invocation','files'} or
        m['schema']!='HF_PORTABLE_V1' or m['source_manifest_sha256']!=sha_value(source) or
        m['proof_manifest_sha256']!=sha_value(proof) or m['mode']!=mode or mode not in ('endpoints','mccormick') or
        type(m['invocation']) is not str or not m['invocation']): raise ValueError('envelope context')
    code={'code/'+n for n in CODE}|{'verify.py'}; files=m['files']
    if (not code<=set(files) or not {'proof/manifest.json','proof/source/manifest.json'}<=set(files) or
        any(n not in code and not n.startswith('proof/') for n in files)): raise ValueError('code/proof inventory')
    actual=set(); total=len(raw)
    for p in Path(root).rglob('*'):
        tick()
        if p.is_symlink(): raise ValueError('symlink inventory')
        if p.is_file(): actual.add(str(p.relative_to(root)))
    if actual!=set(files)|{'manifest.json'}: raise ValueError('complete portable inventory')
    for name,ref in sorted(files.items()):
        tick()
        if (set(ref)!={'sha256','bytes'} or type(ref['bytes']) is not int or
                not 0<ref['bytes']<=MEMBER_LIMIT): raise ValueError('member descriptor')
        path=member(root,name,MEMBER_LIMIT); count=0; h=hashlib.sha256()
        with path.open('rb') as f:
            while True:
                tick(); chunk=f.read(2**20)
                if not chunk: break
                count+=len(chunk)
                if count>ref['bytes']: raise ValueError('growing portable member')
                h.update(chunk)
        if count!=ref['bytes'] or h.hexdigest()!=sha_value(ref['sha256']): raise ValueError('portable member identity')
        total+=count
        if total>TOTAL_LIMIT: raise ValueError('total portable bytes')
    tick(); return m,total


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve(); mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise PermissionError('read-only checker')
        stdlib=not {'site-packages','dist-packages'}.intersection(path.parts) and any(path.is_relative_to(p) for p in STDLIB)
        if not path.is_relative_to(ROOT) and not stdlib: raise PermissionError('outside bundle/stdlib')
    if event.startswith(('subprocess.','socket.','ctypes.')) or event in ('os.system','os.exec','os.fork','os.posix_spawn','os.remove','os.rename','os.mkdir','os.rmdir','os.link','os.symlink'):
        raise PermissionError('external execution or mutation forbidden')
    if event=='import' and (args[0].split('.')[0] in ('act','torch','numpy','scipy','highspy','gurobipy','ctypes') or
        args[0] in ('scoped_source.factored_build','scoped_source.endpoint_source_build','scoped_source.endpoint_build')):
        raise ImportError('model/producer/solver forbidden')


def verify(root,manifest_sha,source_sha,proof_sha,mode,deadline):
    m,total=envelope(root,manifest_sha,source_sha,proof_sha,mode,deadline)
    sys.path.insert(0,str(root/'code'))
    from scoped_source.factored_check import check
    result=check(root/'proof',expected_source_manifest=source_sha,expected_proof_manifest=proof_sha,
                 expected_mode=mode,deadline=deadline)
    # Check code/envelope and member closure at return as well. No second LP
    # reconstruction; these are streaming identity checks under the same clock.
    final,final_total=envelope(root,manifest_sha,source_sha,proof_sha,mode,deadline)
    if final!=m or final_total!=total: raise ValueError('portable identity changed during check')
    if time.monotonic()>=deadline: raise TimeoutError('late complete proof')
    return {'schema':'HF_PORTABLE_CHECK_V1','manifest_sha256':manifest_sha,'source_manifest_sha256':source_sha,
        'proof_manifest_sha256':proof_sha,'mode':mode,'invocation':m['invocation'],'result':result,
        'bundle_bytes':total,'isolated':bool(sys.flags.isolated),'site_disabled':bool(sys.flags.no_site),
        'solver_imported':False,'producer_executed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest-sha','source-sha','proof-sha','mode'): p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',type=float); a=p.parse_args(); start=time.monotonic()
    deadline=start+300 if a.deadline is None else a.deadline
    if not sys.flags.isolated or not sys.flags.no_site: raise ValueError('python -I -S required')
    if not math.isfinite(deadline) or deadline>start+300: raise ValueError('bounded checker budget')
    sys.addaudithook(guard)
    result=verify(ROOT,a.manifest_sha,a.source_sha,a.proof_sha,a.mode,deadline)
    result['check_seconds']=time.monotonic()-start
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
