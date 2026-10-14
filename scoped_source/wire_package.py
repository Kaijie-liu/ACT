"""Package-level inventory, budget and portable independent-check interface."""
import hashlib
import os
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY,compact,decode,digest,require,sha,Work
from scoped_source.factored_io import read
from scoped_source.wire_proof import check_manifest

BUNDLE='WIRE_PORTABLE_BUNDLE_V1'


def bundle(archive):
    """Publish after all mathematical inputs and optional checker code exist."""
    archive.audit(); require('manifest.json' in archive.entries,'complete proof manifest')
    entries=[dict(archive.entries[n]) for n in sorted(archive.entries)]
    raw=compact(dict(schema=BUNDLE,manifest='manifest.json',members=entries))
    require(len(raw)<=2**20,'bounded package inventory')
    return archive.put('bundle.json',raw)


def inventory(root,expected,work,*,schema=BUNDLE):
    work.tick(); root=Path(root); raw=read(root,'bundle.json',2**20,sha(expected)); work.tick()
    d=decode(raw)
    require(compact(d)==raw and type(d) is dict and set(d)=={'schema','manifest','members'} and
            d['schema']==schema and d['manifest']=='manifest.json' and type(d['members']) is list and
            0<len(d['members'])<POLICY['members'],'package schema/count')
    refs={}; total=len(raw)
    for ref in d['members']:
        work.tick(); require(type(ref) is dict and set(ref)=={'file','bytes','sha256'},'package member fields')
        name=ref['file']; p=Path(name)
        require(type(name) is str and len(name)<=200 and not p.is_absolute() and str(p)==name and
                all(s not in ('','.','..') for s in p.parts) and name not in refs and name!='bundle.json','package member path/unique')
        require(type(ref['bytes']) is int and ref['bytes']>=0,'package member size')
        sha(ref['sha256']); refs[name]=ref; total+=ref['bytes']
        require(total<=POLICY['total_bytes'],'package aggregate cap')
    require('manifest.json' in refs,'package manifest binding')
    found=set(); stack=[(root,0)]; seen=0
    while stack:
        directory,depth=stack.pop(); work.tick()
        require(not directory.is_symlink() and depth<=8,'package directory/symlink/depth')
        with os.scandir(directory) as entries:
            for entry in entries:
                work.tick(); seen+=1; require(seen<=2*POLICY['members'],'bounded filesystem inventory')
                require(not entry.is_symlink(),'package symlink')
                path=Path(entry.path); name=str(path.relative_to(root))
                if entry.is_dir(follow_symlinks=False): stack.append((path,depth+1)); continue
                require(entry.is_file(follow_symlinks=False) and (name in refs or name=='bundle.json'),'extra package member')
                found.add(name)
                if name=='bundle.json': continue
                ref=refs[name]; require(entry.stat(follow_symlinks=False).st_size==ref['bytes'],'package changed length')
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    count=0
                    while True:
                        work.tick(); part=stream.read(min(65536,ref['bytes']-count+1))
                        if not part: break
                        count+=len(part); require(count<=ref['bytes'],'package grew during read'); h.update(part)
                require(count==ref['bytes'] and h.hexdigest()==ref['sha256'],'package member identity')
    require(found==set(refs)|{'bundle.json'},'complete package inventory')
    require(read(root,'bundle.json',2**20,expected)==raw,'package final inventory identity'); work.tick()
    return refs,total


def check_package(root,*,expected,work):
    start=time.monotonic(); refs,total=inventory(root,expected,work); checked_inventory=time.monotonic()
    ref=refs['manifest.json']; raw=read(root,'manifest.json',2**20,ref['sha256']); work.tick()
    result=check_manifest(root,raw,expected=ref['sha256'],work=work); checked_proof=time.monotonic()
    inventory(root,expected,work); work.tick()
    return dict(result=result,package_sha256=expected,package_bytes=total,
                cost=dict(initial_inventory_seconds=checked_inventory-start,
                          proof_seconds=checked_proof-checked_inventory,
                          final_inventory_seconds=time.monotonic()-checked_proof,
                          total_seconds=time.monotonic()-start),work=dict(work.counts))


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('expected')
    parser.add_argument('--seconds',type=float,default=300); args=parser.parse_args()
    require(0<args.seconds<=300,'unchanged portable budget')
    result=check_package(args.root,expected=args.expected,work=Work(time.monotonic()+args.seconds))
    print(compact(result).decode())


if __name__=='__main__': main()
