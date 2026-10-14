"""Default-off direct-source / typed-candidate portable check, standard library.

The caller supplies the trusted expected package hash and checker implementation.
Packaged code hashes detect drift, not the correctness of a malicious checker.
"""
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY,compact,decode,digest,identity,require,sha,Work
from scoped_source.factored_io import read
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.scaled_endpoints import contract
from scoped_source.direct_candidates import Driver,MAX_TILE
from scoped_source.wire_package import inventory

MANIFEST='DIRECT_ENDPOINT_MANIFEST_V1'
BUNDLE='DIRECT_PORTABLE_BUNDLE_V1'
CODE_FILES=tuple('scoped_source/'+name+'.py' for name in (
    'direct_package','direct_source_format','check_direct_source','direct_candidates',
    'wire_forest','wire_package','wire_proof','wire_archive','check_source_dag','check_affine_references',
    'check_scaled_references','forest_endpoints','scaled_endpoints','bounded_runtime','pair_workspace',
    'tensor_rows','factored_io','factored_source','hz_request_store'))+('source_enclosure/format.py','upstream_source/checker.py')


def check_manifest(root,raw,*,expected,work,enabled=False):
    require(enabled is True,'direct portable checking is opt-in'); work.tick()
    require(type(raw) is bytes and len(raw)<=2**20 and digest(raw)==sha(expected),'direct manifest bytes/identity')
    m=decode(raw)
    require(type(m) is dict and compact(m)==raw and set(m)=={'schema','source','forest','contract','candidate_width','check_width','tiles','code'}
            and m['schema']==MANIFEST and type(m['candidate_width']) is int and m['candidate_width']==8
            and type(m['check_width']) is int and m['check_width']==2,'direct manifest schema/widths')
    def member(ref,cap):
        require(type(ref) is dict and set(ref)=={'file','bytes','sha256'} and type(ref['bytes']) is int and 0<ref['bytes']<=cap,'direct member fields')
        work.tick(); value=read(root,ref['file'],cap,sha(ref['sha256'])); work.tick()
        require(len(value)==ref['bytes'],'direct member length'); return value
    require(type(m['code']) is list and len(m['code'])==len(CODE_FILES) and
            [r['file'] for r in m['code']]==['code/'+n for n in CODE_FILES],'complete checker code inventory')
    for ref in m['code']: member(ref,64*2**10)
    require(type(m['source']) is dict and set(m['source'])=={'directory','sha256'} and m['source']['directory']=='source','local direct source')
    require(m['forest']['file']=='forest.bin' and m['contract']['file']=='contract.json','source/contract member names')
    f=CheckedDirectForest(member(m['forest'],POLICY['total_bytes']),Path(root)/'source',m['source']['sha256'],
                          expected=m['forest']['sha256'],work=work,enabled=True)
    cr=member(m['contract'],8192); c=decode(cr)
    require(compact(c)==cr and c==contract(f),'complete bound properties')
    d=Driver(f,c,expected=identity(c),work=work,pair_scoped=True)
    try:
        require(type(m['tiles']) is list and len(m['tiles'])==d.session.tile_count and
                [r['file'] for r in m['tiles']]==[f'tiles/{i:04d}.bin' for i in range(d.session.tile_count)],'all ordered unique tiles')
        for ref in m['tiles']:
            _,_,token=d.prepare(); value=member(ref,MAX_TILE)
            d.receive(token,value,lambda i,b:(digest(b),len(b)))
        result=d.finish(); work.tick()
        return dict(result=result,owner=dict(d.owner.stats),driver_cost=dict(d.costs),candidate_bytes=d.total_bytes)
    finally:
        if not d.closed: d.abort()


def check_package(root,*,expected,work,enabled=False):
    require(enabled is True,'direct package is opt-in'); start=time.monotonic()
    refs,total=inventory(root,expected,work,schema=BUNDLE); initial=time.monotonic()
    ref=refs['manifest.json']; raw=read(root,'manifest.json',2**20,ref['sha256'])
    checked=check_manifest(root,raw,expected=ref['sha256'],work=work,enabled=True); middle=time.monotonic()
    inventory(root,expected,work,schema=BUNDLE); work.tick()
    return dict(**checked,package_sha256=expected,package_bytes=total,work=dict(work.counts),
                cost=dict(initial_inventory_seconds=initial-start,proof_seconds=middle-initial,
                          final_inventory_seconds=time.monotonic()-middle,total_seconds=time.monotonic()-start))


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('expected')
    p.add_argument('--enable-direct',action='store_true'); p.add_argument('--seconds',type=float,default=300)
    a=p.parse_args(); require(0<a.seconds<=300,'unchanged portable deadline')
    print(compact(check_package(a.root,expected=a.expected,work=Work(time.monotonic()+a.seconds),enabled=a.enable_direct)).decode())


if __name__=='__main__': main()
