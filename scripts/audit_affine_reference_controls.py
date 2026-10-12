"""Offline source and operator audit; stdlib, no native/producer/solver calls."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from scoped_source.hz_request_store import Work,require
from scoped_source.check_affine_references import CheckedAffineForest
from scoped_source.affine_reference_controls import oracle

ANCHORS={'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
         'deep':'91a9cd3551e5f63e500e6db435a3039d6e158d7d671e88b7780a735267f4ba62'}
PATHS={'mixed':'mixed-fixture-mixed','deep':'deep-fixture/source'}
PROTOCOL='1b09da542fa2a15aa64c0fdfa0938000f128e3c4dcb53cf1a0449bc94d380425'


def audit(root):
    start=time.monotonic(); root=Path(root); load=lambda n:json.loads((root/n).read_bytes())
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),'self-contained nonsymlink archive')
    e=load('execution.json'); t=load('terminal.json'); require(t['status']=='PASS_BOUNDED_V2_COMPONENT','successful terminal')
    require(t['code']==e['code'] and e['archive_limit']==t['archive_cap']==16*2**20,'code/cap inventory')
    require(e['protocol']==e['code']['docs/hz_affine_references_design_20261013.md']==PROTOCOL,'protocol identity')
    require(e['source_anchors']==ANCHORS,'fixed recipe anchors')
    require(e['dense']=='NOT_ADMITTED_STATIC_COMPLETE_QUERY_CAPACITY' and e['dense_zero_with_prefix_minimum']==8650647,
            'dense refusal not an execution')
    require([c['name'] for c in t['calls']]==['unit','mixed','relocated-mixed','deep','relocated-deep'],'fixed call inventory')
    for call in t['calls']:
        require(call['status']=='COMPLETE' and call['returncode']==0 and call['hard_cap_seconds']==300
                and 0<call['parent_observed_seconds']<300,'terminal cost/status')
    require(math.isfinite(t['outer_before_terminal_seconds']) and
            sum(c['parent_observed_seconds'] for c in t['calls'])<=t['outer_before_terminal_seconds'],'parent total cost')
    for name,h in e['code'].items():
        require(hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==h,'archived code identity')
    result=[]
    for name in ('mixed','deep'):
        r=load(name+'.receipt.json'); op=load(name+'.operator.json'); raw=(root/(name+'.forest.json')).read_bytes()
        require(r['source']==ANCHORS[name] and r['source_path']==PATHS[name],'fixed source/path')
        w=Work(time.monotonic()+300)
        f=CheckedAffineForest(raw,root/r['source_path'],r['source'],expected=r['forest_sha256'],work=w)
        require(r['status']=='CHECKED_V2_SOURCE_AND_OPERATOR' and dict(f.receipt)['stats']==r['source_check']['stats'],'source stats')
        require(op['source']==r['source'] and op['forest']==r['forest_sha256'],'operator source binding')
        _,_,forward,adjoint,details=oracle(f,w)
        require(details==r['oracle']==op['oracle'],'complete differential inventory')
        require(op['native']==r['operator'] and op['native']['expanded_entries']==0
                and op['native']['device']=='cpu' and op['native']['dtype']=='float64','native representation boundary')
        for key,values in [('forward',forward),('adjoint',adjoint)]:
            require([[str(q) for q in row] for row in values]==op['exact_'+key],'independent exact values')
            require(len(values)==len(op[key]),'native matrix row inventory')
            for exact,actual in zip(values,op[key]):
                require(len(exact)==len(actual)==2,'fixed two-column matrix')
                for x,y in zip(exact,actual):
                    require(type(y) in (int,float) and abs(x-F(y))<=F(1,10**12)*(1+abs(x)),'native differential tolerance')
        require(set(r['work'])=={'arithmetic','coefficient_visits','row_visits'} and
                all(type(v) is int and v>=0 for v in r['work'].values()) and
                sum(r['work'].values())<=8000000 and 0<r['worker_before_receipt_seconds']<300,'worker work/cost')
        parent_call=next(c for c in t['calls'] if c['name']==name)
        require(r['worker_before_receipt_seconds']<=parent_call['parent_observed_seconds'],'worker within parent cost')
        require(all(type(v) in (float,int) and v>=0 for v in r['phases'].values()) and
                sum(r['phases'].values())<=r['worker_before_receipt_seconds'],'phase cost')
        require(all(r[k]==0 for k in ('real_models','gpu_calls','full_request_certificates')),'claim boundary')
        dest=root/('relocated-'+name)
        require((dest/'forest.json').read_bytes()==raw,'relocated forest identity')
        for p in (root/r['source_path']).rglob('*'):
            if p.is_file(): require((dest/'source'/p.relative_to(root/r['source_path'])).read_bytes()==p.read_bytes(),'relocated source bytes')
        for path,h in e['code'].items():
            require(hashlib.sha256((dest/'code'/path).read_bytes()).hexdigest()==h,'relocated checker code')
        portable=load('relocated-'+name+'.stdout')
        require(portable['status']=='CHECKED_SOURCE_FOREST' and portable['stats']==r['source_check']['stats'] and
                portable['source']==r['source'] and portable['request']==f.receipt['request'] and
                portable['source_receipt']==r['source_check']['source_receipt'] and
                portable['representation']=='EXACT_AFFINE_REFERENCES_TRIANGLE_RADIUS_V2','portable source check')
        result.append(dict(name=name,source=r['source'],forest=r['forest_sha256'],stats=r['source_check']['stats'],
                           oracle=details,execution_work=sum(r['work'].values()),audit_work=sum(w.counts.values())))
    require(not ({'numpy','scipy','torch','highspy'}&set(sys.modules)),'independent stdlib audit')
    payload=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(payload==t['payload_before_terminal_bytes']+(root/'terminal.json').stat().st_size
            and payload<=16*2**20,'complete archive byte accounting')
    return dict(status='PASS_INDEPENDENT_V2_COMPONENT_AUDIT',sources=result,archive_bytes=payload,
                seconds=time.monotonic()-start,real_models=0,gpu_calls=0,full_request_certificates=0,
                dense='NOT_ADMITTED',guarantee='captured source enclosure and same-V2 operator differential; no output proof')


if __name__=='__main__': print(json.dumps(audit(sys.argv[1]),indent=2))
