"""Portable ONE source-step check; not a complete request or network proof."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)
REQUIRED={'source.json','target.json','proof.json','range_check.py','verify_range.py',
          'proof_format.py','local_check.py','act/back_end/solver/lp_certificate.py',
          'act/back_end/solver/sparse_lp_certificate.py'}


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise PermissionError('read-only check')
        if not p.is_relative_to(ROOT) and not any(p.is_relative_to(d) for d in STDLIB):
            raise PermissionError('outside relocated proof')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):
        raise PermissionError('no external process/network')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','source_ranges','source_enclosure','full_source'):
        raise ImportError('no producer/model/solver')


def verify(root,expected_hash):
    from range_check import check_affine,check_relu
    from local_check import csr
    load=lambda n:json.loads((root/n).read_bytes())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/'manifest.json')!=expected_hash:raise ValueError('external statement/checker identity')
    m=load('manifest.json')
    if m['schema']!='SCOPED_RANGE_STEP_BUNDLE_V1' or set(m['files'])!=REQUIRED:
        raise ValueError('complete range-step inventory')
    for name,h in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('source-step file identity')
    s,t,p=load('source.json'),load('target.json'),load('proof.json');step=m['step']
    if step['kind']=='affine':
        if set(step)!={'kind','context','tag','operator','bias'}:raise ValueError('affine statement')
        op=csr(step['operator'],[len(step['bias']),len(s['hz']['c'])])
        r=check_affine(s,t,op,step['bias'],p,step['context'],step['tag'])
    elif step['kind']=='relu':
        if set(step)!={'kind','context','tag'}:raise ValueError('ReLU statement')
        r=check_relu(s,t,p,step['context'],step['tag'])
    else:raise ValueError('unsupported source step')
    return {**r,'remaining_trust':['supplied source encloses intended request domain','declared operation matches intended graph','checker/interpreter execution'],
        'scope':'One checked source-step range/graph; NOT whole-model safety, routed coverage, or deployed floating semantics.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-hash',required=True);a=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('python -I -S required')
    start=time.monotonic();sys.addaudithook(guard);sys.path.insert(0,str(ROOT))
    r=verify(ROOT,a.manifest_hash);r['check_seconds']=time.monotonic()-start
    print(json.dumps(r,sort_keys=True))
