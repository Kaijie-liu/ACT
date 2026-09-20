"""Isolated complete source + all fresh duals; numerical solver is not trusted."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):raise PermissionError('read-only checker')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):raise PermissionError('outside moved package/stdlib')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):raise PermissionError('external execution blocked')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','full_bounds','full_source','source_enclosure'):
        raise ImportError('model/solver/producer blocked')


def verify(root,manifest_hash):
    from verify_full import verify as check_source
    from bound_check import aggregate
    load=lambda n:json.loads((root/n).read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/'manifest.json')!=manifest_hash:raise ValueError('bound package external identity')
    m=load('manifest.json')
    if m['schema']!='FULL_SOURCE_DUAL_BUNDLE_V1':raise ValueError('proof bundle schema')
    for name,h in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('proof file identity/path')
    required={'bound_check.py','verify_bounds.py','act/back_end/solver/lp_certificate.py',
              'act/back_end/solver/sparse_lp_certificate.py','source/manifest.json'}
    pm=load('source/manifest.json');required|={'source/'+n for n in pm['files']}
    if pm['request']!=m['request'] or pm['pair']!=m['pair']:raise ValueError('parent/request binding')
    t=time.monotonic();source=check_source(root/'source',m['parent_manifest_sha256']);source_seconds=time.monotonic()-t
    base=load('source/lp_base.json');obs=load('source/obligations.json')
    required|={r['file'] for r in m['outcomes'] if r['file'] is not None}
    if set(m['files'])!=required:raise ValueError('complete source/dual file inventory')
    t=time.monotonic();result=aggregate(base,obs,m,m['outcomes'],load);bound_seconds=time.monotonic()-t
    return {**result,'source_checks':{k:source[k] for k in ('prefix_steps_checked','remaining_steps_checked','route_exclusions_checked','joint')},
        'source_check_seconds':source_seconds,'bound_check_seconds':bound_seconds,
        'remaining_trust':['declared graphs/parameters correspond to intended registered program','checker/interpreter execution'],
        'scope':'Complete declared real MoE positivity ONLY if every new output bound is checked positive. Not native floating semantics or high-accuracy/route-changing evidence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-hash',required=True);a=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('python -I -S required')
    start=time.monotonic();sys.addaudithook(guard)
    sys.path[:0]=[str(ROOT),str(ROOT/'source'),str(ROOT/'source/prefix')]
    result=verify(ROOT,a.manifest_hash);result['check_seconds']=time.monotonic()-start
    result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result,sort_keys=True))
