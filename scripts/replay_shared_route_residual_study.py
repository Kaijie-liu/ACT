"""Read-only, fresh-process reproduction of the frozen synthetic archive.

This reuses the two mathematical checkers, not the candidate providers. It is
not a third proof implementation or an independent human technical review.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load, save, sha
from shared_route_residual.review import audit_one, comparable


def replay():
    started=time.monotonic()
    if not sys.flags.no_site:raise ValueError('use python -S')
    cfg=load(ROOT/'configs/backend_controls/shared_route_residual_synthetic_r1.json')
    archive=load(ROOT/'docs/shared_route_residual_archive_20260924_r1.json')
    report=load(ROOT/'docs/shared_route_residual_audit_20260924_r1.json')
    root=ROOT/'data/moe/results/shared_route_residual_synthetic_20260924_r1'
    if archive['root']!=str(root) or cfg['output']!=str(root):raise ValueError('frozen output binding')
    paths=[]
    for row in archive['files']:
        p=Path(row['path'])
        if p.is_absolute() or '..' in p.parts:raise ValueError('archive path')
        if (root/p).is_symlink() or (root/p).stat().st_size!=row['bytes'] or sha(root/p)!=row['sha256']:
            raise ValueError('raw file changed: '+str(p))
        paths.append(str(p))
    actual=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    if actual!=sorted(paths):raise ValueError('file inventory')
    for p,digest in cfg['sources'].items():
        if sha(ROOT/p)!=digest:raise ValueError('frozen code: '+p)
    records={}
    for call in cfg['calls']:
        result=audit_one(root/call['id'])
        if result['status']!='COMPLETED_ROUTER_SEGMENT':raise ValueError('unexpected status')
        records[(call['fixture'],call['repeat'],call['mode'])]=result
    pairs=0
    for kind in cfg['fixture_sha256']:
        for repeat in range(3):
            a=records[(kind,repeat,'pairwise')]['result'];b=records[(kind,repeat,'shared')]['result']
            if comparable(a)!=comparable(b):raise ValueError('differential')
            pairs+=1
    if report['status']!='PASS' or report['calls']!=len(records) or report['raw_file_count']!=len(paths):
        raise ValueError('summary')
    if any(n.split('.')[0] in ('torch','numpy','scipy','act','highspy') for n in sys.modules):
        raise ValueError('model/native dependency loaded')
    if 'shared_route_residual.propose' in sys.modules or 'checked_route_frontier.build' in sys.modules:
        raise ValueError('candidate provider loaded')
    return {'status':'PASS','issues':[],'calls_rechecked':len(records),'exact_paired_differentials':pairs,
        'files_rehashed':len(paths),'frozen_sources':len(cfg['sources']),
        'seconds_separate_from_request':time.monotonic()-started,'real_requests':0,
        'native_solver_calls':0,'new_complete_output_certificates':0,
        'note':'fresh saved-only rerun of checker implementations, not human review'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path);args=p.parse_args()
    result=replay()
    if args.receipt:
        if not args.receipt.resolve().is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('write scope')
        save(args.receipt,result)
    print(result)
