"""Saved-only independent reread and compact accounting; no new queries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_metamoe_receipt_reserve import audit
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from act.back_end.solver.isolated_feasibility import NativeSession

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/backend_controls/metamoe_receipt_reserve_r1.json'
OUTPUT=ROOT/'docs/metamoe_receipt_reserve_archive_20260924_r1.json'


def archive():
    start=time.monotonic();cfg=json.loads(CONFIG.read_text());root=Path(cfg['output_root'])
    saved=json.loads((root/'independent_audit.json').read_text())
    with patch.object(NativeSession,'query',side_effect=AssertionError('saved audit must not solve')):
        fresh=audit(CONFIG,root/'independent_replay.json')
    compare=lambda r:{k:v for k,v in r.items() if k!='separate_audit_seconds'}
    if compare(fresh)!=compare(saved):raise ValueError('independent reread differs')
    files={str(p.relative_to(root)):{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in sorted(root.rglob('*')) if p.is_file()}
    rows=[]
    for row in fresh['rows']:
        details=row['details'];experts=details['experts'];folder=root/row['variant']/f"{row['id']}_act"
        queries=[]
        for p in sorted((folder/'protected').glob('evaluation_*/query_*/return.json')):
            q=json.loads(p.read_text())
            if q['scope']['phase']=='base':continue
            queries.append({'scope':q['scope'],'terminal':q['terminal'],'returned_status':q['returned_status'],
                'seconds':q['return_elapsed_seconds'],'accepted_after_receipt':q['accepted_after_receipt']})
        rows.append({'id':row['id'],'variant':row['variant'],'status':row['status'],'grade':row['grade'],
            'seconds':row['seconds'],'worker_starts':row['native_caps']['worker_starts'],
            'missing_native_runtime':row['native_caps']['native_cost_missing'],
            'properties':dict(sum((Counter(e['property_status_counts']) for e in experts),Counter())),
            'expert_seconds':[e['expert_elapsed_before_publication'] for e in experts],
            'query_terminals':dict(Counter(q['terminal'] for q in queries))})
    groups={v:{'status_counts':dict(Counter(r['status'] for r in rows if r['variant']==v)),
        'total_seconds':sum(r['seconds'] for r in rows if r['variant']==v),
        'worker_starts':sum(r['worker_starts'] for r in rows if r['variant']==v),
        'missing_native_runtime':sum(r['missing_native_runtime'] for r in rows if r['variant']==v)} for v in cfg['variants']}
    lookup={(r['id'],r['variant']):r for r in rows}
    differences=[lookup[r['id'],'receipt_reserve']['seconds']-lookup[r['id'],'full_native']['seconds'] for r in cfg['requests']]
    return {'audit':'PASS','issues':0,'config_sha256':sha256(CONFIG),'raw_root':str(root),
        'launch':json.loads((root/'launch.json').read_text()),'files':files,'rows':rows,'groups':groups,
        'paired':fresh['paired'],'paired_seconds_reserve_minus_full':differences,
        'median_paired_seconds':statistics.median(differences),'cost':fresh['cost'],
        'original_audit_seconds':saved['separate_audit_seconds'],
        'new_archive_seconds':time.monotonic()-start,'archive_source_sha256':sha256(__file__),
        'decision':'STOP_THIS_FACTOR: no endpoint gain or observed reduction in worker restarts/missing native returns; keep opt-in, no default change',
        'limits':'four observed inputs, one execution per arm; timing variability not quantified; no relaxation/source-complete inference'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();v=archive()
    if a.check:
        old=json.loads(OUTPUT.read_text());fields=lambda r:{k:x for k,x in r.items() if k!='new_archive_seconds'}
        if fields(old)!=fields(v):raise ValueError('archive drift')
        print('PASS: archive reproduced; no new queries')
    else:
        if OUTPUT.exists():raise FileExistsError(OUTPUT)
        write(OUTPUT,v);print(json.dumps(v['groups']))
