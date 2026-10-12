"""Offline stdlib archive/source recheck; no generation, native library or solve."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

LIMIT=64*2**20
ANCHORS={'small40':'28c41d0da9493f2b80de1dab910a769868f12240ac00dc55727b9a716dc8fe65',
         'dense':'0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc'}
PROTOCOL='6a161cf33baf9a845c64cd7c4a543d4d1713834cbd80561bdb69517320ca69d7'
FILES={'scoped_source/source_dag.py','scoped_source/check_source_dag.py','scoped_source/source_dag_controls.py',
    'scoped_source/hz_request_store.py','scoped_source/hz_store_source.py','scoped_source/check_hz_store_source.py',
    'scoped_source/hz_store_sparse.py','scoped_source/hz_store_write.py','scoped_source/tensor_rows.py',
    'scoped_source/tensor_row_controls.py','scoped_source/factored_source.py','scoped_source/factored_io.py',
    'source_enclosure/format.py','upstream_source/checker.py','scripts/test_source_dag.py',
    'scripts/run_source_dag_controls.py','scripts/audit_source_dag_controls.py','scripts/run_tensor_rows_controls.py',
    'docs/hz_source_dag_design_20261013.md'}


def seconds(value, limit=300):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=limit


def envelope(execution,terminal):
    from scoped_source.hz_request_store import require
    require(execution['source_anchors']==terminal['source_anchors']==ANCHORS,'frozen source anchors')
    require(execution['archive_limit']==terminal['archive_cap']==LIMIT,'frozen archive capacity')
    require(set(execution['code'])==FILES and terminal['code']==execution['code'],'complete code inventory')
    require(execution['protocol']==execution['code']['docs/hz_source_dag_design_20261013.md']==PROTOCOL,'frozen protocol')
    require(set(execution['source_members'])==set(ANCHORS),'source case inventory')
    require(terminal['status']=='PASS_COMPONENT_CONTROLS','archive not successful')
    require(terminal['real_models']==terminal['gpu_calls']==terminal['full_request_certificates']==0,'terminal claim boundary')
    require(seconds(terminal['outer_before_terminal_seconds'],1200),'total stage cost')
    require([x['name'] for x in terminal['calls']]==['unit','small40','dense','relocated'],'call inventory')
    require(all(x['status']=='COMPLETE' and x['returncode']==0 and x['hard_cap_seconds']==300
        and seconds(x['parent_observed_seconds']) for x in terminal['calls']),'call completion/cost')


def receipt_cost(r):
    from scoped_source.hz_request_store import require
    work=r['work']
    require(set(work)=={'row_visits','coefficient_visits','arithmetic'} and
            all(type(x) is int and x>=0 for x in work.values()) and sum(work.values())<=8000000,'cumulative work budget')
    require(work==r['check']['work'] and seconds(r['worker_seconds']),'generation/check cost consistency')
    p=r['phases']
    require(set(p)=={'generation_and_encode_seconds','file_write_seconds','independent_parse_source_check_seconds',
                   'native_seconds','native_coordinates'},'phase inventory')
    require(all(seconds(p[k]) for k in p if k!='native_coordinates') and type(p['native_coordinates']) is int
            and p['native_coordinates']>=0 and p['native_seconds']<=p['generation_and_encode_seconds'],'phase costs')
    require(sum(p[k] for k in ('generation_and_encode_seconds','file_write_seconds','independent_parse_source_check_seconds'))
            <=r['worker_seconds'],'additive versus nested cost')


def audit(root):
    from scoped_source.check_source_dag import check
    from scoped_source.hz_request_store import Work,digest,require
    root=Path(root); require(not root.is_symlink(),'archive symlink')
    members=list(root.rglob('*')); require(not any(p.is_symlink() for p in members),'archive member symlink')
    require(sum(p.stat().st_size for p in members if p.is_file())<=LIMIT,'archive byte limit')
    execution=json.loads((root/'execution.json').read_bytes()); terminal=json.loads((root/'terminal.json').read_bytes())
    envelope(execution,terminal)
    for name,h in execution['code'].items(): require(digest((root/'code'/name).read_bytes())==h,'archived code')
    for name,h in execution['code'].items():
        require(digest((root/'relocated'/'code'/name).read_bytes())==h,'relocated code identity')
    for case,members in execution['source_members'].items():
        require({str(p.relative_to(root/'sources'/case)) for p in (root/'sources'/case).rglob('*') if p.is_file()}==set(members),'source inventory')
        for name,h in members.items(): require(digest((root/'sources'/case/name).read_bytes())==h,'source member')
    relocated=root/'relocated'/'source'; small=execution['source_members']['small40']
    require({str(p.relative_to(relocated)) for p in relocated.rglob('*') if p.is_file()}==set(small),'relocated source inventory')
    for name,h in small.items(): require(digest((relocated/name).read_bytes())==h,'relocated source bytes')
    results=[]
    for name in ('small40','dense'):
        begin=time.monotonic(); r=json.loads((root/(name+'.receipt.json')).read_bytes()); raw=(root/(name+'.forest.json')).read_bytes()
        require(r['source']==execution['source_anchors'][name] and r['forest_sha256']==digest(raw),'receipt identity')
        receipt_cost(r)
        result=check(raw,root/'sources'/name,r['source'],expected=r['forest_sha256'],
                     work=Work(begin+300),cut=None if name=='small40' else ['expert0',2])
        old=dict(r['check']); old.pop('work'); comparable=dict(result); comparable.pop('work')
        require(old==comparable,'independent result differential')
        if name=='small40':
            require(digest((root/'relocated'/'forest.json').read_bytes())==r['forest_sha256'],'relocated forest identity')
            moved=json.loads((root/'relocated.stdout').read_bytes()); moved.pop('work')
            require(moved==comparable,'relocated checker result')
        require(r['status']==result['status'] and r['full_request_certificates']==r['gpu_calls']==r['real_models']==0,'claim boundary')
        results.append(dict(name=name,status=result['status'],source=r['source'],forest=r['forest_sha256'],
                            stats=result['stats'],seconds=time.monotonic()-begin,independent_work=result['work']))
    require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=execution['archive_limit'],'archive byte limit')
    require(not {'numpy','scipy','torch','highspy'}&set(sys.modules),'offline native import')
    return dict(status='PASS_SOURCE_FOREST_RECHECK',audit_version=2,cases=results,full_request_certificates=0)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1])); p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    print(json.dumps(audit(p.parse_args().root),indent=2))
