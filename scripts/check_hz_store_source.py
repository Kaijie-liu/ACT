"""Portable source-only checking CLI; standard library, no checkpoint or solver."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source.hz_request_store import Store, Work
from scoped_source.check_hz_store_source import check_source

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('declaration')
    p.add_argument('--store',required=True); p.add_argument('--source',required=True); a=p.parse_args()
    work=Work(time.monotonic()+300)
    store=Store(a.root,expected_manifest=a.store,expected_request=a.source,work=work)
    doc=json.loads(Path(a.declaration).read_bytes())
    out=check_source(store,doc,expected_source=a.source,work=work)
    out['work_counts']=work.counts
    out['forbidden_modules']=[x for x in ('numpy','scipy','torch','gurobipy','highspy') if x in sys.modules]
    print(json.dumps(out,sort_keys=True))
