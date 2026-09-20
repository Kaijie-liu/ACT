"""Fresh saved-control review; no SoPlex calls or LP optimization."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from lp_sandwich.check import identity, strict_json

ROOT = Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def review(receipt_path, output):
    started=time.monotonic()
    r=strict_json(receipt_path.read_bytes())
    if r['status']!='PASS' or r['real_queries']!=0: raise ValueError('control scope/result')
    for table in ('source_sha256','artifacts'):
        for name,digest in r[table].items():
            if sha(ROOT/name)!=digest: raise ValueError('saved identity drift: '+name)
    root=Path(r['raw_root']);checks=[]
    moved=Path(tempfile.mkdtemp(prefix='soplex_review_',dir=ROOT/'data/moe/results'))
    shutil.copyfile(ROOT/'lp_sandwich/check.py',moved/'verify.py')
    for item in r['native_controls']:
        case=root/item['case'];bundle=strict_json((case/'moved/bundle.json').read_bytes())
        if bundle['lp']!=strict_json((case/'original.json').read_bytes()):
            raise ValueError('bundle does not contain original LP')
        dest=moved/(item['case']+'.json');shutil.copyfile(case/'moved/bundle.json',dest)
        args=[sys.executable,'-I','-S',str(moved/'verify.py'),str(dest),
              '--bundle-sha256',sha(dest),'--statement-sha256',identity(bundle['statement']),
              '--timeout-seconds','10']
        p=subprocess.run(args,cwd=moved,capture_output=True,text=True,timeout=15,check=True)
        actual=strict_json(p.stdout);old=strict_json((case/'checked.json').read_bytes())
        if {k:v for k,v in actual.items() if k!='seconds'}!={k:v for k,v in old.items() if k!='seconds'}:
            raise ValueError('fresh checker differs: '+item['case'])
        (moved/(item['case']+'.checked.json')).write_text(json.dumps(actual,sort_keys=True,indent=2))
        checks.append({'case':item['case'],'primal_status':actual['primal_status'],
                       'upper_bound':actual['upper_bound'],'seconds':actual['seconds']})
    # Tamper moved bytes and retain the stale expected digest: must reject.
    original=(moved/'third.json').read_bytes();dest=moved/'tampered.json'
    dest.write_bytes(original+b' ')
    b=strict_json(original)
    args=[sys.executable,'-I','-S',str(moved/'verify.py'),str(dest),
          '--bundle-sha256',hashlib.sha256(original).hexdigest(),
          '--statement-sha256',identity(b['statement']),'--timeout-seconds','10']
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    if p.returncode!=2 or strict_json(p.stdout)['status']!='REJECTED': raise ValueError('tamper accepted')
    # Reuse a valid bundle with an already exhausted check deadline.
    args[4]=str(moved/'third.json');args[-1]='0.0000001'
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    if p.returncode!=3 or strict_json(p.stdout)['status']!='TIMEOUT': raise ValueError('deadline accepted')
    result=dict(status='PASS',issues=[],receipt_sha256=sha(receipt_path),
        review_source_sha256=sha(Path(__file__)),checks=checks,artifact_count=len(r['artifacts']),
        native_calls=0,relocated_root=str(moved),mutation_rejected=True,deadline_rejected=True,
        seconds=time.monotonic()-started,
        scope='Saved identity + original-LP feasibility/objective checks; NOT native optimality or network proof.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(review(a.receipt,a.output),indent=2))
