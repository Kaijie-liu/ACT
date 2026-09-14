"""Frozen R3: three direct rational residual LPs; no model/range refinement."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from act.pipeline.moe.experiment1 import PROJECT_ROOT,WRITE_ROOT,_inside,_sha256,_git_value
from act.pipeline.moe.paired_followup import save,source_identity
from act.pipeline.moe.check_request_lp import check_directory,RATIONAL_TRUSTED,property_row
from act.pipeline.moe.request_lp_control import frozen_request
from act.back_end.solver.lp_certificate import identity,propose
from act.back_end.solver.rational_mccormick import build
from act.back_end.solver.check_rational_mccormick import check_construction

PARENT=PROJECT_ROOT/'data/moe/results/request_lp_order_20260914_r2'
REVIEW=PROJECT_ROOT/'act/pipeline/moe/results/request_lp_order_review_20260914_r2.json'


def parent():
    review=json.loads(REVIEW.read_text())
    for name,sha in review['raw_hashes'].items():
        path=(PARENT/name).resolve()
        if not path.is_relative_to(PARENT) or _sha256(path)!=sha:raise ValueError('parent artifact drift')
    checked=check_directory(PARENT,expected_request_id=identity(frozen_request()))
    if checked!=review['check']:raise ValueError('parent check drift')
    return json.loads((PARENT/'manifest.json').read_text())


def worker(root):
    old=parent();shutil.copytree(PARENT,root/'parent')
    m=copy.deepcopy(old);m.update(schema='request_lp_rational_v3',trusted_base=RATIONAL_TRUSTED,
        parent={'manifest':'parent/manifest.json','sha256':_sha256(PARENT/'manifest.json')})
    for item in m['proofs'].values():
        for field in ('export','certificate'):
            if item[field]:item[field]['file']='parent/'+item[field]['file']
    save(root/'manifest.json',m)
    for row in m['obligations']:
        if row['kind']!='residual':continue
        p=row['pair'];idx=row['property_index'];q=property_row(m['request']['classes'],m['request']['clean_prediction'],idx)
        src=json.loads((root/m['proofs'][row['difference_lower']]['export']['file']).read_text())['source']
        record=build(src,q,0,row['lambda_bounds'],row['difference_bounds'])
        key=f'rational_{p[0]}_{p[1]}_p{idx}';ep=root/(key+'.export.json');save(ep,record)
        ref=lambda path:{'file':path.name,'sha256':_sha256(path)}
        item={'request_id':m['request_id'],'kind':'rational_weighted','scope':{'pair':p},'property_index':idx,
              'status':'PENDING','export':ref(ep),'certificate':None,'hz_sha256':record['source_sha256']}
        m['proofs'][key]=item;row['source']=key;save(root/'manifest.json',m)
        start=time.monotonic()
        # Construction errors fail-stop. A failed LP proposal remains UNKNOWN.
        check_construction(record,source_hash=item['hz_sha256'],q=q,offset=0,gate=row['lambda_bounds'],difference=row['difference_bounds'])
        try:
            cert=propose(record['lp'],time_limit=10);cp=root/(key+'.certificate.json');save(cp,cert)
            checked=check_construction(record,cert,source_hash=item['hz_sha256'],q=q,offset=0,gate=row['lambda_bounds'],difference=row['difference_bounds'])
            item.update(status='CHECKED',certificate=ref(cp),checked_lower_bound=checked['bound']['checked_lower_bound'])
        except ValueError as exc:item.update(status='UNKNOWN',error=str(exc))
        item['seconds']=time.monotonic()-start;save(root/'manifest.json',m)
        print(key,item['status'],item.get('checked_lower_bound'),flush=True)


def run(root):
    if _git_value('branch','--show-current')!='feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise RuntimeError('clean feature branch required')
    root=_inside(root,WRITE_ROOT);root.mkdir(exist_ok=False)
    save(root/'launch.json',{'head':_git_value('rev-parse','HEAD'),'source_sha256':source_identity(),
        'parent_review_sha256':_sha256(REVIEW),'request_id':identity(frozen_request()),'outer_seconds':600,'max_new_queries':3})
    start=time.monotonic()
    with (root/'worker.log').open('w') as log:
        try:
            done=subprocess.run([sys.executable,'-m',__name__ if __name__!='__main__' else 'act.pipeline.moe.request_lp_rational','--worker',str(root)],
                stdout=log,stderr=subprocess.STDOUT,timeout=600,check=False)
            terminal={'returncode':done.returncode}
        except subprocess.TimeoutExpired:terminal={'status':'TIMEOUT'}
    terminal['seconds']=time.monotonic()-start;save(root/'terminal.json',terminal)
    if terminal.get('returncode')==0:
        result=check_directory(root,expected_request_id=identity(frozen_request()));save(root/'check.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run',type=Path);g.add_argument('--worker',type=Path);a=p.parse_args()
    run(a.run) if a.run else worker(a.worker)
