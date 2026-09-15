"""Freeze a new pre-F0 construction proof, keeping every old source immutable."""
from scripts.conv_request_sign_lp_contract import (ROOT,read,save,sha,git,publication_gate,
    verify_freeze as old_freeze,sources as old_sources,DEFAULT as OLD)
from scripts.conv_pre_f0_contract import verify_freeze as failed_freeze,DEFAULT as FAILED_RAW

PROTOCOL=ROOT/'scripts/conv_pre_f0_r2_protocol.json'
DEFAULT=ROOT/'data/moe/results/conv_pre_f0_rational_20260915_r2'
FREEZE=ROOT/'act/pipeline/moe/results/conv_pre_f0_freeze_20260915_r2.json'
PARENT=ROOT/'act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json'
FILES=('scripts/conv_pre_f0_r2_protocol.json','scripts/conv_pre_f0_r2_contract.py',
       'scripts/run_conv_pre_f0_r2.py','scripts/check_conv_pre_f0_r2.py',
       'scripts/test_conv_pre_f0_r2.py','docs/conv_pre_f0_r2.md','scripts/review_conv_pre_f0_failure.py')


def sources():return {**old_sources(),**{p:sha(ROOT/p) for p in FILES}}


def parents():
    f=old_freeze();p=read(PROTOCOL)
    failed_freeze()
    failed=ROOT/'act/pipeline/moe/results/conv_pre_f0_failure_20260915_r1.json'
    if sha(failed)!=p['failed_r1_review_sha256']:raise ValueError('failure review changed')
    inv_failed=read(failed)['artifact_inventory']
    if {str(x.relative_to(FAILED_RAW)) for x in FAILED_RAW.rglob('*') if x.is_file()}!={v['path'] for v in inv_failed}:
        raise ValueError('failed R1 inventory changed')
    for v in inv_failed:
        if sha(FAILED_RAW/v['path'])!=v['sha256']:raise ValueError('failed R1 artifact changed')
    if sha(PARENT)!=p['parent_review_sha256']:raise ValueError('parent review drift')
    inv=read(PARENT)['artifact_inventory']
    if {str(x.relative_to(OLD)) for x in OLD.rglob('*') if x.is_file()}!={v['path'] for v in inv}:
        raise ValueError('parent inventory drift')
    for v in inv:
        if sha(OLD/v['path'])!=v['sha256']:raise ValueError('old proof evidence changed')
    return p,f


def job():
    p,f=parents();j=next(j for j in f['jobs'] if j['case']['job_id']==p['job_id'])
    if j['case']['dataset_index']!=p['dataset_index'] or j['case']['expected_pairs']!=p['expected_pairs']:
        raise ValueError('wrong request')
    return j


def verify_freeze():
    f=read(FREEZE)
    if f!={'protocol':parents()[0],'job':job(),'sources':sources()}:
        raise ValueError('pre-F0 freeze drift')
    return f


def validate_job(directory):
    f=verify_freeze();directory=directory.resolve()
    if directory!=DEFAULT/f['protocol']['job_id']:raise ValueError('unregistered request path')
    j=read(directory/'job.json')
    if j!={**f['job'],'protocol':f['protocol'],'freeze_sha256':sha(FREEZE)}:raise ValueError('job drift')
    return j
