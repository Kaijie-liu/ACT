"""Frozen full-obligation evidence study; no production verdict changes."""
import time

from scripts.conv_sign_lp_contract import (ROOT, RAW, read, save, sha, git,
    verify_freeze as verify_sign_freeze, source_identity as sign_sources)

PROTOCOL=ROOT/'scripts/conv_request_sign_lp_protocol.json'
DEFAULT=ROOT/'data/moe/results/conv_request_sign_lp_20260915_r1'
FREEZE=ROOT/'act/pipeline/moe/results/conv_request_sign_lp_freeze_20260915_r1.json'
PARENT=ROOT/'act/pipeline/moe/results/conv_sign_lp_review_20260915_r1.json'
FILES=('scripts/conv_request_sign_lp_protocol.json','scripts/conv_request_sign_lp_contract.py',
       'scripts/run_conv_request_sign_lp.py','scripts/check_conv_request_sign_lp.py',
       'scripts/test_conv_request_sign_lp.py','docs/conv_request_sign_lp_r1.md',
       'scripts/run_conv_sign_lp.py','scripts/check_conv_sign_lp.py','scripts/conv_sign_lp_contract.py')


def sources():
    return {**sign_sources(),**{p:sha(ROOT/p) for p in FILES}}


def parents():
    old=verify_sign_freeze(); protocol=read(PROTOCOL)
    if sha(PARENT)!=protocol['parent_sign_review_sha256']:raise ValueError('sign review changed')
    prior=read(PARENT)
    oldroot=ROOT/'data/moe/results/conv_sign_lp_20260915_r1'
    for v in prior['artifact_inventory']:
        if sha(oldroot/v['path'])!=v['sha256']:raise ValueError('old sign evidence changed')
    return protocol,old


def jobs():
    protocol,old=parents(); previous={j['case']['job_id']:j for j in old['jobs']}; result=[]
    for case in protocol['cases']:
        j=previous[case['job_id']];req=j['parent_request']
        if req['sample']['dataset_index']!=case['dataset_index']:raise ValueError('fixed input mismatch')
        result.append({'case':case,'parent_request':req,'parent_request_sha256':j['parent_request_sha256'],
                       'competitors':[i for i in range(10) if i!=req['sample']['label']]})
    return result


def verify_freeze():
    f=read(FREEZE)
    if f['protocol']!=parents()[0] or f['sources']!=sources() or f['jobs']!=jobs():
        raise ValueError('full-obligation freeze drift')
    return f


def validate_job(directory):
    f=verify_freeze();directory=directory.resolve()
    expected=next((j for j in f['jobs'] if j['case']['job_id']==directory.name),None)
    if directory.parent!=DEFAULT or expected is None:raise ValueError('outside frozen requests')
    job=read(directory/'job.json')
    if job!={**expected,'protocol':f['protocol'],'freeze_sha256':sha(FREEZE)}:raise ValueError('request job drift')
    req=job['parent_request']
    if sha(req['config']['path'])!=req['config']['sha256']:raise ValueError('original config drift')
    return job


def publication_gate():
    head=git('rev-parse','HEAD')
    remote=git('ls-remote','--exit-code','origin','refs/heads/feat/moe-route-verification')
    fields=remote.split()
    if len(fields)!=2 or fields[0]!=head or fields[1]!='refs/heads/feat/moe-route-verification':
        raise ValueError('freeze is not confirmed on remote; do not launch')
    return {'local_head':head,'remote_head':fields[0],'remote_ref':fields[1],
            'confirmed_unix':time.time(),'gate':'REMOTE_EQUALS_LOCAL_BEFORE_RUN_ROOT_CREATED'}
