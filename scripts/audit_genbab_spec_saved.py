"""Saved-only structural audit; never reruns controls, imports torch or solves."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def raw(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def check(condition,message):
    if not condition: raise ValueError(message)


def audit(path):
    archive=json.loads(path.read_text())
    frozen_path=ROOT/'docs/genbab_spec_freeze_20261021.json'
    frozen=json.loads(frozen_path.read_text())
    anchor=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    check(archive['freeze_sha256']==anchor,'freeze identity')
    for name,digest in frozen['files'].items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'frozen source '+name)
    check(set(archive['documents'])=={'worker.json','session.json','result.json'},'complete core records')
    for name,record in archive['documents'].items():
        data=raw(record['value'])
        check(len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name)
    worker=archive['documents']['worker.json']['value']
    session=archive['documents']['session.json']['value']
    result=archive['documents']['result.json']['value']
    check(worker['status']==result['status']=='PASS','original online terminal')
    check(worker['freeze_sha256']==result['freeze_sha256']==anchor,'bound worker/terminal')
    check(result['session']==session,'owned receipt retained')
    check(result['worker_sha256']==archive['documents']['worker.json']['sha256'],'worker identity')
    check(result['caps']==frozen['caps'] and result['policy']==frozen['policy'],'resources')
    check(result['deadline']==result['entry']+300 and worker['deadline']==result['entry']+260,
          'nonresetting budget')
    check(worker['rlimit_as']==[frozen['caps']['as_bytes']]*2,'hard address-space limit')
    check(result['fd_before']==result['fd_after'],'FD balance')
    check(0<=result['seconds_before_export']<300,'terminal timing')
    check(session['status']=='COMPLETED' and session['returncode']==0
          and session['execution_error'] is None and session['cleanup_error'] is None
          and session['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
          and session['wait4_usage'] is not None,'owned exit and cleanup')
    records=worker['records']
    check([r['name'] for r in records]==frozen['tests'] and len(records)==result['tests']==25,
          'all preregistered controls')
    check(all(r['passed'] and not r['errors'] and 0<=r['seconds']<300 for r in records),
          'control outcomes')
    check(worker['native_calls']==result['native_calls']==result['real_stage_calls']==0,
          'no native/real calls')
    check(worker['synthetic'] and not worker['complete_network_proof']
          and not worker['mathematical_bound_checked'],'evidence scope')
    check(set(archive['prefixes'])=={'prefix_%02d.json'%i for i in range(1,26)},'all prefixes')
    for i in range(1,26):
        record=archive['prefixes']['prefix_%02d.json'%i]; data=raw(records[:i])
        check(len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],
              'lossless deduplicated prefix')
    original=sum(r['bytes'] for r in archive['documents'].values())+sum(r['bytes'] for r in archive['prefixes'].values())
    check(original+path.stat().st_size<frozen['caps']['artifact_bytes'],'run plus deduplicated archive')
    return dict(status='PASS',tests=25,original_bytes=original,archive_bytes=path.stat().st_size,
                execution_head=result['execution_head'],seconds_before_export=result['seconds_before_export'],
                scope='SAVED_STRUCTURAL_AUDIT_NOT_NETWORK_PROOF')


if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True))
