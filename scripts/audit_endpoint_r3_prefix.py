"""Read-only saved R3 failure audit. No solving, factories, signals or retries.

Imports only stdlib and stdlib-only record contracts. Preserves late controller
publication as evidence of incomplete cleanup, never promotes it to online PASS.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scripts.audit_endpoint_midpoint_controls import audit as saved_audit
from scoped_source.endpoint_batch_observer import receive

RUN=Path('/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261020_r3')
FREEZE='192016eb5c8b3383a9d417610421c86ba57f4aec1fb94249dcf178f4b66dce6a'


def packed(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b): return hashlib.sha256(b).hexdigest()
def need(ok,message):
    if not ok: raise ValueError(message)


def main():
    begin=time.monotonic(); deadline=begin+60
    inventory={}
    def read(path,cap=2**20):
        need(time.monotonic()<deadline,'offline audit deadline')
        path=Path(path); s=path.stat()
        need(path.is_file() and not path.is_symlink() and s.st_nlink==1 and s.st_size<=cap,
             'bounded unaliased saved file')
        with path.open('rb') as stream: raw=stream.read(cap+1)
        need(len(raw)==s.st_size<=cap,'stable saved file size')
        return raw
    def obj(path,cap=2**20):
        raw=read(path,cap); return json.loads(raw),sha(raw)
    f,fh=obj(RUN/'freeze.json',131072); need(fh==FREEZE,'R3 identity')
    need(read(ROOT/'docs/hz_endpoint_midpoint_freeze_r3_20261020.json',131072)==packed(f),'same frozen file')
    stdout=read(str(RUN)+'.stdout.txt',131072); lines=stdout.splitlines()
    need(len(lines)==2,'both host records')
    e,x=map(json.loads,lines); caller=e['caller']
    need(e['schema']=='ENDPOINT_PROVISIONED_HOST_RETURN_V1'
         and e['caller_sha256']==sha(packed(caller)) and e['live_receipt'] is True,
         'original live caller record')
    need(x['schema']=='ENDPOINT_ROOT_EXPORT_COST_V1' and x['envelope_sha256']==sha(lines[0])
         and x['within_original_budget'] is True and 0<=x['export_seconds']
         and x['request_through_export_seconds']==e['request_seconds']+x['export_seconds']<2100,
         'first line export cost inside original batch budget')
    p=e['provisioning']
    need(p['full_cold_start_seconds'] is None and p['interpreter_startup_seconds'] is None
         and p['missing_is_not_zero'] is True and 0<=p['observed_prefix_seconds']<60
         and p['scope']=='STANDARD_LIBRARY_ROOT_ONLY_NO_REQUEST_WORK','provisioning boundary')
    need(e['host']['python']==f['records'][0]['profile']['environment']['python']
         and e['host']['flags']==['-I','-S','-B'],'same root host')
    stderr=read(str(RUN)+'.stderr.txt',65536); need(not stderr,'empty root stderr')
    a,ah=obj(RUN/'admission.json'); t,th=obj(RUN/'batch-received.json',65536)
    need(caller['online_batch_acceptance'] is False and caller['batch_sha256'] is None
         and caller['seconds']==e['request_seconds'],'negative live receipt')
    receive(caller,e['caller_sha256'],t,th,a,ah,None)
    prefix=saved_audit(RUN,f,FREEZE)  # No caller argument with missing batch.json.
    need(prefix['status']=='INCOMPLETE_OR_FAILED' and not prefix['online_batch_acceptance'],
         'missing final records never promoted')
    run=t['owned_execution']
    need(run['status']==run['cleanup_status']=='CLEANUP_INCOMPLETE'
         and run['error']=='SESSION_CLEANUP_ValueError' and run['returncode'] is None
         and run['sampled_session_plus_caller_rss']<=f['limits']['rss_bytes']
         and t['candidate_complete'] is False and not t['worker_summary_received'],
         'retained failed outer observation')
    positions=sorted(p.name for p in (RUN/'runs').iterdir())
    need(positions==['components'],'one attempted position only')
    cr=RUN/'runs/components'
    need(not (cr/'slot.json').exists() and not (RUN/'batch.json').exists(),'no accepted slot/final batch')
    inv,ih=obj(cr/'component-invocation.json',131072)
    actor,ch=obj(cr/'component.json'); controller,coh=obj(cr/'controller.json')
    need(inv['freeze']==f and inv['freeze_sha256']==fh
         and inv['owner_session']==run['session'] and actor['invocation']==controller['invocation']==ih
         and controller['component_sha256']==ch,'complete component source binding')
    code=read(ROOT/'scripts/endpoint_midpoint_components.py',131072)
    need(sha(code)==f['code']['scripts/endpoint_midpoint_components.py'],'frozen component code')
    tree=ast.parse(code)
    ids=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(v,ast.Name) and v.id=='IDS' for v in n.targets))
    out=actor['output']
    need(actor['status']=='COMPLETE' and actor['error'] is None and actor['work_complete'] is True
         and out['status']=='PASS' and out['binding']==sha(packed(f['component']))
         and [r['name'] for r in out['tests']]==list(ids) and len(ids)==22
         and out['source_factory_calls']==3 and out['successful_search_tiles']==3
         and out['failed_search_tiles']==1 and out['fixed_package_optimizer_calls']==0
         and out['models']==out['gpu']==0
         and actor['work']==out['work']==controller['work'],'actor recorded complete controls')
    need(inv['started']<=actor['begin']<=actor['end']<caller['received_at']<controller['end']
         and run['finished_at']<controller['end']<inv['run_deadline'],
         'late controller cannot repair outer cleanup failure')
    for rel,limit in (('components',out['storage']),('',controller['storage'])):
        source=cr/rel
        need(limit['bytes']==sum(limit['files'].values()) and set(limit['files'])==set(limit['hashes']),
             'complete declared prefix inventory')
        for name,size in limit['files'].items():
            target=source/name
            need(target.resolve().is_relative_to(cr.resolve()),'local prefix file')
            raw=read(target); need(len(raw)==size and sha(raw)==limit['hashes'][name],'saved prefix unchanged')
    for path in sorted(RUN.rglob('*')):
        need(not path.is_symlink(),'no late symlink')
        if not path.is_file(): continue
        raw=read(path); inventory[str(path.relative_to(RUN))]=dict(bytes=len(raw),sha256=sha(raw))
        need(len(inventory)<=4096 and sum(r['bytes'] for r in inventory.values())<=128*2**20,
             'bounded entire retained failure inventory')
    missing=[r['id'] for r in f['records'][1:]]
    record=dict(schema='ENDPOINT_R3_SAVED_FAILURE_RECORDS_V1',freeze=fh,
        admission=a,terminal=t,actor=actor,controller=controller,
        invocation_without_freeze={k:v for k,v in inv.items() if k!='freeze'},invocation_sha256=ih,
        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),inventory=inventory)
    result=dict(schema='ENDPOINT_R3_PREFIX_AUDIT_V1',status='VERIFIED_FAILED_BATCH_NOT_ALGORITHM_RESULT',
        execution_commit='fb42be8b5590f5b19f2266960cbe282edc8fca33',freeze=fh,
        records_sha256=sha(packed(record)),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),
        saved_audit=prefix,admissions=1,positions_started=1,positions_accepted=0,
        actor_reported_controls=22,component_position_accepted=False,
        controls_scope='SAVED_ACTOR_REPORT_AND_BINDINGS_NOT_REEXECUTED',
        normal_comparisons_executed=0,unexecuted=missing,models=0,gpu=0,retries=0,
        online_batch_acceptance=False,work_complete=False,work=None,
        component_recorded_work=actor['work'],component_actor_seconds=actor['end']-actor['begin'],
        request_seconds=e['request_seconds'],export_seconds=x['export_seconds'],
        request_through_first_export_seconds=x['request_through_export_seconds'],
        second_export_and_process_exit_seconds=None,full_cold_start_seconds=None,
        provision_observed_prefix_seconds=p['observed_prefix_seconds'],
        late_controller_after_caller_seconds=controller['end']-caller['received_at'],
        quiescent_cleanup_seconds=None,cleanup_complete=False,
        sampled_peak_bytes=run['sampled_session_plus_caller_rss'],
        saved_artifact_bytes=sum(r['bytes'] for r in inventory.values()),saved_artifact_files=len(inventory),
        scientific_decision='UNTESTED_NO_PAIRED_EFFECT_DATA',
        error_detail='ValueError cause/phase/PID not saved; stat/status exit race plausible but not established',
        offline_audit_seconds=time.monotonic()-begin)
    print(packed(dict(result=result,records=record)).decode())


if __name__=='__main__': main()
