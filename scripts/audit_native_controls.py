"""Read-only terminal/cost audit. Not independent re-proof or online rescue."""
import argparse
import json
from pathlib import Path

from scoped_source.hz_request_store import digest, identity, compact, require, POLICY


def fault_reached(run, spec, result, elapsed):
    run=Path(run); fault=spec['profile']['control']
    text='\n'.join(p.read_text(errors='replace') for pattern in ('*.log','*.progress','production.json','replay.json','reception.json')
                   for p in run.glob(pattern))
    markers=dict(source_binding='frozen generated source',setup_error='CONTROL_INJECTED_native_setup',
        proposal_block='propose_check/0',close_error='CONTROL_INJECTED_native_close',
        partial_candidate='CONTROL_INJECTED_partial_candidate',replay_tamper='CONTROL_INJECTED_replay_integrity_tamper',
        reception_block='CONTROL_INJECTED_reception_block',limit_mismatch='hard limits inherited and verified')
    if fault in markers: require(markers[fault] in text,'intended fault stage not reached: '+fault)
    owned=result['owned_execution']
    actual=owned.get('actual_cleanup_status',owned['cleanup_status'])
    require(actual=='LEADER_REAPED_NO_LIVE_GROUP' and not owned['remaining_group']['live'],'actual controlled group cleanup')
    if fault in ('proposal_block','reception_block'):
        require(owned['status']=='TIMEOUT' and elapsed>=299,'actual shared watchdog deadline')
    if fault=='cleanup_unconfirmed': require(owned.get('cleanup_fault_simulated') is True,'explicit cleanup simulation')
    if fault=='late_receipt':
        inv=json.loads((run/'invocation.json').read_bytes())
        require(json.loads((run/'controller.json').read_bytes())['end']>inv['deadline'],'late timestamp injected')
    return True


def audit(directory, freeze_path):
    root=Path(directory); raw=Path(freeze_path).read_bytes(); freeze=json.loads(raw)
    batch=json.loads((root/'batch.json').read_bytes()); require(batch['freeze_sha256']==digest(raw),'batch freeze')
    roster={r['id']:r for r in freeze['records']}; seen=set(); checks=[]; problems=[]; pairs={}
    for row in batch['outcomes']:
        name=row['id']; require(name in roster and name not in seen,'one attempt per frozen position'); seen.add(name)
        payload=(root/'receipts'/(name+'.json')).read_bytes()
        require(digest(payload)==row['receipt_sha256'],'caller receipt identity')
        rec=json.loads(payload); spec=roster[name]; result=rec['result']; run=root/'runs'/name
        require(rec['profile']==spec['sha256']==identity(spec['profile']),'frozen profile binding')
        if result is None or not row['ok']:
            problems.append(dict(id=name,reason='incomplete/unexpected execution',error=rec['error'])); continue
        invraw=(run/'invocation.json').read_bytes(); inv=json.loads(invraw)
        require(result['invocation']==digest(invraw) and inv['profile']==spec['profile'],'invocation and caller bound')
        require(row['full_caller_seconds']<300 and result['caller_seconds']<=row['full_caller_seconds']
                and result['overrun_seconds']==0,'whole caller cost and deadline')
        if spec['kind']=='normal':
            require(result['online_acceptance'] and result['storage_complete'],'complete actual caller acceptance')
            cumulative={k:0 for k in ('row_visits','coefficient_visits','arithmetic')}; previous=digest(invraw)
            outputs=[]
            for stage in ('production','replay','reception'):
                data=(run/(stage+'.json')).read_bytes(); r=json.loads(data)
                require(r['status']=='COMPLETE' and r['invocation']==digest(invraw) and r['previous']==previous
                    and r['initial_work']==cumulative,'independent cumulative phase chain')
                require(all(r['work'][k]>=cumulative[k] and r['delta_work'][k]==r['work'][k]-cumulative[k] for k in cumulative),
                    'nonresetting work')
                require(r['begin']>=inv['started'] and r['end']<inv['run_deadline'],'timely phase')
                require(r['limits']['address_space_bytes']==8*2**30,'same AS limits')
                cumulative=r['work']; previous=digest(data); outputs.append(r['output'])
            require(sum(cumulative.values())<=POLICY['visits'] and result['cumulative_work']==cumulative,'original work cap')
            require(outputs[0]['result']==outputs[1]['result']==outputs[2]['result']==result['result'],'all required results recomputed')
            actual={p.relative_to(run).as_posix():p.stat().st_size for p in run.rglob('*') if p.is_file()}
            require(actual==result['storage']['files'] and sum(actual.values())==result['storage']['bytes']<=POLICY['total_bytes'],
                    'whole durable request storage')
            for n,h in result['storage']['hashes'].items(): require(digest((run/n).read_bytes())==h,'stored file identity')
            key=spec['profile']['source']['case']; pairs.setdefault(key,[]).append(dict(arm=spec['profile']['method']['producer']['representation'],
                result=result['result'],seconds=row['full_caller_seconds'],work=cumulative,bytes=sum(actual.values()),
                sampled_rss_with_parent=result['owned_execution']['sampled_peak_rss'],
                phases=outputs[0]['phases']))
        else:
            require(not result['online_acceptance'],'fault rejected')
            fault_reached(run,spec,result,row['full_caller_seconds'])
        checks.append(name)
    require(sorted(batch['unexecuted'])==sorted(set(roster)-seen),'all unexecuted positions retained')
    component_ok=False
    if batch['component'] is not None:
        cr=(root/'component-receipt.json').read_bytes()
        require(digest(cr)==batch['component_receipt_sha256'],'component caller receipt binding')
        component=json.loads(cr)
        component_ok=(component.get('status')=='COMPLETED' and component['full_caller_seconds']<300
            and component['cleanup_status']=='LEADER_REAPED_NO_LIVE_GROUP' and not component['remaining_group']['live']
            and component.get('report',{}).get('status')=='PASS' and len(component.get('report',{}).get('tests',[]))==32)
        if component_ok:
            require(json.loads((root/'components/report.json').read_bytes())==component['report'],'durable component report')
            require(sum(component['report']['work'].values())<=POLICY['visits'],'whole component Work')
            require(component['report']['address_space_bytes']==8*2**30,'actual component AS limit')
            actual={p.relative_to(root/'components').as_posix():p.stat().st_size
                    for p in (root/'components').rglob('*') if p.is_file()}
            stored=component['report']['storage_before_report']
            require({n:b for n,b in actual.items() if n!='report.json'}==stored['files']
                    and sum(stored['files'].values())==stored['bytes'],'component complete retained artifact inventory')
            for n,h in stored['hashes'].items():
                require(digest((root/'components'/n).read_bytes())==h,'component retained identity')
            require(sum(actual.values())+(root/'component.log').stat().st_size+len(cr)<=POLICY['total_bytes']
                    and len(actual)+2<=POLICY['members'],'whole component artifact and receipt cap')
    complete=batch['status']=='COMPLETE_PENDING_AUDIT' and not problems and len(checks)==26 and component_ok
    return dict(schema='NATIVE_FINITE_AUDIT_V1',status='PASS' if complete else 'INCOMPLETE_OR_FAILED',
        checked_requests=len(checks),planned_requests=26,problems=problems,unexecuted=batch['unexecuted'],pairs=pairs,
        component_status=None if batch['component'] is None else batch['component'].get('status','ERROR'),
        total_seconds=batch['seconds'],full_topology_admitted=False,real_model_runs=0,gpu_runs=0,
        guarantee='declared-source controls only; terminal audit not independent human review')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('freeze'); a=p.parse_args()
    print(compact(audit(a.directory,a.freeze)).decode())
