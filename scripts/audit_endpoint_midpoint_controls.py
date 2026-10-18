"""Saved-record audit of one finite batch; never re-solves or rescues online failure."""
import argparse
import json
import math
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, digest, identity, require
from scoped_source.native_protocol import (read_json, counts, COUNTERS, STAGES,
    validate_receipt, REPORT_CAP, LOG_CAP, META_CAPS)
from scoped_source.endpoint_batch import (validate, ORDER, FREEZE_CAP, RECEIPT_CAP,
    COMPONENT_META, BATCH_META_CAP)


def tick(deadline):
    if deadline is not None: require(time.monotonic()<deadline,'bounded terminal audit')


def marker(path, name, deadline):
    tick(deadline)
    require(not path.is_symlink() and path.is_file() and path.stat().st_nlink==1
            and path.stat().st_size<=LOG_CAP,'bounded fault log')
    raw=path.read_bytes(); require(len(raw)<=LOG_CAP,'fault log grew')
    found=[]
    for line in raw.splitlines():
        tick(deadline)
        try: r=json.loads(line)
        except (ValueError,UnicodeError): continue
        if type(r) is dict and r.get('control')==name: found.append(r)
    require(len(found)==1,'unique actual fault arrival marker')
    return found[0]


def group(run):
    require(run['schema']=='OWNED_BOUNDED_CPU_V1'
            and run['cleanup_status']=='LEADER_REAPED_NO_LIVE_GROUP'
            and not run['remaining_group']['live'] and run['sampled_peak_rss']<=2*2**30,
            'owned cleanup and sampled group plus owner RSS')


def inspect(root, spec, freeze, observed, started, *, deadline=None):
    """Bounded online audit also reused offline. No native or exact math execution."""
    root=Path(root); tick(deadline); profile=spec['profile']; run=observed['owned_execution']
    group(run)
    require(run['run_deadline']==started+298 and run['cleanup_deadline']==started+298.5,
            'same whole-position execution/cleanup schedule')
    if spec['id']=='components':
        inv,ih=read_json(root/'component-invocation.json',expected=observed['invocation'],cap=FREEZE_CAP,deadline=deadline)
        require(inv['freeze']==freeze and inv['freeze_sha256']==identity(freeze)
                and inv['started']==started and inv['outer_deadline']==started+300,'component invocation')
        require(run['status']=='COMPLETED','component process complete')
        r,rh=read_json(root/'component.json',deadline=deadline)
        c,_=read_json(root/'controller.json',deadline=deadline)
        require(r['schema']=='ENDPOINT_COMPONENT_ACTOR_V1' and r['status']=='COMPLETE'
                and r['invocation']==ih and r['work_complete'] is True
                and started<=r['begin']<=r['end']<started+298
                and c['schema']=='ENDPOINT_COMPONENT_CONTROLLER_V1' and c['status']=='COMPLETE'
                and c['invocation']==ih and c['component_sha256']==rh and c['work']==r['work']
                and c['end']<started+298,'timely complete component chain')
        out=r['output']; from scripts.endpoint_midpoint_components import IDS
        require(out['schema']=='ENDPOINT_COMPONENT_RESULT_V1' and out['status']=='PASS'
                and out['binding']==identity(freeze['component']) and out['initial_work']==dict.fromkeys(COUNTERS,0)
                and [t['name'] for t in out['tests']]==list(IDS)
                and out['source_factory_calls']==3 and out['successful_search_tiles']==3
                and out['failed_search_tiles']==1 and out['fixed_package_optimizer_calls']==0
                and out['models']==out['gpu']==0 and out['work']==r['work'],'all finite component obligations')
        counts(r['work'])
        return dict(kind='COMPONENT_CONTROL',work_complete=True,work=r['work'],result=None,
                    all_cases=22,guarantee='synthetic controls, not model certification')
    inv,ih=read_json(root/'invocation.json',expected=observed['invocation'],deadline=deadline)
    require(inv['profile']==profile and inv['started']==started and inv['run_deadline']==started+298
            and inv['cleanup_deadline']==started+298.5 and inv['deadline']==started+299
            and inv['outer_deadline']==started+300 and inv['approval']==freeze['native_registry'],
            'native invocation with charged outer reserve')
    terminal,th=read_json(root/'terminal.json',deadline=deadline)
    require(terminal['invocation']==ih and terminal['owned_execution']==run
            and observed['attempted_terminal_ref']['sha256']==th,'durable pre-return native terminal')
    require(observed['overrun_seconds']==0,'no late native acceptance')
    if spec['id'] not in ('midpoint_block','partial_candidate'):
        require(observed['online_acceptance'] is True and observed['status']=='COMPLETE'
                and observed['work_complete'] is True and run['status']=='COMPLETED','observed complete normal control')
        initial=dict.fromkeys(COUNTERS,0); previous=ih; outputs=[]
        for phase in STAGES:
            r,h=read_json(root/(phase+'.json'),deadline=deadline)
            initial=validate_receipt(r,stage=phase,invocation=ih,previous=previous,initial=initial,
                                     deadline=started+298,started=started)
            require(r['limits']['address_space_bytes']==8*2**30,'same actor AS envelope')
            previous=h; outputs.append(r['output'])
        require(initial==observed['cumulative_work'] and all(o['result']==observed['result'] for o in outputs),
                'all obligations independently received, not necessarily positive')
        result=observed['result']; stats=outputs[0]['snapshots']['native']
        manifest,_=read_json(root/'artifacts/package/manifest.json',deadline=deadline)
        tiles=len(manifest['tiles'])
        require(stats['setup_calls']==stats['close_calls']==1 and stats['propose_calls']==tiles
                and result['missing_endpoints']==0
                and result['checked_endpoints']==result['required_endpoints'],'complete candidate/control roster')
        if 'search' in stats:
            s=stats['search']; paired=s['midpoint_updates']
            require(s['method']==identity(profile['method'])
                    and s['kind']==profile['method']['producer']['kind']
                    and s['evaluations']==129*tiles and s['forward_calls']==128*tiles
                    and type(paired) is int and 0<=paired<=tiles
                    and s['gradient_updates']==128*tiles-paired
                    and s['midpoint_pairs']==result['required_endpoints']-result['required']
                    and paired<=s['midpoint_pairs']<=4*paired,'aggregate completed search accounting')
        else:
            require(spec['id'].endswith('_baseline'),'midpoint search counters required')
        c,_=read_json(root/'controller.json',deadline=deadline)
        require(c['invocation']==ih and c['status']=='COMPLETE' and c['last_receipt']==previous
                and c['work']==initial and c['result']==observed['result']
                and c['received_all_obligations'] is True and c['end']<started+298,'controller bound to actor chain')
        return dict(kind='NORMAL_COMPLETE',work_complete=True,work=initial,result=observed['result'],
            native=stats,phases=outputs[0]['phases'],
            guarantee='checked declared synthetic source, no checkpoint or floating execution proof')
    require(observed['online_acceptance'] is False and observed['work_complete'] is False
            and observed['cumulative_work'] is None,'fault never accepts partial evidence')
    for name in ('replay.json','reception.json','controller.json','artifacts/package/manifest.json',
                 'artifacts/package/bundle.json','artifacts/package/result/index.json'):
        require(not (root/name).exists(),'no later phase or complete package after fault')
    fault=spec['id']; m=marker(root/'production.log',fault,deadline)
    contract,_=read_json(root/'artifacts/package/contract.json',deadline=deadline)
    source,_=read_json(root/'artifacts/package/source/manifest.json',expected=profile['source']['source'],deadline=deadline)
    require(m['method']==identity(profile['method']) and m['request']==contract['request']
            ==identity(source['declaration']['request']) and contract['source']==profile['source']['source'],
            'fault reached bound source and objective plan')
    prefix=counts(m['work'])
    if fault=='midpoint_block':
        s=m['search']
        require(m['step']==127 and m['pair']==[0,1] and s['method']==identity(profile['method'])
                and [s[k] for k in ('evaluations','forward_calls','gradient_updates','midpoint_updates')]
                    ==[128,128,127,0] and run['status']=='TIMEOUT'
                and (run['exit_observed_at'] is None or type(run['exit_observed_at']) in (int,float)
                     and math.isfinite(run['exit_observed_at']) and run['exit_observed_at']>=started+298)
                and run['started_at']+run['execution_seconds']>=started+298,
                'actual noncooperative midpoint deadline, not setup failure')
        require(not (root/'production.json').exists(),'blocked actor has no completed receipt')
    else:
        require(m['invocation']==ih and m['source']==profile['source']['source'] and m['ordinal']==0
                and m['expected_bytes']==m['actual_bytes']+1 and m['actual_bytes']>0
                and m['expected_sha256']!=m['actual_sha256'],'bound truncated first frame')
        path=root/'artifacts/package/tiles/0000.bin'
        require(not path.is_symlink() and path.stat().st_size==m['actual_bytes']<=4*2**20,
                'retained partial frame length')
        tick(deadline); require(digest(path.read_bytes())==m['actual_sha256'],'retained truncated frame hash')
        r,_=read_json(root/'production.json',deadline=deadline)
        require(r['status']=='ERROR' and r['invocation']==ih and r['output'] is None
                and r['error']['kind']=='OSError' and 'CONTROL_INJECTED_partial_candidate_publication' in r['error']['message']
                and run['status']=='ERROR','intended candidate publication exception')
        after=counts(r['work'],complete=False)
        require(all(after[k]>=prefix[k] for k in COUNTERS),'fault nonresetting observed Work')
        prefix=after
    return dict(kind='EXPECTED_FAULT_REJECTED',work_complete=False,work=None,
        observed_prefix_work=prefix,unknown_remainder=True,result=None,
        guarantee='failure control only, never an online proof')


def storage_caps(storage, component, *, reserve_receipt=False):
    require(storage['bytes']==sum(storage['files'].values()) and set(storage['files'])==set(storage['hashes']),
            'complete byte/member ledger')
    require(storage['bytes']+(RECEIPT_CAP if reserve_receipt else 0)<=POLICY['total_bytes']
            and len(storage['files'])+int(reserve_receipt)<=POLICY['members'],'whole position storage cap')
    caps=COMPONENT_META if component else dict(META_CAPS,**{'slot.json':RECEIPT_CAP})
    for n,cap in caps.items():
        if n in storage['files']: require(storage['files'][n]<=cap,'metadata member bound: '+n)


def audit(directory, freeze, expected):
    validate(freeze,expected,authorize=False,verify=False)
    root=Path(directory); b,_=read_json(root/'batch.json')
    saved,_=read_json(root/'freeze.json',expected=expected,cap=FREEZE_CAP)
    require(saved==freeze,'archived freeze identity')
    require(set(b)=={'schema','freeze','started','admission_sha256','positions','stop','unexecuted',
                    'before_publish','models','gpu','retries','quota_increases'}
            and b['schema']=='ENDPOINT_FINITE_BATCH_V1' and b['freeze']==expected
            and all(type(b[k]) is int and b[k]==0 for k in ('models','gpu','retries','quota_increases')),
            'bound finite batch receipt')
    members=list(root.iterdir())
    require({p.name for p in members}=={'runs','freeze.json','admission.json','batch.json','batch-received.json'}
            and all(not p.is_symlink() for p in members) and (root/'runs').is_dir(),
            'exact batch root inventory')
    a,ah=read_json(root/'admission.json')
    require(ah==b['admission_sha256'] and a['freeze']==expected and a['attempts']==1
            and a['seconds']<=60 and a['available_bytes']>=freeze['admission']['available_bytes']
            and a['disk_free_bytes']>=freeze['admission']['disk_free_bytes'],'one recorded admission')
    require([x['id'] for x in b['positions']]==list(ORDER[:len(b['positions'])])
            and len(b['positions'])<=7 and b['unexecuted']==list(ORDER[len(b['positions']):]),
            'one ordered prefix, all missing positions retained')
    dirs=list((root/'runs').iterdir()); attempted={r['id'] for r in b['positions']}
    require(all(p.is_dir() and not p.is_symlink() and p.name in attempted for p in dirs),
            'no extra or aliased run directory')
    checks=[]; problems=[]; previous=b['started']
    from scoped_source.direct_supervised import census
    for row,spec in zip(b['positions'],freeze['records']):
        run=root/'runs'/row['id']
        require(set(row)=={'id','ok','started','observed_end','seconds','overrun_seconds','error','receipt_sha256'}
                and type(row['ok']) is bool,'closed observed position')
        require(row['started']>=previous and row['observed_end']>=row['started'],'nonoverlapping position intervals')
        previous=row['observed_end']
        if not row['ok'] or row['receipt_sha256'] is None:
            problems.append(dict(id=row['id'],reason='failed or missing durable online receipt',
                storage_complete=False,storage_bytes=None,work_complete=False,work=None,
                cost_scope='UNKNOWN_NOT_ZERO; retained prefix may be audited separately, never rescued'))
            require(row==b['positions'][-1],'stop on first unexpected failure'); continue
        r,_=read_json(run/'slot.json',expected=row['receipt_sha256'],cap=RECEIPT_CAP)
        require(r['freeze']==expected and r['id']==row['id'] and r['profile']==spec['sha256']
                and r['schema']=='ENDPOINT_POSITION_RECEIPT_V1'
                and r['started']==row['started'] and r['status']=='PASS' and r['error'] is None
                and row['overrun_seconds']==0 and r['storage_complete'] is True
                and r['before_publish']<=row['observed_end']<row['started']+300
                and row['seconds']==row['observed_end']-row['started'],'durable whole-position reception')
        result=inspect(run,spec,freeze,r['observed'],r['started'])
        require(result==r['audit'],'independent terminal recomputation')
        actual=census(run); storage_caps(actual,row['id']=='components')
        prior={k:v for k,v in actual['files'].items() if k!='slot.json'}
        require(prior==r['storage']['files'] and sum(prior.values())==r['storage']['bytes']
                and {k:v for k,v in actual['hashes'].items() if k!='slot.json'}==r['storage']['hashes'],
                'all retained artifacts including failed prefixes')
        checks.append(dict(id=row['id'],seconds=row['seconds'],bytes=actual['bytes'],audit=result,
                           sampled_managed_phase_rss=r['observed']['owned_execution']['sampled_peak_rss']))
    # Cumulative init, every position and final metadata reception are all charged.
    terminal,th=read_json(root/'batch-received.json',cap=RECEIPT_CAP)
    require(terminal['batch_sha256']==identity(b) and terminal['freeze']==expected
            and b['before_publish']<=terminal['observed_at'] and b['started']<=previous,
            'batch durable reception binding')
    from os import scandir
    metadata={e.name:e.stat(follow_symlinks=False).st_size for e in scandir(root) if e.is_file(follow_symlinks=False)}
    require(set(metadata)=={'admission.json','freeze.json','batch.json','batch-received.json'}
            and sum(metadata.values())<=BATCH_META_CAP,'bounded separate batch metadata')
    consistent=(not problems and len(checks)==7 and not b['unexecuted'] and b['stop'] is None
              and terminal['observed_at']<b['started']+2100 and terminal['timely'] is True)
    # A pre-write timestamp cannot attest its own write/fsync. No authenticated
    # post-return observer is bound yet: even seven consistent slots stay pending.
    # Future execution freeze must wire that finite caller, not remove this gate.
    return dict(schema='ENDPOINT_FINITE_AUDIT_V1',
        status='CONSISTENT_PENDING_CALLER_OBSERVATION' if consistent else 'INCOMPLETE_OR_FAILED',
        checks=checks,problems=problems,unexecuted=b['unexecuted'],
        seconds=terminal['observed_at']-b['started'],batch_metadata_bytes=sum(metadata.values()),
        online_batch_acceptance=False,post_return_observation_bound=False,
        models=0,gpu=0,retries=0,scope='saved terminal audit, not a scientific efficacy gate or independent human review')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('freeze'); p.add_argument('--sha256',required=True)
    args=p.parse_args(); freeze,_=read_json(args.freeze,expected=args.sha256,cap=FREEZE_CAP)
    from scoped_source.hz_request_store import compact
    print(compact(audit(args.directory,freeze,args.sha256)).decode())
