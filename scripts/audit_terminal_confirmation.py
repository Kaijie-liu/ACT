"""Saved-record consistency audit only; never executes controls or native code."""

def check(result,freeze):
    issues=[]
    def need(ok,message):
        if not ok: issues.append(message)
    need(result['schema']=='TERMINAL_CONTROL_V1','schema')
    need(result['caps']==freeze['caps'],'caps')
    need(result['deadline']==result['started']+300,'shared deadline')
    need(result['started']<=result.get('admission',{}).get('at',0)<=result['started']+60,'admission')
    need(result['before_export']<result['deadline'],'export cutoff')
    need(result['algorithm_work_consumed']==0 and not any(result[k] for k in
        ('algorithm_executed','model_executed','gpu_executed')),'scope')
    need(result['peak_sampled_rss_bytes']<=freeze['caps']['rss_bytes'],'sampled RSS')
    cases=result['cases']; sessions=result['sessions']
    need([r['id'] for r in cases]==freeze['cases'][:len(cases)],'case prefix')
    need([r['id'] for r in sessions]==freeze['native'][:len(sessions)],'native prefix')
    need(result['unexecuted_cases']==freeze['cases'][len(cases):],'case missingness')
    need(result['unexecuted_native']==freeze['native'][len(sessions):],'native missingness')
    refusals={
        'default_missing_reject':'ValueError','no_events':'ValueError','hup_only':'ValueError',
        'poll_error':'ValueError','poll_invalid':'ValueError','wrong_fd':'ValueError',
        'multiple_events':'ValueError','changed_sid':'ValueError','changed_start':'ValueError',
        'binding_disappeared':'FileNotFoundError','binding_permission':'PermissionError',
        'open_permission':'PermissionError','poll_permission':'PermissionError',
        'malformed_binding':'ValueError','oversized_binding':'ValueError','malformed_rss':'ValueError',
        'deadline_before':'TimeoutError','deadline_after_poll':'TimeoutError'}
    for r in cases:
        if r['status']!='PASS': continue
        d=r['detail']
        if r['id'] in freeze['cases'][:-2]:
            required={'fd_closed','poll_waits_ms','opens','reads','error','observation'}
            need(isinstance(d,dict) and required<=d.keys(),'control detail '+r['id'])
            if not isinstance(d,dict) or not required<=d.keys(): continue
            need((None if d['error'] is None else d['error']['kind'])==refusals.get(r['id']),
                 'expected refusal '+r['id'])
            need(d['fd_closed'] and len(d['poll_waits_ms'])<=1 and
                all(0<=x<=10 for x in d['poll_waits_ms']),'bounded confirmation '+r['id'])
            need(d['opens'] in (0,1),'at most one descriptor '+r['id'])
            no_open={'known_rss','default_missing_reject','terminal_Z','terminal_X','terminal_x',
                     'malformed_rss','deadline_before'}
            no_poll=no_open|{'changed_sid','changed_start','binding_disappeared','binding_permission',
                            'open_permission','malformed_binding','oversized_binding'}
            need(d['opens']==(0 if r['id'] in no_open else 1),'open coverage '+r['id'])
            need(len(d['poll_waits_ms'])==(0 if r['id'] in no_poll else 1),'poll coverage '+r['id'])
            reads=0 if r['id']=='deadline_before' else (2 if r['id']=='malformed_rss' else
                (7 if r['id']=='terminal_with_live_peer' else
                 (3 if r['id'] in no_open|{'open_permission'} else 4)))
            need(d['reads']==reads,'read coverage '+r['id'])
            if r['id']=='deadline_after_poll':
                need(len(d['poll_waits_ms'])==1 and
                     d['error']['phase']=='terminal_pidfd_confirmation','actual post-poll deadline')
            if d['observation']:
                for record in d['observation'].get('terminal_without_rss',[]):
                    need(record['rss'] is None and record['evidence']=='IDENTITY_BOUND_PIDFD_POLLIN'
                        and record['poll_events']==1,'terminal evidence '+r['id'])
                    need(record['identity']['sid']==101 and record['identity']['start_ticks']==55,
                         'terminal identity '+r['id'])
            if r['id'] not in refusals:
                o=d['observation']
                need(isinstance(o,dict),'required successful observation '+r['id'])
                if isinstance(o,dict):
                    terminal=r['id'] in {'confirmed_missing','confirmed_status_gone','terminal_with_live_peer'}
                    live=[101] if r['id']=='known_rss' else ([102] if r['id']=='terminal_with_live_peer' else [])
                    zombies=[101] if r['id'] in {'terminal_Z','terminal_X','terminal_x'} else []
                    need(o.get('live')==live and o.get('zombies')==zombies and
                         o.get('rss')==(12288 if live else 0) and
                         len(o.get('terminal_without_rss',[]))==int(terminal),'successful classification '+r['id'])
            if r['id']=='terminal_with_live_peer':
                need(isinstance(d['observation'],dict) and
                     d['observation'].get('live')==[102] and d['observation'].get('rss')==12288,
                     'terminal confirmation does not remove live peer')
        elif r['id']=='partial_scan_and_first_errors':
            need(d['execution_error']['kind']=='PermissionError' and
                 d['cleanup_error']['kind']=='ValueError' and
                 [x['pid'] for x in d['terminal_without_rss']]==[101,103],
                 'partial error and evidence retention')
        elif r['id']=='bounded_terminal_records':
            need(d['retained']==512 and d['overflow_error']=='bounded terminal evidence'
                 and d['original_error']=='PermissionError'
                 and d['retention_error']['kind']=='ValueError','evidence cap and original error')
    for row in sessions:
        r=row['receipt']; name=row['id']
        need(r['observation_contract']=='CONFIRMED_IDENTITY_V3','new observation binding')
        need(r['cleanup_contract']=='IDENTITY_ONLY_V1','unchanged cleanup')
        need(result['started']<=r['started_at']<=r['finished_at']<result['deadline'],'session time')
        need(r['started_at']<=r['run_deadline']<=r['cleanup_deadline']<=result['deadline'],'nested deadlines')
        need(r['finished_at']<=r['cleanup_deadline'],'session final deadline')
        need(0<=r['sampled_session_plus_caller_rss']<=result['peak_sampled_rss_bytes'],'peak composition')
        records=r.get('terminal_without_rss',[])
        need(len(records)<=512,'native evidence cap')
        identities=set()
        for record in records:
            identity=record['identity']; missing=record['missing_observation']
            need(record['rss'] is None and record['evidence']=='IDENTITY_BOUND_PIDFD_POLLIN'
                 and record['poll_events'] in (1,17) and 0<=record['poll_limit_ms']<=10,
                 'native terminal evidence')
            need(identity['sid']==r['session'] and
                 r['started_at']<=record['confirmed_at']<=r['finished_at'],'native terminal scope')
            need(missing['rss_present'] is False and
                 all(missing[k]['sid']==identity['sid'] and
                     missing[k]['start_ticks']==identity['start_ticks']
                     for k in ('first_identity','last_identity')),'native missing identity binding')
            identities.add((record['pid'],identity['sid'],identity['start_ticks']))
        need(len(identities)==len(records),'deduplicated native evidence')
        if result['status']=='PASS':
            need(r['status']==('ERROR' if name=='native_seven' else 'COMPLETED'),'native status')
            need(r['returncode']==(7 if name=='native_seven' else 0),'native exit code')
            need(r['execution_error'] is None and r['cleanup_error'] is None,'native observation errors')
            need(r['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
                and not r['remaining_session']['live'],'native cleanup')
    if result['status']=='PASS':
        need(len(cases)==len(freeze['cases']) and all(r['status']=='PASS' for r in cases),'full controls')
        need(len(sessions)==len(freeze['native']),'full native roster')
        need(result.get('initial_fds')==result.get('final_fds'),'FD balance')
        need(result.get('owned_children_remaining')==[],'no direct children')
        need('failure' not in result,'no failed PASS')
    return dict(schema='TERMINAL_SAVED_AUDIT_V1',consistent=not issues,issues=issues,
                accepted=result['status']=='PASS' and not issues,
                native_confirmation_events=sum(len(s['receipt'].get('terminal_without_rss',[])) for s in sessions),
                independent_scope='saved-record consistency, not reexecution or algorithm proof')
