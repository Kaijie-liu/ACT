"""One four-group identity-cleanup repair control, no algorithm/model/GPU execution."""
import time
PROVISION_ENTRY=time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys

ROOT=Path('/data1/Kane/MOE/ACT')
PYTHON='/data1/Kane/miniconda3/envs/act-py312/bin/python'
FREEZE=ROOT/'docs/hz_owned_cleanup_freeze_20261020.json'
RUN=Path('/data1/Kane/MOE/results/owned_cleanup_20261020_r1')
IDS=('memory_diagnostics','identity_cleanup','signal_binding','execution_outcomes')
CAPS=dict(seconds=300,as_bytes=8*2**30,rss_bytes=2*2**30,artifact_bytes=128*2**20,work=8000000)
SUBCASES=dict(
    memory_diagnostics=['live_no_rss','status_gone_live','confirmed_zombie','known_rss','partial_memory_unknown'],
    identity_cleanup=['live_without_status','foreign_excluded','reused_pid','changed_sid',
                      'permission_denied','expired_deadline','partial_roster_unknown'],
    signal_binding=['same_identity_signal','foreign_reject','reused_pid_reject','deadline_no_open'],
    execution_outcomes=['execution_rejection_cleanup','two_errors','deadline_cleanup','normal_memory_acceptance'])


def packed(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    for kind,cap in ((resource.RLIMIT_AS,CAPS['as_bytes']),(resource.RLIMIT_FSIZE,CAPS['artifact_bytes'])):
        limits=resource.getrlimit(kind)
        require(all(v==resource.RLIM_INFINITY or v>=cap for v in limits),'no raising inherited limits')
        resource.setrlimit(kind,(cap,cap))
    started=time.monotonic(); deadline=started+300
    def hard_stop(_sig,_frame):
        os.write(2,b'OBSERVER_CONTROL_HARD_DEADLINE_NO_ACCEPTANCE\n'); os._exit(124)
    signal.signal(signal.SIGALRM,hard_stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    rows=[]; receipts=[]; reaps=[]; created=False; peak=0
    result=dict(schema='OWNED_CLEANUP_CONTROL_V1',status='FAIL',started=started,caps=CAPS,
                provisioning_seconds=started-PROVISION_ENTRY,full_cold_start_seconds=None,
                cases=rows,sessions=receipts,post_failure_reaps=reaps,
                algorithm_executed=False,model_executed=False,gpu_executed=False,
                seven_position_batch_authorized=False,algorithm_work_consumed=0)
    try:
        require(len(sys.argv)==3 and sys.argv[1]=='--execute-approved-once','explicit opt in')
        require(sys.executable==PYTHON,'existing environment')
        with FREEZE.open('rb') as f: raw=f.read(131073)
        expected=sys.argv[2]
        require(len(raw)<=131072 and hashlib.sha256(raw).hexdigest()==expected,'freeze identity')
        freeze=json.loads(raw); result['freeze_sha256']=expected
        require(freeze['schema']=='OWNED_CLEANUP_FREEZE_V1' and freeze['caps']==CAPS
                and freeze['cases']==list(IDS) and freeze['subcases']==SUBCASES
                and freeze['attempts']==1 and freeze['result_directory']==str(RUN),'finite freeze')
        for path,digest in freeze['files'].items(): require(sha(path)==digest,'source binding: '+path)
        require(not (ROOT/'scoped_proof/__init__.py').exists(),'frozen namespace package')
        sys.path.append(str(ROOT))
        from scoped_proof import owned_session as owned
        from scoped_proof.io import save
        import shutil
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
        disk=shutil.disk_usage(RUN.parent).free
        require(available>=4*2**30 and disk>=CAPS['artifact_bytes'],'fresh admission')
        require(time.monotonic()<started+60,'bounded admission')
        result['admission']=dict(available_bytes=available,disk_free_bytes=disk,at=time.monotonic())
        RUN.mkdir()  # Exclusive one-shot claim. Never resume a failed attempt.
        created=True
        initial_fds=len(os.listdir('/proc/self/fd'))
        result['initial_fds']=initial_fds
        env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

        def tick():
            nonlocal peak
            require(time.monotonic()<started+240,'work deadline with finalization reserve')
            peak=max(peak,owned.parent_rss())
            require(peak<=CAPS['rss_bytes'],'sampled RSS limit')

        def stat(pid=101,state='R',sid=101,start=55):
            return (str(pid)+' (fixed control) '+' '.join(
                [state,'1','101',str(sid)]+['0']*15+[str(start)])+'\n').encode()

        class Reader:
            def __init__(self,steps,pids=(101,)):
                self.steps=list(steps); self.ids=pids; self.reads=0
            def pids(self): return iter(self.ids)
            def read(self,pid,name,cap):
                require(bool(self.steps),'no extra reads/retries')
                want_pid,want_name,value=self.steps.pop(0); self.reads+=1
                require((pid,name)==(want_pid,want_name),'ordered reads')
                if isinstance(value,Exception): raise value
                return value

        def memory_reader(status,last=None):
            return Reader([(101,'stat',stat()),(101,'status',status),
                           (101,'stat',stat() if last is None else last)])

        def observed(source,cleanup=False,until=None):
            fn=owned._observe_cleanup if cleanup else owned._observe
            value=fn(101,started+230 if until is None else until,source)
            require(not source.steps,'all prescribed reads consumed')
            return value

        def rejected(source,cleanup=False,until=None):
            try: observed(source,cleanup,until)
            except owned._ObservationFailure as exc:
                return dict(error=exc.detail,reads=source.reads,remaining_reads=len(source.steps))
            raise AssertionError('uncertain observation accepted')

        def partial(cleanup):
            evidence=owned._SessionEvidence()
            if cleanup:
                good=Reader([(101,'stat',stat()),(101,'stat',stat())])
                bad=Reader([(101,'stat',stat()),(101,'stat',stat()),
                    (102,'stat',stat(pid=102)),(102,'stat',PermissionError('cannot confirm'))],(101,102))
            else:
                good=memory_reader(b'VmRSS: 12 kB\n')
                bad=Reader([(101,'stat',stat()),(101,'status',b'VmRSS: 12 kB\n'),(101,'stat',stat()),
                    (102,'stat',stat(pid=102)),(102,'status',b'Name: no RSS\n'),
                    (102,'stat',stat(pid=102))],(101,102))
            fn=owned._observe_cleanup if cleanup else owned._observe
            evidence.scan(101,started+230,lambda sid,d:fn(sid,d,good))
            before=evidence.fields()
            try: evidence.scan(101,started+230,lambda sid,d:fn(sid,d,bad))
            except owned._ObservationFailure as exc:
                evidence.failed('cleanup' if cleanup else 'execution',exc,'partial',102)
            else: raise AssertionError('partial scan accepted')
            after=evidence.fields()
            require(before['remaining_session']['live']==[101] and after['remaining_session'] is None
                    and not after['remaining_session_known'],'partial does not retain old roster')
            return dict(before=before,after=after,reads=bad.reads)

        def memory_diagnostics():
            d={}
            for name,status in (('live_no_rss',b'Name: control\n'),
                                ('status_gone_live',FileNotFoundError('status disappeared'))):
                tick(); r=rejected(memory_reader(status)); o=r['error']['observation']
                require(r['error']['kind']=='ValueError' and r['reads']==3
                    and o['first_identity']==o['last_identity']==dict(state='R',sid=101,start_ticks=55)
                    and o['status_read']==('READ' if name=='live_no_rss' else 'DISAPPEARED')
                    and o['rss_present'] is False,'missing RSS remains rejected with diagnostic')
                d[name]=r
            d['confirmed_zombie']=observed(memory_reader(b'Name: control\n',stat(state='Z')))
            d['known_rss']=observed(memory_reader(b'VmRSS: 12 kB\n'))
            require(d['confirmed_zombie']==dict(live=[],zombies=[101],rss=0)
                and d['known_rss']==dict(live=[101],zombies=[],rss=12288),'unchanged memory classifications')
            d['partial_memory_unknown']=partial(False)
            return d

        def identity_cleanup():
            d={}; r=Reader([(101,'stat',stat()),(101,'stat',stat(state='S'))])
            d['live_without_status']=observed(r,True)
            require(r.reads==2 and d['live_without_status']['live']==[101]
                and d['live_without_status']['rss'] is None
                and d['live_without_status']['rss_known'] is False,'cleanup supplies no memory evidence')
            d['foreign_excluded']=observed(Reader([(101,'stat',stat(sid=102))]),True)
            require(not d['foreign_excluded']['live'],'foreign excluded')
            for name,last,kind in (
                ('reused_pid',stat(start=56),'ValueError'),
                ('changed_sid',stat(sid=102),'ValueError'),
                ('permission_denied',PermissionError('cannot confirm'),'PermissionError')):
                tick(); d[name]=rejected(Reader([(101,'stat',stat()),(101,'stat',last)]),True)
                require(d[name]['error']['kind']==kind,name)
            d['expired_deadline']=rejected(Reader([]),True,time.monotonic()-1)
            require(d['expired_deadline']['error']['kind']=='TimeoutError'
                    and d['expired_deadline']['reads']==0,'deadline before any read')
            d['partial_roster_unknown']=partial(True)
            return d

        def signal_binding():
            class Api:
                def __init__(self): self.fd=None; self.opens=0; self.sent=[]
                def pidfd_open(self,pid,flags):
                    self.opens+=1; self.fd=os.open('/dev/null',os.O_RDONLY); return self.fd
                def pidfd_send_signal(self,fd,sig,info,flags):
                    require(fd==self.fd,'bound descriptor')
                    self.sent.append(dict(pid=101,signal=sig))
            d={}
            for name,value,expired in (
                ('same_identity_signal',stat(),False),
                ('foreign_reject',stat(sid=102),False),
                ('reused_pid_reject',stat(start=56),False),
                ('deadline_no_open',None,True)):
                tick(); api=Api(); r=Reader([] if expired else [(101,'stat',value)])
                err=None; accepted=None
                try: accepted=owned._kill_owned(101,101,time.monotonic()-1 if expired else started+230,
                                               pidfd=api,expected_start=55,reader=r)
                except (ValueError,TimeoutError) as exc: err=type(exc).__name__
                closed=api.fd is None
                if api.fd is not None:
                    try: os.fstat(api.fd)
                    except OSError: closed=True
                require(closed,'every opened descriptor closed')
                require((accepted is True and len(api.sent)==1 and err is None)
                    if name=='same_identity_signal' else
                    (not api.sent and err==('TimeoutError' if expired else 'ValueError')),'identity before signal')
                require(api.opens==(0 if expired else 1),'no open after deadline')
                d[name]=dict(opens=api.opens,signals=api.sent,error=err,fd_closed=closed,reads=r.reads)
            return d

        def aftercare(receipt):
            # Existing direct-child bookkeeping only; never upgrades the receipt.
            if receipt['returncode'] is not None: return
            stop=min(started+235,time.monotonic()+5)
            while time.monotonic()<stop:
                try: pid,code=os.waitpid(receipt['session'],os.WNOHANG)
                except ChildProcessError:
                    reaps.append(dict(pid=receipt['session'],status=None,disposition='ALREADY_NOT_WAITABLE',
                                      at=time.monotonic(),changes_original_acceptance=False)); break
                if pid:
                    reaps.append(dict(pid=pid,status=code,disposition='WAITPID_REAPED',
                                      at=time.monotonic(),changes_original_acceptance=False)); break
                time.sleep(.005)
            else: raise TimeoutError('bounded direct fixture reap')
            reaps[-1]['post_observation']=owned.observe_cleanup(receipt['session'],stop)
            require(not reaps[-1]['post_observation']['live'],'no live fixture aftercare')

        def execution_outcomes():
            nonlocal peak
            d={}; members=[]; cleanup_scans=[]; memory_scans=[]
            class NoRSS(owned._ProcReader):
                def read(self,pid,name,cap):
                    return b'Name: injected missing RSS\n' if name=='status' else super().read(pid,name,cap)
            def member_rejection(sid,until):
                nonlocal peak
                actual=owned.observe(sid,until); memory_scans.append(actual)
                peak=max(peak,actual['rss']+owned.parent_rss())
                require(peak<=CAPS['rss_bytes'],'sampled fixture RSS')
                children=[p for p in actual['live'] if p!=sid]
                if children:
                    members.extend(children)
                    return owned._observe(sid,until,NoRSS())
                return actual
            def cleanup(sid,until):
                value=owned.observe_cleanup(sid,until)
                require(len(cleanup_scans)<32,'finite cleanup trace')
                cleanup_scans.append(value); return value
            code=("import subprocess,sys,signal,time;signal.alarm(8);"
                  "p=subprocess.Popen([sys.executable,'-I','-S','-B','-c',"
                  "'import signal,time;signal.alarm(8);time.sleep(7)']);time.sleep(7)")
            tick()
            r=owned._execute([PYTHON,'-I','-S','-B','-c',code],
                run_deadline=min(started+220,time.monotonic()+3),
                cleanup_deadline=min(started+230,time.monotonic()+6),env=env,rss_limit=CAPS['rss_bytes'],
                _observer=member_rejection,_cleanup_observer=cleanup)
            receipts.append(r); aftercare(r)
            require(members and r['status']=='ERROR' and r['execution_error']['message']=='live process RSS unavailable'
                and r['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION' and r['cleanup_error'] is None
                and not r['remaining_session']['live'] and r['remaining_session']['rss'] is None,
                'real member cleanup does not upgrade rejected execution')
            require(any(set(members)&set(s['live']) for s in cleanup_scans) and r['kill_signals']>=2,
                    'actual member signal path, not leader only')
            d['execution_rejection_cleanup']=dict(session_index=0,members=sorted(set(members)),
                cleanup_scans=cleanup_scans,complete_memory_samples=len(memory_scans))
            for name in ('two_errors','deadline_cleanup','normal_memory_acceptance'):
                tick()
                def fault(sid,until):
                    raise owned._ObservationFailure(ValueError('execution rejected'),'fixed_execution',sid)
                def cleanup_fault(sid,until):
                    raise owned._ObservationFailure(PermissionError('cleanup identity unavailable'),'fixed_cleanup',sid)
                observer=fault if name=='two_errors' else owned.observe
                cleaner=cleanup_fault if name=='two_errors' else owned.observe_cleanup
                r=owned._execute([PYTHON,'-I','-S','-B','-c',
                    'import signal,time;signal.alarm(2);time.sleep(1)' if name=='deadline_cleanup' else
                    'import signal,time;signal.alarm(2);time.sleep(.05)'],
                    run_deadline=min(started+220,time.monotonic()+(.03 if name=='deadline_cleanup' else 3)),
                    cleanup_deadline=min(started+230,time.monotonic()+6),
                    env=env,rss_limit=CAPS['rss_bytes'],_observer=observer,_cleanup_observer=cleaner)
                receipts.append(r); aftercare(r)
                peak=max(peak,r['sampled_session_plus_caller_rss'])
                expected_status={'two_errors':'CLEANUP_INCOMPLETE','deadline_cleanup':'TIMEOUT',
                                 'normal_memory_acceptance':'COMPLETED'}[name]
                require(r['status']==expected_status,name)
                if name=='two_errors':
                    require(r['execution_error']['kind']=='ValueError' and r['cleanup_error']['kind']=='PermissionError'
                        and r['remaining_session'] is None and not r['remaining_session_known'],'two errors and unknown roster')
                else:
                    require(r['cleanup_error'] is None and r['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
                        and r['remaining_session']['rss'] is None and not r['remaining_session']['rss_known'],
                        'cleanup identity is not memory evidence')
                if name=='normal_memory_acceptance':
                    require(r['execution_error'] is None and r['returncode']==0
                        and r['sampled_session_plus_caller_rss']>0 and r['exit_observed_at'] is not None,
                        'normal complete execution still has real memory observations')
                d[name]=dict(session_index=len(receipts)-1)
            return d

        actions=(memory_diagnostics,identity_cleanup,signal_binding,execution_outcomes)
        for name,action in zip(IDS,actions):
            begin=time.monotonic(); tick()
            try:
                details=action(); require(list(details)==SUBCASES[name],'closed subcase roster'); tick()
                rows.append(dict(id=name,status='PASS',seconds=time.monotonic()-begin,details=details))
            except BaseException as exc:
                rows.append(dict(id=name,status='FAIL',seconds=time.monotonic()-begin,
                                 error=dict(kind=type(exc).__name__,message=str(exc)[:256])))
                raise
        result['final_fds']=len(os.listdir('/proc/self/fd'))
        require(result['final_fds']==initial_fds,'descriptor balance')
        require(not Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split(),'no fixture children')
        tick(); result['status']='PASS'
    except BaseException as exc:
        result['failure']=dict(kind=type(exc).__name__,message=str(exc)[:256])
    result['unexecuted']=list(IDS[len(rows):]); result['sampled_managed_rss_bytes']=peak
    result['online_acceptance']=result['status']=='PASS' and time.monotonic()<deadline
    result['before_export']=time.monotonic(); result['seconds_before_export']=result['before_export']-started
    reception=None
    if created:
        require(len(packed(result))<=131072,'bounded result bytes')
        saved=save(RUN/'result.json',result)
        require(json.loads((RUN/'result.json').read_bytes())==result,'same-budget readback')
        received=time.monotonic()
        reception=dict(result_sha256=saved['sha256'],result_bytes=saved['bytes'],received_at=received,
                       seconds=received-started,online_acceptance=result['online_acceptance'] and received<deadline)
        save(RUN/'reception.json',reception)
    finished=time.monotonic()
    returned=dict(result=result,reception=reception,finished_at=finished,seconds=finished-started,
                  online_acceptance=bool(reception and reception['online_acceptance'] and finished<deadline))
    require(len(packed(returned))<=131072,'bounded stdout bytes')
    sys.stdout.buffer.write(packed(returned)+b'\n'); sys.stdout.buffer.flush()
    exported=time.monotonic()
    stamp=dict(schema='OWNED_CLEANUP_EXPORT_V1',after_export=exported,seconds=exported-started,
               online_acceptance=returned['online_acceptance'] and exported<deadline)
    sys.stderr.buffer.write(packed(stamp)+b'\n'); sys.stderr.buffer.flush()
    raise SystemExit(0 if stamp['online_acceptance'] else 1)


if __name__=='__main__': main()
