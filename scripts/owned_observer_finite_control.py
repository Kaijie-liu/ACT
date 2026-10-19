"""One four-group observer repair control, no algorithm/model/GPU execution."""
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
FREEZE=ROOT/'docs/hz_owned_observer_freeze_20261020.json'
RUN=Path('/data1/Kane/MOE/results/owned_observer_20261020_r1')
IDS=('confirmed_exit','uncertain_rejection','dual_error_preservation','unknown_and_no_upgrade')
CAPS=dict(seconds=300,as_bytes=8*2**30,rss_bytes=2*2**30,artifact_bytes=128*2**20,work=8000000)
SUBCASES=dict(
    confirmed_exit=['live_to_zombie','live_to_gone','missing_status_then_gone','live_with_rss'],
    uncertain_rejection=['missing_rss_live','missing_status_live','reused_pid','changed_sid',
                         'permission_denied','malformed_stat','oversized_status','expired_deadline'],
    dual_error_preservation=['actual_execute_two_errors','bounded_error_text'],
    unknown_and_no_upgrade=['partial_scan_unknown','actual_execute_cleanup_does_not_upgrade'])


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
    result=dict(schema='OWNED_OBSERVER_CONTROL_V1',status='FAIL',started=started,caps=CAPS,
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
        require(freeze['schema']=='OWNED_OBSERVER_FREEZE_V1' and freeze['caps']==CAPS
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
            # /proc stat fields 3..22: starttime is index19 after comm.
            fields=[state,'1','101',str(sid)]+['0']*15+[str(start)]
            return (str(pid)+' (fixed control) '+' '.join(fields)+'\n').encode()

        class Reader:
            def __init__(self,steps,pids=(101,)):
                self.steps=list(steps); self.ids=pids; self.reads=0
            def pids(self): return iter(self.ids)
            def read(self,pid,name,cap):
                require(bool(self.steps),'no extra reads/retries')
                want_pid,want_name,value=self.steps.pop(0); self.reads+=1
                require((pid,name)==(want_pid,want_name),'ordered identity reads')
                if isinstance(value,Exception): raise value
                return value

        def reader(last,status=b'Name: control\n',first=None):
            return Reader([(101,'stat',stat() if first is None else first),
                           (101,'status',status),(101,'stat',last)])

        def scan(source,until=None):
            out=owned._observe(101,started+230 if until is None else until,source)
            require(not source.steps,'all prescribed reads consumed')
            return out

        def rejected(source,kind,until=None):
            try: scan(source,until)
            except owned._ObservationFailure as exc:
                require(exc.detail['kind']==kind,'correct failure kind')
                return dict(error=exc.detail,reads=source.reads,remaining_reads=len(source.steps))
            raise AssertionError('uncertain observation accepted')

        def confirmed_exit():
            out={}
            for name,source,want in (
                ('live_to_zombie',reader(stat(state='Z')),dict(live=[],zombies=[101],rss=0)),
                ('live_to_gone',reader(FileNotFoundError('gone')),dict(live=[],zombies=[],rss=0)),
                ('missing_status_then_gone',reader(ProcessLookupError('gone'),FileNotFoundError('status')),
                 dict(live=[],zombies=[],rss=0)),
                ('live_with_rss',reader(stat(state='S'),b'VmRSS:\t12 kB\n'),
                 dict(live=[101],zombies=[],rss=12288))):
                tick(); actual=scan(source); require(actual==want,name); out[name]=actual
            return out

        def uncertain_rejection():
            out={}
            for name,source,kind in (
                ('missing_rss_live',reader(stat()),'ValueError'),
                ('missing_status_live',reader(stat(),FileNotFoundError('status')),'ValueError'),
                ('reused_pid',reader(stat(start=56)),'ValueError'),
                ('changed_sid',reader(stat(sid=102)),'ValueError'),
                ('permission_denied',reader(PermissionError('cannot confirm')),'PermissionError'),
                ('malformed_stat',reader(b'101 (bad) R\n'),'IndexError'),
                ('oversized_status',reader(stat(),b'x'*65537),'ValueError')):
                tick(); out[name]=rejected(source,kind)
            out['expired_deadline']=rejected(Reader([]), 'TimeoutError',time.monotonic()-1)
            return out

        def short_session(observer):
            nonlocal peak
            tick()
            def sampled(sid,until):
                nonlocal peak
                peak=max(peak,owned.observe(sid,until)['rss']+owned.parent_rss())
                require(peak<=CAPS['rss_bytes'],'sampled owned session plus root RSS')
                return observer(sid,until)
            # Two short owned stdlib fixtures only, not endpoint workers. Own
            # alarm also bounds lifetime if the injected cleanup observation fails.
            receipt=owned._execute([PYTHON,'-I','-S','-B','-c',
                'import signal,time;signal.alarm(2);time.sleep(1)'],
                run_deadline=min(started+220,time.monotonic()+5),
                cleanup_deadline=min(started+230,time.monotonic()+10),
                env=env,rss_limit=CAPS['rss_bytes'],_observer=sampled)
            receipts.append(receipt)
            # This is a direct child identified by the Popen receipt. No signal
            # fallback; later reap is recorded, never repairs the original result.
            if receipt['returncode'] is None:
                stop=min(started+235,time.monotonic()+5)
                while time.monotonic()<stop:
                    try: pid,code=os.waitpid(receipt['session'],os.WNOHANG)
                    except ChildProcessError:
                        # Popen.__del__ may already have reaped the child after
                        # the receipt was formed. Do not invent its exit code.
                        reaps.append(dict(pid=receipt['session'],status=None,
                            disposition='ALREADY_NOT_WAITABLE',at=time.monotonic(),
                            changes_original_acceptance=False)); break
                    if pid:
                        reaps.append(dict(pid=pid,status=code,at=time.monotonic(),
                            disposition='WAITPID_REAPED',changes_original_acceptance=False)); break
                    time.sleep(.005)
                else: raise TimeoutError('bounded fixture reap')
                # Read-only aftercare. No bare PID/SID signalling and no upgrade
                # of the earlier incomplete-cleanup receipt.
                reaps[-1]['post_observation']=owned.observe(receipt['session'],stop)
                require(not reaps[-1]['post_observation']['live'],'no live fixture aftercare')
            tick()
            return receipt

        def dual_error_preservation():
            calls=[]
            def fault(sid,until):
                calls.append(sid)
                exc=ValueError('execution fault') if len(calls)==1 else PermissionError('cleanup fault')
                raise owned._ObservationFailure(exc,'fixed_execution' if len(calls)==1 else 'fixed_cleanup',sid)
            r=short_session(fault)
            require(len(calls)==2 and r['status']=='CLEANUP_INCOMPLETE','both real execute paths')
            require(r['execution_error']['kind']=='ValueError' and r['cleanup_error']['kind']=='PermissionError'
                    and r['error']=='ValueError','first error not overwritten')
            require(r['remaining_session'] is None and r['remaining_session_known'] is False,'unknown final scan')
            require(r['execution_error']['pid']==r['cleanup_error']['pid']==r['session'],'error PID binding')
            record=owned._SessionEvidence()
            record.failed('execution',ValueError('a'*1000+'\n'),'p'*1000,101)
            record.failed('execution',RuntimeError('later error'),'later',102)
            detail=record.fields()['execution_error']
            require(detail['kind']=='ValueError' and len(detail['message'])==256
                    and len(detail['phase'])==64 and detail['pid']==101,'bounded first error')
            return dict(actual_execute_two_errors=dict(session_index=len(receipts)-1,calls=len(calls)),
                        bounded_error_text=detail)

        def unknown_and_no_upgrade():
            record=owned._SessionEvidence()
            record.scan(101,started+230,lambda sid,until:scan(reader(stat(),b'VmRSS: 12 kB\n')))
            complete=record.fields()
            source=Reader([(101,'stat',stat()),(101,'status',b'VmRSS: 12 kB\n'),(101,'stat',stat()),
                           (102,'stat',stat(pid=102)),(102,'status',b'Name: no RSS\n'),
                           (102,'stat',stat(pid=102))],pids=(101,102))
            try: record.scan(101,started+230,lambda sid,until:owned._observe(sid,until,source))
            except owned._ObservationFailure as exc: record.failed('execution',exc,'partial',102)
            else: raise AssertionError('partial scan accepted')
            failed=record.fields()
            require(failed['remaining_session'] is None and not failed['remaining_session_known']
                    and not failed['execution_error']['observation_known'],'stale roster invalidated')
            count=0
            def first_failure(sid,until):
                nonlocal count
                count+=1
                if count==1: raise owned._ObservationFailure(ValueError('execution rejected'),'fixed_execution',sid)
                return owned.observe(sid,until)
            r=short_session(first_failure)
            require(r['status']=='ERROR' and r['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
                    and r['execution_error']['kind']=='ValueError' and r['cleanup_error'] is None
                    and r['remaining_session_known'] and not r['remaining_session']['live'],
                    'successful cleanup does not upgrade execution')
            return dict(partial_scan_unknown=dict(before=complete,after=failed,reads=source.reads),
                        actual_execute_cleanup_does_not_upgrade=dict(session_index=len(receipts)-1))

        actions=(confirmed_exit,uncertain_rejection,dual_error_preservation,unknown_and_no_upgrade)
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
    stamp=dict(schema='OWNED_OBSERVER_EXPORT_V1',after_export=exported,seconds=exported-started,
               online_acceptance=returned['online_acceptance'] and exported<deadline)
    sys.stderr.buffer.write(packed(stamp)+b'\n'); sys.stderr.buffer.flush()
    raise SystemExit(0 if stamp['online_acceptance'] else 1)


if __name__=='__main__': main()
