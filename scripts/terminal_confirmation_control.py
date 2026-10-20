"""Once-only observation compatibility control, not an algorithm experiment."""
import time
ENTRY=time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import select
import signal
import sys

ROOT=Path('/data1/Kane/MOE/ACT')
PYTHON='/data1/Kane/miniconda3/envs/act-py312/bin/python'
FREEZE=ROOT/'docs/terminal_confirmation_freeze_20261020.json'
RUN=Path('/data1/Kane/MOE/results/terminal_confirmation_20261020_r1')
CAPS=dict(seconds=300,as_bytes=8*2**30,rss_bytes=2*2**30,artifact_bytes=128*2**20,work=8000000)
CASES=('known_rss','default_missing_reject','terminal_Z','terminal_X','terminal_x',
       'confirmed_missing','confirmed_status_gone','no_events','hup_only','poll_error',
       'poll_invalid','wrong_fd','multiple_events','changed_sid','changed_start',
       'binding_disappeared','binding_permission','open_permission','poll_permission',
       'malformed_binding','oversized_binding','malformed_rss','deadline_before',
       'deadline_after_poll','terminal_with_live_peer','partial_scan_and_first_errors',
       'bounded_terminal_records')
NATIVE=('stdlib_zero','native_zero','native_seven')


def require(value,message):
    if not value: raise ValueError(message)


def packed(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def main():
    started=ENTRY; deadline=started+300
    for kind,limit in ((resource.RLIMIT_AS,CAPS['as_bytes']),(resource.RLIMIT_FSIZE,CAPS['artifact_bytes'])):
        old=resource.getrlimit(kind)
        require(all(v==resource.RLIM_INFINITY or v>=limit for v in old),'no inherited limit increase')
        resource.setrlimit(kind,(limit,limit))
    def hard_stop(_sig,_frame):
        os.write(2,b'TERMINAL_CONTROL_HARD_DEADLINE_NO_ACCEPTANCE\n'); os._exit(124)
    signal.signal(signal.SIGALRM,hard_stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    result=dict(schema='TERMINAL_CONTROL_V1',status='FAIL',started=started,deadline=deadline,caps=CAPS,
                full_cold_start_seconds=None,measurement_boundary='FIRST_PYTHON_MONOTONIC',
                cases=[],sessions=[],algorithm_work_consumed=0,
                algorithm_executed=False,model_executed=False,gpu_executed=False)
    created=False; peak=0
    try:
        require(len(sys.argv)==3 and sys.argv[1]=='--execute-approved-once','explicit opt in')
        require(sys.executable==PYTHON,'existing environment')
        raw=FREEZE.read_bytes()
        require(len(raw)<=131072 and hashlib.sha256(raw).hexdigest()==sys.argv[2],'freeze identity')
        freeze=json.loads(raw)
        require(freeze['schema']=='TERMINAL_FREEZE_V1' and freeze['caps']==CAPS
                and freeze['cases']==list(CASES) and freeze['native']==list(NATIVE)
                and freeze['attempts']==1 and freeze['result_directory']==str(RUN),'finite contract')
        for path,digest in freeze['files'].items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'source binding: '+path)
        sys.path.append(str(ROOT))
        from scoped_proof import owned_session as owned
        from scoped_proof.io import save
        from scripts.audit_terminal_confirmation import check
        import shutil
        import subprocess
        def git(*args):
            return subprocess.check_output(['git',*args],cwd=ROOT,
                timeout=max(.001,min(5,started+60-time.monotonic()))).decode().strip()
        require(git('branch','--show-current')=='feat/moe-route-verification' and
                not git('status','--porcelain'),'clean authorized branch')
        result['execution_head']=git('rev-parse','HEAD')
        result['freeze_sha256']=sys.argv[2]
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:'))
        disk=shutil.disk_usage(RUN.parent).free
        require(available>=4*2**30 and disk>=CAPS['artifact_bytes'],'fresh resource admission')
        require(time.monotonic()<started+60,'one bounded admission')
        result['admission']=dict(available_bytes=available,disk_bytes=disk,at=time.monotonic())
        RUN.mkdir(); created=True
        result['initial_fds']=len(os.listdir('/proc/self/fd'))

        def tick():
            nonlocal peak
            require(time.monotonic()<started+240,'control work cutoff')
            peak=max(peak,owned.parent_rss())
            require(peak<=CAPS['rss_bytes'],'sampled RSS cap')

        def stat(pid=101,sid=101,start=55,state='R'):
            return (str(pid)+' (fixed) '+' '.join(
                [state,'1','101',str(sid)]+['0']*15+[str(start)])+'\n').encode()

        class Reader:
            def __init__(self,steps,ids=(101,)):
                self.steps=list(steps); self.ids=ids; self.reads=0
            def pids(self): return iter(self.ids)
            def read(self,pid,name,cap):
                require(bool(self.steps),'no extra reads')
                p,n,value=self.steps.pop(0); self.reads+=1
                require((pid,name)==(p,n),'ordered reads')
                if isinstance(value,Exception): raise value
                return value

        class Api:
            backend='CONTROL_ONLY'
            def __init__(self,deny=False):
                self.fd=None; self.opens=0; self.deny=deny
            def pidfd_open(self,pid,flags):
                require((pid,flags)==(101,0),'fixed open')
                self.opens+=1
                if self.deny: raise PermissionError('fixed open denied')
                self.fd=os.open('/dev/null',os.O_RDONLY)
                return self.fd

        def closed(api):
            if api.fd is None: return True
            try: os.fstat(api.fd)
            except OSError: return True
            return False

        def observe_case(name):
            enabled=name!='default_missing_reject'
            status=b'Name: missing\n'; last=stat(); bound=stat()
            until=started+230; expected_error=None; events='ready'
            if name=='known_rss': status=b'VmRSS: 12 kB\n'
            if name.startswith('terminal_') and name[-1] in ('Z','X','x'): last=stat(state=name[-1])
            if name=='confirmed_status_gone': status=FileNotFoundError('fixed status gone')
            if name=='default_missing_reject': expected_error='ValueError'
            if name in ('no_events','hup_only','poll_error','poll_invalid','wrong_fd','multiple_events'):
                events=name; expected_error='ValueError'
            if name=='changed_sid': bound=stat(sid=102); expected_error='ValueError'
            if name=='changed_start': bound=stat(start=56); expected_error='ValueError'
            if name=='binding_disappeared': bound=FileNotFoundError('fixed stat gone'); expected_error='FileNotFoundError'
            if name=='binding_permission': bound=PermissionError('fixed stat denied'); expected_error='PermissionError'
            if name in ('open_permission','poll_permission'): expected_error='PermissionError'
            if name=='malformed_binding': bound=b'invalid'; expected_error='ValueError'
            if name=='oversized_binding': bound=b'x'*8193; expected_error='ValueError'
            if name=='malformed_rss': status=b'VmRSS: 12 MB\n'; expected_error='ValueError'
            if name=='deadline_before': until=time.monotonic()-1; expected_error='TimeoutError'
            if name=='deadline_after_poll': until=time.monotonic()+.05; expected_error='TimeoutError'
            steps=[(101,'stat',stat()),(101,'status',status),(101,'stat',last)]
            if name=='deadline_before': steps=[]
            elif name=='malformed_rss': steps=steps[:2]
            elif enabled and name!='known_rss' and last==stat() and name!='open_permission':
                steps.append((101,'stat',bound))
            ids=(101,)
            if name=='terminal_with_live_peer':
                steps.extend([(102,'stat',stat(pid=102)),(102,'status',b'VmRSS: 12 kB\n'),
                              (102,'stat',stat(pid=102))]); ids=(101,102)
            reader=Reader(steps,ids); api=Api(name=='open_permission'); waits=[]
            def poll(fd,ms):
                waits.append(ms); require(0<=ms<=10 and fd==api.fd,'bounded bound poll')
                if name=='poll_permission': raise PermissionError('fixed poll denied')
                if name=='deadline_after_poll':
                    time.sleep(max(0,until-time.monotonic())+.002)
                return {'ready':[(fd,select.POLLIN)],'no_events':[],
                    'hup_only':[(fd,select.POLLHUP)],'poll_error':[(fd,select.POLLIN|select.POLLERR)],
                    'poll_invalid':[(fd,select.POLLIN|select.POLLNVAL)],'wrong_fd':[(fd+1,select.POLLIN)],
                    'multiple_events':[(fd,select.POLLIN),(fd,select.POLLIN)]}[events]
            value=None; error=None
            try:
                value=owned._observe(101,until,reader,terminal_confirmation=enabled,_pidfd=api,_poll=poll)
            except owned._ObservationFailure as exc: error=exc.detail
            require(closed(api) and not reader.steps,'descriptor close and full fixed read trace')
            require((None if error is None else error['kind'])==expected_error,'expected refusal: '+name)
            if name=='deadline_after_poll':
                require(len(waits)==1 and error['phase']=='terminal_pidfd_confirmation',
                        'post-poll deadline actually reached')
            if error is None:
                if name=='known_rss':
                    require(value['live']==[101] and value['rss']==12288 and api.opens==0,'unchanged known RSS')
                elif name in ('terminal_Z','terminal_X','terminal_x'):
                    require(value['zombies']==[101] and api.opens==0,'unchanged stat terminal')
                else:
                    records=value['terminal_without_rss']
                    require(len(records)==1 and records[0]['rss'] is None and not value['zombies']
                        and value['live']==([102] if name=='terminal_with_live_peer' else [])
                        and value['rss']==(12288 if name=='terminal_with_live_peer' else 0),
                        'terminal evidence is not missing RSS zero or absent live peer')
            if name in ('deadline_before','default_missing_reject','malformed_rss'):
                require(api.opens==0,'no unauthorized open')
            return dict(observation=value,error=error,reads=reader.reads,opens=api.opens,
                        poll_waits_ms=waits,fd_closed=closed(api))

        def evidence_case(name):
            record=dict(pid=101,identity=dict(state='R',sid=101,start_ticks=55),rss=None,
                        evidence='CONTROL_ONLY')
            evidence=owned._SessionEvidence(True)
            if name=='bounded_terminal_records':
                evidence.retain_terminals([dict(record,pid=p) for p in range(owned.MEMBERS)])
                evidence.retain_terminals([dict(record,pid=0)])
                try: evidence.retain_terminals([dict(record,pid=owned.MEMBERS)])
                except ValueError as exc: error=str(exc)
                else: raise AssertionError('terminal evidence overflow accepted')
                require(len(evidence.terminals)==512,'cap not silently truncated')
                failure=owned._ObservationFailure(PermissionError('original scan denied'),'fixed',102)
                failure.detail['terminal_without_rss']=[dict(record,pid=owned.MEMBERS)]
                def bad(*_): raise failure
                try: evidence.scan(101,started+230,bad)
                except owned._ObservationFailure as exc:
                    require(exc.detail['kind']=='PermissionError' and
                        exc.detail['terminal_retention_error']['kind']=='ValueError',
                        'capacity error must not erase original failure')
                    secondary=exc.detail['terminal_retention_error']
                else: raise AssertionError('overflow partial scan accepted')
                return dict(retained=512,overflow_error=error,original_error='PermissionError',
                            retention_error=secondary)
            good=dict(live=[102],zombies=[],rss=12288,terminal_without_rss=[record])
            evidence.scan(101,started+230,lambda *_:good)
            failure=owned._ObservationFailure(PermissionError('fixed later scan denied'),'fixed',102)
            second=dict(record,pid=103)
            failure.detail['terminal_without_rss']=[second]
            def bad(*_): raise failure
            try: evidence.scan(101,started+230,bad)
            except owned._ObservationFailure as exc: evidence.failed('execution',exc,'fixed',102)
            require(evidence.remaining is None,'failed scan invalidates prior roster')
            evidence.failed('execution',ValueError('must not replace first'),'later',101)
            evidence.failed('cleanup',ValueError('fixed cleanup failure'),'cleanup',101)
            evidence.scan(101,started+230,lambda *_:dict(live=[],zombies=[],rss=None))
            fields=evidence.fields()
            require(fields['execution_error']['kind']=='PermissionError' and
                    fields['cleanup_error']['kind']=='ValueError' and
                    fields['terminal_without_rss']==[record,second],
                    'later empty roster cannot erase partial-scan evidence or failures')
            return fields

        for name in CASES:
            tick(); begin=time.monotonic()
            try:
                detail=evidence_case(name) if name in CASES[-2:] else observe_case(name)
                tick(); result['cases'].append(dict(id=name,status='PASS',seconds=time.monotonic()-begin,detail=detail))
            except BaseException as exc:
                result['cases'].append(dict(id=name,status='FAIL',seconds=time.monotonic()-begin,
                    error=dict(kind=type(exc).__name__,message=str(exc)[:256])))
                raise
        env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        for name in NATIVE:
            tick()
            code='import signal;signal.alarm(45);print("stdlib",flush=True)'
            if name!='stdlib_zero':
                code=('import signal;signal.alarm(45);import numpy,scipy,torch;'
                      'a=numpy.zeros(1);b=torch.zeros(1,device="cpu");'
                      'print(numpy.__version__,scipy.__version__,torch.__version__,flush=True);'
                      'raise SystemExit('+('7' if name=='native_seven' else '0')+')')
            now=time.monotonic()
            receipt=owned.execute([PYTHON,'-I','-B',*(['-S'] if name=='stdlib_zero' else []),'-c',code],
                run_deadline=min(started+220,now+50),cleanup_deadline=min(started+230,now+55),
                env=env,rss_limit=CAPS['rss_bytes'],terminal_confirmation=True)
            result['sessions'].append(dict(id=name,receipt=receipt))
            peak=max(peak,receipt['sampled_session_plus_caller_rss']); tick()
            require(receipt['status']==('ERROR' if name=='native_seven' else 'COMPLETED')
                and receipt['returncode']==(7 if name=='native_seven' else 0)
                and receipt['execution_error'] is None and receipt['cleanup_error'] is None
                and receipt['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
                and not receipt['remaining_session']['live'],'real compatibility: '+name)
        result['final_fds']=len(os.listdir('/proc/self/fd'))
        require(result['final_fds']==result['initial_fds'],'descriptor balance')
        result['owned_children_remaining']=Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split()
        require(not result['owned_children_remaining'],'no direct children remain')
        tick(); result['status']='PASS'
    except BaseException as exc:
        result['failure']=dict(kind=type(exc).__name__,message=str(exc)[:256])
    result['peak_sampled_rss_bytes']=peak
    result['before_export']=time.monotonic()
    result['unexecuted_cases']=list(CASES[len(result['cases']):])
    result['unexecuted_native']=list(NATIVE[len(result['sessions']):])
    reception=None
    if created:
        require(len(packed(result))<131072,'bounded record')
        saved=save(RUN/'result.json',result)
        reread=json.loads((RUN/'result.json').read_bytes())
        verdict=check(reread,freeze)
        received=time.monotonic()
        reception=dict(result_sha256=saved['sha256'],result_bytes=saved['bytes'],audit=verdict,
                       received_at=received,online_acceptance=verdict['accepted'] and received<deadline)
        save(RUN/'reception.json',reception)
    returned=dict(result=result,reception=reception,finished_at=time.monotonic())
    require(len(packed(returned))<131072,'bounded export')
    sys.stdout.buffer.write(packed(returned)+b'\n'); sys.stdout.buffer.flush()
    stamp=dict(after_export=time.monotonic(),online_acceptance=bool(reception and
        reception['online_acceptance'] and time.monotonic()<deadline))
    sys.stderr.buffer.write(packed(stamp)+b'\n'); sys.stderr.buffer.flush()
    raise SystemExit(0 if stamp['online_acceptance'] else 1)


if __name__=='__main__': main()
