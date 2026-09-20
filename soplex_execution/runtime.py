"""Original-clock supervision and durable phase/cost records (stdlib only)."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from evidence_cohort.ownership import info, stop_owned, descendants
from soplex_execution.candidate import LimitError
from soplex_fidelity.io import save

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/data1/Kane/miniconda3/envs/act-py312/bin/python')


def env():
    return {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'',
            'PYTHONHASHSEED':'0'}


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.nice(10)


def output_sizes(root, enforce=True):
    sizes={str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*') if p.is_file()}
    if enforce:
        if any(v>128*1024**2 for v in sizes.values()):raise LimitError('individual output cap')
        if sizes.get('point.txt',0)>64*1024**2:raise LimitError('point output cap')
        if sum(sizes.values())>512*1024**2:raise LimitError('total output cap')
    return sizes


def wait(process, deadline, root):
    """Poll owned tree, output caps and one absolute deadline. Never kill unrelated PIDs."""
    record=info(process.pid);peaks={};cleanup=[];reason=None
    if record is None and process.poll() is None:raise RuntimeError('missing process identity')
    while process.poll() is None:
        try:
            output_sizes(root)
            end=deadline() if callable(deadline) else deadline
            if time.monotonic()>=end:raise TimeoutError('absolute process deadline')
            if record:
                for item in descendants(record['pid']):
                    try:
                        fields=Path(f"/proc/{item['pid']}/status").read_text().splitlines()
                        rss=max([int(s.split()[1]) for s in fields if s.startswith(('VmHWM:','VmRSS:'))] or [0])
                        key=f"{item['pid']}:{item['start']}"
                        peaks[key]=max(peaks.get(key,0),rss)
                    except (FileNotFoundError,ProcessLookupError):pass
            time.sleep(min(.05,max(0,end-time.monotonic())))
        except BaseException as exc:
            cleanup=stop_owned(record) if record else []
            process.wait();reason=type(exc).__name__+': '+str(exc)
            if not isinstance(exc,(TimeoutError,LimitError)):raise
            break
    process.wait()
    try:output_sizes(root)
    except LimitError as exc:
        if reason is None:reason='LimitError: '+str(exc)
    end=deadline() if callable(deadline) else deadline
    if reason is None and time.monotonic()>end:reason='TimeoutError: process returned late'
    return dict(returncode=process.returncode,termination=reason,owned_cleanup=cleanup,
                sampled_peak_rss_kib_by_pid_start=peaks,
                rss_scope='sampled per-process HWM/RSS, not address space or sum of simultaneous peaks')


class Clock:
    def __init__(self,root,start,proposal,work):
        self.root=Path(root);self.start=start;self.proposal=start+proposal;self.work=start+work
        self.deadline=self.proposal

    def tick(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('shared absolute deadline')

    @contextmanager
    def phase(self,name,proposal=False):
        self.deadline=self.proposal if proposal else self.work
        begin=time.monotonic();save(self.root/f'{name}.entered.json',dict(start_offset=begin-self.start,deadline_offset=self.deadline-self.start))
        # OS timer also covers long stdlib parsing/serialization, not only cooperative ticks.
        signal.setitimer(signal.ITIMER_REAL,max(.000001,self.deadline-time.monotonic()))
        try:
            self.tick();yield;self.tick()
            save(self.root/f'{name}.done.json',dict(end_offset=time.monotonic()-self.start,
                                                  seconds=time.monotonic()-begin))
        finally:signal.setitimer(signal.ITIMER_REAL,0)

    def native(self,name,command):
        self.tick();started=time.monotonic()
        save(self.root/f'{name}.command.json',dict(command=list(map(str,command)),
             start_offset=started-self.start,deadline_offset=self.deadline-self.start))
        # GNU time captures short processes whose RSS could escape polling.
        with (self.root/f'{name}.stdout').open('xb') as out,(self.root/f'{name}.stderr').open('xb') as err:
            process=subprocess.Popen(['/usr/bin/time','-f','%e %M','-o',str(self.root/f'{name}.resources'),
                                      *map(str,command)],stdout=out,stderr=err,env=env(),start_new_session=True)
            record=info(process.pid)
            try:result=wait(process,self.deadline,self.root)
            except BaseException:
                if record:stop_owned(record)
                process.wait();raise
        result['seconds']=time.monotonic()-started
        save(self.root/f'{name}.process.json',result)
        if result['termination']:
            if result['termination'].startswith('LimitError'):raise LimitError(result['termination'])
            raise TimeoutError(result['termination'])
        if result['returncode'] in (-signal.SIGXFSZ,128+signal.SIGXFSZ):raise LimitError('native file cap')
        return result


def costs(root,end):
    rows={}
    for name in ('load','export','import','readback','solve','capture','package','check','review'):
        entered=root/f'{name}.entered.json';done=root/f'{name}.done.json'
        if not entered.exists():rows[name]=dict(seconds=None,observed_seconds=None,state='NOT_ENTERED');continue
        a=json.loads(entered.read_text());b=json.loads(done.read_text()) if done.exists() else None
        rows[name]=dict(start_offset=a['start_offset'],end_offset=None if b is None else b['end_offset'],
             seconds=None if b is None else b['seconds'],
             observed_seconds=max(0,(end if b is None else b['end_offset'])-a['start_offset']),
             state='CENSORED' if b is None else 'COMPLETE')
    return rows
