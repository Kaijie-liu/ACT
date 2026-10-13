"""Single-owner, no-overwrite archive accounting and bounded child reception.

Logical bytes, not filesystem allocated blocks. Failed prefixes remain charged.
This module never grants a mathematical result merely because a file exists.
"""
import math
import hashlib
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time

from scoped_source.hz_request_store import POLICY, compact, digest, require


class Archive:
    def __init__(self,root,work,*,limit=POLICY['total_bytes'],reserve=8192):
        require(type(limit) is int and 0<limit<=POLICY['total_bytes'] and
                type(reserve) is int and 0<reserve<limit,'unchanged archive cap/reserve')
        self.work=work; self.root=Path(root); self.limit=limit; self.reserve=reserve
        self.bytes=0; self.entries={}; self.attempted=set(); self.prefixes={}; self.pid=os.getpid(); self.terminal=False
        self.started=time.monotonic(); self.events=[]; self.deadline=work.deadline
        work.tick(); self.root.mkdir(parents=False,exist_ok=False); work.tick()

    def _path(self,name):
        require(type(name) is str and len(name)<=200 and not Path(name).is_absolute() and
                str(Path(name))==name and all(v not in ('','.','..') for v in Path(name).parts),'archive local path')
        p=self.root
        require(not p.is_symlink() and os.getpid()==self.pid,'archive owner/root')
        for part in Path(name).parts:
            p=p/part; require(not p.is_symlink(),'archive symlink')
        return p

    def put(self,name,raw,*,terminal=False):
        self.work.tick(); require(type(raw) is bytes and not self.terminal,'immutable bytes/open archive')
        require(len(self.attempted)<POLICY['members'] and name not in self.attempted,'archive members/no overwrite')
        p=self._path(name); require(not p.exists(),'archive existing member')
        ceiling=self.limit if terminal else self.limit-self.reserve
        require(self.bytes+len(raw)<=ceiling and (not terminal or len(raw)<=self.reserve),'archive aggregate bytes/reserved terminal')
        self.attempted.add(name); start=time.monotonic(); count=0; created=False
        p.parent.mkdir(parents=True,exist_ok=True); self.work.tick()
        try:
            with p.open('xb',buffering=0) as stream:
                created=True
                while count<len(raw):
                    self.work.tick()
                    n=stream.write(memoryview(raw)[count:count+65536]); require(n and n>0,'archive short write')
                    count+=n; self.bytes+=n
                    self.work.tick()
                os.fsync(stream.fileno()); self.work.tick()
            value=dict(file=name,bytes=count,sha256=digest(raw)); self.work.tick()
            self.work.tick()  # A late, success-looking file is not an accepted receipt.
            self.entries[name]=value
            if terminal: self.terminal=True
            return dict(value)
        finally:
            self.prefixes[name]=dict(bytes=count,created=created,sha256=(self.entries[name]['sha256'] if name in self.entries
                else hashlib.sha256(memoryview(raw)[:count]).hexdigest()))
            self.events.append(dict(file=name,bytes=count,seconds=time.monotonic()-start,
                                    bytes_fully_written=count==len(raw),timely_published=name in self.entries,
                                    complete=name in self.entries))

    def copy(self,source,name):
        self.work.tick(); source=Path(source); require(not source.is_symlink() and source.is_file(),'copy regular source')
        size=source.stat().st_size
        require(size<=self.limit-self.reserve-self.bytes,'copy admission')
        with source.open('rb') as stream: raw=stream.read(size+1)
        self.work.tick(); require(len(raw)==size,'copy changed source'); return self.put(name,raw)

    def relocate(self,prefix,names):
        """Copy files independently within this same charged archive, keep both."""
        self.work.tick(); names=tuple(names)
        require(len(set(names))==len(names) and all(n in self.entries for n in names),'relocation inventory')
        require(sum(self.entries[n]['bytes'] for n in names)+self.bytes<=self.limit-self.reserve,'two-copy aggregate cap')
        result=[]
        for name in names:
            self.work.tick(); source=self._path(name); expected=self.entries[name]
            with source.open('rb') as stream: raw=stream.read(expected['bytes']+1)
            self.work.tick(); require(len(raw)==expected['bytes'] and digest(raw)==expected['sha256'],'relocation source identity')
            ref=self.put(prefix+'/'+name,raw)
            require(source.stat().st_ino!=self._path(ref['file']).stat().st_ino,'relocation independent file')
            result.append(ref)
        self.work.tick(); return result

    def audit(self):
        """Reconcile even failed prefixes; external additions/edits are errors."""
        self.work.tick(); actual={}
        require(not self.root.is_symlink() and self.root.is_dir(),'archive audit root')
        for directory,dirs,files in os.walk(self.root,followlinks=False):
            self.work.tick()
            for item in dirs+files:
                require(not (Path(directory)/item).is_symlink(),'archive audit symlink')
            for item in files:
                p=Path(directory)/item; name=str(p.relative_to(self.root))
                require(name in self.attempted and stat.S_ISREG(p.stat(follow_symlinks=False).st_mode),'unowned/nonregular archive member')
                actual[name]=p.stat().st_size
                self.work.tick()
                ref=self.entries.get(name,self.prefixes.get(name))
                require(ref is not None and actual[name]==ref['bytes'],'archive member grew/shrank')
                h=hashlib.sha256()
                fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                with os.fdopen(fd,'rb') as stream:
                    require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),'archive changed file type')
                    for raw in iter(lambda:stream.read(65536),b''):
                        self.work.tick(); h.update(raw)
                require(h.hexdigest()==ref['sha256'],'archive member changed')
        prefixes={n for n,v in self.prefixes.items() if v['created']}
        require(set(actual)==set(self.entries)|prefixes and sum(actual.values())==self.bytes<=self.limit,'archive complete byte ledger')
        self.work.tick(); return self.ledger()

    def ledger(self):
        """Can report after deadline. Never upgrades an incomplete operation."""
        return dict(logical_bytes=self.bytes,limit=self.limit,reserved_terminal=self.reserve,
                    written=len(self.entries),attempted=len(self.attempted),events=list(self.events),
                    elapsed_seconds=time.monotonic()-self.started,
                    elapsed_after_deadline=max(0.,time.monotonic()-self.deadline))


def run_child(argv,archive,*,log_prefix,log_limit=2**20,cwd=None):
    """Only terminate this owned process group; every received log byte is bounded.

    The caller's Work/deadline is reused. No shell, numerical imports or child
    result acceptance here. Cleanup wall time, including overruns, is returned.
    """
    work=archive.work; work.tick()
    require(type(argv) in (tuple,list) and argv and all(type(x) is str for x in argv), 'child argv')
    require(type(log_limit) is int and 0<log_limit<=archive.limit-archive.reserve,'log quota')
    start=time.monotonic(); process=None; selector=selectors.DefaultSelector(); count=0; chunks=0
    status='ERROR'; error=None; code=None
    try:
        process=subprocess.Popen(argv,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        os.set_blocking(process.stdout.fileno(),False); selector.register(process.stdout,selectors.EVENT_READ)
        while selector.get_map():
            work.tick()
            for key,_ in selector.select(min(.05,max(0.,work.deadline-time.monotonic()))):
                work.tick(); raw=os.read(key.fileobj.fileno(),min(16384,log_limit-count+1))
                if not raw: selector.unregister(key.fileobj); continue
                count+=len(raw); require(count<=log_limit,'child log cap (not truncated success)')
                archive.put(f'{log_prefix}/{chunks:04d}.log',raw); chunks+=1
        work.tick(); code=process.wait(timeout=max(.001,work.deadline-time.monotonic())); work.tick()
        status='COMPLETE' if code==0 else 'ERROR'
    except BaseException as exc:
        status='TIMEOUT' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'ERROR'; error=type(exc).__name__+': '+str(exc)[:500]
    finally:
        selector.close()
        if process is not None:
            # Descendants can keep pipes alive after the leader exits.
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(); code=process.returncode; process.stdout.close()
    if status=='COMPLETE' and time.monotonic()>=work.deadline:
        status='TIMEOUT'; error='owned cleanup exceeded cumulative deadline'
    return dict(status=status,error=error,returncode=code,log_bytes_received=count,
                log_chunks=chunks,seconds=time.monotonic()-start,
                cleanup_after_deadline=max(0.,time.monotonic()-work.deadline))


def receive(raw,expected,work):
    """Bounded timely reception; no deadline reset or trust in status text."""
    work.tick(); require(type(raw) is bytes and len(raw)<=8192 and digest(raw)==expected,'terminal reception')
    from scoped_source.hz_request_store import decode
    result=decode(raw); require(compact(result)==raw,'canonical terminal'); work.tick(); return result


def receive_file(archive,ref):
    """Re-read the durable owned terminal, then audit the complete archive."""
    work=archive.work; work.tick()
    require(type(ref) is dict and set(ref)=={'file','bytes','sha256'} and
            ref==archive.entries.get(ref['file']) and type(ref['bytes']) is int and 0<=ref['bytes']<=8192,
            'owned bounded terminal reference')
    path=archive._path(ref['file']); fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),'terminal regular file')
        raw=stream.read(8193)
    work.tick(); require(len(raw)==ref['bytes'],'terminal actual length')
    result=receive(raw,ref['sha256'],work); archive.audit(); work.tick(); return result
