"""One-shot wiring in an ALREADY RUNNING, pre-limited trusted host. No CLI.

This module does not manufacture an earlier clock, install limits retrospectively,
or establish a host's trust. The host must retain the external pre-request origin
and expected hash. A new Python importing this file is NOT that external origin.
Only the existing observer spawns a worker; no extra supervisor is introduced.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import time
import uuid

ROOT=Path('/data1/Kane/MOE/ACT')
RUN_DIRECTORY='/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r2'
FREEZE_FILE='/data1/Kane/MOE/ACT/docs/hz_endpoint_midpoint_freeze_v3_20261019.json'
REGISTRY=ROOT/'docs/hz_endpoint_execution_approvals.json'
SELF='scripts/endpoint_midpoint_launcher.py'
AS_CAP=8*2**30
FILE_CAP=128*2**20
CLAIMED=False


def require(value, message):
    if not value: raise ValueError(message)


def packed(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def sha(value):
    require(type(value) is str and len(value)==64
            and all(c in '0123456789abcdef' for c in value),'hex sha256')
    return value


def read(path, cap, deadline):
    require(time.monotonic()<deadline,'startup deadline')
    require(path.is_absolute() and path.resolve()==path
            and not any(p.is_symlink() for p in (path,*path.parents)),'canonical source')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1
                and before.st_size<=cap,'bounded regular source')
        with os.fdopen(fd,'rb',closefd=False) as stream: raw=stream.read(cap+1)
        after=os.fstat(fd); named=path.stat(follow_symlinks=False)
        key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        require(len(raw)<=cap and len(raw)==before.st_size and key(before)==key(after)==key(named),
                'stable bounded source')
    finally:
        os.close(fd)
    require(time.monotonic()<deadline,'startup read completed timely')
    return raw


def issuer_identity(raw):
    """Code/contract identity only, NOT evidence that a live host exists."""
    return hashlib.sha256(packed(dict(schema='ENDPOINT_RETAINED_HOST_WIRING_V1',
        path=SELF,sha256=hashlib.sha256(raw).hexdigest(),clock='CLOCK_MONOTONIC',
        origin='BEFORE_REQUEST_PREPARATION_AND_SPAWN',as_bytes=AS_CAP,file_bytes=FILE_CAP,
        mode='IN_PROCESS_FROM_EXTERNALLY_RETAINED_ORIGIN',attempts=1))).hexdigest()


def import_state(code, deadline):
    """Refuse preloaded/shadowed project code; never patch global import state.

    Python/stdlib import machinery remains trusted. File hashes alone do not
    authenticate previously loaded modules or timestamp-valid bytecode caches.
    """
    packages=('scripts','scoped_source','scoped_proof','source_enclosure','upstream_source','router_source')
    machinery=importlib.machinery
    require(sys.flags.isolated==1 and sys.flags.no_site==1 and sys.flags.optimize==0
            and sys.dont_write_bytecode and sys.pycache_prefix is None,'isolated standard-library host')
    require(sys.meta_path==[machinery.BuiltinImporter,machinery.FrozenImporter,machinery.PathFinder],
            'standard trusted import machinery')
    require(len(sys.modules)<=512,'bounded pre-import module inventory')
    for name in tuple(sys.modules):
        top=name.split('.')[0]
        require(top not in ('torch','numpy','scipy','act'),'no preloaded model/native execution')
        if top not in packages: continue
        require(name in packages or (name=='scripts.endpoint_midpoint_launcher'
                and sys.modules[name].__dict__ is globals()),'no stale or foreign project modules')
    for name in packages:
        require(time.monotonic()<deadline,'import state deadline')
        spec=machinery.PathFinder.find_spec(name,sys.path)
        require(spec is not None and spec.origin is None and spec.loader is None
                and set(spec.submodule_search_locations)=={str(ROOT/name)},'only bound namespace path')
        if name in sys.modules:
            module=sys.modules[name]
            require(getattr(module,'__file__',None) is None
                    and set(module.__path__)=={str(ROOT/name)},'already loaded namespace identity')
    for name in code:
        require(time.monotonic()<deadline,'bytecode exclusion deadline')
        require(name.endswith('.py'),'bound Python source module')
        path=ROOT/name
        spec=machinery.PathFinder.find_spec(name[:-3].replace('/','.'),[str(path.parent)])
        require(spec is not None and spec.origin==str(path)
                and type(spec.loader) is machinery.SourceFileLoader,
                'no extension/package/sourceless shadow of checked source')
        require(not Path(importlib.util.cache_from_source(str(ROOT/name))).exists(),
                'no unbound timestamp-valid project bytecode')


def launch_from_retained_start(expected, *, origin, origin_sha256, enabled=False):
    """Return a live caller anchor, or a negative early-failure record.

    Never reads an origin from a file, CLI or /proc clock. `origin` must already
    be retained by the trusted host. Its hash is an integrity check, not authority.
    The externally held invocation clock includes this module/host's cold startup;
    if the host cannot supply that provenance it must NOT invoke this function.
    """
    global CLAIMED
    require(enabled is True,'default-off launcher')
    require(CLAIMED is False,'one attempt per host; no retry after early failure')
    CLAIMED=True
    entry=time.monotonic()  # observation only; never assigned to started
    started=None; start_hash=None; failure_stage='origin'; result=None
    try:
        keys={'schema','started','clock','origin','boot_id','time_namespace',
              'host_pid','host_start_ticks','issuer','freeze'}
        require(type(origin) is dict and set(origin)==keys,'closed retained origin')
        saved=dict(origin)
        require(set(saved)==keys,'closed detached origin')
        require(type(saved['started']) in (int,float) and math.isfinite(saved['started'])
                and 0<=saved['started']<=entry<saved['started']+60,'original startup window')
        # Only immutable bounded scalars enter serialization; no caller-owned aliases.
        require(all(type(saved[k]) is str and len(saved[k])<=128
                    for k in keys-{'started','host_pid','host_start_ticks'}),'bounded origin strings')
        require(type(saved['host_pid']) is int and saved['host_pid']==os.getpid()
                and type(saved['host_start_ticks']) is int
                and 0<=saved['host_start_ticks']<2**64,'same dedicated host')
        require(saved['schema']=='ENDPOINT_RETAINED_ORIGIN_V1'
                and saved['clock']=='CLOCK_MONOTONIC'
                and saved['origin']=='BEFORE_REQUEST_PREPARATION_AND_SPAWN'
                and saved['freeze']==sha(expected),'retained request and clock')
        raw=packed(saved)
        require(len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==sha(origin_sha256),
                'retained external origin identity')
        started=saved['started']; deadline=started+60
        failure_stage='inherited_limits'
        require(resource.getrlimit(resource.RLIMIT_AS)==(AS_CAP,AS_CAP)
                and resource.getrlimit(resource.RLIMIT_FSIZE)==(FILE_CAP,FILE_CAP),
                'limits must already cover host startup; never set/raise/repair here')
        with os.scandir('/proc/self/task') as tasks:
            require(next(tasks,None) is not None and next(tasks,None) is None,'single-thread dedicated host')
        failure_stage='authority'
        authority=json.loads(read(REGISTRY,65536,deadline))
        require(type(authority) is dict and set(authority)=={'schema','batches'}
                and authority['schema']=='ENDPOINT_BATCH_APPROVALS_V1'
                and type(authority['batches']) is list and len(authority['batches'])<=16
                and authority['batches']==sorted(set(authority['batches']))
                and expected in authority['batches'],'separate batch execution authority')
        for h in authority['batches']: sha(h)
        freeze_raw=read(Path(FREEZE_FILE),131072,deadline)
        require(hashlib.sha256(freeze_raw).hexdigest()==expected,'exact frozen request bytes')
        freeze=json.loads(freeze_raw)
        require(freeze['schema']=='ENDPOINT_FINITE_FREEZE_V3'
                and freeze['startup']['issuer']==sha(saved['issuer'])
                and freeze['startup']['caller_limits_before_spawn'] is True,'frozen startup contract')
        own=read(ROOT/SELF,131072,deadline)
        require(issuer_identity(own)==saved['issuer'],'frozen wiring identity, not a host claim')
        failure_stage='code'
        code=freeze['code']
        require(type(code) is dict and 1<=len(code)<=256 and SELF in code
                and 'scripts/endpoint_midpoint_caller.py' in code,'complete bounded code roster')
        for name,h in code.items():
            require(type(name) is str and not Path(name).is_absolute()
                    and '..' not in Path(name).parts and str(Path(name))==name,'relative code path')
            require(hashlib.sha256(read(ROOT/name,131072,deadline)).hexdigest()==sha(h),'bound code')
        for name in ('scripts','scoped_source','scoped_proof','source_enclosure','upstream_source','router_source'):
            require(not (ROOT/name/'__init__.py').exists(),'no unbound initializer')
        import_state(code,deadline)
        failure_stage='anchor'
        anchor=dict(schema='ENDPOINT_TRUSTED_PRELAUNCH_V1',freeze=expected,issuer=saved['issuer'],
            root=RUN_DIRECTORY,code_root=str(ROOT),clock=saved['clock'],origin=saved['origin'],
            started=started,nonce=uuid.uuid4().hex,boot_id=saved['boot_id'],
            time_namespace=saved['time_namespace'],launcher_pid=saved['host_pid'],
            launcher_start_ticks=saved['host_start_ticks'])
        raw=packed(anchor); require(len(raw)<=4096,'existing admission anchor capacity')
        start_hash=hashlib.sha256(raw).hexdigest()
        failure_stage='caller'
        # This is the existing caller, in this host: no extra spawn or supervisor.
        from scripts.endpoint_midpoint_caller import receive_once
        module=sys.modules['scripts.endpoint_midpoint_caller']; path=str(ROOT/'scripts/endpoint_midpoint_caller.py')
        require(module.__file__==module.__spec__.origin==receive_once.__code__.co_filename==path
                and type(module.__loader__) is importlib.machinery.SourceFileLoader
                and receive_once.__globals__ is module.__dict__,'actual caller from checked source')
        result=receive_once(expected,start_anchor=anchor,start_sha256=start_hash,enabled=True)
        require(result['schema']=='ENDPOINT_CALLER_RETURN_V2'
                and result['start_sha256']==start_hash and result['started']==started,
                'same live caller return')
        require(len(packed(result))<=65536,'existing caller receipt capacity')
        # Extend the existing receipt event through THIS bounded reception work.
        end=time.monotonic()
        result=dict(result,received_at=end,seconds=end-started,
                    online_batch_acceptance=result['online_batch_acceptance'] is True and end<started+2100)
        if time.monotonic()>=started+2100: result['online_batch_acceptance']=False
        return result
    except BaseException as exc:
        end=time.monotonic()
        # No source/read/setup failure can become a positive result or zero cost.
        return dict(schema='ENDPOINT_LAUNCH_FAILURE_V1',status='INCOMPLETE_OR_FAILED',
            stage=failure_stage,error=type(exc).__name__,start_sha256=start_hash,
            started=started,observed_entry=entry,observed_end=end,
            seconds=None if started is None else end-started,
            observed_prefix_seconds=end-entry,missing_cost_is_not_zero=True,
            online_batch_acceptance=False,all_work_complete=False)


if __name__=='__main__':
    raise SystemExit('No standalone launch: retained external clock and pre-start caps are mandatory')
