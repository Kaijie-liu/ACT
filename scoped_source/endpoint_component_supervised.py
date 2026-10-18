"""Finite component wiring using existing owned-group and bounded-pipe supervision."""
from pathlib import Path
import os
import time

from scoped_source.hz_request_store import POLICY, identity, require
from scoped_source.native_protocol import (PYTHON, read_json, write_json, restore,
    COUNTERS, counts, hard_limits, REPORT_CAP)
from scoped_source.native_supervised import run_actor, clean_env
from scoped_proof.owned_bounded import execute as owned
from scoped_proof.io import ROOT
from scoped_source.endpoint_batch import validate, FREEZE_CAP


def command(role, root, invhash):
    require(role in ('controller','component'),'closed component role')
    bootstrap='''import resource,sys,time
started=time.monotonic()
resource.setrlimit(resource.RLIMIT_AS,(8589934592,8589934592))
role=sys.argv[1]
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576 if role=="controller" else 134217728,134217728))
import hashlib,json
from pathlib import Path
root=Path(sys.argv[2]); repo=Path(sys.argv[4]); p=root/"component-invocation.json"
if p.is_symlink() or p.stat().st_size>131072: raise ValueError("component invocation cap")
raw=p.read_bytes()
if len(raw)>131072 or hashlib.sha256(raw).hexdigest()!=sys.argv[3]: raise ValueError("component invocation hash")
inv=json.loads(raw); code=inv["freeze"]["code"]
if type(code) is not dict or not 1<=len(code)<=256: raise ValueError("component roster")
for n,h in code.items():
    q=repo/n
    if time.monotonic()>=inv["run_deadline"]: raise TimeoutError("component bootstrap deadline")
    if Path(n).is_absolute() or ".." in Path(n).parts or any(a.is_symlink() for a in (q,*q.parents)): raise ValueError("component code path")
    if not q.is_file() or q.stat().st_nlink!=1 or q.stat().st_size>131072 or hashlib.sha256(q.read_bytes()).hexdigest()!=h: raise ValueError("component code binding")
for n in ("scripts","scoped_source","scoped_proof","source_enclosure","upstream_source","router_source"):
    if (repo/n/"__init__.py").exists(): raise ValueError("unbound initializer")
if "scoped_source/endpoint_component_supervised.py" not in code: raise ValueError("component entry missing")
sys.path.append(str(repo))
from scoped_source.endpoint_component_supervised import actor
actor(role,root,sys.argv[3],started)
'''
    return [PYTHON,'-I','-S','-B','-c',bootstrap,role,str(root),invhash,str(ROOT)]


def actor(role, root, expected, boot_started):
    root=Path(root); inv,_=read_json(root/'component-invocation.json',expected=expected,cap=FREEZE_CAP)
    require(set(inv)=={'schema','root','freeze','freeze_sha256','started','run_deadline',
        'cleanup_deadline','deadline','outer_deadline','owner_session'} and inv['schema']=='ENDPOINT_COMPONENT_INVOCATION_V1'
        and inv['root']==str(root) and inv['run_deadline']==inv['started']+298
        and inv['cleanup_deadline']==inv['started']+298.5 and inv['deadline']==inv['started']+299
        and inv['outer_deadline']==inv['started']+300
        and type(inv['owner_session']) is int and inv['owner_session']>0
        and os.getsid(0)==inv['owner_session'],'component cumulative schedule and enclosing session')
    freeze=validate(inv['freeze'],inv['freeze_sha256'],deadline=inv['run_deadline'])
    profile=freeze['records'][0]['profile']
    limits=hard_limits(profile,'controller' if role=='controller' else 'production')
    if role=='controller':
        execution=run_actor(root,'component',expected,profile,inv['run_deadline'],
                            launch_command=command('component',root,expected))
        report,h=read_json(root/'component.json',deadline=inv['run_deadline'])
        require(report['status']=='COMPLETE' and report['invocation']==expected
                and report['work_complete'] is True and report['output']['status']=='PASS',
                'complete component actor, not partial receipt')
        from scoped_source.direct_supervised import census
        storage=census(root,inv['run_deadline'])
        require(storage['bytes']+2*REPORT_CAP<=POLICY['total_bytes']
                and len(storage['files'])+2<=POLICY['members'],'component final metadata reserve')
        write_json(root/'controller.json',dict(schema='ENDPOINT_COMPONENT_CONTROLLER_V1',
            status='COMPLETE',invocation=expected,component_sha256=h,work=counts(report['work']),
            storage=storage,limits=limits,execution=execution,started=boot_started,end=time.monotonic()),
            deadline=inv['run_deadline'])
        return
    work=restore({k:0 for k in COUNTERS},inv['run_deadline']); result=None; error=None
    try:
        from scripts.endpoint_midpoint_components import run
        result=run(root/'components',profile,freeze['component'],identity(freeze['component']),work)
        work.tick()
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,message=str(exc)[:400])
    complete=error is None
    # A failed portable child may have performed unknown Work. Parent counters
    # remain an observed prefix only; never resume this position or reset Work.
    report=dict(schema='ENDPOINT_COMPONENT_ACTOR_V1',status='COMPLETE' if complete else 'ERROR',
        invocation=expected,output=result,error=error,work=dict(work.counts),work_complete=complete,
        work_scope='CUMULATIVE' if complete else 'OBSERVED_PREFIX_UNKNOWN_CHILD_REMAINDER',
        limits=limits,begin=boot_started,end=time.monotonic(),deadline=inv['run_deadline'])
    write_json(root/'component.json',report,deadline=inv['run_deadline'])
    if not complete: raise RuntimeError('component incomplete; stop batch')


def execute(root, freeze, expected, started, *, owner_session):
    require(type(owner_session) is int and owner_session>0 and os.getsid(0)==owner_session,
            'explicit enclosing component session')
    require(time.monotonic()<started+298,'component preparation inside original position')
    validate(freeze,expected,deadline=started+298)
    root=Path(root)
    require(root.is_absolute() and root.resolve()==root and root.is_relative_to('/data1/Kane/MOE')
            and not root.exists(),'new component position')
    root.mkdir()
    inv=dict(schema='ENDPOINT_COMPONENT_INVOCATION_V1',root=str(root),freeze=freeze,freeze_sha256=expected,
        started=started,run_deadline=started+298,cleanup_deadline=started+298.5,
        deadline=started+299,outer_deadline=started+300,owner_session=owner_session)
    ref=write_json(root/'component-invocation.json',inv,deadline=started+298,cap=FREEZE_CAP)
    run=owned(command('controller',root,ref['sha256']),root/'controller.log',
        run_deadline=started+298,cleanup_deadline=started+298.5,env=clean_env(),
        rss_limit=freeze['limits']['rss_bytes'],bounded_errors=True,owner_session=owner_session)
    return dict(invocation=ref['sha256'],owned_execution=run,started=started,
                online_acceptance=False,control_only=True)
