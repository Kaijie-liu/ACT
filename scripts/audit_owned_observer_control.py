"""Static closure and saved-record audit only; never rerun control actions."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('scoped_proof/owned_session.py','scoped_proof/owned_bounded.py',
         'scoped_proof/pidfd_compat.py','scoped_proof/io.py',
         'scripts/owned_observer_finite_control.py','scripts/audit_owned_observer_control.py')
HOST=('/data1/Kane/miniconda3/envs/act-py312/bin/python',
      '/data1/Kane/miniconda3/envs/act-py312/lib/python3.12/ctypes/__init__.py',
      '/data1/Kane/miniconda3/envs/act-py312/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so',
      '/usr/lib/x86_64-linux-gnu/libc.so.6')
FREEZE=ROOT/'docs/hz_owned_observer_freeze_20261020.json'


def packed(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v): return hashlib.sha256(packed(v)).hexdigest()


def read(path):
    raw=Path(path).read_bytes(); assert len(raw)<=1024**2
    def unique(items):
        out={}
        for k,v in items:
            assert k not in out,'duplicate field'; out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def static():
    hashes={}; trees={}
    assert not (ROOT/'scoped_proof/__init__.py').exists(),'frozen namespace package'
    for name in SOURCES:
        raw=(ROOT/name).read_bytes(); tree=ast.parse(raw,filename=name); compile(tree,name,'exec')
        trees[name]=tree; hashes[str(ROOT/name)]=hashlib.sha256(raw).hexdigest()
    for path in HOST:
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024**2),b''): h.update(block)
        hashes[path]=h.hexdigest()
    own=trees['scoped_proof/owned_session.py']
    public=next(n for n in own.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    private=next(n for n in own.body if isinstance(n,ast.FunctionDef) and n.name=='_execute')
    assert '_observer' not in [a.arg for a in public.args.kwonlyargs]
    call=public.body[-1].value
    assert call.func.id=='_execute'
    assert next(k.value.id for k in call.keywords if k.arg=='_observer')=='observe'
    probe=[n.lineno for n in ast.walk(private) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=='probe']
    spawn=[n.lineno for n in ast.walk(private) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert len(probe)==len(spawn)==1 and probe[0]<spawn[0]
    for name in ('scoped_proof/owned_session.py','scripts/owned_observer_finite_control.py'):
        for n in ast.walk(trees[name]):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                assert n.func.attr not in ('kill','killpg','syscall','setattr','patch'),name
    tree=trees['scripts/owned_observer_finite_control.py']
    def assignment(key):
        return next(n.value for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(x,ast.Name) and x.id==key for x in n.targets))
    cases=ast.literal_eval(assignment('IDS'))
    subcases={k.arg:ast.literal_eval(k.value) for k in assignment('SUBCASES').keywords}
    assert len(cases)==len(set(cases))==4 and list(subcases)==list(cases)
    caps=dict(seconds=300,as_bytes=8*2**30,rss_bytes=2*2**30,artifact_bytes=128*2**20,work=8000000)
    assert read(ROOT/'docs/hz_endpoint_execution_approvals.json')['batches']==[]
    return dict(schema='OWNED_OBSERVER_FREEZE_V1',cases=list(cases),subcases=subcases,caps=caps,
                attempts=1,result_directory='/data1/Kane/MOE/results/owned_observer_20261020_r1',files=hashes,
                scope='EXISTING_OBSERVER_REPAIR_ONLY',algorithm_batch_authorized=False)


def audit(directory,stdout,stderr,exit_code):
    root=Path(directory); freeze=read(FREEZE)
    assert static()==freeze,'frozen implementation identity'
    out=read(stdout); stamp=read(stderr); result=read(root/'result.json'); receipt=read(root/'reception.json')
    assert out['result']==result and out['reception']==receipt
    assert str(root)==freeze['result_directory']
    assert result['freeze_sha256']==hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    assert result['caps']==freeze['caps']
    assert receipt['result_sha256']==digest(result) and receipt['result_bytes']==len(packed(result))
    assert (root/'result.json').read_bytes()==packed(result)
    start=result['started']
    assert 0<=result['provisioning_seconds'] and result['full_cold_start_seconds'] is None
    assert start<=result['before_export']<=receipt['received_at']<=out['finished_at']<=stamp['after_export']<start+300
    for r,key in ((receipt,'received_at'),(out,'finished_at'),(stamp,'after_export')):
        assert r['seconds']==r[key]-start
    assert result['seconds_before_export']==result['before_export']-start
    rows=result['cases']; ids=[r['id'] for r in rows]
    assert ids==freeze['cases'][:len(ids)] and result['unexecuted']==freeze['cases'][len(ids):]
    assert all(not result[k] for k in ('algorithm_executed','model_executed','gpu_executed','seven_position_batch_authorized'))
    complete=result['status']=='PASS'
    if complete:
        assert int(exit_code)==0 and all(v['online_acceptance'] is True for v in (result,receipt,out,stamp))
        assert ids==freeze['cases'] and all(r['status']=='PASS' for r in rows)
        assert all(set(r['details'])==set(freeze['subcases'][r['id']]) for r in rows)
        d=rows[0]['details']
        assert d['live_to_zombie']==dict(live=[],zombies=[101],rss=0)
        assert d['live_to_gone']==d['missing_status_then_gone']==dict(live=[],zombies=[],rss=0)
        assert d['live_with_rss']==dict(live=[101],zombies=[],rss=12288)
        d=rows[1]['details']
        for name,kind in zip(freeze['subcases']['uncertain_rejection'],
                             ['ValueError']*4+['PermissionError','IndexError','ValueError','TimeoutError']):
            assert d[name]['error']['kind']==kind
        assert d['expired_deadline']['reads']==0
        sessions=result['sessions']; assert len(sessions)==2
        first,second=sessions
        assert first['status']=='CLEANUP_INCOMPLETE' and first['error']=='ValueError'
        assert first['execution_error']['kind']=='ValueError' and first['cleanup_error']['kind']=='PermissionError'
        assert first['execution_error']['phase']=='fixed_execution' and first['cleanup_error']['phase']=='fixed_cleanup'
        assert first['execution_error']['pid']==first['cleanup_error']['pid']==first['session']
        assert first['remaining_session'] is None and first['remaining_session_known'] is False
        assert second['status']=='ERROR' and second['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
        assert second['remaining_session_known'] is True and not second['remaining_session']['live']
        assert second['execution_error']['kind']=='ValueError' and second['cleanup_error'] is None
        partial=rows[3]['details']['partial_scan_unknown']
        assert partial['before']['remaining_session']['live']==[101]
        assert partial['after']['remaining_session'] is None and not partial['after']['remaining_session_known']
        assert not partial['after']['execution_error']['observation_known']
        for session in sessions:
            assert start<=session['started_at']<=session['finished_at']<session['cleanup_deadline']<start+300
            assert session['observation_contract']=='CONFIRMED_IDENTITY_V2'
            for detail in (session['execution_error'],session['cleanup_error']):
                if detail:
                    assert len(detail['message'])<=256 and len(detail['phase'])<=64 and len(detail['kind'])<=64
        assert len(result['post_failure_reaps'])==1
        reap=result['post_failure_reaps'][0]
        assert reap['pid']==first['session'] and reap['changes_original_acceptance'] is False
        assert reap['disposition'] in ('WAITPID_REAPED','ALREADY_NOT_WAITABLE')
        assert (reap['status'] is None)==(reap['disposition']=='ALREADY_NOT_WAITABLE')
        assert not reap['post_observation']['live']
        assert first['finished_at']<=reap['at']<second['started_at']
        assert result['initial_fds']==result['final_fds']
        assert result['sampled_managed_rss_bytes']<=freeze['caps']['rss_bytes']
        assert result['algorithm_work_consumed']==0
    else:
        assert not out['online_acceptance'],'no offline upgrade'
    files={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in (root/'result.json',root/'reception.json',Path(stdout),Path(stderr))}
    size=sum(v['bytes'] for v in files.values()); assert size<freeze['caps']['artifact_bytes']
    assert set(p.name for p in root.iterdir())=={'result.json','reception.json'}
    return dict(status='PASS_SAVED_OBSERVER_CONTROLS' if complete else 'RETAINED_INCOMPLETE',
                freeze_sha256=result['freeze_sha256'],result_sha256=digest(result),files=files,
                artifact_bytes=size,passed=sum(r['status']=='PASS' for r in rows),unexecuted=result['unexecuted'],
                seconds=stamp['seconds'],process_exit_code=int(exit_code),runtime_reexecuted=False,
                algorithm_efficacy=False,seven_position_batch_authorized=False)


if __name__=='__main__':
    value=static() if sys.argv[1:]==['--static'] else audit(*sys.argv[1:])
    print(json.dumps(value,sort_keys=True,indent=2))
