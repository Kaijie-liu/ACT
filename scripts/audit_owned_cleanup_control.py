"""AST freeze and saved-record audit, never reexecutes the finite controls."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('scoped_proof/owned_session.py','scoped_proof/owned_bounded.py',
         'scoped_proof/pidfd_compat.py','scoped_proof/io.py',
         'scripts/owned_cleanup_finite_control.py','scripts/audit_owned_cleanup_control.py')
HOST=('/data1/Kane/miniconda3/envs/act-py312/bin/python',
      '/data1/Kane/miniconda3/envs/act-py312/lib/python3.12/ctypes/__init__.py',
      '/data1/Kane/miniconda3/envs/act-py312/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so',
      '/usr/lib/x86_64-linux-gnu/libc.so.6')
FREEZE=ROOT/'docs/hz_owned_cleanup_freeze_20261020.json'


def packed(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v): return hashlib.sha256(packed(v)).hexdigest()


def read(path):
    p=Path(path); assert not p.is_symlink() and p.stat().st_size<=2**20
    raw=p.read_bytes()
    def unique(items):
        d={}
        for k,v in items:
            assert k not in d,'duplicate field'; d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def static():
    hashes={}; trees={}
    assert not (ROOT/'scoped_proof/__init__.py').exists()
    for name in SOURCES:
        raw=(ROOT/name).read_bytes(); tree=ast.parse(raw,filename=name); compile(tree,name,'exec')
        trees[name]=tree; hashes[str(ROOT/name)]=hashlib.sha256(raw).hexdigest()
    for name in HOST:
        h=hashlib.sha256()
        with open(name,'rb') as f:
            for block in iter(lambda:f.read(2**20),b''): h.update(block)
        hashes[name]=h.hexdigest()
    functions={n.name:n for n in trees[SOURCES[0]].body if isinstance(n,ast.FunctionDef)}
    public=functions['execute']; call=public.body[-1].value
    assert '_observer' not in [a.arg for a in public.args.kwonlyargs]
    assert '_cleanup_observer' not in [a.arg for a in public.args.kwonlyargs]
    assert {k.arg:k.value.id for k in call.keywords if k.arg in ('_observer','_cleanup_observer')}=={
        '_observer':'observe','_cleanup_observer':'observe_cleanup'}
    cleanup=functions['_observe_cleanup']
    assert not any(isinstance(n,ast.Constant) and n.value=='status' for n in ast.walk(cleanup))
    assert any(isinstance(n,ast.Constant) and n.value=='IDENTITY_ONLY_CLEANUP' for n in ast.walk(cleanup))
    for name in (SOURCES[0],'scripts/owned_cleanup_finite_control.py'):
        for node in ast.walk(trees[name]):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in ('kill','killpg','syscall','setattr','patch')
    private=functions['_execute']
    probes=[n.lineno for n in ast.walk(private) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='probe']
    spawns=[n.lineno for n in ast.walk(private) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert len(probes)==len(spawns)==1 and probes[0]<spawns[0]
    tree=trees['scripts/owned_cleanup_finite_control.py']
    assignments={t.id:n.value for n in tree.body if isinstance(n,ast.Assign)
                 for t in n.targets if isinstance(t,ast.Name)}
    cases=ast.literal_eval(assignments['IDS'])
    subcases={k.arg:ast.literal_eval(k.value) for k in assignments['SUBCASES'].keywords}
    assert len(cases)==4 and list(subcases)==list(cases) and sum(map(len,subcases.values()))==20
    assert read(ROOT/'docs/hz_endpoint_execution_approvals.json')['batches']==[]
    return dict(schema='OWNED_CLEANUP_FREEZE_V1',cases=list(cases),subcases=subcases,
        caps=dict(seconds=300,as_bytes=8*2**30,rss_bytes=2*2**30,artifact_bytes=128*2**20,work=8000000),
        attempts=1,result_directory='/data1/Kane/MOE/results/owned_cleanup_20261020_r1',files=hashes,
        scope='EXISTING_RSS_DIAGNOSTIC_AND_IDENTITY_CLEANUP_ONLY',algorithm_batch_authorized=False)


def audit(directory,stdout,stderr,exit_code):
    root=Path(directory); freeze=read(FREEZE)
    assert static()==freeze,'frozen code and host files'
    out=read(stdout); stamp=read(stderr); result=read(root/'result.json'); receipt=read(root/'reception.json')
    assert out['result']==result and out['reception']==receipt
    assert str(root)==freeze['result_directory'] and result['caps']==freeze['caps']
    assert result['freeze_sha256']==hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    assert receipt['result_sha256']==digest(result) and receipt['result_bytes']==len(packed(result))
    assert (root/'result.json').read_bytes()==packed(result)
    start=result['started']
    assert result['provisioning_seconds']>=0 and result['full_cold_start_seconds'] is None
    assert start<=result['before_export']<=receipt['received_at']<=out['finished_at']<=stamp['after_export']<start+300
    assert result['seconds_before_export']==result['before_export']-start
    for r,k in ((receipt,'received_at'),(out,'finished_at'),(stamp,'after_export')):
        assert r['seconds']==r[k]-start
    rows=result['cases']; ids=[r['id'] for r in rows]
    assert ids==freeze['cases'][:len(ids)] and result['unexecuted']==freeze['cases'][len(ids):]
    assert all(result[k] is False for k in ('algorithm_executed','model_executed','gpu_executed','seven_position_batch_authorized'))
    assert result['algorithm_work_consumed']==0
    complete=result['status']=='PASS'
    if complete:
        assert int(exit_code)==0 and all(v['online_acceptance'] for v in (result,receipt,out,stamp))
        assert ids==freeze['cases'] and all(r['status']=='PASS' for r in rows)
        assert all(set(r['details'])==set(freeze['subcases'][r['id']]) for r in rows)
        d=rows[0]['details']
        for name,how in (('live_no_rss','READ'),('status_gone_live','DISAPPEARED')):
            error=d[name]['error']; o=error['observation']
            assert error['kind']=='ValueError' and error['message']=='live process RSS unavailable'
            assert o['first_identity']==o['last_identity']==dict(state='R',sid=101,start_ticks=55)
            assert o['status_read']==how and o['rss_present'] is False and d[name]['reads']==3
        assert d['confirmed_zombie']==dict(live=[],zombies=[101],rss=0)
        assert d['known_rss']==dict(live=[101],zombies=[],rss=12288)
        c=rows[1]['details']; bound=c['live_without_status']
        assert bound['live']==[101] and bound['identities']==[dict(pid=101,sid=101,start_ticks=55)]
        assert bound['rss'] is None and bound['rss_known'] is False
        assert bound['observation_scope']=='IDENTITY_ONLY_CLEANUP' and not c['foreign_excluded']['live']
        for name,kind in (('reused_pid','ValueError'),('changed_sid','ValueError'),
                          ('permission_denied','PermissionError'),('expired_deadline','TimeoutError')):
            assert c[name]['error']['kind']==kind
        assert c['expired_deadline']['reads']==0
        for p,stage in ((d['partial_memory_unknown'],'execution'),(c['partial_roster_unknown'],'cleanup')):
            assert p['before']['remaining_session']['live']==[101]
            assert p['after']['remaining_session'] is None and not p['after']['remaining_session_known']
            assert not p['after'][stage+'_error']['observation_known']
        signals=rows[2]['details']
        for name,r in signals.items():
            assert r['fd_closed'] and r['opens']==(0 if name=='deadline_no_open' else 1)
            if name=='same_identity_signal': assert len(r['signals'])==1 and r['error'] is None
            else: assert not r['signals'] and r['error']==('TimeoutError' if name=='deadline_no_open' else 'ValueError')
        sessions=result['sessions']; assert len(sessions)==4
        assert [r['status'] for r in sessions]==['ERROR','CLEANUP_INCOMPLETE','TIMEOUT','COMPLETED']
        first,second,third,normal=sessions
        assert first['execution_error']['message']=='live process RSS unavailable' and first['kill_signals']>=2
        assert second['execution_error']['kind']=='ValueError' and second['cleanup_error']['kind']=='PermissionError'
        assert second['remaining_session'] is None and not second['remaining_session_known']
        assert third['execution_error']['kind']=='TimeoutError'
        # Deadline was created before _execute recorded its own start. Check
        # actual deadline crossing, not a spurious 30ms minimum execution time.
        assert third['started_at']+third['execution_seconds']>=third['run_deadline']
        assert 0<third['run_deadline']-third['started_at']<.04
        assert normal['returncode']==0 and normal['execution_error'] is None and normal['sampled_session_plus_caller_rss']>0
        real=rows[3]['details']['execution_rejection_cleanup']; assert real['session_index']==0
        assert real['members'] and real['complete_memory_samples']>0
        assert any(set(real['members'])&set(s['live']) for s in real['cleanup_scans'])
        for r in sessions:
            assert start<=r['started_at']<=r['finished_at']<r['cleanup_deadline']<start+300
            assert r['cleanup_contract']=='IDENTITY_ONLY_V1'
            if r['cleanup_error'] is None:
                assert r['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION' and not r['remaining_session']['live']
                assert r['remaining_session']['rss'] is None and r['remaining_session']['rss_known'] is False
            for error in (r['execution_error'],r['cleanup_error']):
                if error: assert len(packed(error))<=2048
        reaps=result['post_failure_reaps']; assert len(reaps)==1
        assert reaps[0]['pid']==second['session'] and reaps[0]['changes_original_acceptance'] is False
        assert not reaps[0]['post_observation']['live']
        assert result['initial_fds']==result['final_fds']
        assert result['sampled_managed_rss_bytes']<=freeze['caps']['rss_bytes']
    else:
        assert not out['online_acceptance'],'offline audit cannot rescue a failed execution'
    files={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in (root/'result.json',root/'reception.json',Path(stdout),Path(stderr))}
    size=sum(v['bytes'] for v in files.values()); assert size<freeze['caps']['artifact_bytes']
    assert set(p.name for p in root.iterdir())=={'result.json','reception.json'}
    return dict(status='PASS_SAVED_IDENTITY_CLEANUP_CONTROLS' if complete else 'RETAINED_INCOMPLETE',
        freeze_sha256=result['freeze_sha256'],result_sha256=digest(result),files=files,artifact_bytes=size,
        passed=sum(r['status']=='PASS' for r in rows),unexecuted=result['unexecuted'],seconds=stamp['seconds'],
        process_exit_code=int(exit_code),runtime_reexecuted=False,algorithm_efficacy=False,
        seven_position_batch_authorized=False)


if __name__=='__main__':
    value=static() if sys.argv[1:]==['--static'] else audit(*sys.argv[1:])
    print(json.dumps(value,sort_keys=True,indent=2))
