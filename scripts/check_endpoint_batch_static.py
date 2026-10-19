"""AST/compile-only preparation checks. Never imports any project runtime."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=('scoped_source/endpoint_batch.py','scoped_source/endpoint_component_supervised.py',
    'scoped_source/endpoint_batch_observer.py','scoped_proof/owned_session.py',
    'scripts/endpoint_midpoint_caller.py',
    'scripts/run_endpoint_midpoint_controls.py','scripts/audit_endpoint_midpoint_controls.py',
    'scripts/endpoint_midpoint_components.py','scoped_source/native_supervised.py',
    'scoped_source/native_production.py','scoped_proof/owned_bounded.py')


def check():
    parsed={}; hashes={}; bootstraps=0
    for name in FILES:
        raw=(ROOT/name).read_bytes(); tree=ast.parse(raw,filename=name)
        compile(tree,name,'exec'); parsed[name]=tree; hashes[name]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(tree):
            if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)
                    and isinstance(node.value.value,str)
                    and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in node.targets)):
                compile(node.value.value,name+':bootstrap','exec'); bootstraps+=1
    assert bootstraps==4
    registry=json.loads((ROOT/'docs/hz_endpoint_execution_approvals.json').read_bytes())
    assert registry==dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[])
    # No tests or runtime body is called to obtain these literals.
    component=parsed['scripts/endpoint_midpoint_components.py']
    ids=next(ast.literal_eval(n.value) for n in component.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='IDS' for t in n.targets))
    passed=[n.args[0].value for n in ast.walk(component) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='passed' and n.args
        and isinstance(n.args[0],ast.Constant)]
    assert len(ids)==len(passed)==22 and set(ids)==set(passed)
    protocol=parsed['scoped_source/endpoint_batch.py']
    order=next(ast.literal_eval(n.value) for n in protocol.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='ORDER' for t in n.targets))
    assert order==('components','guarded_baseline','guarded_midpoint','mixed_midpoint',
                   'mixed_baseline','midpoint_block','partial_candidate')
    # Directory-only evidence stays pending. No execution to test the receiver.
    audit=(ROOT/'scripts/audit_endpoint_midpoint_controls.py').read_text()
    assert "'CONSISTENT_PENDING_CALLER_OBSERVATION'" in audit
    assert 'post_return_observation_bound=False' in audit
    assert 'online_batch_acceptance=False' in audit
    assert "'PASS_TRUSTED_CALLER_OBSERVATION'" in audit
    observer=(ROOT/'scoped_source/endpoint_batch_observer.py').read_text()
    assert "schema='ENDPOINT_CALLER_PROPOSAL_V2'" in observer
    assert 'def observe_return(proposal, *, start_anchor, start_sha256)' in observer
    assert "anchor['all_work_complete'] is False" in observer
    worker=(ROOT/'scripts/run_endpoint_midpoint_controls.py').read_text()
    assert 'ENTRY_STARTED' not in worker and "write_json(root/'batch-received.json'" not in worker
    assert 'No standalone timer/CLI' in worker
    # No new supervisor/clock fallback: only an external pre-request anchor can
    # supply started. These are source checks, NOT executed rejection controls.
    tree=parsed['scoped_source/endpoint_batch_observer.py']
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    execute=functions['execute']
    assert [a.arg for a in execute.args.kwonlyargs]==['start_anchor','start_sha256','enabled']
    assert execute.args.kw_defaults[:2]==[None,None]
    origins=[n.value for n in ast.walk(execute) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='started' for t in n.targets)]
    assert len(origins)==1 and ast.unparse(origins[0])=="start['started']"
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr=='setrlimit' for n in ast.walk(execute))
    reception=functions['observe_return']
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='live_clock' for n in ast.walk(reception))
    checked_at=next(n.lineno for n in ast.walk(reception) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='live_clock')
    observed_at=next(n.lineno for n in ast.walk(reception) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='observed' for t in n.targets))
    assert checked_at<observed_at
    assert 'start_sha256' in audit and 'startup_seconds' in audit
    assert "start_sha256=a['start_sha256']" in worker
    caller=parsed['scripts/endpoint_midpoint_caller.py']
    entry=next(n for n in caller.body if isinstance(n,ast.FunctionDef) and n.name=='receive_once')
    assert [a.arg for a in entry.args.kwonlyargs]==['start_anchor','start_sha256','enabled']
    assert entry.args.kw_defaults[:2]==[None,None] and len(entry.body)==2
    calls=(entry.body[0].value,entry.body[1].value)
    assert [n.func.id for n in calls]==['execute','observe_return']
    assert all({k.arg:ast.unparse(k.value) for k in n.keywords}.items()
        >= {'start_anchor':'start_anchor','start_sha256':'start_sha256'}.items() for n in calls)
    assignments={t.id:n.value for n in protocol.body if isinstance(n,ast.Assign)
        for t in n.targets if isinstance(t,ast.Name)}
    def literal(node):
        if isinstance(node,ast.Constant): return node.value
        if isinstance(node,ast.Name):
            return 1048576 if node.id=='REPORT_CAP' else literal(assignments[node.id])
        if isinstance(node,ast.BinOp):
            a,b=literal(node.left),literal(node.right)
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,ast.Pow): return a**b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            assert not node.args and all(k.arg is not None for k in node.keywords)
            return {k.arg:literal(k.value) for k in node.keywords}
        raise AssertionError('closed static literal')
    current=literal(assignments['LIMITS'])
    old=json.loads((ROOT/'docs/hz_endpoint_midpoint_freeze_candidate_20261019.json').read_bytes())['limits']
    assert set(current)==set(old)
    assert {k for k in old if current[k]!=old[k]}=={'observation'}
    assert current['observation']=='TRUSTED_PRELAUNCH_AND_POST_PUBLICATION_RETURN_V2'
    session=(ROOT/'scoped_proof/owned_session.py').read_text()
    assert 'pidfd_send_signal' in session and 'os.kill(' not in session and 'os.killpg(' not in session
    owned=(ROOT/'scoped_proof/owned_bounded.py').read_text()
    assert "**({} if owner_session is None else {'owner_session':owner_session})" in owned
    native_metadata=7*1048576+6*65536+65536
    component_metadata=131072+6*1048576+3*65536
    assert native_metadata==7798784 and component_metadata==6619136
    assert max(native_metadata,component_metadata)<8*1048576
    assert 14<=16 and 10<=16
    batch_metadata=131072+2*1048576+65536
    assert batch_metadata==2293760 and batch_metadata<4*1048576
    pending=list(FILES)+['act/back_end/moe/source_row_endpoints.py',
                        'act/back_end/moe/scaled_forest_endpoints.py','scoped_source/native_portable.py']
    found=set()
    while pending:
        name=pending.pop()
        if name in found: continue
        raw=(ROOT/name).read_bytes(); assert len(raw)<=128*1024
        tree=ast.parse(raw,filename=name); compile(tree,name,'exec'); found.add(name)
        for node in ast.walk(tree):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else
                     [node.module] if isinstance(node,ast.ImportFrom) else [])
            for module in modules:
                if module and module.split('.')[0] in ('scripts','scoped_source','scoped_proof',
                        'source_enclosure','upstream_source','router_source'):
                    target=module.replace('.','/')+'.py'
                    if (ROOT/target).is_file(): pending.append(target)
    return dict(status='PASS',scope='AST_AND_COMPILE_ONLY_NOT_CONTROL_RESULTS',files=len(FILES),
        bootstraps=bootstraps,dependency_files=len(found),components=22,positions=7,
        native_metadata_upper_bytes=native_metadata,component_metadata_upper_bytes=component_metadata,
        batch_metadata_upper_bytes=batch_metadata,
        startup='REQUIRED_EXTERNAL_ANCHOR_NO_FALLBACK',numeric_limits_unchanged=True,
        approvals=0,project_runtime_imports=0,controls_executed=0,models=0,gpu=0,sha256=hashes)


if __name__=='__main__': print(json.dumps(check(),sort_keys=True,indent=2))
