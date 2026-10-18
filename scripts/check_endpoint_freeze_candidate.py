"""Read-only preparation/check of a NOT-READY endpoint freeze candidate.

Stdlib AST/literal data and file hashes only: no project imports, exec/eval,
source factories, optimizer, registry writes or process launches. --emit prints
proposed artifacts for review; applying them is a separate repository edit.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
SELF='scripts/check_endpoint_freeze_candidate.py'
FREEZE='docs/hz_endpoint_midpoint_freeze_candidate_20261019.json'
REVIEW='docs/hz_endpoint_midpoint_freeze_review_20261019.json'
NATIVE='docs/hz_native_execution_approvals.json'
BATCH='docs/hz_endpoint_execution_approvals.json'
HISTORY={
    'docs/hz_native_controls_freeze_20261018.json':'77c82312195e7698af6dc26d2f81aa25ec3f80e1b0749df5215ec6b478dc8834',
    'docs/hz_scaled_batch_20261013.json':'9400561bb4dcfeaa993fb9c8a31d6d8e316ea117c68e8893b32752ec4964b21b'}
CURRENT={NATIVE:'cc81fb912403cbff4e3b5568dcb1016f450772973558b9960200ca1900f94c0d',
         BATCH:'ac4fd56a29e87ffb3d6d39bcaf8e32a74e2463baa8c09ee433f6964518ba054b',
         'docs/hz_native_resource_execution_approvals.json':'4ed38318d04d0b48b34fffe6359b1dc53bb01984a44b712c7138745bd5fb7af4'}
PREFIXES=('scoped_source','scoped_proof','source_enclosure','upstream_source','router_source')


def compact(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def ident(obj): return sha(compact(obj))
def need(ok,why):
    if not ok: raise ValueError(why)
def tick(deadline): need(time.monotonic()<deadline,'finite read-only preparation deadline')


def read(path, cap=512*1024):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1 and path.stat().st_size<=cap,
         'bounded unaliased input: '+str(path))
    with path.open('rb') as stream: raw=stream.read(cap+1)
    need(len(raw)<=cap,'input grew'); return raw


def literal(node, lookup):
    """Only the closed literal expressions used in the inspected contracts."""
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,ast.Name): return lookup(node.id)
    if isinstance(node,(ast.Tuple,ast.List)): return [literal(x,lookup) for x in node.elts]
    if isinstance(node,ast.BinOp):
        a,b=literal(node.left,lookup),literal(node.right,lookup)
        need(type(a) in (int,float) and type(b) in (int,float),'numeric constant operands')
        if isinstance(node.op,ast.Mult): return a*b
        if isinstance(node.op,ast.Add): return a+b
        if isinstance(node.op,ast.Pow):
            need(type(b) is int and 0<=b<=64,'small static exponent'); return a**b
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
        if node.func.id=='dict' and not node.args and all(k.arg is not None for k in node.keywords):
            return {k.arg:literal(k.value,lookup) for k in node.keywords}
        if node.func.id=='MappingProxyType' and len(node.args)==1 and not node.keywords:
            return literal(node.args[0],lookup)
    raise ValueError('not a permitted literal expression: '+ast.dump(node))


def syntax(name):
    raw=read(ROOT/name,128*1024); tree=ast.parse(raw,filename=name); compile(tree,name,'exec')
    return tree


def constant(name,key,extras=None):
    assignments={t.id:n.value for n in syntax(name).body if isinstance(n,ast.Assign)
                 for t in n.targets if isinstance(t,ast.Name)}
    extras={} if extras is None else extras
    def lookup(k):
        if k in extras: return extras[k]
        need(k in assignments,'declared literal constant: '+k)
        return literal(assignments[k],lookup)
    return lookup(key)


def closure(seeds, prefixes, deadline):
    pending=list(seeds); found={}
    while pending:
        tick(deadline); name=pending.pop()
        if name in found: continue
        need(len(found)<256,'finite code closure')
        tree=syntax(name); found[name]=sha(read(ROOT/name,128*1024))
        for node in ast.walk(tree):
            mods=([a.name for a in node.names] if isinstance(node,ast.Import) else
                  [node.module] if isinstance(node,ast.ImportFrom) else [])
            for module in mods:
                if module and module.split('.')[0] in prefixes:
                    target=module.replace('.','/')+'.py'
                    if (ROOT/target).is_file(): pending.append(target)
    return dict(sorted(found.items()))


def methods(old):
    name='scoped_source/native_method.py'; tree=syntax(name)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='method')
    need([a.arg for a in fn.args.kwonlyargs]==['endpoint_midpoint']
         and ast.literal_eval(fn.args.kw_defaults[0]) is False,'default-off search')
    lookup=lambda k: 'native_source_row_blocks' if k=='arm' else constant(name,k)
    base=literal(next(n.value for n in fn.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets)),lookup)
    need(base==old,'old source-row method remains exact reference')
    new=json.loads(compact(base))
    change=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name)
                and n.test.id=='endpoint_midpoint')
    need(len(change.body)==2,'single closed method delta')
    for expression,target in zip(change.body,(new,new['producer']),strict=True):
        need(isinstance(expression,ast.Expr) and isinstance(expression.value,ast.Call)
             and isinstance(expression.value.func,ast.Attribute) and expression.value.func.attr=='update'
             and not expression.value.args,'literal update only')
        target.update({k.arg:literal(k.value,lookup) for k in expression.value.keywords})
    need(base['producer']['evaluations']==new['producer']['evaluations']==129
         and base['producer']['updates']==new['producer']['updates']==128,'no extra search calls')
    return base,new


def file_hash(path, size, expected, deadline):
    path=Path(path); tick(deadline)
    need(path.is_file() and path.stat().st_size==size,'historical artifact size: '+str(path))
    h=hashlib.sha256(); n=0
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(65536),b''):
            tick(deadline); n+=len(raw); need(n<=size,'artifact grew'); h.update(raw)
    need(n==size and h.hexdigest()==expected,'historical artifact hash: '+str(path))


def build(deadline):
    history={}
    for n,h in HISTORY.items():
        raw=read(ROOT/n); need(sha(raw)==h,'frozen historical input'); history[n]=json.loads(raw)
    old=history[next(iter(HISTORY))]
    source={k:old['source_previews'][k]['source'] for k in ('guarded','mixed')}
    source['half']=next(x['source'] for x in history['docs/hz_scaled_batch_20261013.json']['sources'] if x['name']=='half')
    source_records={}
    for case,h in source.items():
        root=Path('/data1/Kane/MOE/runs/hz_scaled_batch_20261013_r1')/case/'bundle/source'
        raw=read(root/'manifest.json'); need(sha(raw)==h,'historical declared source identity')
        m=json.loads(raw); members={'manifest.json'}; size=len(raw)
        for tensor in m['tensors']:
            for chunk in tensor['chunks']:
                name=chunk['file']; p=Path(name)
                need(not p.is_absolute() and '..' not in p.parts and name not in members,'unique local source member')
                file_hash(root/name,chunk['bytes'],chunk['sha256'],deadline)
                members.add(name); size+=chunk['bytes']
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        need(actual==members and not any(p.is_symlink() for p in root.rglob('*')),'exact archived source inventory')
        source_records[case]=dict(directory=str(root),sha256=h,bytes=size,members=len(members))
    env=old['records'][0]['profile']['environment']
    need(len(env['files'])==6 and env['torch_version']=='2.9.1+cu128','same six primary environment artifacts')
    for r in env['files']: file_hash(r['file'],r['bytes'],r['sha256'],deadline)
    registry={}
    for n,h in CURRENT.items():
        raw=read(ROOT/n)
        need(sha(raw)==h,'authority changed since preparation')
        registry[n]=dict(sha256=sha(raw),value=json.loads(raw))
    need(registry[BATCH]['value']==dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[]),'batch authority stays empty')
    need(len(registry[NATIVE]['value']['profiles'])==26,'preserve all old profiles')
    fn=next(n for n in syntax('scoped_source/native_protocol.py').body if isinstance(n,ast.FunctionDef) and n.name=='closure')
    seeds=next(ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='pending' for t in n.targets))
    code=closure(seeds,PREFIXES,deadline)
    proto='scoped_source/endpoint_batch.py'; ext={'REPORT_CAP':1048576}
    batch_code=closure(list(code)+constant(proto,'SEEDS'),('scripts',)+PREFIXES,deadline)
    need('scripts/endpoint_midpoint_caller.py' in batch_code,'caller code actually bound')
    caller=syntax('scripts/endpoint_midpoint_caller.py')
    call=next(n for n in caller.body if isinstance(n,ast.FunctionDef) and n.name=='receive_once')
    need(len(call.body)==2 and isinstance(call.body[0],ast.Assign) and isinstance(call.body[1],ast.Return)
         and call.body[0].value.func.id=='execute' and call.body[1].value.func.id=='observe_return',
         'return observation strictly after observer returns')
    run_dir=constant('scripts/endpoint_midpoint_caller.py','RUN_DIRECTORY')
    need(not Path(run_dir).exists(),'new future directory, no resume')
    base=next(r['profile'] for r in old['records'] if r['id']=='normal-guarded-native_source_row_blocks')
    baseline,midpoint=methods(base['method']); order=constant(proto,'ORDER')
    need(order==['components','guarded_baseline','guarded_midpoint','mixed_midpoint',
                 'mixed_baseline','midpoint_block','partial_candidate'],'same seven positions')
    need(constant('scoped_source/hz_request_store.py','POLICY')==base['policy'],'policy unchanged')
    records=[]
    for i,(case,on) in enumerate(zip(('guarded','guarded','guarded','mixed','mixed','guarded','guarded'),
                                    (True,False,True,True,False,True,True),strict=True)):
        p=dict(schema='NATIVE_SHARED_EXECUTION_V1',method=midpoint if on else baseline,
            source=dict(kind='fixed_control',case=case,source=source[case]),policy=base['policy'],
            seconds=300,as_bytes=8589934592,rss_bytes=2147483648,file_bytes=134217728,
            implementation=code,environment=env)
        if i>=5: p['control']=order[i]
        records.append(dict(id=order[i],profile=p,sha256=ident(p)))
    added=sorted({r['sha256'] for r in records}); need(len(added)==6,'six unique profiles, component reuses guarded midpoint')
    projected_native=dict(schema='NATIVE_EXECUTION_APPROVAL_REGISTRY_V1',
        profiles=sorted(set(registry[NATIVE]['value']['profiles'])|set(added)))
    need(len(projected_native['profiles'])==32,'append exactly six, never replace old approvals')
    component_code=dict(code)
    for n in ('scripts/endpoint_midpoint_components.py','scoped_source/scaled_endpoint_controls.py'):
        component_code[n]=batch_code[n]
    ids=constant('scripts/endpoint_midpoint_components.py','IDS'); need(len(ids)==22,'same component count')
    f=dict(schema='ENDPOINT_FINITE_FREEZE_V2',limits=constant(proto,'LIMITS',ext),
        admission=constant(proto,'ADMISSION'),policy=base['policy'],records=records,
        component=dict(schema='ENDPOINT_COMPONENT_BINDING_V1',profile=records[0]['sha256'],
            sources=source,code=component_code,cases=ids),code=batch_code,native_registry=ident(projected_native))
    need(len(compact(f))<=131072,'same freeze cap')
    review=dict(schema='ENDPOINT_FREEZE_CANDIDATE_REVIEW_V1',status='NOT_READY_CALLER_HOST_UNBOUND',
        base_head='5e8b2784bbdb6c3403d86d49f1f329829c326f7b',freeze_sha256=ident(f),freeze_bytes=len(compact(f)),
        code_counts=dict(native=len(code),component=len(component_code),batch=len(batch_code)),
        static_checker_sha256=sha(read(ROOT/SELF,128*1024)),historical_inputs=HISTORY,
        sources=source_records,environment_files_checked=6,environment_bytes=sum(r['bytes'] for r in env['files']),
        registry_before={n:r['sha256'] for n,r in registry.items()},added_profiles=added,
        authority_projection=dict(native=projected_native,native_sha256=ident(projected_native),
            batch=dict(schema='ENDPOINT_BATCH_APPROVALS_V1',batches=[ident(f)]),
            encoding='COMPACT_SORTED_UTF8_NO_TRAILING_NEWLINE'),
        caller=dict(file='scripts/endpoint_midpoint_caller.py',sha256=batch_code['scripts/endpoint_midpoint_caller.py'],
            function='receive_once',directory=run_dir,host_identity=None,
            host_ready=False,fresh_interpreter_is_not_preexisting_service=True,
            online_reception='LIVE_MEMORY_POST_RETURN',offline_archival='SEPARATELY_CHARGED_NO_PROMOTION'),
        execution_ready=False,execution_authorized=False,registry_writes=0,source_factory_calls=0,
        project_runtime_imports=0,controls_executed=0,model_runs=0,gpu_runs=0)
    tick(deadline); return {FREEZE:f,REVIEW:review}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--emit',action='store_true'); args=p.parse_args()
    started=time.monotonic(); artifacts=build(started+60)
    if args.emit:
        # Preserve canonical number spelling across tooling; still data, no writes.
        print(compact({n:compact(v).decode() for n,v in artifacts.items()}).decode())
    else:
        for n,expected in artifacts.items():
            raw=read(ROOT/n,131072); need(json.loads(raw)==expected and raw==compact(expected),'candidate drift: '+n)
        review=artifacts[REVIEW]
        print(json.dumps(dict(status='PASS_STATIC_CANDIDATE_NOT_EXECUTION_READY',seconds=time.monotonic()-started,
            freeze_sha256=review['freeze_sha256'],freeze_bytes=review['freeze_bytes'],
            code_counts=review['code_counts'],environment_files=6,sources=review['sources'],
            active_native_profiles=26,active_batch_approvals=0,new_unique_profiles=6,
            runtime_controls=0,project_runtime_imports=0,blocker='TRUSTED_CALLER_HOST_NOT_BOUND'),sort_keys=True))
