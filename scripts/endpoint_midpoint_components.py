"""Prepared finite component body; no CLI or execution authority.

The future hard-limited batch runner must authenticate this file, the binding,
and its approved native profile before calling run(). Nothing runs on import.
Fixed-dual packages below are evidence fixtures, NOT optimizer performance.
"""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import resource
import subprocess
import sys
import time

IDS = (
    'default_identity', 'width8_mates', 'synchronous_midpoint', 'singleton_gate',
    'width2_mates', 'missing_mate', 'duplicate_slot', 'wrong_property',
    'cross_pair', 'objective_binding', 'foreign_roots', 'frame_binding',
    'revoked_lease', 'private_output', 'positive_inequality', 'finite_values',
    'midpoint_exception', 'cooperative_stop', 'same_candidate_oracle',
    'portable_recomputation', 'portable_binding', 'completed_counts',
)


def run(root, profile, binding, expected_binding, work):
    """One shared Work/deadline. Caller must supply the supervised invocation.

    Deliberately not wired to any batch approval yet. The component has three
    successful one-tile searches and one expected-error search; package/oracle
    work uses fixed duals, with no extra search. All three sources are synthetic.
    """
    started=time.monotonic(); initial=dict(work.counts)
    from scoped_source.hz_request_store import POLICY, compact, digest, identity, require
    from scoped_source.native_protocol import validate_profile, verify_code
    from scoped_source.native_supervised import environment, clean_env
    from scoped_source.native_method import method, validate, MIDPOINT_SCHEMA
    profile = validate_profile(profile)  # Existing authority rejects unregistered profiles.
    require(identity(binding) == expected_binding and set(binding) ==
        {'schema', 'profile', 'sources', 'code', 'cases'}
        and binding['schema'] == 'ENDPOINT_COMPONENT_BINDING_V1'
        and binding['profile'] == identity(profile) and binding['cases'] == list(IDS),
        'externally frozen complete component binding')
    require(set(binding['sources']) == {'guarded', 'mixed', 'half'}, 'finite component source roster')
    require(profile['method'] == method('native_source_row_blocks', endpoint_midpoint=True),
            'component midpoint identity')
    require(profile['as_bytes'] == 8*2**30 and profile['rss_bytes'] == 2*2**30
            and profile['file_bytes'] == 128*2**20
            and resource.getrlimit(resource.RLIMIT_AS) == (8*2**30, 8*2**30)
            and resource.getrlimit(resource.RLIMIT_FSIZE) == (128*2**20,128*2**20),
            'component inherited approved OS envelope')
    repo = Path(__file__).resolve().parents[1]
    extra={'scripts/endpoint_midpoint_components.py','scoped_source/scaled_endpoint_controls.py'}
    require(set(binding['code']) == set(profile['implementation']) | extra
            and all(binding['code'][n]==h for n,h in profile['implementation'].items())
            and len(binding['code']) <= 256, 'complete component code binding')
    for name, h in binding['code'].items():
        work.tick(); p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
                'local component implementation path')
        path = repo/p
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024
                and digest(path.read_bytes()) == h, 'frozen component implementation')
    verify_code(repo, profile['implementation'], work.deadline)
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_relative_to('/data1/Kane/MOE')
            and root != Path('/data1/Kane/MOE') and not root.exists(), 'new component output only')
    root.mkdir(); work.tick()
    environment(profile, work)
    import torch
    from scoped_source.native_loader import load
    native = load(repo, profile['implementation'], work)
    from scoped_source.scaled_endpoint_controls import expanded, scalar_bound
    from scoped_source.shared_build import build
    from scoped_source.check_shared_source import CheckedSharedForest
    from scoped_source.direct_candidates import Driver, Tile, PREFIX, RECORD, RATIONAL
    from scoped_source.direct_lifecycle import prepare, revoke
    from scoped_source.native_lifecycle import submit_owned_duals
    from scoped_source.scaled_endpoints import contract
    from scoped_source.direct_supervised import census
    from scoped_source.wire_archive import Archive
    from scoped_source.native_package import CODE_FILES, MANIFEST, BUNDLE, check_package
    from scoped_source.native_result import Writer

    tests = {}; searches = []; sources = {}; progress_bytes=0
    def passed(name, **info):
        nonlocal progress_bytes
        work.tick(); require(name in IDS and name not in tests, 'unique finite component result')
        tests[name] = dict(name=name, **info)
        raw=compact(dict(name=name,work=dict(work.counts),at=time.monotonic(),complete=False))+b'\n'
        require(progress_bytes+len(raw)<=65536,'bounded partial component progress')
        with (root/'progress.jsonl').open('xb' if progress_bytes==0 else 'ab',buffering=0) as stream:
            require(stream.write(raw)==len(raw),'component progress short write')
        progress_bytes+=len(raw); work.tick()
    def rejects(fn, reason):
        try: fn()
        except (ValueError, TypeError, AttributeError, RuntimeError, TimeoutError) as exc:
            require(reason in str(exc), 'expected rejection boundary, not unrelated failure')
            return type(exc).__name__
        raise AssertionError('unexpected acceptance: ' + reason)
    for case in ('guarded', 'mixed', 'half'):
        source=root/('source-'+case)
        if case=='mixed':
            from scoped_source.source_dag_controls import mixed
            source,h=mixed(source)
        else:
            from scoped_source.direct_supervised import fixed_source
            h=fixed_source(source,case)
        work.tick()
        require(h == binding['sources'][case], 'frozen component source identity')
        packet, _ = build(source, h, work=work, enabled=True)
        forest = CheckedSharedForest(packet, source, h, expected=packet.identity, work=work, enabled=True)
        sources[case] = (Path(source), h, packet, forest)
    def driver(case):
        f = sources[case][3]; c = contract(f)
        return Driver(f, c, expected=identity(c), work=work, pair_scoped=True)
    def tensor_dual(d, tasks, token):
        def tensor(kind):
            n = len(d.forest.rows(tasks[0].pair, kind))
            work.visit('coefficient_visits', n*len(tasks))
            return torch.tensor([[(-1/16 if kind == 'ine' else ((i+j)%3-1)/8)
                for j in range(len(tasks))] for i in range(n)], dtype=torch.float64).reshape(n, len(tasks))
        return native.DualTile(d, tasks, token, tensor('eq'), tensor('ine'))
    def discard_publish(i, raw): return digest(raw), len(raw)
    def fixed_step(d, tasks, lease, token):
        dual = tensor_dual(d, tasks, token)
        try: return submit_owned_duals(d, token, dual, discard_publish, native=native)
        finally: dual.close(); revoke(lease)
    def one_search(case, *, singleton=False, fault=None):
        d = driver(case); p = native.Proposer(profile['method'], enabled=True, midpoint_control=fault)
        lease = None
        try:
            p.setup(d.forest, work)
            for _ in range(d.session.tile_count):
                tasks, lease, token = prepare(d)
                mates = native.endpoint_mates(tasks, lease)
                if singleton and mates:
                    fixed_step(d, tasks, lease, token); lease = None; continue
                if fault:
                    rejects(lambda:p.propose(tasks, lease), 'CONTROL_INJECTED_midpoint_error')
                    require(p.tile is None and p.state == 'PROPOSING', 'no completed candidate on exception')
                else:
                    dual = p.propose(tasks, lease)
                    submit_owned_duals(d, token, dual, discard_publish, native=native); p.release()
                searches.append(dict(case=case,singleton=singleton,fault=fault,search=dict(p.stats['search'])))
                return len(tasks), len(mates)
            raise AssertionError('required frozen tile absent')
        finally:
            revoke(lease)
            if not d.closed: d.abort()
            p.close()

    # Old identity is a literal external expectation, not method(False) vs itself.
    for arm in ('native_coo_reference', 'native_source_row_blocks'):
        old = dict(schema='NATIVE_SHARED_METHOD_V1',scope='HZ_NATIVE_SOURCE_INTEGRATION_20261017_V1',
            producer=dict(kind='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3',representation=arm,
                device='cpu',dtype='float64',threads=1,profile=0,updates=128,evaluations=129),
            candidate_width=8,production=dict(kind='pair_scoped',check_width=2),
            replay=dict(kind='pair_scoped',check_width=2),
            results=dict(schema='SHARED_RESULT_STREAM_V1',page_rows=8,page_bytes=65536))
        require(compact(method(arm)) == compact(old), 'unchanged default method bytes')
        wrong = method(arm); wrong['producer']['endpoint_exchange'] = {}
        rejects(lambda:validate(wrong), 'complete native method binding')
    passed('default_identity')
    width, pairs = one_search('guarded'); require((width,pairs)==(8,4), 'full-width grouped targets')
    passed('width8_mates')
    best = torch.tensor([[-1.,-3.,-4.,-8.,-8.,-16.,-16.,-32.]],dtype=torch.float64)
    saved = best.clone(); mates = ((0,1),(2,3),(4,5),(6,7))
    mixed = native.midpoint_update(best,'ine',mates,8,work)
    require(torch.equal(mixed,torch.tensor([[-2.,-2.,-6.,-6.,-12.,-12.,-24.,-24.]],dtype=torch.float64))
            and torch.equal(best,saved), 'simultaneous exact binary midpoint control')
    passed('synchronous_midpoint')
    width,pairs = one_search('half',singleton=True)
    require(pairs==0 and width>0, 'actual checked singleton gate')
    passed('singleton_gate')
    require(one_search('mixed')==(2,1), 'actual two-column checked tile'); passed('width2_mates')

    d = driver('guarded'); tasks, lease, token = prepare(d)
    def changed_tasks(new):
        # Test-only corruption of private state, never a production callback.
        # Restore before further cases; no Work/deadline/source is reset.
        pending = d.pending
        try:
            d.pending = (new, token); object.__setattr__(lease,'_tasks',new)
            return native.endpoint_mates(new,lease)
        finally:
            d.pending = pending; object.__setattr__(lease,'_tasks',tasks)
    try:
        rejects(lambda:changed_tasks(tasks[:-1]), 'complete ordered endpoint mates'); passed('missing_mate')
        rejects(lambda:changed_tasks((tasks[0],tasks[0],*tasks[2:])), 'objective/query/slot binding'); passed('duplicate_slot')
        other = next(j for j in range(d.forest.declaration['classes'])
                     if j not in (d.forest.declaration['label'],tasks[0].other))
        rejects(lambda:changed_tasks((replace(tasks[0],other=other),*tasks[1:])), 'objective/query/slot binding')
        passed('wrong_property')
        rejects(lambda:changed_tasks((tasks[0],replace(tasks[1],pair=(1,2)),*tasks[2:])), 'pair/property/weight binding')
        passed('cross_pair')
        rejects(lambda:changed_tasks((replace(tasks[0],offset='1'),*tasks[1:])), 'objective/query/slot binding')
        passed('objective_binding')
        oldforest = d.forest
        try:
            d.forest = sources['mixed'][3]
            rejects(lambda:native.endpoint_mates(tasks,lease), 'root lease request/tile/owner binding')
        finally: d.forest = oldforest
        passed('foreign_roots')
        dual = tensor_dual(d,tasks,token)
        try:
            rejects(lambda:dual.check(d,tasks,object()), 'owned native tile identity/lifetime')
        finally: dual.close()
        revoke(lease); rejects(lambda:native.endpoint_mates(tasks,lease),'revoked root lease')
        passed('revoked_lease')
    finally:
        revoke(lease)
        if not d.closed: d.abort()
    mixed[0,0] = 99.; require(torch.equal(best,saved), 'no output alias to best')
    passed('private_output')
    rejects(lambda:native.midpoint_update(torch.ones((1,2),dtype=torch.float64),'ine',((0,1),),2,work),
            'nonpositive endpoint best duals')
    # Full-frame positive-dual rejection is also checked below, not just helper sign.
    for value in (float('nan'),float('inf')):
        rejects(lambda:native.midpoint_update(torch.tensor([[value,-1.]],dtype=torch.float64),
                'ine',((0,1),),2,work), 'finite endpoint best duals')
    for value in (-float.fromhex('0x1.fffffffffffffp+1023'), -float.fromhex('0x0.0000000000001p-1022')):
        out = native.midpoint_update(torch.tensor([[value,value]],dtype=torch.float64),'ine',((0,1),),2,work)
        require(torch.isfinite(out).all() and (out<=0).all(), 'extreme finite midpoint')
        d=driver('mixed'); tt,ll,tok=prepare(d); dual=tensor_dual(d,tt,tok)
        try:
            require(dual.inequality.shape[0]>0 and len(tt)==2,'extreme full-frame fixture')
            dual.inequality[0,:].copy_(out[0])
            issued=submit_owned_duals(d,tok,dual,discard_publish,native=native)
            require(len(issued)==2,'extreme finite candidates fully checked')
        finally:
            dual.close(); revoke(ll)
            if not d.closed: d.abort()
    passed('finite_values', extreme_values_checked_by_original_exact_consumer=True)
    one_search('guarded',fault='midpoint_error'); passed('midpoint_exception')
    # A bounded test double simulates early/midway denial, without
    # forging/resetting the live cumulative Work or spending 8M fictitious visits.
    class Denied:
        def __init__(self, error, at): self.error=error; self.at=at; self.calls=0
        def gate(self):
            self.calls+=1
            if self.calls==self.at: raise self.error
        def tick(self): work.tick(); self.gate()
        def visit(self,kind,amount=1): work.visit(kind,amount); self.gate()
    for error in (TimeoutError('injected deadline denial'),ValueError('injected Work denial')):
        for at in (1,6):
            denied=Denied(error,at); before=dict(work.counts)
            rejects(lambda:native.midpoint_update(best,'ine',mates,8,denied), str(error))
            require(denied.calls==at and torch.equal(best,saved)
                    and all(work.counts[k]>=v for k,v in before.items()), 'shared Work and no best mutation on denial')
    passed('cooperative_stop', scope='INJECTED_EARLY_AND_MIDWAY_NOT_MEASURED_OS_OR_8M_EXHAUSTION')

    # Self-contained mixed fixed-candidate package. This does not run Proposer.
    outside=census(root,work.deadline)
    remaining=POLICY['total_bytes']-outside['bytes']-8*2**20
    remaining_members=POLICY['members']-len(outside['files'])-16
    require(remaining_members>0,'component source and metadata member reserve')
    class ComponentArchive(Archive):
        def put(self,*args,**kwargs):
            require(len(self.attempted)<remaining_members,'whole component member cap')
            return super().put(*args,**kwargs)
    archive=ComponentArchive(root/'evidence',work,limit=remaining)
    def put(name,raw): return dict(archive.put('package/'+name,raw),file=name)
    source,h,packet,f=sources['mixed']
    inv=identity(dict(component=expected_binding,fixture='fixed_candidates_not_search_execution'))
    inventory=census(source,work.deadline)
    for name in sorted(inventory['files']): archive.copy(source/name,'package/source/'+name)
    fr=packet.publish(put); c=contract(f); cr=put('contract.json',compact(c)); refs=[]
    d=driver('mixed'); expanded_rows=expanded(f,work); oracle=0; first_raw=None
    try:
        for _ in range(d.session.tile_count):
            tasks,lease,token=prepare(d); dual=tensor_dual(d,tasks,token)
            raw_holder=[]
            def publish(i,raw):
                ref=put(f'tiles/{i:04d}.bin',raw); refs.append(ref); raw_holder.append(raw)
                return ref['sha256'],ref['bytes']
            try: submit_owned_duals(d,token,dual,publish,native=native)
            finally: dual.close(); revoke(lease)
            raw=raw_holder[0]
            if first_raw is None: first_raw=raw
            # Independent scalar oracle consumes the exact fixed dual formula,
            # not the proposed lower value or production pullback implementation.
            for col,(task,receipt) in enumerate(zip(tasks,d.receipts[-len(tasks):],strict=True)):
                for role in ('proposed','zero'):
                    candidate={key:[str(F(-1,16) if kind=='ine' else F((i+col)%3-1,8))
                        if role=='proposed' else '0' for i in range(len(f.rows(task.pair,kind)))]
                        for kind,key in (('eq','equality'),('ine','inequality'))}
                    low,_=scalar_bound(f,task,candidate,expanded_rows,work)
                    require(low==F(receipt.record[role]),'same fixed candidate exact oracle'); oracle+=1
        sink=Writer(put,profile['method'],inv,fr['sha256'],refs,work)
        result=d.finish(sink=sink)
    finally:
        if not d.closed: d.abort()
    require(oracle==12, 'mixed six endpoints and two roles')
    passed('same_candidate_oracle', roles=oracle, fixed_candidates_not_search_execution=True)
    import struct
    eq=PREFIX.unpack_from(first_raw)[6]
    for edit,reason in (
        (lambda b:b.__setitem__(slice(40,72),bytes(32)), 'frame invocation/header binding'),
        (lambda b:struct.pack_into('<I',b,112,eq+1), 'complete frame dual dimensions'),
    ):
        d=driver('mixed'); _,_,token=d.prepare(); corrupted=bytearray(first_raw); edit(corrupted)
        try: rejects(lambda:d.receive(token,bytes(corrupted),discard_publish),reason)
        finally:
            if not d.closed: d.abort()
    # A same-frame numerical permutation is a different LEGAL dual, not a
    # source error. Check it against original rows, including the zero role.
    ine=PREFIX.unpack_from(first_raw)[7]
    require(eq>=2 or ine>=2,'existing source has two constraint rows')
    base=PREFIX.size+RECORD.size+2*RATIONAL+(0 if eq>=2 else 8*eq)
    permuted=bytearray(first_raw)
    struct.pack_into('<dd',permuted,base,-1/16,-1/8)
    left,right=bytes(permuted[base:base+8]),bytes(permuted[base+8:base+16])
    permuted[base:base+8]=right; permuted[base+8:base+16]=left
    raw_permuted=bytes(permuted); d=driver('mixed'); tt,_,token=d.prepare()
    try:
        tile=Tile(raw_permuted,d.owner,f,d.contract,tt,d.ordinal)
        records=[{role:{key:[str(tile.weight(col,role,key,j)) for j in range(len(f.rows(t.pair,kind)))]
            for kind,key in (('eq','equality'),('ine','inequality'))} for role in ('proposed','zero')}
            for col,t in enumerate(tt)]
        receipts=d.receive(token,raw_permuted,discard_publish)
        for t,rec,receipt in zip(tt,records,receipts,strict=True):
            for role in ('proposed','zero'):
                low,_=scalar_bound(f,t,rec[role],expanded_rows,work)
                require(low==F(receipt.record[role]),'permuted dual original-coefficient oracle')
    finally:
        if not d.closed: d.abort()
    passed('frame_binding', numerical_dual_permutations_are_not_intrinsically_invalid=True,
           permuted_roles_checked=2*len(tt))
    corrupted=bytearray(first_raw)
    struct.pack_into('<d',corrupted,PREFIX.size+RECORD.size+2*RATIONAL+8*eq,1.)
    d=driver('mixed'); _,_,token=d.prepare()
    try:
        rejects(lambda:d.receive(token,bytes(corrupted),discard_publish),'sign')
    finally:
        if not d.closed: d.abort()
    passed('positive_inequality')
    code=[]
    for name in CODE_FILES:
        ref=archive.copy(repo/name,'package/code/'+name)
        require(ref['sha256']==profile['implementation'][name],'frozen portable code')
        code.append(dict(ref,file='code/'+name))
    put('manifest.json',compact(dict(schema=MANIFEST,source=dict(directory='source',sha256=h),forest=fr,
        contract=cr,tiles=refs,code=code,method=profile['method'],invocation=inv,result=dict(sink.ref))))
    entries=[dict(r,file=n.removeprefix('package/')) for n,r in sorted(archive.entries.items())]
    bundle=put('bundle.json',compact(dict(schema=BUNDLE,manifest='manifest.json',members=entries)))
    archive.relocate('relocated',tuple(archive.entries))
    relocated=archive.root/'relocated/package'
    ext=dict(bundle=bundle['sha256'],method=profile['method'],invocation=inv,source=h,
        code={n:profile['implementation'][n] for n in CODE_FILES},
        budget=dict(initial_work=dict(work.counts),deadline=work.deadline))
    extref=archive.put('external.json',compact(ext))
    stdout=root/'portable.stdout'; stderr=root/'portable.stderr'
    bootstrap='import resource,runpy,sys; resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576)); p=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(p,run_name="__main__")'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        # No new session and no unbounded subprocess.run timeout cleanup. The
        # existing owned group retains and reaps every descendant on failure.
        proc=subprocess.Popen([sys.executable,'-I','-S','-B','-c',bootstrap,
            str(relocated/'code/scoped_source/native_portable.py'),'--enabled',str(relocated),
            str(archive.root/extref['file'])],cwd=relocated,env=clean_env(),stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,start_new_session=False)
        proc.wait(timeout=max(.001,work.deadline-time.monotonic()))
    work.tick(); require(stdout.stat().st_size<=2**20 and stderr.stat().st_size<=2**20,'bounded child logs')
    from scoped_source.hz_request_store import decode
    rr=decode(stdout.read_bytes())
    require(set(rr['work'])==set(work.counts) and all(type(v) is int and v>=work.counts[k]
        for k,v in rr['work'].items()) and sum(rr['work'].values())<=POLICY['visits'], 'cumulative portable Work')
    work.counts.update(rr['work']); work.tick()
    require(proc.returncode==0 and rr['result']==result and rr['online_acceptance'] is False,
            'fresh relocated exact recomputation, not an online certificate')
    passed('portable_recomputation', checkpoint_loaded=False, fixed_candidates_not_search_execution=True)
    for wrong_method,wrong_inv in ((method('native_source_row_blocks'),inv),(profile['method'],'0'*64)):
        rejects(lambda:check_package(relocated,expected=bundle['sha256'],expected_method=wrong_method,
            invocation=wrong_inv,expected_code=ext['code'],expected_source=h,work=work,enabled=True),
            'native manifest identity/schema')
    passed('portable_binding')
    require(len(searches)==4,'exact finite search-call roster')
    for s in searches:
        c=s['search']
        if s['fault']:
            require((c['evaluations'],c['forward_calls'],c['gradient_updates'],c['midpoint_updates'])
                    ==(128,128,127,0),'failed seam completed prefix counts')
        else:
            require((c['evaluations'],c['forward_calls'])==(129,128),'complete fixed evaluation count')
            require((c['gradient_updates'],c['midpoint_updates'])==
                    ((128,0) if s['singleton'] else (127,1)), 'singleton/paired update count')
    passed('completed_counts', searches=searches)
    archive.audit(); storage=census(root,work.deadline)
    require(storage['bytes']+2**20<=POLICY['total_bytes'] and len(storage['files'])+1<=POLICY['members'],
            'all component sources/copies/logs and final report charged')
    require(set(tests)==set(IDS),'all finite component obligations')
    work.tick()
    return dict(schema='ENDPOINT_COMPONENT_RESULT_V1',status='PASS',binding=expected_binding,
        tests=[tests[k] for k in IDS],initial_work=initial,work=dict(work.counts),storage=storage,
        seconds_before_report=time.monotonic()-started,source_factory_calls=3,successful_search_tiles=3,
        failed_search_tiles=1,fixed_package_optimizer_calls=0,models=0,gpu=0,
        scope='SYNTHETIC_COMPONENTS_NOT_COMPLETE_SHAPE_OR_RESEARCH_GAIN')
