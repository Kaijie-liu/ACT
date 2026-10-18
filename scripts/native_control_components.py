"""One frozen, bounded CPU component batch. Not a model/performance experiment."""
import json
import mmap
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time
from fractions import Fraction as F


def main(root, freeze):
    started=time.monotonic()
    from scoped_source.native_protocol import validate_profile, verify_code
    from scoped_source.native_supervised import environment
    from scoped_source.hz_request_store import Work, compact, digest, identity, require, POLICY
    from scoped_source.direct_supervised import census
    from scoped_source.native_loader import load
    from scoped_source.native_method import method
    from scoped_source.scaled_endpoint_controls import fixture, expanded, scalar_bound
    from scoped_source.shared_build import build
    from scoped_source.check_shared_source import CheckedSharedForest
    from scoped_source.shared_source import SourceLocal
    from scoped_source.direct_candidates import Driver, Tile, PREFIX, RECORD, RATIONAL
    from scoped_source.direct_lifecycle import prepare, revoke
    from scoped_source.native_lifecycle import submit_owned_duals
    from scoped_source.scaled_endpoints import contract
    profile=validate_profile(freeze['records'][0]['profile']); root=Path(root); root.mkdir()
    repo=Path(__file__).resolve().parents[1]; w=Work(started+298)
    require(resource.getrlimit(resource.RLIMIT_AS)==(8*2**30,8*2**30), 'component actual AS')
    verify_code(repo,profile['implementation'],w.deadline)
    environment(profile,w)
    import torch
    native=load(repo,profile['implementation'],w); tests=[]
    def passed(name, **info):
        w.tick(); require(len(tests)<32, 'frozen component count'); tests.append(dict(name=name,**info))
    def reject(name,fn):
        try: fn()
        except (ValueError,TypeError,AttributeError,RuntimeError): passed(name); return
        raise AssertionError('unexpected acceptance: '+name)
    def forest(case):
        dest=root/case
        if case=='tail':
            from scoped_source.affine_reference_controls import deep
            source,h=deep(dest,center=(1.,float.fromhex('0x1p-1000')),radius='1/4')
        elif case=='guarded':
            from scoped_source.direct_supervised import fixed_source
            source=dest; h=fixed_source(source,case)
        else: source,h=fixture(dest,case)
        packet,_=build(source,h,work=w,enabled=True)
        return CheckedSharedForest(packet,source,h,expected=packet.identity,work=w,enabled=True)
    forests={k:forest(k) for k in ('mixed','clipped','guarded','tail','zero')}
    # Exact symbolic operator comparison uses the actual binary64 local/ref/scale
    # values. It does NOT claim native rounding equals the exact source network.
    def symbolic(op):
        rows=[{} for _ in range(op.shape[0])]
        ix=op.local.indices(); vv=op.local.values()
        for k in range(vv.numel()):
            i,j=(int(ix[t,k]) for t in (0,1)); w.visit('coefficient_visits')
            rows[i][j]=w.exact(rows[i].get(j,F(0))+F(float(vv[k])))
        for ns,bank in getattr(op,'blocks',()):
            for i,n in enumerate(ns.tolist()):
                for j in range(bank.shape[1]):
                    w.visit('coefficient_visits'); rows[n][j]=w.exact(rows[n].get(j,F(0))+F(float(bank[i,j])))
        for children,parents,sc in op.levels:
            for i,p,s in zip(children.tolist(),parents.tolist(),sc[:,0].tolist(),strict=True):
                for j,q in rows[p].items():
                    w.visit('coefficient_visits'); rows[i][j]=w.exact(rows[i].get(j,F(0))+w.exact(F(s)*q))
        for row in rows:
            for j in row: row[j]=w.exact(row[j]*F(float(op.scales[j,0])))
        return [{j:q for j,q in r.items() if q} for r in rows]
    for case,f in forests.items():
        a=native.Operator(f,w); b=native.SourceRowOperator(f,w)
        require(symbolic(a)==symbolic(b), 'all native local/ref/scale symbolic rows')
        exact=expanded(f,w)
        x=[[F(j%7-3,16),F(j%3,8)] for j in range(len(f.factors))]
        alpha=[[F(i%5-2,32),F(i%4,16)] for i in range(len(f.nodes))]
        forward=[[F(0),F(0)] for _ in exact]
        for i,row in enumerate(exact):
            for j,q in row.items():
                for k in range(2):
                    w.visit('coefficient_visits'); forward[i][k]=w.exact(forward[i][k]+w.exact(q*x[j][k]))
        adj=[[F(0),F(0)] for _ in x]
        for i,row in enumerate(exact):
            for j,q in row.items():
                for k in range(2): w.visit('coefficient_visits'); adj[j][k]=w.exact(adj[j][k]+w.exact(q*alpha[i][k]))
        cast=lambda z:torch.tensor([[float(q) for q in row] for row in z],dtype=torch.float64)
        xx,aa=cast(x),cast(alpha)
        for op in (a,b):
            require(torch.allclose(op.forward(xx),cast(forward),rtol=1e-12,atol=1e-12), 'source forward numerical control')
            require(torch.allclose(op.adjoint(aa),cast(adj),rtol=1e-12,atol=1e-12), 'source adjoint numerical control')
        passed('operator-'+case,nodes=len(f.nodes),factors=len(f.factors),source_blocks=b.stats['source_blocks'])
        if case=='tail':
            tails=[q for _,local,_ in f.nodes if type(local) is SourceLocal for _,q in local.tail]
            require(F(1,2**55) in tails and F(1,2**1000) in tails and f.scales[:2]==(F(1,8),F(1,4)), 'nonzero error/lift actually exercised')
            before=symbolic(b)
            b.blocks[0][1][0,0]+=1
            require(symbolic(b)!=before, 'bank mutation detected by exact oracle')
            require(expanded(f,w)==exact, 'private bank cannot mutate checked source')
            passed('bank-alias-and-tiny-tail',tail_terms=len(tails))
        del a,b
    f=forests['mixed']; c=contract(f); frames=[]; allrecords=[]
    d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True)
    for index in range(d.session.tile_count):
        tasks,lease,token=prepare(d)
        if index==0:
            reject('roots-readonly',lambda:setattr(lease,'_live',False))
            rootmap=lease[0][1]
            key=next(iter(rootmap)); original_q=rootmap[key]; pristine=F(original_q.numerator,original_q.denominator)
            object.__setattr__(original_q,'_numerator',original_q.numerator+1)
            require(rootmap[key]==pristine,'leased Fraction is a private value, not a mutable source alias')
            saved=lease
        def tensor(kind):
            return torch.tensor([[(-.0 if (i+k)%3==0 else (1 if (i+k)%2 else -1)/8 if kind=='eq' else -1/16)
                for k in range(len(tasks))] for i in range(len(f.rows(tasks[0].pair,kind)))],dtype=torch.float64).reshape(-1,len(tasks))
        dual=native.DualTile(d,tasks,token,tensor('eq'),tensor('ine'))
        if index==0:
            reject('foreign-token',lambda:dual.check(d,tasks,object()))
            reject('foreign-task-object',lambda:dual.check(d,list(tasks),token))
            from scoped_source.native_method import validate
            bad=method('native_coo_reference'); bad['producer']['updates']=127
            reject('method-binding',lambda:validate(bad))
            oldforest=d.forest
            try:
                d.forest=forests['clipped']
                reject('foreign-forest',lambda:dual.check(d,tasks,token))
            finally: d.forest=oldforest
            oldwork=d.work
            try:
                d.work=object()  # wrong identity only; no replacement Work/counter budget
                reject('foreign-work',lambda:dual.check(d,tasks,token))
            finally: d.work=oldwork
        revoke(lease)
        def publish(i,raw): frames.append(raw); return digest(raw),len(raw)
        submit_owned_duals(d,token,dual,publish,native=native); dual.close()
    original=d.finish()
    reject('revoked-roots',lambda:len(saved))
    exact=expanded(f,w)
    for scoped in (False,True):
        check=Driver(f,c,expected=identity(c),work=w,pair_scoped=scoped)
        for raw in frames:
            tasks,_,tok=check.prepare(); tile=Tile(raw,check.owner,f,check.contract,tasks,check.ordinal)
            records=[]
            for k,t in enumerate(tasks):
                rec=dict(slot=t.slot)
                for role in ('proposed','zero'):
                    rec[role]=dict(query=t.query_id,lower='0',**{key:[str(tile.weight(k,role,key,i))
                        for i in range(len(f.rows(t.pair,kind)))] for kind,key in (('eq','equality'),('ine','inequality'))})
                records.append(rec)
            receipts=check.receive(tok,raw,lambda i,b:(digest(b),len(b)))
            # Driver owns receipts; the expanded oracle computes each role independently.
            for task,rec,receipt in zip(tasks,records,check.receipts[-len(tasks):],strict=True):
                for role in ('proposed','zero'):
                    lower,_=scalar_bound(f,task,rec[role],exact,w)
                    require(lower==F(receipt.record[role]), 'independent exact scalar candidate bound')
        require(check.finish()==original, 'same candidate complete dense/scoped aggregation')
        passed('same-candidate-'+('scoped' if scoped else 'dense'))
    raw=frames[0]; fields=PREFIX.unpack_from(raw); eq,ine=fields[6:8]
    coeff=PREFIX.size+RECORD.size+2*RATIONAL
    variants={
        'foreign-source':lambda b:b.__setitem__(slice(40,72),bytes(32)),
        'foreign-ordinal':lambda b:struct.pack_into('<I',b,104,1),
        'missing-role':lambda b:b.__setitem__(PREFIX.size+65,1),
        'negative-zero':lambda b:struct.pack_into('<d',b,coeff,-0.),
        'nonfinite':lambda b:struct.pack_into('<d',b,coeff,float('nan')),
        'positive-inequality':lambda b:struct.pack_into('<d',b,coeff+8*eq,1.),
        'false-zero':lambda b:struct.pack_into('<d',b,coeff+8*(eq+ine),1.),
        'truncation':lambda b:b.pop(),
        'wrong-contract':lambda b:b.__setitem__(slice(72,104),bytes(32)),
        'wrong-dual-dimension':lambda b:struct.pack_into('<I',b,112,eq+1),
    }
    stride=(len(raw)-PREFIX.size)//fields[5]
    require(fields[5]>=2,'frozen mixed first tile has two slots')
    variants['duplicate-slot']=lambda b:b.__setitem__(slice(PREFIX.size+stride,PREFIX.size+stride+32),b[PREFIX.size:PREFIX.size+32])
    def swap_slots(b):
        a=bytes(b[PREFIX.size:PREFIX.size+32]); c=bytes(b[PREFIX.size+stride:PREFIX.size+stride+32])
        b[PREFIX.size:PREFIX.size+32]=c; b[PREFIX.size+stride:PREFIX.size+stride+32]=a
    variants['reordered-slots']=swap_slots
    for name,edit in variants.items():
        check=Driver(f,c,expected=identity(c),work=w,pair_scoped=True); _,_,tok=check.prepare()
        changed=bytearray(raw); edit(changed)
        reject('frame-'+name,lambda:check.receive(tok,bytes(changed),lambda i,b:(digest(b),len(b))))
        if not check.closed: check.abort()
    # Missing evidence is a complete UNKNOWN, not an error and never a positive.
    check=Driver(f,c,expected=identity(c),work=w,pair_scoped=True)
    for raw in frames:
        tasks,_,tok=check.prepare()
        check.submit(tok,[dict(slot=t.slot,proposed=None,zero=None) for t in tasks],lambda i,b:(digest(b),len(b)))
    require(check.finish()['status']=='UNKNOWN_MISSING_EVIDENCE', 'missing obligation aggregation')
    passed('missing-obligations-unknown')
    # Actual completed package from this frozen batch, copied to a new location.
    # The child imports only the copied checker with -I -S -B and no project path.
    first=freeze['records'][0]
    run=root.parent/'runs'/first['id']
    prod=json.loads((run/'production.json').read_bytes())['output']
    inv=json.loads((run/'invocation.json').read_bytes())
    from scoped_source.native_package import CODE_FILES
    from scoped_source.native_supervised import clean_env
    relocated=root/'relocated'; shutil.copytree(run/'artifacts/package',relocated); w.tick()
    external=dict(bundle=prod['package']['sha256'],method=first['profile']['method'],
        invocation=identity(inv),source=first['profile']['source']['source'],
        code={n:first['profile']['implementation'][n] for n in CODE_FILES})
    def portable(bind,name):
        path=root/(name+'.json')
        path.write_bytes(compact(dict(bind,budget=dict(initial_work=dict(w.counts),deadline=w.deadline)))); w.tick()
        result=subprocess.run([sys.executable,'-I','-S','-B',
            str(relocated/'code/scoped_source/native_portable.py'),'--enabled',str(relocated),str(path)],
            cwd=relocated,env=clean_env(),stdin=subprocess.DEVNULL,capture_output=True,
            timeout=max(.001,w.deadline-time.monotonic()))
        require(len(result.stdout)+len(result.stderr)<=2**20,'bounded portable component output')
        if result.stdout:
            partial=json.loads(result.stdout)
            final=partial['work']
            require(set(final)==set(w.counts) and all(type(v) is int and v>=w.counts[k] for k,v in final.items())
                    and sum(final.values())<=8000000,'portable cumulative Work, never reset')
            w.counts.update(final)
        w.tick(); return result
    r=portable(external,'external')
    require(r.returncode==0,'relocated fresh standard-library check')
    rr=json.loads(r.stdout); require(rr['result']==prod['result'] and rr['online_acceptance'] is False,'same offline result')
    passed('portable-full-recomputation',copied_source=True,checkpoint_loaded=False)
    require(portable(dict(external,source='0'*64),'wrong-source').returncode!=0,'portable foreign source')
    passed('portable-source-binding')
    extra=relocated/'code/scoped_source/__init__.py'; extra.write_bytes(b'raise RuntimeError("must not execute")\n')
    require(portable(external,'extra-code').returncode!=0,'portable unbound initializer')
    passed('portable-extra-code')
    # Non-touching OS allocation proves this actual AS ceiling rejects over-cap maps.
    try: m=mmap.mmap(-1,8*2**30+4096)
    except (OSError,MemoryError): passed('actual-AS-overcap-rejected')
    else: m.close(); raise AssertionError('OS AS limit not enforced')
    storage=census(root,w.deadline)
    require(storage['bytes']+2**20<=POLICY['total_bytes'] and len(storage['files'])+1<=POLICY['members'],
            'component aggregate artifact reserve, not merely per-file RLIMIT')
    require(len(tests)==32,'all frozen component positions completed')
    report=dict(schema='NATIVE_COMPONENT_CONTROLS_V1',status='PASS',tests=tests,work=dict(w.counts),
        storage_before_report=storage,address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
        seconds=time.monotonic()-started,models=0,gpu=0,native_float_tolerance_only=True)
    data=compact(report); require(len(data)<=2**20 and time.monotonic()<w.deadline,'bounded component report')
    (root/'report.json').write_bytes(data)
    print(compact(dict(status='PASS',tests=len(tests),seconds=report['seconds'])).decode())


if __name__=='__main__':
    # Limits before importing any project/numerical module. Bootstrap is trusted.
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(128*2**20,128*2**20))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    main(sys.argv[1],json.loads(Path(sys.argv[2]).read_bytes()))
