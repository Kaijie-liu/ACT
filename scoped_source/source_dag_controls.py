"""Finite source-forest fixtures and test-only eager differential."""
from fractions import Fraction as F
import hashlib
from pathlib import Path
import struct

from scoped_source.hz_request_store import compact, identity, Work
from scoped_source.tensor_rows import TensorRows


def mixed(root, first_relu=False):
    """Three experts, tied router, active/inactive/unstable/zero ReLU rows."""
    from scoped_source.tensor_row_controls import fixture
    from scoped_source.factored_io import write, HEADER_LIMIT
    # Reuse the registered small schema, replace the *test* graph/tensor inventory.
    root=Path(root); fixture(root)
    old=__import__('json').loads((root/'manifest.json').read_bytes())
    # A separate directory is written; no frozen source is overwritten.
    target=root.parent/(root.name+'-mixed'); target.mkdir()
    shape=(1,1,2); tensors={'@center':(shape,[0.,0.])}; graphs=[]
    for name in ('router','expert0','expert1','expert2'):
        prefix='router' if name=='router' else 'experts.'+name[6:]
        layers=[]
        if first_relu and name!='router': layers.append(dict(index=0,kind='ReLU',training=False,inplace=False))
        layers.append(dict(index=len(layers),kind='Flatten',training=False,dimensions=[1,-1]))
        if name=='router': specs=[(3,2,[1.,0.,-1.,0.,0.,0.],[0.,0.,0.])]
        else:
            specs=[(5,2,[1.,0.,1.,0.,1.,0.,0.,0.,0.,1.],[2.,-2.,0.,0.,.3]),
                   (2,5,[.1,.2,.3,.4,.5, -.3,.2,.4,.1,.2],[.1,-.2])]
        for k,(outs,ins,weights,bias) in enumerate(specs):
            j=len(layers); wn=f'{prefix}.{j}.weight'; bn=f'{prefix}.{j}.bias'
            tensors[wn]=((outs,ins),weights); tensors[bn]=((outs,),bias)
            layers.append(dict(index=j,kind='Linear',training=False,weight_name=wn,bias_name=bn,
                               weight={'tensor_ref':wn},bias={'tensor_ref':bn}))
            if k<len(specs)-1: layers.append(dict(index=len(layers),kind='ReLU',training=False,inplace=False))
        graphs.append(dict(name=name,layers=layers))
    records=[]; state=[]; center=None
    for i,(name,(dims,values)) in enumerate(sorted(tensors.items())):
        raw=b''.join(struct.pack('<d',x) for x in values)
        h=hashlib.sha256(b'torch.float64'+compact(list(dims))+raw).hexdigest()
        ident=dict(dtype='torch.float64',shape=list(dims),sha256=h)
        if name=='@center': center=ident
        else: state.append(dict(name=name,**ident))
        ref=write(target,f't/{i:05d}-00000.bin',raw,1048576,binary=True)
        records.append(dict(name=name,dtype='torch.float64',shape=list(dims),byte_order='little',chunks=[dict(**ref,offset=0)]))
    h=hashlib.sha256()
    for x in state: h.update(x['name'].encode()); h.update(x['sha256'].encode())
    request={**old['declaration']['request'],'experts':3,'center':center,'radius':'1',
             'model_state':dict(sha256=h.hexdigest(),tensor_count=len(state),parameter_count=sum(len(v) for k,(s,v) in tensors.items() if k!='@center'))}
    m=dict(schema=old['schema'],tensors=records,declaration=dict(schema='SCOPED_DECLARED_TOP2_V1',request=request,
        center={'tensor_ref':'@center'},state_inventory=state,networks=graphs))
    write(target,'manifest.json',m,HEADER_LIMIT); return target,identity(m)


def old_differential(root, source, raw, destination, work):
    """Test only: all layers/rows/factors vs old producer and independent checker."""
    import json
    from scoped_source.hz_store_source import build
    from scoped_source.check_hz_store_source import check_source
    from scoped_source.hz_request_store import Store
    reader=TensorRows(root,source,deadline=work.deadline); stream=iter(reader.rows()); nets={}; names={'input':'input','common':'common'}
    for layer in reader.plan:
        seq=nets.setdefault(layer.graph,[])
        if layer.kind=='Flatten': continue
        names[f'{layer.graph}/{layer.index}']=f'{layer.graph}/{len(seq)}'
        if layer.kind=='ReLU': seq.append(dict(kind='relu'))
        else:
            rr=[]
            for _ in range(layer.outputs):
                r=next(stream); rr.append([str(r.bias),[[i,str(q)] for i,q in enumerate(r.coefficients) if q]])
            seq.append(dict(kind='affine',rows=rr))
    list(stream); reader.receipt()
    decl=dict(schema='REQUEST_STORE_SOURCE_V1',lower=[str(l) for l,u in reader.inputs],upper=[str(u) for l,u in reader.inputs],
              router=nets['router'],experts=[nets[f'expert{i}'] for i in range(reader.request['experts'])])
    h,_=build(destination,decl,work=work,page_rows=8)
    store=Store(destination,expected_manifest=h,expected_request=identity(decl),work=work)
    check_source(store,decl,expected_source=identity(decl),work=work)
    d=json.loads(raw); nodes=d['nodes']; factors=d['factors']
    def mapped(name):
        for before,after in sorted(names.items(),key=lambda x:-len(x[0])):
            if name==before or name.startswith(before+'/'): return after+name[len(before):]
        return name
    for f,g in zip(factors,store.factors,strict=True):
        fname,kind,scope,owner=f; ownername=d['views'][owner]['name']
        assert (mapped(fname),kind,'common' if scope==-1 else f'expert/{scope}',mapped(ownername))==g
    def expand(n):
        result={}
        for j,q in nodes[n][1]:
            work.visit('coefficient_visits'); result[int(j)]=F(q)
        for p,s in nodes[n][2]:
            for j,q in expand(p).items(): result[j]=work.exact(result.get(j,F(0))+F(s)*q)
        return {j:q for j,q in result.items() if q}
    totals={}; checked=0
    for v in d['views']:
        parent=totals.get(v['parent'],dict(eq=[],ine=[])); allrows={k:parent[k]+v[k] for k in ('eq','ine')}
        allrows['output']=v['output']; totals[len(totals)]=allrows
        if v['op']=='Flatten': continue
        name=v['name']
        if v['op']=='guard': name='entry'+name.split('/')[1]
        else: name=mapped(name)
        old=store.views[name]
        for kind in ('output','eq','ine'):
            for (c,n),r in zip(allrows[kind],getattr(old,kind),strict=True):
                work.visit('row_visits'); assert F(c)==r.rhs and expand(n)==dict(r.coefficients); checked+=1
    return dict(checked_expanded_roots=checked,old_manifest=h,old_source=identity(decl),factors=len(factors))
