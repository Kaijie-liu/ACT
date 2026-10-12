"""Fixed control capture and test-only eager differential, not production intake."""
from fractions import Fraction as F
import hashlib
from pathlib import Path
import struct

from scoped_source.hz_request_store import compact, identity, require, Store
from scoped_source.factored_io import write, HEADER_LIMIT
from scoped_source.factored_source import SCHEMA


def capture(root,doc):
    """Encode the existing binary-rational source controls, without new rounding."""
    root=Path(root); root.mkdir(exist_ok=False)
    require(doc['lower']==['-1'] and doc['upper']==['1'], 'registered one-dimensional controls')
    tensors={'@center':((1,1,1),[F(0)])}; graphs=[]
    for name,seq in [('router',doc['router'])]+[(f'expert{i}',s) for i,s in enumerate(doc['experts'])]:
        prefix='router' if name=='router' else 'experts.'+name[6:]
        layers=[dict(index=0,kind='Flatten',training=False,dimensions=[1,-1])]; ins=1
        for layer in seq:
            j=len(layers)
            if layer['kind']=='relu': layers.append(dict(index=j,kind='ReLU',training=False,inplace=False)); continue
            require(layer['kind']=='affine', 'registered affine source'); rows=layer['rows']; w=[]; b=[]
            for bias,coeff in rows:
                rr=[F(0)]*ins
                for k,q in coeff: rr[k]=F(q)
                w.extend(rr); b.append(F(bias))
            wn=f'{prefix}.{j}.weight'; bn=f'{prefix}.{j}.bias'
            tensors[wn]=((len(rows),ins),w); tensors[bn]=((len(rows),),b); ins=len(rows)
            layers.append(dict(index=j,kind='Linear',training=False,weight_name=wn,bias_name=bn,
                               weight={'tensor_ref':wn},bias={'tensor_ref':bn}))
        graphs.append(dict(name=name,layers=layers))
    records=[]; state=[]; center=None
    for i,(name,(dims,values)) in enumerate(sorted(tensors.items())):
        require(all(F(float(q))==q for q in values), 'lossless registered capture')
        raw=b''.join(struct.pack('<d',float(q)) for q in values)
        h=hashlib.sha256(b'torch.float64'+compact(list(dims))+raw).hexdigest()
        ident=dict(dtype='torch.float64',shape=list(dims),sha256=h)
        if name=='@center': center=ident
        else: state.append(dict(name=name,**ident))
        ref=write(root,f't/{i:05d}-00000.bin',raw,1048576,binary=True)
        records.append(dict(name=name,dtype='torch.float64',shape=list(dims),byte_order='little',chunks=[dict(**ref,offset=0)]))
    h=hashlib.sha256()
    for v in state: h.update(v['name'].encode()); h.update(v['sha256'].encode())
    classes=len(doc['experts'][0][-1]['rows'])
    request=dict(experts=len(doc['experts']),classes=classes,label=0,top_k=2,gate='SELECTED_SOFTMAX',
        tie_policy='ANY_LEGAL_TOPK',training=False,center=center,radius='1',clip=['-1','1'],margin='0',
        model_state=dict(sha256=h.hexdigest(),tensor_count=len(state),
                         parameter_count=sum(len(v) for k,(_,v) in tensors.items() if k!='@center')))
    manifest=dict(schema=SCHEMA,tensors=records,declaration=dict(schema='SCOPED_DECLARED_TOP2_V1',
        request=request,center={'tensor_ref':'@center'},state_inventory=state,networks=graphs))
    write(root,'manifest.json',manifest,HEADER_LIMIT); return identity(manifest)


def fixture(root,case):
    if case=='small40':
        from scoped_source.tensor_row_controls import fixture as f
        return Path(root),f(root,chunk_bytes=40)
    if case=='mixed':
        from scoped_source.source_dag_controls import mixed
        return mixed(root)
    require(case in ('half','guarded'), 'fixed case')
    from scripts.run_store_device_controls import fixture as f
    return Path(root),capture(root,f(case))


def eager(root,source,raw,destination,work):
    from scoped_source.source_dag_controls import old_differential
    receipt=old_differential(root,source,raw,destination,work)
    return Store(destination,expected_manifest=receipt['old_manifest'],expected_request=receipt['old_source'],work=work),receipt


def differential(store,forest,inventory,proof,work):
    from scoped_source.check_store_endpoints import evaluate_tile as old
    from scoped_source.forest_endpoints import evaluate_tile as new
    count=0
    for tasks,records in zip(inventory.tiles(proof['width']),proof['tiles'],strict=True):
        cc=[r['candidate'] for r in records]
        actual,ar=new(forest,tasks,cc,work); expected,er=old(store,tasks,cc,work)
        require(actual==expected, 'same-candidate exact support differential')
        require([{j:v for j,v in r.items() if v} for r in ar]==[{j:v for j,v in r.items() if v} for r in er],
                'same-candidate global residual differential')
        count+=len(tasks)
    return count
