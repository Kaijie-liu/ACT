"""Frozen finite V2 fixtures and test-only exact expansion oracle."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

from scoped_source.hz_request_store import compact, identity


def deep(root,*,center=(0.,0.),radius='1'):
    from scoped_source.tensor_row_controls import fixture
    from scoped_source.factored_io import write, HEADER_LIMIT
    root=Path(root); fixture(root/'template')
    old=json.loads((root/'template/manifest.json').read_bytes()); target=root/'source'; target.mkdir()
    tensors={'@center':((1,1,2),list(center))}; graphs=[]
    for name in ('router','expert0','expert1','expert2'):
        prefix='router' if name=='router' else 'experts.'+name[6:]
        dims=[2,3,3,3] if name=='router' else [2,2,2,2,2]
        layers=[dict(index=0,kind='Flatten',training=False,dimensions=[1,-1])]
        for k,(ins,outs) in enumerate(zip(dims,dims[1:])):
            j=len(layers); wn=f'{prefix}.{j}.weight'; bn=f'{prefix}.{j}.bias'
            values=[(-1. if (r+c+k)%2 else 1.)*(1+(r+2*c+k)%4)/4 for r in range(outs) for c in range(ins)]
            tensors[wn]=((outs,ins),values); tensors[bn]=((outs,),[(r-k)/10 for r in range(outs)])
            layers.append(dict(index=j,kind='Linear',training=False,weight_name=wn,bias_name=bn,
                               weight={'tensor_ref':wn},bias={'tensor_ref':bn}))
            if k==1: layers.append(dict(index=len(layers),kind='ReLU',training=False,inplace=False))
        graphs.append(dict(name=name,layers=layers))
    records=[]; state=[]
    for i,(name,(shape,values)) in enumerate(sorted(tensors.items())):
        raw=b''.join(struct.pack('<d',x) for x in values)
        h=hashlib.sha256(b'torch.float64'+compact(list(shape))+raw).hexdigest()
        ident=dict(dtype='torch.float64',shape=list(shape),sha256=h)
        if name=='@center': center=ident
        else: state.append(dict(name=name,**ident))
        ref=write(target,f't/{i:05d}-00000.bin',raw,1048576,binary=True)
        records.append(dict(name=name,dtype='torch.float64',shape=list(shape),byte_order='little',chunks=[dict(**ref,offset=0)]))
    h=hashlib.sha256()
    for v in state: h.update(v['name'].encode()); h.update(v['sha256'].encode())
    request={**old['declaration']['request'],'experts':3,'center':center,'radius':radius,
        'model_state':dict(sha256=h.hexdigest(),tensor_count=len(state),
            parameter_count=sum(len(v) for k,(s,v) in tensors.items() if k!='@center'))}
    manifest=dict(schema=old['schema'],tensors=records,declaration=dict(schema='SCOPED_DECLARED_TOP2_V1',
        request=request,center={'tensor_ref':'@center'},state_inventory=state,networks=graphs))
    write(target,'manifest.json',manifest,HEADER_LIMIT); return target,identity(manifest)


def expanded(nodes,work):
    """Independent eager oracle for bounded controls ONLY, never production."""
    matrix=[]
    for _,local,refs in nodes:
        work.visit('row_visits'); d={}
        for j,q in local: work.visit('coefficient_visits'); d[j]=q
        for parent,scale in refs:
            for j,q in matrix[parent].items():
                work.visit('coefficient_visits'); d[j]=work.exact(d.get(j,F(0))+work.exact(scale*q))
        matrix.append({j:q for j,q in d.items() if q})
    return matrix


def exact_forward(nodes,x,work):
    values=[]
    for _,local,refs in nodes:
        result=[F(0)]*len(x[0])
        for j,q in local:
            for k,v in enumerate(x[j]):
                work.visit('coefficient_visits'); result[k]=work.exact(result[k]+work.exact(q*v))
        for p,q in refs:
            for k,v in enumerate(values[p]):
                work.visit('coefficient_visits'); result[k]=work.exact(result[k]+work.exact(q*v))
        values.append(result)
    return values


def exact_adjoint(nodes,alpha,columns,work):
    a=[list(v) for v in alpha]; result=[[F(0)]*len(alpha[0]) for _ in range(columns)]
    for n in range(len(nodes)-1,-1,-1):
        _,local,refs=nodes[n]
        for j,q in local:
            for k,v in enumerate(a[n]):
                work.visit('coefficient_visits'); result[j][k]=work.exact(result[j][k]+work.exact(q*v))
        for p,q in refs:
            for k,v in enumerate(a[n]):
                work.visit('coefficient_visits'); a[p][k]=work.exact(a[p][k]+work.exact(q*v))
    return result


def oracle(forest,work):
    """Fully charged finite oracle: fixed two-column forward/transpose checks."""
    from scoped_source.hz_request_store import require
    rows=expanded(forest.nodes,work); columns=len(forest.factors)
    x=[[F((j%7)-3,16),F(j%3,8)] for j in range(columns)]
    a=[[F((i%5)-2,32),F(i%4,16)] for i in range(len(rows))]
    expected=[[F(0),F(0)] for _ in rows]; trans=[[F(0),F(0)] for _ in x]
    for i,row in enumerate(rows):
        for j,q in row.items():
            for k in range(2):
                work.visit('coefficient_visits')
                expected[i][k]=work.exact(expected[i][k]+work.exact(q*x[j][k]))
                trans[j][k]=work.exact(trans[j][k]+work.exact(q*a[i][k]))
    require(exact_forward(forest.nodes,x,work)==expected,'exact recursive forward differential')
    require(exact_adjoint(forest.nodes,a,columns,work)==trans,'exact recursive adjoint differential')
    for row,bound in zip(rows,forest.norms):
        norm=F(0)
        for q in row.values(): norm=work.exact(norm+abs(q))
        require(norm<=bound,'sound triangle radius')
    return x,a,expected,trans,dict(expanded_nnz=sum(map(len,rows)),forward_values=2*len(rows),
                                adjoint_values=2*columns)
