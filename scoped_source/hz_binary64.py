"""Untrusted finite binary64 enclosure producer; no solver or propagation."""
from copy import deepcopy
from fractions import Fraction as F
import math

from source_enclosure.format import identity, pack, clean
from scoped_source.graph import clock
from scoped_source.check_hz_binary64 import REFERENCE, SCHEMA, parse, label


def nearest(q):
    try:f=float(q)
    except OverflowError:raise ValueError('no finite binary64 coefficient') from None
    if not math.isfinite(f):raise ValueError('no finite binary64 coefficient')
    return F(f)


def upward(q):
    f=float(nearest(q))
    if F(f)<q:f=math.nextafter(f,math.inf)
    if not math.isfinite(f):raise ValueError('no finite outward binary64 endpoint')
    return F(f)


def produce(reference, *, expected_reference_sha256, owner, deadline):
    tick=clock(deadline);label(owner)
    if identity(reference)!=expected_reference_sha256:raise ValueError('external reference binding')
    s,ci,bi=parse(reference); target=deepcopy(s); newids=list(ci)
    for key in ('c','b'):
        target[key]=[nearest(v) for v in s[key]]
    for key in ('Gc','Gb','Ac','Ab','Auc','Aub'):
        target[key]=[clean({j:nearest(v) for j,v in row.items()}) for row in s[key]]
    records={}
    for kind,vector,ck,bk in (('output','c','Gc','Gb'),('equality','b','Ac','Ab')):
        records[kind]=[]
        for i,value in enumerate(s[vector]):
            tick(); needed=abs(value-target[vector][i])
            for key in (ck,bk):
                needed+=sum((abs(v-target[key][i].get(j,F(0))) for j,v in s[key][i].items()),F(0))
            radius=upward(needed); name=None
            if radius:
                name=f'{owner}/{expected_reference_sha256}/{kind}/{i}'
                if name in newids+bi:raise ValueError('fresh factor collision')
                target[ck][i][len(newids)]=radius;newids.append(name)
            records[kind].append({'factor':name,'radius':str(radius)})
    target['ub']=[]
    for i,rhs in enumerate(s['ub']):
        tick(); required=rhs
        for key in ('Auc','Aub'):
            required+=sum((abs(v-target[key][i].get(j,F(0))) for j,v in s[key][i].items()),F(0))
        target['ub'].append(upward(required))
    result={'schema':REFERENCE,'state':pack(target,newids,bi),
            'ownership':{'continuous':reference['ownership']['continuous']+[owner]*(len(newids)-len(ci)),
                         'binary':list(reference['ownership']['binary'])}}
    parse(result,target=True);tick()
    proof={'schema':SCHEMA,'reference_sha256':expected_reference_sha256,'owner':owner,
            'target':result,'target_sha256':identity(result),
            'maps':{'continuous':list(range(len(ci))),'binary':list(range(len(bi))),
                    'flat':list(range(len(ci)))+[len(newids)+j for j in range(len(bi))]},
            'output_errors':records['output'],'equality_errors':records['equality']}
    if identity(reference)!=expected_reference_sha256:raise ValueError('reference changed during construction')
    tick()
    return proof


def instantiate(target, *, expected_target_sha256, deadline):
    """Actual SparseHZono round trip, not proof of reference inclusion."""
    tick=clock(deadline)
    if identity(target)!=expected_target_sha256:raise ValueError('target identity')
    h,ci,bi=parse(target,target=True)
    import numpy as np
    import scipy.sparse as sp
    from act.back_end.solver.solver_hz import SparseHZono
    from act.back_end.solver.hz_lp_export import snapshot
    from upstream_source.checker import hz
    args={k:np.array(list(map(float,h[k])),dtype=np.float64) for k in ('c','b','ub')}
    for key in ('Gc','Gb','Ac','Ab','Auc','Aub'):
        data=[];indices=[];ptr=[0]
        for row in h[key]:
            for j,v in sorted(row.items()):indices.append(j);data.append(float(v))
            ptr.append(len(data))
        args[key]=sp.csr_matrix((data,indices,ptr),shape=(len(h[key]),len(ci) if key.endswith('c') else len(bi)))
    live=SparseHZono(**args,frame_id=h['frame_id'],exact=False);snap=snapshot(live)
    if hz(snap)!=h:raise ValueError('live sparse HZ coefficient mismatch')
    tick()
    if identity(target)!=expected_target_sha256:raise ValueError('target mutated during instantiation')
    tick()
    return live,snap
