"""Shared serialization only; no enclosure/construction acceptance logic."""
from fractions import Fraction as F
import hashlib
import json

try:
    from local_check import hz, rational
except ImportError:
    from upstream_source.checker import hz, rational


def compact(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def identity(x):return hashlib.sha256(compact(x)).hexdigest()


def sparse(rows,n):
    data=[];indices=[];ptr=[0]
    for row in rows:
        for i,v in sorted(row.items()):
            v=rational(v)
            if v:indices.append(i);data.append(str(v))
        ptr.append(len(data))
    return {'shape':[len(rows),n],'data':data,'indices':indices,'indptr':ptr}


def pack(h,cids,bids):
    nc,nb=len(cids),len(bids)
    raw={k:[str(rational(v)) for v in h[k]] for k in ('c','b','ub')}
    for k in ('Gc','Ac','Auc'):raw[k]=sparse(h[k],nc)
    for k in ('Gb','Ab','Aub'):raw[k]=sparse(h[k],nb)
    raw.update(frame_id=h['frame_id'],exact=False)
    return {'schema':'CHECKED_SOURCE_HZ_V1','continuous_ids':list(cids),'binary_ids':list(bids),'hz':raw}


def unpack(state):
    if set(state)!={'schema','continuous_ids','binary_ids','hz'} or state['schema']!='CHECKED_SOURCE_HZ_V1':
        raise ValueError('source-state schema')
    h=hz(state['hz']);c,b=state['continuous_ids'],state['binary_ids']
    if (type(c)is not list or type(b)is not list or len(c)!=h['nc'] or len(b)!=h['nb'] or
            any(type(i)is not str or not i or len(i)>200 for i in c+b) or len(set(c+b))!=len(c+b)):
        raise ValueError('factor identity/alias/width')
    if state['hz']['exact'] is not False:raise ValueError('new enclosure is not an exact-image claim')
    return h,c,b


def clean(row):return {i:rational(v) for i,v in row.items() if v}


def empty(n,frame=1):
    return dict(c=[F(0)]*n,Gc=[{} for _ in range(n)],Gb=[{} for _ in range(n)],
                Ac=[],Ab=[],b=[],Auc=[],Aub=[],ub=[],frame_id=frame)
