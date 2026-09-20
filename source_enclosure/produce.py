"""Untrusted exact-source enclosure producer; never an acceptance authority."""
import copy
from fractions import Fraction as F
import math

from source_enclosure.format import pack,unpack,empty,identity,rational


def box(lower,upper):
    if len(lower)!=len(upper) or not lower:raise ValueError('input dimensions')
    h=empty(len(lower));ids=[]
    for i,(a,b) in enumerate(zip(lower,upper)):
        a,b=rational(a),rational(b)
        if a>b:raise ValueError('input order')
        c=float((a+b)/2);need=max(F(c)-a,b-F(c));r=float(need)
        if F(r)<need:r=math.nextafter(r,math.inf)
        if not math.isfinite(c) or not math.isfinite(r):raise ValueError('finite binary64 enclosure unavailable')
        h['c'][i]=F(c);h['Gc'][i]={i:F(r)} if r else {};ids.append(f'input/c/{i}')
    return pack(h,ids,[])


def redundant_guards(state,ac,ab,rhs):
    h,c,b=unpack(state);h=copy.deepcopy(h)
    h['Auc']+=copy.deepcopy(ac);h['Aub']+=copy.deepcopy(ab);h['ub']+=list(map(rational,rhs))
    return pack(h,c,b)


def affine(state,operator,bias,nominal,tag):
    """Retain nominal stored coefficients, append per-row exact error factors."""
    from upstream_source.checker import hz
    s,cids,bids=unpack(state);p=hz(nominal);h=copy.deepcopy(s)
    h['c']=p['c'];h['Gc']=copy.deepcopy(p['Gc']);h['Gb']=copy.deepcopy(p['Gb'])
    ids=list(cids);errors=[]
    for i,(row,b) in enumerate(zip(operator,bias)):
        center=rational(b);gc={};gb={}
        for j,w in row.items():
            center+=w*s['c'][j]
            for result,k in ((gc,'Gc'),(gb,'Gb')):
                for col,value in s[k][j].items():result[col]=result.get(col,F(0))+w*value
        radius=abs(center-p['c'][i])
        for exact,k in ((gc,'Gc'),(gb,'Gb')):
            for col in exact.keys()|p[k][i].keys():radius+=abs(exact.get(col,F(0))-p[k][i].get(col,F(0)))
        errors.append(str(radius))
        if radius:
            h['Gc'][i][len(ids)]=radius;ids.append(f'{tag}/error/{i}')
    return pack(h,ids,bids),{'source':identity(state),'tag':tag,'error_bounds':errors}


def relu(state,tag):
    s,cids,bids=unpack(state);h=copy.deepcopy(s);cids=list(cids);bids=list(bids);ranges=[];branches=[]
    for i,c in enumerate(s['c']):
        radius=sum(map(abs,s['Gc'][i].values()),F(0))+sum(map(abs,s['Gb'][i].values()),F(0))
        lo,hi=c-radius,c+radius;ranges.append([str(lo),str(hi)])
        if lo>=0:branches.append('active');continue
        if hi<=0:
            branches.append('inactive');h['c'][i]=F(0);h['Gc'][i]={};h['Gb'][i]={};continue
        branches.append('unstable');u,v=len(cids),len(cids)+1;z=len(bids)
        cids.extend([f'{tag}/negative/{i}',f'{tag}/positive/{i}']);bids.append(f'{tag}/sign/{i}')
        h['c'][i]=hi/2;h['Gc'][i]={v:-hi/2};h['Gb'][i]={}
        eqc={k:-w for k,w in s['Gc'][i].items()};eqb={k:-w for k,w in s['Gb'][i].items()}
        eqc.update({u:lo/2,v:-hi/2});eqb[z]=lo/2
        h['Ac'].append(eqc);h['Ab'].append(eqb);h['b'].append(c-hi/2)
        h['Auc'].extend([{u:F(-1)},{v:F(-1)}]);h['Aub'].extend([{z:F(-1)},{z:F(1)}]);h['ub'].extend([F(0),F(0)])
    return pack(h,cids,bids),{'source':identity(state),'tag':tag,'ranges':ranges,'branches':branches}


def join(base,left,right):
    h0,c0,b0=unpack(base);a,ca,ba=unpack(left);b,cb,bb=unpack(right)
    cids=c0+ca[len(c0):]+cb[len(c0):];bids=b0+ba[len(b0):]+bb[len(b0):]
    ci={v:i for i,v in enumerate(cids)};bi={v:i for i,v in enumerate(bids)}
    maps={'left_c':[ci[v] for v in ca],'right_c':[ci[v] for v in cb],
          'left_b':[bi[v] for v in ba],'right_b':[bi[v] for v in bb]}
    h=empty(0,h0['frame_id'])
    for src,name in ((a,'left'),(b,'right')):
        h['c']+=src['c']
        for key,m in (('Gc',maps[name+'_c']),('Gb',maps[name+'_b'])):
            h[key]+=[{m[k]:v for k,v in row.items()} for row in src[key]]
    for ck,bk,rhs in (('Ac','Ab','b'),('Auc','Aub','ub')):
        h[ck]=copy.deepcopy(h0[ck]);h[bk]=copy.deepcopy(h0[bk]);h[rhs]=list(h0[rhs]);n=len(h0[rhs])
        for src,name in ((a,'left'),(b,'right')):
            for key,m in ((ck,maps[name+'_c']),(bk,maps[name+'_b'])):
                h[key]+=[{m[k]:v for k,v in row.items()} for row in src[key][n:]]
            h[rhs]+=src[rhs][n:]
    return pack(h,cids,bids),{'base':identity(base),'left':identity(left),'right':identity(right),'maps':maps}
