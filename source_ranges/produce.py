"""Untrusted range-aware proposals. Does not call the checker or any solver."""
import copy
from fractions import Fraction as F
from source_enclosure.format import unpack,pack,identity,sparse,rational,clean


def polynomial(source,row,offset):
    s,c,b=unpack(source);center=rational(offset);gc={};gb={}
    for j,value in row.items():
        value=rational(value);center+=value*s['c'][j]
        for result,key in [(gc,'Gc'),(gb,'Gb')]:
            for k,v in s[key][j].items():result[k]=result.get(k,F(0))+value*v
    return center,clean(gc),clean(gb)


def range_fact(source,row,offset,context,index,bounds,lower_duals,upper_duals):
    """Bind supplied candidate multipliers; this is NOT an accepting API."""
    s,c,b=unpack(source);n=len(c)+len(b);center,gc,gb=polynomial(source,row,offset)
    coeff={**gc,**{j+len(c):v for j,v in gb.items()}}
    lp={'matrix_format':'csr_v1','c':[str(coeff.get(j,F(0))) for j in range(n)],'offset':str(center),
        'lower':[-1]*n,'upper':[1]*n}
    for key,ac,ab,rhs in [('A','Auc','Aub','ub'),('E','Ac','Ab','b')]:
        lp[key]=sparse([{**x,**{j+len(c):v for j,v in y.items()}} for x,y in zip(s[ac],s[ab])],n)
        lp['b' if key=='A' else 'h']=[str(v) for v in s[rhs]]
    neg={**lp,'c':[str(-F(x)) for x in lp['c']],'offset':str(-center)}
    def cert(program,duals):
        return {'lp_sha256':identity(program),'inequality_dual':list(duals[0]),'equality_dual':list(duals[1])}
    return {'schema':'SCOPED_SOURCE_RANGE_V1','query':{'source_sha256':identity(source),'context':context,
        'row':index,'expression':[[j,str(rational(v))] for j,v in sorted(row.items()) if rational(v)],'offset':str(rational(offset))},
        'range':list(map(str,bounds)),'lower':cert(lp,lower_duals),'negative_upper':cert(neg,upper_duals)}


def selected(source,operator,bias,facts):
    if not len(operator)==len(bias)==len(facts):raise ValueError('complete row inventory')
    answer=[]
    for row,offset,fact in zip(operator,bias,facts):
        center,gc,gb=polynomial(source,row,offset);r=sum(map(abs,list(gc.values())+list(gb.values())),F(0))
        answer.append((center-r,center+r) if fact is None else tuple(map(rational,fact['range'])))
    return answer


def proof(source,context,tag,kind,facts,bounds):
    return {'schema':'CHECKED_RANGE_SOURCE_STEP_V1','source':identity(source),'context':context,'tag':tag,
            'kind':kind,'facts':facts,'ranges':[list(map(str,p)) for p in bounds]}


def affine(source,operator,bias,facts,context,tag):
    s,ci,bi=unpack(source);h=copy.deepcopy(s);ids=list(ci)
    bounds=selected(source,operator,bias,facts);h['c']=[];h['Gc']=[];h['Gb']=[]
    for i,(op,offset,(lo,hi)) in enumerate(zip(operator,bias,bounds)):
        c,gc,gb=polynomial(source,op,offset);mid=(lo+hi)/2;radius=(hi-lo)/2;out={}
        if radius:
            slot=len(ids);ids.append(f'{tag}/value/{i}');out[slot]=radius;gc[slot]=-radius
        h['c'].append(mid);h['Gc'].append(out);h['Gb'].append({})
        if gc or gb or mid!=c:h['Ac'].append(gc);h['Ab'].append(gb);h['b'].append(mid-c)
    return pack(h,ids,bi),proof(source,context,tag,'affine',facts,bounds)


def relu(source,facts,context,tag):
    s,ci,bi=unpack(source);h=copy.deepcopy(s);ci=list(ci);bi=list(bi)
    bounds=selected(source,[{i:1} for i in range(len(s['c']))],[0]*len(s['c']),facts)
    for i,(lo,hi) in enumerate(bounds):
        if lo>=0:continue
        if hi<=0:h['c'][i]=F(0);h['Gc'][i]={};h['Gb'][i]={};continue
        u,v,z=len(ci),len(ci)+1,len(bi)
        ci.extend([f'{tag}/negative/{i}',f'{tag}/positive/{i}']);bi.append(f'{tag}/sign/{i}')
        h['c'][i]=hi/2;h['Gc'][i]={v:-hi/2};h['Gb'][i]={}
        gc={k:-x for k,x in s['Gc'][i].items()};gb={k:-x for k,x in s['Gb'][i].items()}
        gc.update({u:lo/2,v:-hi/2});gb[z]=lo/2
        h['Ac'].append(clean(gc));h['Ab'].append(clean(gb));h['b'].append(s['c'][i]-hi/2)
        h['Auc'].extend([{u:F(-1)},{v:F(-1)}]);h['Aub'].extend([{z:F(-1)},{z:F(1)}]);h['ub'].extend([F(0),F(0)])
    return pack(h,ci,bi),proof(source,context,tag,'relu',facts,bounds)
