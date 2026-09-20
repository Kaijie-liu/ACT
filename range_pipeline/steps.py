"""Untrusted affine range consumer; parse a layer's source once, retain all rows."""
import copy
from fractions import Fraction as F
from source_enclosure.format import unpack,pack,clean,identity,rational


def affine(source,operator,bias,facts,context,tag):
    if not operator or not len(operator)==len(bias)==len(facts):raise ValueError('row inventory')
    s,ci,bi=unpack(source);h=copy.deepcopy(s);ids=list(ci);bounds=[]
    h['c']=[];h['Gc']=[];h['Gb']=[]
    for i,(op,b,fact) in enumerate(zip(operator,bias,facts)):
        center=rational(b);gc={};gb={}
        for j,w in op.items():
            w=rational(w);center+=w*s['c'][j]
            for result,key in [(gc,'Gc'),(gb,'Gb')]:
                for k,v in s[key][j].items():result[k]=result.get(k,F(0))+w*v
        gc,gb=clean(gc),clean(gb);radius=sum(map(abs,list(gc.values())+list(gb.values())),F(0))
        lo,hi=(center-radius,center+radius) if fact is None else tuple(map(rational,fact['range']))
        mid=(lo+hi)/2;r=(hi-lo)/2;out={}
        if r:j=len(ids);ids.append(f'{tag}/value/{i}');out[j]=r;gc[j]=-r
        h['c'].append(mid);h['Gc'].append(out);h['Gb'].append({});bounds.append([str(lo),str(hi)])
        if gc or gb or center!=mid:h['Ac'].append(gc);h['Ab'].append(gb);h['b'].append(mid-center)
    return pack(h,ids,bi),{'schema':'CHECKED_RANGE_SOURCE_STEP_V1','source':identity(source),
        'context':context,'tag':tag,'kind':'affine','facts':facts,'ranges':bounds}
