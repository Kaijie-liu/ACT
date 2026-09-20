"""Independent bulk affine check, mathematically identical to scoped-range V1."""
import copy
from fractions import Fraction as F
try:
    from proof_format import unpack,identity,rational,clean
    from range_check import header,check_range,finish
except ImportError:
    from source_enclosure.format import unpack,identity,rational,clean
    from source_ranges.check import header,check_range,finish


def check(source,target,operator,bias,proof,context,tag):
    s,t,c,b,ct,bt=header(source,target,proof,context,tag,'affine')
    if (set(context)!={'request_id','scope','layer'} or any(type(v)is not str or not v for v in context.values()) or
            not operator or not len(operator)==len(bias)==len(proof['facts'])==len(proof['ranges'])):
        raise ValueError('context/all row obligations')
    expected=copy.deepcopy(s);ids=list(c);checks=[]
    expected['c']=[];expected['Gc']=[];expected['Gb']=[]
    for i,(op,offset,fact) in enumerate(zip(operator,bias,proof['facts'])):
        center=rational(offset);gc={};gb={}
        for j,value in op.items():
            if type(j)is not int or not 0<=j<len(s['c']):raise ValueError('affine input index')
            w=rational(value);center+=w*s['c'][j]
            for out,key in [(gc,'Gc'),(gb,'Gb')]:
                for k,v in s[key][j].items():out[k]=out.get(k,F(0))+w*v
        gc,gb=clean(gc),clean(gb);radius=sum(map(abs,list(gc.values())+list(gb.values())),F(0))
        if fact is None:
            lo,hi=center-radius,center+radius;checked={'kind':'GENERATOR_BOX','range':[str(lo),str(hi)]}
        else:(lo,hi),checked=check_range(source,op,offset,fact,context,i)
        if proof['ranges'][i]!=[str(lo),str(hi)]:raise ValueError('consumed range differs')
        checks.append(checked);mid=(lo+hi)/2;r=(hi-lo)/2;out={}
        if r:j=len(ids);ids.append(f'{tag}/value/{i}');out[j]=r;gc[j]=-r
        expected['c'].append(mid);expected['Gc'].append(out);expected['Gb'].append({})
        if gc or gb or mid!=center:expected['Ac'].append(gc);expected['Ab'].append(gb);expected['b'].append(mid-center)
    return finish(target,expected,ct,bt,ids,b,checks,new_continuous_factors=len(ids)-len(c),new_binary_factors=0,kind='affine')
