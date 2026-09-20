"""Independent scoped range and source-step checks; no producer or solver calls."""
import copy
from fractions import Fraction as F
try:
    from proof_format import unpack, identity, rational, sparse, clean
except ImportError:
    from source_enclosure.format import unpack, identity, rational, sparse, clean
from act.back_end.solver.sparse_lp_certificate import evaluate


def projection(source, weights, bias):
    s, ci, bi = unpack(source); n = len(ci)+len(bi)
    if not n: raise ValueError('finite nonempty factor box required')
    center = rational(bias); coeff = {}
    for j, value in weights.items():
        if type(j) is not int or not 0 <= j < len(s['c']): raise ValueError('property column')
        w = rational(value); center += w*s['c'][j]
        for key, shift in [('Gc',0),('Gb',len(ci))]:
            for k,v in s[key][j].items(): coeff[k+shift] = coeff.get(k+shift,F(0))+w*v
    coeff = clean(coeff)
    lp = {'matrix_format':'csr_v1','c':[str(coeff.get(j,F(0))) for j in range(n)],
          'offset':str(center),'lower':[-1]*n,'upper':[1]*n}
    for key, ck, bk, rhs in [('A','Auc','Aub','ub'),('E','Ac','Ab','b')]:
        lp[key] = sparse([{**a, **{j+len(ci):v for j,v in b.items()}} for a,b in zip(s[ck],s[bk])],n)
        lp['b' if key=='A' else 'h'] = [str(x) for x in s[rhs]]
    rad = sum(map(abs,coeff.values()),F(0))
    return lp, (center-rad,center+rad)


def query(source, weights, bias, context, row):
    if (set(context) != {'request_id','scope','layer'} or
            any(type(v) is not str or not v for v in context.values()) or type(row) is not int or row < 0):
        raise ValueError('explicit external request/scope/layer/row required')
    # Canonical expression identity retains the source-output indices, not just
    # an accidentally equal projected LP or a native solver row number.
    expression = [[j,str(rational(v))] for j,v in sorted(weights.items()) if rational(v)]
    return {'source_sha256':identity(source),'context':context,'row':row,
            'expression':expression,'offset':str(rational(bias))}


def check_range(source, weights, bias, fact, context, row):
    lp, box = projection(source,weights,bias)
    if fact is None: return box, {'kind':'GENERATOR_BOX','range':list(map(str,box))}
    if (set(fact) != {'schema','query','range','lower','negative_upper'} or
            fact['schema'] != 'SCOPED_SOURCE_RANGE_V1' or fact['query'] != query(source,weights,bias,context,row)):
        raise ValueError('source/request/scope/layer/property binding')
    if type(fact['range']) is not list or len(fact['range']) != 2: raise ValueError('range endpoints')
    lo,hi = map(rational,fact['range'])
    if not box[0] <= lo <= hi <= box[1]: raise ValueError('ordered nonwidening range required')
    neg = {**lp,'c':[str(-F(x)) for x in lp['c']],'offset':str(-F(lp['offset']))}
    values=[]
    for program,cert in [(lp,fact['lower']),(neg,fact['negative_upper'])]:
        if set(cert) != {'lp_sha256','inequality_dual','equality_dual'}: raise ValueError('multipliers only')
        values.append(evaluate(program,cert)[0])
    if values[0] < lo or values[1] < -hi: raise ValueError('inward or unproved range')
    return (lo,hi), {'kind':'CHECKED_SOURCE_DUAL_RANGE','range':[str(lo),str(hi)],
        'lower_checked':str(values[0]),'negative_upper_checked':str(values[1]),'fact_sha256':identity(fact)}


def ranges(source, operator, bias, facts, context):
    if not operator or not len(operator)==len(bias)==len(facts): raise ValueError('all row obligations required')
    answer=[];checks=[]
    for i,(op,b,f) in enumerate(zip(operator,bias,facts)):
        # Validate external identity even for a row using the sound fallback.
        query(source,op,b,context,i)
        bounds, checked = check_range(source,op,b,f,context,i);answer.append(bounds);checks.append(checked)
    return answer, checks


def header(source,target,proof,context,tag,kind):
    s,c,b=unpack(source);t,ct,bt=unpack(target)
    if (set(proof) != {'schema','source','context','tag','kind','facts','ranges'} or
            proof['schema'] != 'CHECKED_RANGE_SOURCE_STEP_V1' or proof['source'] != identity(source) or
            proof['context'] != context or proof['tag'] != tag or proof['kind'] != kind):
        raise ValueError('step binding')
    if t['frame_id'] != s['frame_id'] or ct[:len(c)] != c or bt[:len(b)] != b:
        raise ValueError('source factor identity/frame')
    return s,t,c,b,ct,bt


def finish(target, expected, ct, bt, ci, bi, checked, **extra):
    t,_,_=unpack(target)
    if ct != ci or bt != bi: raise ValueError('fresh factor placement/identity')
    if any(t[k] != expected[k] for k in ('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub')):
        raise ValueError('source relation or constraints changed')
    return {'status':'CHECKED_RANGE_SOURCE_STEP','state_sha256':identity(target),'rows':len(checked),
        'checked_range_rows':sum(r['kind']=='CHECKED_SOURCE_DUAL_RANGE' for r in checked),
        'fallback_rows':sum(r['kind']=='GENERATOR_BOX' for r in checked),'ranges':checked,
        'new_continuous_factors':extra.pop('new_continuous_factors'),**extra,
        'complete_network_certificate':False,'production_verdict_changed':False}


def check_affine(source,target,operator,bias,proof,context,tag):
    s,t,c,b,ct,bt=header(source,target,proof,context,tag,'affine')
    bounds,checked=ranges(source,operator,bias,proof['facts'],context)
    if proof['ranges'] != [list(map(str,p)) for p in bounds]: raise ValueError('consumed range differs')
    expected=copy.deepcopy(s);ci=list(c);expected['c']=[];expected['Gc']=[];expected['Gb']=[]
    for i,(op,offset,(lo,hi)) in enumerate(zip(operator,bias,bounds)):
        lp,_=projection(source,op,offset);center=F(lp['offset']);mid=(lo+hi)/2;rad=(hi-lo)/2
        gc={j:F(v) for j,v in enumerate(lp['c'][:len(c)]) if F(v)}
        gb={j:F(v) for j,v in enumerate(lp['c'][len(c):]) if F(v)}
        out={}
        if rad:
            j=len(ci);ci.append(f'{tag}/value/{i}');out[j]=rad;gc[j]=-rad
        expected['c'].append(mid);expected['Gc'].append(out);expected['Gb'].append({})
        # Even zero-width, nonconstant expressions retain a defining equality.
        if gc or gb or mid != center:
            expected['Ac'].append(gc);expected['Ab'].append(gb);expected['b'].append(mid-center)
    return finish(target,expected,ct,bt,ci,b,checked,new_continuous_factors=len(ci)-len(c),
                  new_binary_factors=0,kind='affine')


def check_relu(source,target,proof,context,tag):
    s,t,c,b,ct,bt=header(source,target,proof,context,tag,'relu')
    bounds,checked=ranges(source,[{i:1} for i in range(len(s['c']))],[0]*len(s['c']),proof['facts'],context)
    if proof['ranges'] != [list(map(str,p)) for p in bounds]: raise ValueError('consumed range differs')
    expected=copy.deepcopy(s);ci=list(c);bi=list(b);counts=dict(active=0,inactive=0,unstable=0)
    for i,(lo,hi) in enumerate(bounds):
        kind='active' if lo>=0 else 'inactive' if hi<=0 else 'unstable';counts[kind]+=1
        if kind=='active':continue
        if kind=='inactive':expected['c'][i]=F(0);expected['Gc'][i]={};expected['Gb'][i]={};continue
        u,v,z=len(ci),len(ci)+1,len(bi)
        ci += [f'{tag}/negative/{i}',f'{tag}/positive/{i}'];bi.append(f'{tag}/sign/{i}')
        expected['c'][i]=hi/2;expected['Gc'][i]={v:-hi/2};expected['Gb'][i]={}
        gc={k:-x for k,x in s['Gc'][i].items()};gb={k:-x for k,x in s['Gb'][i].items()}
        gc[u]=lo/2;gc[v]=-hi/2;gb[z]=lo/2
        expected['Ac'].append(clean(gc));expected['Ab'].append(clean(gb));expected['b'].append(s['c'][i]-hi/2)
        expected['Auc'] += [{u:F(-1)},{v:F(-1)}];expected['Aub'] += [{z:F(-1)},{z:F(1)}];expected['ub'] += [F(0),F(0)]
    return finish(target,expected,ct,bt,ci,bi,checked,new_continuous_factors=len(ci)-len(c),
                  new_binary_factors=len(bi)-len(b),kind='relu',**counts)
