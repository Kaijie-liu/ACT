"""Independent row-composition checker, not a new rounding/solver policy."""
from fractions import Fraction as F
import math

from source_enclosure.format import identity,pack,unpack,empty
from scoped_source.graph import clock
from scoped_source.check_hz_binary64 import REFERENCE,label,check as check_local

SCHEMA='ROW_COMPOSED_BINARY64_HZ_V1'
KINDS=(('output','c','Gc','Gb'),('equality','b','Ac','Ab'),('inequality','ub','Auc','Aub'))


def parse(value,*,target=False):
    if set(value)!={'schema','state','ownership'} or value['schema']!=REFERENCE:
        raise ValueError('global owned HZ schema')
    h,c,b=unpack(value['state']); owners=value['ownership']
    if (len(c)+len(b)>128 or not 1<=len(h['c'])<=128 or len(h['b'])+len(h['ub'])>256
            or pack(h,c,b)!=value['state'] or set(owners)!={'continuous','binary'}):
        raise ValueError('unchanged source capacity/canonical form')
    for kind,ids in (('continuous',c),('binary',b)):
        if type(owners[kind]) is not list or len(owners[kind])!=len(ids):raise ValueError('global ownership dimensions')
        for owner in owners[kind]:label(owner)
    for _,v,ck,bk in KINDS:
        for q in h[v]+[x for k in (ck,bk) for row in h[k] for x in row.values()]:
            if max(q.numerator.bit_length(),q.denominator.bit_length())>4096:raise ValueError('coefficient capacity')
            if target:
                try:f=float(q)
                except OverflowError:raise ValueError('finite binary64 global target') from None
                if not math.isfinite(f) or F(f)!=q:raise ValueError('finite binary64 global target')
    return h,c,b


def projected(reference,kind,row):
    """Independently derive exact local source, never read a proposed mapping."""
    s,c,b=parse(reference);_,v,ck,bk=next(k for k in KINDS if k[0]==kind)
    cm=sorted(s[ck][row]);bm=sorted(s[bk][row]);h=empty(1,s['frame_id'])
    a={j:s[ck][row][i] for j,i in enumerate(cm)};d={j:s[bk][row][i] for j,i in enumerate(bm)}
    h[v]=[s[v][row]];h[ck]=[a];h[bk]=[d]
    return {'schema':REFERENCE,'state':pack(h,[c[i] for i in cm],[b[i] for i in bm]),
            'ownership':{'continuous':[reference['ownership']['continuous'][i] for i in cm],
                         'binary':[reference['ownership']['binary'][i] for i in bm]}},cm,bm


def check(reference,proof,*,expected_reference_sha256,expected_owner,deadline):
    tick=clock(deadline);label(expected_owner);anchor=identity(proof)
    if identity(reference)!=expected_reference_sha256:raise ValueError('global reference binding')
    s,c,b=parse(reference)
    if (set(proof)!={'schema','reference_sha256','owner','rows','target','target_sha256','maps'}
            or proof['schema']!=SCHEMA or proof['reference_sha256']!=expected_reference_sha256
            or proof['owner']!=expected_owner or proof['target_sha256']!=identity(proof['target'])):
        raise ValueError('global enclosure identity/schema')
    expected=empty(len(s['c']),s['frame_id']);newc=list(c);newowners=list(reference['ownership']['continuous'])
    roster=[(kind,i) for kind,v,_,_ in KINDS for i in range(len(s[v]))]
    if [(r['kind'],r['row']) for r in proof['rows']]!=roster:raise ValueError('complete ordered row inventory')
    count=0
    for item,(kind,i) in zip(proof['rows'],roster):
        tick()
        if set(item)!={'kind','row','continuous_map','binary_map','local'} or type(item['row']) is not int:
            raise ValueError('row schema')
        ref,cm,bm=projected(reference,kind,i)
        if (item['continuous_map']!=cm or item['binary_map']!=bm
                or any(type(j) is not int for j in item['continuous_map']+item['binary_map'])):
            raise ValueError('row original factor mapping')
        lp=item['local']; result=check_local(ref,lp,expected_reference_sha256=identity(ref),
                                           expected_owner=f'row/{kind}/{i}',deadline=deadline)
        h,lc,lb=unpack(lp['target']['state']);mapping=list(cm)
        fresh=len(lc)-len(cm)
        if fresh not in (0,1) or kind!='output' and (h['c']!=[F(0)] or h['Gc']!=[{}] or h['Gb']!=[{}]):
            raise ValueError('single row compensation/dummy output')
        if fresh:
            newc.append(f'{expected_owner}/{expected_reference_sha256}/{kind}/{i}')
            newowners.append(expected_owner);mapping.append(len(newc)-1);count+=1
        _,v,ck,bk=next(k for k in KINDS if k[0]==kind)
        mapped_c={mapping[j]:q for j,q in h[ck][0].items()}
        mapped_b={bm[j]:q for j,q in h[bk][0].items()}
        if kind=='output':
            expected[v][i]=h[v][0];expected[ck][i]=mapped_c;expected[bk][i]=mapped_b
        else:
            expected[v].append(h[v][0]);expected[ck].append(mapped_c);expected[bk].append(mapped_b)
    target={'schema':REFERENCE,'state':pack(expected,newc,b),
            'ownership':{'continuous':newowners,'binary':reference['ownership']['binary'][:]}}
    parse(target,target=True);parse(proof['target'],target=True)
    maps={'continuous':list(range(len(c))),'binary':list(range(len(b))),
          'flat':list(range(len(c)))+[len(newc)+j for j in range(len(b))]}
    if (proof['target']!=target or proof['maps']!=maps
            or any(type(j) is not int for values in proof['maps'].values() for j in values)):
        raise ValueError('full global assembly/ownership/zero layout')
    result={'status':'CHECKED_ROW_COMPOSED_BINARY64_ENCLOSURE_GIVEN_REFERENCE',
            'reference_sha256':expected_reference_sha256,'proof_sha256':anchor,
            'rows':len(roster),'added_continuous':count,'target_sha256':identity(target),
            'network_proof':False,'deployed_float_SAFE':False}
    if identity(reference)!=expected_reference_sha256 or identity(proof)!=anchor:raise ValueError('row proof/reference mutated')
    tick();return result
