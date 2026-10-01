"""Untrusted sparse row lift; original global factors are never copied apart."""
from source_enclosure.format import empty,pack,unpack,identity
from scoped_source.graph import clock
from scoped_source.check_hz_binary64 import REFERENCE,label
from scoped_source.hz_binary64 import produce as local_lift
from scoped_source.check_hz_row_enclosure import SCHEMA,KINDS,parse


def produce(reference,*,expected_reference_sha256,owner,deadline):
    tick=clock(deadline);label(owner)
    if identity(reference)!=expected_reference_sha256:raise ValueError('row source identity')
    s,c,b=parse(reference);target=empty(len(s['c']),s['frame_id']);ci=list(c)
    co=list(reference['ownership']['continuous']);records=[]
    for kind,v,ck,bk in KINDS:
        for i,rhs in enumerate(s[v]):
            tick();cm=sorted(s[ck][i]);bm=sorted(s[bk][i]);r=empty(1,s['frame_id'])
            r[v]=[rhs];r[ck]=[{j:s[ck][i][k] for j,k in enumerate(cm)}]
            r[bk]=[{j:s[bk][i][k] for j,k in enumerate(bm)}]
            ref={'schema':REFERENCE,'state':pack(r,[c[k] for k in cm],[b[k] for k in bm]),
                 'ownership':{'continuous':[reference['ownership']['continuous'][k] for k in cm],
                              'binary':[reference['ownership']['binary'][k] for k in bm]}}
            p=local_lift(ref,expected_reference_sha256=identity(ref),owner=f'row/{kind}/{i}',deadline=deadline)
            h,lc,lb=unpack(p['target']['state']);mapping=cm[:]
            if len(lc)>len(cm):
                mapping.append(len(ci));ci.append(f'{owner}/{expected_reference_sha256}/{kind}/{i}');co.append(owner)
            tc={mapping[k]:q for k,q in h[ck][0].items()};tb={bm[k]:q for k,q in h[bk][0].items()}
            if kind=='output':target[v][i]=h[v][0];target[ck][i]=tc;target[bk][i]=tb
            else:target[v].append(h[v][0]);target[ck].append(tc);target[bk].append(tb)
            records.append({'kind':kind,'row':i,'continuous_map':cm,'binary_map':bm,'local':p})
    result={'schema':REFERENCE,'state':pack(target,ci,b),
            'ownership':{'continuous':co,'binary':reference['ownership']['binary'][:]}}
    parse(result,target=True)
    proof={'schema':SCHEMA,'reference_sha256':expected_reference_sha256,'owner':owner,'rows':records,
           'target':result,'target_sha256':identity(result),
           'maps':{'continuous':list(range(len(c))),'binary':list(range(len(b))),
                   'flat':list(range(len(c)))+[len(ci)+j for j in range(len(b))]}}
    if identity(reference)!=expected_reference_sha256:raise ValueError('row source mutated')
    tick();return proof
