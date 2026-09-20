"""Independent full classification inventory and sparse McCormick construction."""
from fractions import Fraction as F
try:
    from proof_format import unpack, identity
    from local_check import csr
except ImportError:
    from source_enclosure.format import unpack, identity
    from upstream_source.checker import csr


def check(state, base, obligations, pair, classes, label):
    h,ci,bi=unpack(state); nc,nb=len(ci),len(bi); n=nc+nb
    source_id,base_id=identity(state),identity(base)
    if type(classes)is not int or classes<2 or type(label)is not int or not 0<=label<classes or len(h['c'])!=2*classes:
        raise ValueError('classification request shape')
    if (base['schema']!='SHARED_NEW_OUTPUT_LP_BASE_V1' or base['source_sha256']!=source_id or
            base['variables']!=n+2 or base['factor_bounds']!=[-1,1] or base['continuous_ids']!=ci or base['binary_ids']!=bi or
            base['binary_relaxation']!='MINUS_PLUS_ONE_TO_CONTINUOUS'):
        raise ValueError('LP base new source/factor binding')
    for key,ac,ab,rhs in [('A','Auc','Aub','ub'),('E','Ac','Ab','b')]:
        actual=csr(base[key],[len(h[rhs]),n+2]); values=list(map(F,base['b' if key=='A' else 'h']))
        if values!=h[rhs]:raise ValueError('LP base RHS')
        for j,row in enumerate(actual):
            if row!={**h[ac][j],**{i+nc:v for i,v in h[ab][j].items()}}:raise ValueError('LP inherited constraint mapping')
    o=obligations
    if (o['schema']!='COMPLETE_NEW_OUTPUT_OBLIGATIONS_V1' or o['pair']!=pair or o['classes']!=classes or
            o['label']!=label or o['source_sha256']!=source_id or o['base_sha256']!=base_id or
            o['old_LP_certificates_used']!=0 or [r['competitor'] for r in o['rows']]!=[i for i in range(classes) if i!=label]):
        raise ValueError('missing/duplicate/wrong new output obligation')
    for r in o['rows']:
        k=r['competitor']
        if (r['pair']!=pair or r['label']!=label or r['source_sha256']!=source_id or r['base_sha256']!=base_id or
                r['gate']!=['0','1'] or r['lower_bound_certificate'] is not None):raise ValueError('obligation binding/historical certificate')
        constants=[];coeff=[]
        for e in range(2):
            y,z=e*classes+label,e*classes+k; constants.append(h['c'][y]-h['c'][z]); v={}
            for mat,shift in [('Gc',0),('Gb',nc)]:
                for col in h[mat][y].keys()|h[mat][z].keys():
                    x=h[mat][y].get(col,F(0))-h[mat][z].get(col,F(0))
                    if x:v[col+shift]=x
            coeff.append(v)
        u0=constants[1]; d0=constants[0]-u0; u=coeff[1]
        d={i:coeff[0].get(i,F(0))-coeff[1].get(i,F(0)) for i in coeff[0].keys()|coeff[1].keys()}
        radius=sum(map(abs,d.values()),F(0)); lo,hi=d0-radius,d0+radius
        if list(map(F,r['difference']))!=[lo,hi] or [[F(x) for x in p] for p in r['extra_bounds']]!=[[0,1],[min(F(0),lo),max(F(0),hi)]]:
            raise ValueError('unchecked product rectangle/corner bound')
        if F(r['offset'])!=u0 or csr(r['objective'],[1,n+2])!=[{**u,n+1:F(1)}]:raise ValueError('property objective')
        # w >= lo*lambda ; w >= d+hi*lambda-hi;
        # w <= d+lo*lambda-lo ; w <= hi*lambda.
        want=[{n:lo,n+1:F(-1)}, {**d,n:hi,n+1:F(-1)},
              {**{j:-v for j,v in d.items()},n:-lo,n+1:F(1)}, {n:-hi,n+1:F(1)}]
        want=[{j:v for j,v in row.items() if v} for row in want]
        if csr(r['A_extra'],[4,n+2])!=want or list(map(F,r['b_extra']))!=[0,hi-d0,d0-lo,0]:
            raise ValueError('McCormick plane sign/coefficient/RHS')
    return {'status':'CHECKED_ALL_NEW_OUTPUT_LP_CONSTRUCTIONS','obligations':classes-1,
        'new_source_sha256':source_id,'base_sha256':base_id,'variables':n+2,
        'base_equalities':len(h['b']),'base_inequalities':len(h['ub']),
        'lower_bounds_checked':0,'positive_certificates':0,'old_LP_certificates_used':0,
        'scope':'All new real normalized top2 classification LP outer obligations, NOT positive lower-bound certificates.'}
