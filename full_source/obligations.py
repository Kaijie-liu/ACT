"""Untrusted shared-matrix LP obligations on the NEW complete expert state."""
from fractions import Fraction as F
from source_enclosure.format import unpack, identity, sparse


def build(state, pair, classes, label):
    h, ci, bi = unpack(state); nc, nb = len(ci), len(bi); n = nc+nb
    if len(h['c']) != 2*classes or not 0 <= label < classes: raise ValueError('classification shape/label')
    source_id=identity(state)
    base = {'schema':'SHARED_NEW_OUTPUT_LP_BASE_V1', 'source_sha256':source_id,
            'variables':n+2,'factor_bounds':[-1,1], 'binary_relaxation':'MINUS_PLUS_ONE_TO_CONTINUOUS',
            'continuous_ids':ci,'binary_ids':bi}
    for key,a,b,r in [('E','Ac','Ab','b'),('A','Auc','Aub','ub')]:
        rows=[{**x,**{j+nc:v for j,v in y.items()}} for x,y in zip(h[a],h[b])]
        base[key]=sparse(rows,n+2); base['h' if key=='E' else 'b']=[str(v) for v in h[r]]
    rows=[];base_id=identity(base)
    for competitor in range(classes):
        if competitor==label: continue
        u0=h['c'][classes+label]-h['c'][classes+competitor]
        d0=h['c'][label]-h['c'][competitor]-u0
        u={}; d={}
        for key,shift in [('Gc',0),('Gb',nc)]:
            for i,sign in [(classes+label,1),(classes+competitor,-1)]:
                for j,v in h[key][i].items():u[j+shift]=u.get(j+shift,F(0))+sign*v
            for i,sign in [(label,1),(competitor,-1),(classes+label,-1),(classes+competitor,1)]:
                for j,v in h[key][i].items():d[j+shift]=d.get(j+shift,F(0))+sign*v
        rad=sum(map(abs,d.values()),F(0)); lo,hi=d0-rad,d0+rad
        a,b=F(0),F(1) # universal normalized softmax range, no historical gate evidence
        planes=[];rhs=[]
        for s,t,sgn in [(a,lo,1),(b,hi,1),(b,lo,-1),(a,hi,-1)]:
            row={j:sgn*s*v for j,v in d.items() if s*v}
            row[n]=sgn*t;row[n+1]=-sgn;planes.append(row);rhs.append(str(sgn*s*(t-d0)))
        objective={j:v for j,v in u.items() if v};objective[n+1]=F(1)
        rows.append({'pair':pair,'label':label,'competitor':competitor,'source_sha256':source_id,
            'base_sha256':base_id,'gate':['0','1'],'difference':[str(lo),str(hi)],
            'extra_bounds':[['0','1'],[str(min(F(0),lo,hi)),str(max(F(0),lo,hi))]],
            'objective':sparse([objective],n+2),'offset':str(u0),
            'A_extra':sparse(planes,n+2),'b_extra':rhs,'lower_bound_certificate':None})
    return base,{'schema':'COMPLETE_NEW_OUTPUT_OBLIGATIONS_V1','pair':pair,'classes':classes,
        'label':label,'source_sha256':source_id,'base_sha256':base_id,'rows':rows,
        'old_LP_certificates_used':0}


def materialize(base, obligation):
    """Solver input adapter, not proof checker. Does not call a solver."""
    from upstream_source.checker import csr
    n=base['variables']; c=csr(obligation['objective'],[1,n])[0]
    a,b=base['A'],obligation['A_extra']
    A={'shape':[a['shape'][0]+4,n],'data':a['data']+b['data'],'indices':a['indices']+b['indices'],
       'indptr':a['indptr']+[len(a['data'])+p for p in b['indptr'][1:]]}
    return {'matrix_format':'csr_v1','c':[str(c.get(i,F(0))) for i in range(n)],'offset':obligation['offset'],
        'A':A,'b':base['b']+obligation['b_extra'],'E':base['E'],'h':base['h'],
        'lower':[-1]*(n-2)+[p[0] for p in obligation['extra_bounds']],
        'upper':[1]*(n-2)+[p[1] for p in obligation['extra_bounds']]}
