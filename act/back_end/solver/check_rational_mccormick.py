"""Independent construction check: no builder, floating F0, torch or solver.

Caller must discharge gate/difference bounds and bind the given HZ to the
request/pair domain. This checker validates that exact rectangle's relaxation.
"""
from fractions import Fraction
from act.back_end.solver.lp_certificate import identity, rational, check
from act.back_end.solver.check_hz_lp_export import _entries


def check_construction(record, certificate=None, *, source_hash, q, offset, gate, difference):
    src, lp = record['source'],record['lp']
    if record['schema']!='shared_hz_rational_mccormick_v1' or identity(src)!=source_hash or record['source_sha256']!=source_hash:
        raise ValueError('shared source identity mismatch')
    vec=lambda xs:[rational(x) for x in xs]
    if vec(record['q'])!=vec(q) or rational(record['offset'])!=rational(offset): raise ValueError('property mismatch')
    if vec(record['gate'])!=vec(gate) or vec(record['difference'])!=vec(difference): raise ValueError('range binding mismatch')
    a,b=vec(gate); l,h=vec(difference)
    if not 0<=a<=b<=1 or l>h: raise ValueError('invalid rectangle')
    matrices={k:_entries(src[k]) for k in ('Gc','Gb','Ac','Ab','Auc','Aub')}
    outputs,nc=matrices['Gc'][0]; outputs_b,nb=matrices['Gb'][0]; n=nc+nb; m=len(q)
    if outputs!=2*m or outputs_b!=outputs or len(src['c'])!=outputs: raise ValueError('expert dimensions')
    if record['relaxation']!='BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX' or record['n_relaxed_binaries']!=nb:
        raise ValueError('binary relaxation declaration')
    # Independently accumulate each expert's property, then subtract.
    projections=[[Fraction(0)]*n for _ in range(2)]
    constants=[Fraction(0),Fraction(0)]
    for expert in range(2):
        for i,weight in enumerate(vec(q)):
            constants[expert]+=weight*rational(src['c'][expert*m+i])
    for key,shift in [('Gc',0),('Gb',nc)]:
        for (i,j),v in matrices[key][1].items(): projections[i//m][j+shift]+=rational(q[i % m])*v
    u=projections[1]; u0=constants[1]+rational(offset)
    d=[x-y for x,y in zip(*projections)]; d0=constants[0]-constants[1]
    for name,v,c in [('u',u,u0),('d',d,d0)]:
        if vec(record[name]['coefficients'])!=v or rational(record[name]['constant'])!=c: raise ValueError('projection mismatch')
    corners=[a*l,a*h,b*l,b*h]
    if vec(lp['lower'])!=[-1]*n+[a,min(corners)] or vec(lp['upper'])!=[1]*n+[b,max(corners)]:
        raise ValueError('variable bounds changed')
    if lp.get('matrix_format')!='csr_v1' or vec(lp['c'])!=u+[0,1] or rational(lp['offset'])!=u0: raise ValueError('objective mismatch')
    for label,left,right,rhs in [('A','Auc','Aub','ub'),('E','Ac','Ab','b')]:
        (nr,width),entries=matrices[left]; shape_b,b_entries=matrices[right]
        if width!=nc or shape_b!=(nr,nb) or len(src[rhs])!=nr: raise ValueError('base shape mismatch')
        expected=dict(entries); expected.update({(i,j+nc):v for (i,j),v in b_entries.items()})
        expected_rhs=vec(src[rhs])
        if label=='A':
            # Derive the two supporting lower planes and two upper planes
            # directly; do not call or mirror a serialized builder result.
            factors=[(a,l,-1),(b,h,-1),(-b,-l,1),(-a,-h,1)]
            offsets=[a*(l-d0),b*(h-d0),b*(d0-l),a*(d0-h)]
            for k,(scale,lam,w) in enumerate(factors):
                for j,v in enumerate(d): expected[nr+k,j]=scale*v
                expected[nr+k,n]=lam; expected[nr+k,n+1]=w
            expected_rhs+=offsets
        shape,actual=_entries(lp[label]); target_rhs=lp['b' if label=='A' else 'h']
        if shape!=(len(expected_rhs),n+2) or vec(target_rhs)!=expected_rhs: raise ValueError('constraint shape/RHS mismatch')
        if {k:v for k,v in actual.items() if v}!={k:v for k,v in expected.items() if v}: raise ValueError('constraint coefficient mismatch')
    result=check(lp,certificate) if certificate is not None else None
    return {'status':'CHECKED','source_sha256':source_hash,'lp_sha256':identity(lp),'bound':result,
            'scope':'Exact construction from supplied shared HZ and externally checked rectangle; upstream lowering remains trusted.'}
