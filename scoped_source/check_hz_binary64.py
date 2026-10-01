"""Independent exact inclusion check for a finite owned HZ binary64 enclosure.

No producer, numeric kernel or optimizer import. This does not check a network.
"""
from fractions import Fraction as F
import math

from source_enclosure.format import identity, unpack, pack
from scoped_source.graph import clock

REFERENCE = 'OWNED_EXACT_HZ_REFERENCE_V1'
SCHEMA = 'HZ_BINARY64_OUTER_ENCLOSURE_V1'
VECTORS = ('c','b','ub')
MATRICES = ('Gc','Gb','Ac','Ab','Auc','Aub')


def label(value):
    if type(value) is not str or not value or len(value)>48:
        raise ValueError('explicit finite owner/tag required')
    return value


def parse(reference, *, target=False):
    if (set(reference)!={'schema','state','ownership'} or reference['schema']!=REFERENCE
            or set(reference['ownership'])!={'continuous','binary'}):
        raise ValueError('owned reference schema')
    state=reference['state']; h,c,b=unpack(state)
    if (len(c)+len(b)>(16 if target else 8) or not 1<=len(h['c'])<=4
            or len(h['b'])>4 or len(h['ub'])>4 or pack(h,c,b)!=state):
        raise ValueError('finite canonical reference capacity')
    for kind,ids in (('continuous',c),('binary',b)):
        owners=reference['ownership'][kind]
        if type(owners) is not list or len(owners)!=len(ids):
            raise ValueError('factor ownership dimensions')
        for owner in owners:label(owner)
    for key in VECTORS+MATRICES:
        for row in h[key]:
            for v in row.values() if isinstance(row,dict) else [row]:
                if max(v.numerator.bit_length(),v.denominator.bit_length())>4096:
                    raise ValueError('finite coefficient capacity')
                if target:
                    try:f=float(v)
                    except OverflowError:raise ValueError('nonfinite binary64 target') from None
                    if not math.isfinite(f) or F(f)!=v:
                        raise ValueError('target is not exact stored binary64')
    return h,c,b


def check(reference, proof, *, expected_reference_sha256, expected_owner, deadline):
    tick=clock(deadline); label(expected_owner); anchor=identity(proof)
    if identity(reference)!=expected_reference_sha256:raise ValueError('external reference binding')
    s,ci,bi=parse(reference)
    if (set(proof)!={'schema','reference_sha256','owner','target','target_sha256','maps','output_errors','equality_errors'}
            or proof['schema']!=SCHEMA or proof['reference_sha256']!=expected_reference_sha256
            or proof['owner']!=expected_owner or identity(proof['target'])!=proof['target_sha256']):
        raise ValueError('enclosure identity/schema')
    t,ct,bt=parse(proof['target'],target=True)
    if (t['frame_id']!=s['frame_id'] or bt!=bi or ct[:len(ci)]!=ci
            or proof['target']['ownership']['continuous'][:len(ci)]!=reference['ownership']['continuous']
            or proof['target']['ownership']['binary']!=reference['ownership']['binary']
            or any(len(t[k])!=len(s[k]) for k in VECTORS)
            or len(proof['output_errors'])!=len(s['c']) or len(proof['equality_errors'])!=len(s['b'])):
        raise ValueError('original factor/owner/frame/row inventory')
    maps={'continuous':list(range(len(ci))),'binary':list(range(len(bi))),
          'flat':list(range(len(ci)))+[len(ct)+j for j in range(len(bi))]}
    if (proof['maps']!=maps or any(type(v) is not int for values in proof['maps'].values() for v in values)):
        raise ValueError('continuous/binary/flat factor maps')
    fresh=[]; out_slack={}; eq_slack={}; diagnostics=[]
    # Independence: compute residuals from every source and target coefficient,
    # not from claimed rounding operations or producer-generated needed radii.
    for kind,vector,ck,bk,claims in (
            ('output','c','Gc','Gb',proof['output_errors']),
            ('equality','b','Ac','Ab',proof['equality_errors'])):
        for i,claim in enumerate(claims):
            tick()
            if set(claim)!={'factor','radius'} or type(claim['radius']) is not str:
                raise ValueError('compensation inventory')
            radius=F(claim['radius'])
            if str(radius)!=claim['radius'] or radius<0:raise ValueError('compensation sign/canonical value')
            residual=abs(s[vector][i]-t[vector][i])
            for key,width in ((ck,len(ci)),(bk,len(bi))):
                residual+=sum((abs(s[key][i].get(j,F(0))-t[key][i].get(j,F(0))) for j in range(width)),F(0))
            if radius<residual:raise ValueError('inward compensation')
            expected=None
            if radius:
                expected=f'{expected_owner}/{expected_reference_sha256}/{kind}/{i}'
                fresh.append(expected); slot=len(ci)+len(fresh)-1
                (out_slack if kind=='output' else eq_slack)[i]=(slot,radius)
            if claim['factor']!=expected:raise ValueError('compensation identity')
            diagnostics.append({'kind':kind,'row':i,'required':str(residual),'stored':str(radius)})
    if (ct!=ci+fresh or proof['target']['ownership']['continuous']!=
            reference['ownership']['continuous']+[expected_owner]*len(fresh)):
        raise ValueError('fresh factor ownership/inventory')
    # No output factor in constraints; an equality slack belongs only to its row.
    for key in MATRICES:
        for i,row in enumerate(t[key]):
            actual={j:v for j,v in row.items() if key.endswith('c') and j>=len(ci)}
            required={}
            use=out_slack if key=='Gc' else eq_slack if key=='Ac' else {}
            if i in use:
                j,v=use[i];required={j:v}
            if actual!=required:raise ValueError('error/slack zero layout')
    inequality=[]
    for i,rhs in enumerate(s['ub']):
        tick(); required=rhs
        for key,width in (('Auc',len(ci)),('Aub',len(bi))):
            required+=sum((abs(s[key][i].get(j,F(0))-t[key][i].get(j,F(0))) for j in range(width)),F(0))
        if t['ub'][i]<required:raise ValueError('inward inequality/guard')
        inequality.append({'row':i,'required':str(required),'stored':str(t['ub'][i])})
    tick()
    if identity(reference)!=expected_reference_sha256 or identity(proof)!=anchor:
        raise ValueError('enclosure input changed during checking')
    tick()
    return {'status':'CHECKED_BINARY64_OUTER_ENCLOSURE_GIVEN_REFERENCE',
            'reference_sha256':expected_reference_sha256,'proof_sha256':anchor,
            'target_sha256':proof['target_sha256'],'added_continuous':len(fresh),
            'binary_preserved':len(bi),'compensation':diagnostics,'inequalities':inequality,
            'set_equality_claimed':False,'network_proof':False,'deployed_float_SAFE':False,
            'hard_budget_supervision':False,
            'remaining_trust':['given exact reference and declared ownership','exact checker implementation']}


def check_embedding(reference, proof, original, extended, *, expected_reference_sha256, expected_owner, deadline):
    """Supplementary exact points, never a substitute for the inclusion proof."""
    anchor=identity(proof)
    check(reference,proof,expected_reference_sha256=expected_reference_sha256,
          expected_owner=expected_owner,deadline=deadline)
    def point(ref,assignment):
        h,ci,bi=parse(ref,target=ref is proof['target'])
        if set(assignment)!={'continuous','binary'}:raise ValueError('point fields')
        x,z=(list(map(F,assignment[k])) for k in ('continuous','binary'))
        if len(x)!=len(ci) or len(z)!=len(bi) or any(abs(v)>1 for v in x) or any(v not in (-1,1) for v in z):
            raise ValueError('factor assignment domains')
        def rows(ck,bk):
            return [sum((v*x[j] for j,v in a.items()),F(0))+sum((v*z[j] for j,v in b.items()),F(0))
                    for a,b in zip(h[ck],h[bk])]
        if rows('Ac','Ab')!=h['b'] or any(v>b for v,b in zip(rows('Auc','Aub'),h['ub'])):
            raise ValueError('infeasible factor assignment')
        return [c+v for c,v in zip(h['c'],rows('Gc','Gb'))]
    s,ci,bi=parse(reference)
    if (extended['continuous'][:len(ci)]!=original['continuous'] or extended['binary']!=original['binary']):
        raise ValueError('embedding changed original factors')
    left,right=point(reference,original),point(proof['target'],extended)
    if left!=right:raise ValueError('embedding output mismatch')
    if identity(reference)!=expected_reference_sha256 or identity(proof)!=anchor:
        raise ValueError('embedding input changed')
    clock(deadline)()
    return {'status':'CHECKED_EXACT_ASSIGNMENT_EMBEDDING','outputs':list(map(str,left))}
