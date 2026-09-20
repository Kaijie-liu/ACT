"""Independent source-step checker. Does not import the producer or solver."""
import copy
from fractions import Fraction as F

try:
    from proof_format import unpack,identity,rational,clean
    from local_check import hz,input_cover
except ImportError:
    from source_enclosure.format import unpack,identity,rational,clean
    from upstream_source.checker import hz,input_cover


def check_box(lower,upper,target):
    h,c,b=unpack(target)
    if b or c!=[f'input/c/{i}' for i in range(len(lower))]:raise ValueError('input factor identity')
    result=input_cover(lower,upper,target['hz'])
    if result['inward_coordinates']:raise ValueError('input enclosure inward')
    # Fix coordinate-to-factor binding for later router/source use.
    for i,row in enumerate(h['Gc']):
        if row and (list(row)!=[i] or row[i]<0):raise ValueError('input coordinate map')
    return {'status':'CHECKED_INPUT_ENCLOSURE','coordinates':len(lower),'state_sha256':identity(target)}


def inherited(source,target):
    s,c,b=unpack(source);t,ct,bt=unpack(target)
    if t['frame_id']!=s['frame_id'] or ct[:len(c)]!=c or bt[:len(b)]!=b:
        raise ValueError('source factors/frame not preserved')
    return s,t,c,b,ct,bt


def same_constraints(s,t):
    if any(s[k]!=t[k] for k in ('Ac','Ab','b','Auc','Aub','ub')):
        raise ValueError('inherited constraints changed')


def check_guards(source,target,ac,ab,rhs):
    s,t,c,b,ct,bt=inherited(source,target)
    if c!=ct or b!=bt or any(s[k]!=t[k] for k in ('c','Gc','Gb','Ac','Ab','b')):
        raise ValueError('guard step changed source values/factors')
    if len(ac)!=len(rhs) or len(ab)!=len(rhs):raise ValueError('guard inventory')
    slacks=[]
    for a,d,r in zip(ac,ab,rhs):
        if any(not 0<=k<len(c) for k in a) or any(not 0<=k<len(b) for k in d):raise ValueError('guard columns')
        slack=rational(r)-sum((abs(rational(v)) for v in list(a.values())+list(d.values())),F(0))
        if slack<0:raise ValueError('new guard is not proved redundant')
        slacks.append(str(slack))
    if (t['Auc']!=s['Auc']+list(ac) or t['Aub']!=s['Aub']+list(ab) or
            t['ub']!=s['ub']+list(map(rational,rhs))):raise ValueError('actual guard rows differ')
    return {'status':'CHECKED_REDUNDANT_GUARD_ATTACHMENT','rows':len(rhs),'slacks':slacks,
            'state_sha256':identity(target)}


def check_affine(source,target,operator,bias,nominal,certificate,tag):
    s,t,c,b,ct,bt=inherited(source,target);p=hz(nominal)
    if (set(certificate)!={'source','tag','error_bounds'} or certificate['source']!=identity(source)
            or certificate['tag']!=tag or (p['nc'],p['nb'])!=(len(c),len(b)) or bt!=b):
        raise ValueError('affine source/proposal binding')
    n=len(operator)
    if n!=len(bias) or len(p['c'])!=n or len(t['c'])!=n or len(certificate['error_bounds'])!=n:
        raise ValueError('affine obligation dimensions')
    same_constraints(s,t)
    ids=list(c);checked=[]
    for i in range(n):
        # Form exact residual polynomial directly, coefficient by coefficient.
        residual_c=rational(bias[i])-p['c'][i]
        rc={k:-v for k,v in p['Gc'][i].items()};rb={k:-v for k,v in p['Gb'][i].items()}
        for j,w in operator[i].items():
            if type(j)is not int or not 0<=j<len(s['c']):raise ValueError('operator column')
            w=rational(w);residual_c+=w*s['c'][j]
            for row,which in ((rc,'Gc'),(rb,'Gb')):
                for k,v in s[which][j].items():row[k]=row.get(k,F(0))+w*v
        needed=abs(residual_c)+sum(map(abs,rc.values()),F(0))+sum(map(abs,rb.values()),F(0))
        radius=rational(certificate['error_bounds'][i])
        if radius!=needed:raise ValueError('missing/incorrect exact affine compensation')
        row=dict(p['Gc'][i])
        if radius:
            row[len(ids)]=radius;ids.append(f'{tag}/error/{i}')
        if t['c'][i]!=p['c'][i] or t['Gc'][i]!=row or t['Gb'][i]!=p['Gb'][i]:
            raise ValueError('compensation generator placement/value')
        checked.append(radius)
    if ct!=ids:raise ValueError('extra/missing/colliding compensation factors')
    return {'status':'CHECKED_AFFINE_OUTER_ENCLOSURE','rows':n,'error_factors':len(ct)-len(c),
            'maximum_compensation':str(max(checked,default=F(0))),'state_sha256':identity(target)}


def check_relu(source,target,certificate,tag):
    s,t,c,b,ct,bt=inherited(source,target);n=len(s['c'])
    if (set(certificate)!={'source','tag','ranges','branches'} or certificate['source']!=identity(source)
            or certificate['tag']!=tag or len(certificate['ranges'])!=n or len(certificate['branches'])!=n):
        raise ValueError('ReLU source/range inventory')
    expected=copy.deepcopy(s);ci=list(c);bi=list(b);counts=dict(active=0,inactive=0,unstable=0)
    for i,(lo,hi) in enumerate(certificate['ranges']):
        lo,hi=rational(lo),rational(hi)
        radius=sum(map(abs,s['Gc'][i].values()),F(0))+sum(map(abs,s['Gb'][i].values()),F(0))
        if lo>s['c'][i]-radius or hi<s['c'][i]+radius or lo>hi:raise ValueError('unchecked/inward ReLU range')
        branch='active' if lo>=0 else 'inactive' if hi<=0 else 'unstable'
        if certificate['branches'][i]!=branch:raise ValueError('ReLU classification')
        counts[branch]+=1
        if branch=='active':continue
        if branch=='inactive':
            expected['c'][i]=F(0);expected['Gc'][i]={};expected['Gb'][i]={};continue
        a,z=len(ci),len(bi);v=a+1
        ci.extend([f'{tag}/negative/{i}',f'{tag}/positive/{i}']);bi.append(f'{tag}/sign/{i}')
        expected['c'][i]=hi/2;expected['Gc'][i]={v:-hi/2};expected['Gb'][i]={}
        # a_pre = lo/2*(xi_negative+sign) + hi/2*(1-xi_positive).
        row_c={k:-value for k,value in s['Gc'][i].items()};row_b={k:-value for k,value in s['Gb'][i].items()}
        row_c[a]=lo/2;row_c[v]=-hi/2;row_b[z]=lo/2
        expected['Ac'].append(clean(row_c));expected['Ab'].append(clean(row_b));expected['b'].append(s['c'][i]-hi/2)
        expected['Auc']+=[{a:F(-1)},{v:F(-1)}];expected['Aub']+=[{z:F(-1)},{z:F(1)}];expected['ub']+=[F(0),F(0)]
    if ct!=ci or bt!=bi:raise ValueError('ReLU fresh factor identity')
    if any(expected[k]!=t[k] for k in ('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub')):
        raise ValueError('ReLU exact graph/constraint construction')
    return {'status':'CHECKED_RELU_GRAPH_OVER_SOURCE_ENCLOSURE','rows':n,**counts,
            'state_sha256':identity(target)}


def check_join(base,left,right,target,certificate):
    h0,c0,b0=unpack(base);a,ca,ba=unpack(left);b,cb,bb=unpack(right);t,ct,bt=unpack(target)
    if certificate.keys()!={'base','left','right','maps'} or any(certificate[k]!=identity(v) for k,v in
            (('base',base),('left',left),('right',right))):raise ValueError('join source binding')
    for h,ci,bi in ((a,ca,ba),(b,cb,bb)):
        if h['frame_id']!=h0['frame_id'] or ci[:len(c0)]!=c0 or bi[:len(b0)]!=b0:raise ValueError('join base factor prefix')
        for ck,bk,r in (('Ac','Ab','b'),('Auc','Aub','ub')):
            n=len(h0[r])
            if h[ck][:n]!=h0[ck] or h[bk][:n]!=h0[bk] or h[r][:n]!=h0[r]:raise ValueError('join base constraint prefix')
    if len(a['c'])!=len(b['c']) or t['frame_id']!=h0['frame_id']:raise ValueError('join output/frame')
    expected_c=c0+ca[len(c0):]+cb[len(c0):];expected_b=b0+ba[len(b0):]+bb[len(b0):]
    if (len(set(expected_c+expected_b))!=len(expected_c+expected_b) or ct!=expected_c or bt!=expected_b):
        raise ValueError('private factors alias across experts/kinds')
    ci={v:i for i,v in enumerate(ct)};bi={v:i for i,v in enumerate(bt)}
    maps={'left_c':[ci[v] for v in ca],'right_c':[ci[v] for v in cb],
          'left_b':[bi[v] for v in ba],'right_b':[bi[v] for v in bb]}
    if certificate['maps']!=maps:raise ValueError('incorrect factor map')
    def mapped(rows,m):return [{m[k]:v for k,v in row.items()} for row in rows]
    if t['c']!=a['c']+b['c']:raise ValueError('join expert output order')
    for key,kind in (('Gc','c'),('Gb','b')):
        if t[key]!=mapped(a[key],maps['left_'+kind])+mapped(b[key],maps['right_'+kind]):raise ValueError('join output factors')
    for ck,bk,r in (('Ac','Ab','b'),('Auc','Aub','ub')):
        n=len(h0[r])
        for key,kind in ((ck,'c'),(bk,'b')):
            required=h0[key]+mapped(a[key][n:],maps['left_'+kind])+mapped(b[key][n:],maps['right_'+kind])
            if t[key]!=required:raise ValueError('join constraint factor map/coverage')
        if t[r]!=h0[r]+a[r][n:]+b[r][n:]:raise ValueError('join constraint RHS/coverage')
    return {'status':'CHECKED_SHARED_INPUT_PRIVATE_FACTOR_JOIN','output_rows':len(t['c']),
            'shared_continuous':len(c0),'shared_binary':len(b0),'continuous':len(ct),'binary':len(bt),
            'equalities':len(t['b']),'inequalities':len(t['ub']),'state_sha256':identity(target)}
