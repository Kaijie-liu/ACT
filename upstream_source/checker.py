"""Local source-lowering checks, not a complete network proof.

Only standard-library exact arithmetic. No production lowering routine is
imported. Nonzero coefficient errors are reported, NOT silently tolerated.
"""
from fractions import Fraction as F
import math


def rational(value):
    if type(value)is F:return value
    if type(value) not in (int, float, str) or (type(value) is float and not math.isfinite(value)):
        raise ValueError('finite rational coefficient required')
    return F(value)


def csr(value, shape=None):
    dims=value['shape']
    if (len(dims)!=2 or any(type(n)is not int or n<0 for n in dims) or
            (shape is not None and dims!=list(shape))):
        raise ValueError('matrix shape')
    data,indices,ptr=value['data'],value['indices'],value['indptr']
    if (len(data)!=len(indices) or len(ptr)!=dims[0]+1 or ptr[0]!=0 or ptr[-1]!=len(data)
            or any(type(i)is not int for i in ptr+indices) or
            any(not 0<=a<=b<=len(data) for a,b in zip(ptr,ptr[1:]))):
        raise ValueError('CSR lengths/pointers')
    rows=[]
    for a,b in zip(ptr,ptr[1:]):
        ids=indices[a:b]
        if ids!=sorted(set(ids)) or any(not 0<=i<dims[1] for i in ids):
            raise ValueError('CSR duplicate/column/order')
        rows.append({i:rational(v) for i,v in zip(ids,data[a:b]) if rational(v)!=0})
    return rows


def hz(value):
    if set(value)!={'c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','frame_id','exact'}:
        raise ValueError('HZ schema')
    n=len(value['c']);nc=value['Gc']['shape'][1];nb=value['Gb']['shape'][1]
    if type(value['frame_id'])is not int or n<=0:raise ValueError('frame/output identity')
    result={k:[rational(v) for v in value[k]] for k in ('c','b','ub')}
    for name,shape in {'Gc':(n,nc),'Gb':(n,nb),'Ac':(len(result['b']),nc),
            'Ab':(len(result['b']),nb),'Auc':(len(result['ub']),nc),'Aub':(len(result['ub']),nb)}.items():
        result[name]=csr(value[name],shape)
    result.update(frame_id=value['frame_id'],nc=nc,nb=nb)
    return result


def input_cover(lower,upper,source):
    """Exact set containment for an unconstrained diagonal input HZ only."""
    s=hz(source);n=len(s['c']);used=set();failures=[]
    if len(lower)!=n or len(upper)!=n or s['nb'] or s['b'] or s['ub']:
        raise ValueError('input cover requires unconstrained diagonal source')
    for i,(row,center,lo,hi) in enumerate(zip(s['Gc'],s['c'],lower,upper)):
        lo,hi=rational(lo),rational(hi)
        if lo>hi or len(row)>1 or used.intersection(row):raise ValueError('box order/independent factors')
        used.update(row)
        radius=sum(map(abs,row.values()),F(0))
        low_gap=max(F(0),center-radius-lo);high_gap=max(F(0),hi-center-radius)
        if low_gap or high_gap:
            failures.append({'coordinate':i,'lower_gap':str(low_gap),'upper_gap':str(high_gap)})
    maximum=max((max(F(v['lower_gap']),F(v['upper_gap'])) for v in failures),default=F(0))
    return {'status':'INPUT_BOX_COVER_NOT_ESTABLISHED' if failures else 'CHECKED_INPUT_BOX_COVER',
        'coordinates':n,'inward_coordinates':len(failures),'maximum_inward_gap':str(maximum),
        'failures':failures,'scope':'Exact represented box versus supplied reconstructed input HZ; not a historical input-state trace.'}


def affine_error(source,target,operator,bias):
    """For the SAME factors, bound |exact(W*s+b) - stored target| per row.

    A nonzero bound does not establish target containment: an explicit error
    generator/other valid enclosure would still be needed. All old constraints
    and both continuous/binary factor positions must be preserved exactly.
    """
    s,t=hz(source),hz(target)
    if (s['frame_id'],s['nc'],s['nb'])!=(t['frame_id'],t['nc'],t['nb']):
        raise ValueError('affine frame/column binding')
    for name in ('Ac','Ab','b','Auc','Aub','ub'):
        if s[name]!=t[name]:raise ValueError('affine changed inherited constraints')
    if len(operator)!=len(t['c']) or len(bias)!=len(operator):raise ValueError('affine output dimensions')
    errors=[];different_coefficients=0
    for i,(row,b) in enumerate(zip(operator,bias)):
        if any(type(j)is not int or not 0<=j<len(s['c']) for j in row):raise ValueError('affine input column')
        row={j:rational(v) for j,v in row.items()}
        center=rational(b)+sum((v*s['c'][j] for j,v in row.items()),F(0))
        delta=center-t['c'][i];different_coefficients+=int(delta!=0);radius=abs(delta)
        for name in ('Gc','Gb'):
            expected={}
            for j,v in row.items():
                for k,w in s[name][j].items():expected[k]=expected.get(k,F(0))+v*w
            for k in expected.keys()|t[name][i].keys():
                diff=expected.get(k,F(0))-t[name][i].get(k,F(0))
                different_coefficients+=int(diff!=0);radius+=abs(diff)
        errors.append(str(radius))
    maximum=max(map(F,errors),default=F(0))
    return {'status':'CHECKED_EXACT_AFFINE_STEP' if not maximum else 'CHECKED_AFFINE_ERROR_ENCLOSURE_ONLY',
        'rows':len(errors),'nonzero_error_rows':sum(F(v)!=0 for v in errors),
        'different_coefficients':different_coefficients,'maximum_same_factor_error':str(maximum),
        'row_error_bounds':errors,'target_containment_established':not maximum,
        'scope':'Local exact-arithmetic same-factor comparison; nonzero error is NOT repaired or accepted as an exact step.'}


def conv_operator(shape,graph,weight,weight_shape,bias):
    """Independent CHW convolution semantics (no SciPy/production matrix builder)."""
    if (len(shape)!=4 or shape[0]!=1 or len(weight_shape)!=4 or
            graph['padding_mode']!='zeros' or graph['training'] is not False):
        raise ValueError('convolution graph')
    _,channels,height,width=shape;out_channels,in_group,kh,kw=weight_shape
    groups=graph['groups'];stride=graph['stride'];padding=graph['padding'];dilation=graph['dilation']
    if (type(groups)is not int or groups<=0 or out_channels%groups or channels!=in_group*groups or
            len(weight)!=math.prod(weight_shape) or len(bias)!=out_channels or
            any(len(v)!=2 for v in (stride,padding,dilation)) or
            any(type(v)is not int or v<=0 for v in stride+dilation) or
            any(type(v)is not int or v<0 for v in padding)):
        raise ValueError('convolution dimensions/options')
    oh=(height+2*padding[0]-dilation[0]*(kh-1)-1)//stride[0]+1
    ow=(width+2*padding[1]-dilation[1]*(kw-1)-1)//stride[1]+1
    if min(oh,ow)<=0:raise ValueError('empty convolution')
    rows=[];constants=[]
    # Traverse kernel offsets for each output, independently indexed in CHW.
    for o in range(out_channels):
        group=o//(out_channels//groups)
        for y in range(oh):
            for x in range(ow):
                row={}
                for c in range(in_group):
                    for a in range(kh):
                        for b in range(kw):
                            iy=y*stride[0]+a*dilation[0]-padding[0]
                            ix=x*stride[1]+b*dilation[1]-padding[1]
                            if 0<=iy<height and 0<=ix<width:
                                j=((group*in_group+c)*height+iy)*width+ix
                                value=rational(weight[((o*in_group+c)*kh+a)*kw+b])
                                if value:row[j]=value
                rows.append(row);constants.append(rational(bias[o]))
    return rows,constants


def pair_guards(router,joint,pair):
    """Check actual saved rows are redundant, and retained in joint's prefix.

    This says nothing about remaining ReLU constraints, membership big-M
    guards used in other sources, or an unrecorded input/factor history.
    """
    r,j=hz(router),hz(joint);e=len(r['c'])
    if (pair!=sorted(set(pair)) or len(pair)!=2 or any(type(i)is not int or not 0<=i<e for i in pair)
            or r['nb'] or r['b'] or j['frame_id']!=r['frame_id'] or j['nc']<r['nc']):
        raise ValueError('affine router/pair/frame required')
    comparisons=[(a,b) for a in pair for b in range(e) if b not in pair]
    if len(r['ub'])!=len(comparisons) or len(j['ub'])<len(comparisons):raise ValueError('guard row inventory')
    rows=[]
    for i,(a,b) in enumerate(comparisons):
        if (r['Auc'][i]!=j['Auc'][i] or r['Aub'][i] or j['Aub'][i] or r['ub'][i]!=j['ub'][i]):
            raise ValueError('guard prefix/column mapping differs')
        residual=[]
        for k in r['Gc'][a].keys()|r['Gc'][b].keys()|r['Auc'][i].keys():
            exact=r['Gc'][b].get(k,F(0))-r['Gc'][a].get(k,F(0))
            residual.append(abs(exact-r['Auc'][i].get(k,F(0))))
        rhs_error=abs((r['c'][a]-r['c'][b])-r['ub'][i])
        slack=r['ub'][i]-sum(map(abs,r['Auc'][i].values()),F(0))
        rows.append({'row':i,'inside':a,'outside':b,'exact_box_slack':str(slack),
            'coefficient_difference_l1':str(sum(residual,F(0))), 'rhs_difference':str(rhs_error)})
    redundant=all(F(v['exact_box_slack'])>=0 for v in rows)
    return {'status':'CHECKED_SAVED_PAIR_GUARDS_REDUNDANT' if redundant else 'GUARD_REDUNDANCY_NOT_ESTABLISHED',
        'pair':pair,'guard_rows':rows,'all_factor_assignments_retained':redundant,
        'remaining_joint_equalities_unchecked':len(j['b']),
        'remaining_joint_inequalities_unchecked':len(j['ub'])-len(rows),
        'scope':'Actual pair rows only. No membership-guard, ReLU, shared-private-frame or full-network proof.'}
