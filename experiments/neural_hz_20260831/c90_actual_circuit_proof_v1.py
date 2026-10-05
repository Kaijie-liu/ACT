"""Source-independent exact elimination of literal native circuit equations."""
from fractions import Fraction as F
import numpy as np


def rebase(packet,old_n_cont,offset,*,pool):
    pool.charge('c90_disjoint_global_auxiliary_names',8*(packet['columns'].size+packet['pivots'].size)+128)
    result=dict(packet)
    for name in ('columns','pivots'):
        value=packet[name].astype(np.int64)
        value[value>=old_n_cont]+=offset
        if np.any(value<0) or np.any(value>=2**31):raise ValueError('global coordinate domain exceeded')
        result[name]=value.astype(np.int32)
    return result


def _bounded(value):
    if value.numerator.bit_length()>512 or value.denominator.bit_length()>512:
        raise MemoryError('unchanged512-bit exact polynomial domain exceeded')
    return value


def prove(packet,original_rows,original_gauges,*,old_n_cont,first_aux,new_factors,pool,enabled=False):
    """Prove every output and a universally redundant unique box extension."""
    if not enabled:return None
    rows=len(packet['rhs'])
    pool.charge('c90_complete_native_headers',128+8*(rows+len(packet['native'])))
    if (first_aux<old_n_cont or rows!=new_factors+len(original_rows)
        or len(original_gauges)!=len(original_rows)
        or len(packet['indptr'])!=rows+1 or packet['indptr'][0]!=0
        or packet['indptr'][-1]!=len(packet['native'])
        or len(packet['columns'])!=len(packet['native'])
        or np.any(np.diff(packet['indptr'])<=0)
        or not np.isfinite(packet['native']).all()
        or np.any((np.abs(packet['native'])<2.**-20)|(np.abs(packet['native'])>2.**40))
        or np.any(packet['ab_indptr']) or not np.isfinite(packet['rhs']).all()):
        raise ValueError('complete actual native circuit domain required')
    expressions={};products=0;output_nnz=0;seen_outputs=set();maximum_support=0
    for r in range(rows):
        a,b=map(int,packet['indptr'][r:r+2]);cols=packet['columns'][a:b];vals=packet['native'][a:b]
        pool.charge('c90_actual_native_row_and_box',64+16*(b-a))
        if np.any(np.diff(cols)<=0):raise ValueError('unique canonical native row required')
        pivot=int(packet['pivots'][r]);coefs={int(c):F(float(v)) for c,v in zip(cols,vals,strict=True)}
        if pivot not in coefs or coefs[pivot]<=0:raise ValueError('actual positive pivot missing')
        if r<new_factors:
            if pivot!=first_aux+r:raise ValueError('complete global auxiliary prefix differs')
            value=coefs.pop(pivot)
            if (any(c>=old_n_cont and c not in expressions for c in coefs)
                or any(c>=pivot for c in coefs)
                or abs(F(float(packet['rhs'][r])))+sum(map(abs,coefs.values()),F(0))>value):
                raise ValueError('topology or universally redundant box failed')
            sign=-1/value;poly={-1:F(float(packet['rhs'][r]))/value}
        else:
            if pivot>=old_n_cont or pivot in seen_outputs:raise ValueError('original unique output pivot differs')
            seen_outputs.add(pivot);sign=F(1);poly={-1:-F(float(packet['rhs'][r]))}
        count=sum(1 if col<old_n_cont else len(expressions.get(col,{})) for col in coefs)
        if any(col>=old_n_cont and col not in expressions for col in coefs):raise ValueError('unbound new factor')
        pool.charge('c90_independent_exact_polynomial_products',64*count);products+=count
        for col,value in coefs.items():
            scalar=_bounded(value*sign)
            terms=((col,F(1)),) if col<old_n_cont else expressions[col].items()
            for source,coefficient in terms:
                poly[source]=_bounded(poly.get(source,F(0))+scalar*coefficient)
        poly={c:v for c,v in poly.items() if v}
        maximum_support=max(maximum_support,len(poly))
        if r<new_factors:
            expressions[pivot]=poly
        else:
            original=original_rows[r-new_factors];q=int(original_gauges[r-new_factors])
            pool.charge('c90_complete_original_equation_comparison',64+32*len(original['coefficients']))
            expected={int(c):F(float(v))/F(2)**q for c,v in original['coefficients']}
            if original['rhs']:expected[-1]=-F(float(original['rhs']))/F(2)**q
            actual={c:_bounded(v/F(2)**int(packet['gauges'][r])) for c,v in poly.items()}
            if actual!=expected:raise ValueError('complete actual original source equation differs')
            if int(original['pivot'])!=pivot:raise ValueError('original source pivot binding differs')
            output_nnz+=len(expected)
    return dict(all_original_output_equations_proved=len(original_rows),
        all_auxiliary_equations_and_redundant_boxes_proved=new_factors,
        original_projected_equation_nnz=output_nnz,exact_polynomial_products=products,
        maximum_expanded_polynomial_support=maximum_support,
        native_only_inverse_sufficient=True,universal_unique_box_extension=True,
        original_source_equivalence=True,concrete_network_witness=False,formal_gain=0)
