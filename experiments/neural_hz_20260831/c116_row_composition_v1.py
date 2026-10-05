# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent full native row identities and a complete one-step inverse."""
from fractions import Fraction as F
import math
import numpy as np


def exact(value):
    value=F(value)
    if max(abs(value.numerator).bit_length(),value.denominator.bit_length())>512:
        raise ValueError('complete exact rational exceeds512bits')
    return value


def checked(row,limit,*,pool,rename=None,auxiliary=False):
    if set(row)!= {'coefficients','rhs','slot','gauge'}:
        raise ValueError('complete continuous native row schema required')
    pairs=row['coefficients'];pool.charge('c116_every_native_header_and_coefficient',64*len(pairs)+64)
    cols=[c for c,v in pairs]
    if (type(row['slot']) is not int or type(row['gauge']) is not int
            or not -4096<=row['gauge']<=4096 or cols!=sorted(set(cols))
            or any(type(c) is not int or not 0<=c<limit for c in cols)
            or row['slot'] not in cols):
        raise ValueError('complete bounded canonical native row/pivot required')
    if (not math.isfinite(row['rhs']) or any(not math.isfinite(v)
            or not 2.**-20<=abs(v)<=2.**40 for c,v in pairs)):
        raise ValueError('actual native coefficient/RHS window invalid')
    gauge=F(2)**row['gauge']
    values={c:exact(F(v)/gauge) for c,v in pairs};rhs=exact(F(row['rhs'])/gauge)
    pivot=values[row['slot']]
    if pivot<=0:raise ValueError('positive actual native pivot required')
    if auxiliary:
        parents={c:v for c,v in values.items() if c!=row['slot']}
        if any(c>=row['slot'] for c in parents):raise ValueError('complete triangular definition required')
        if abs(rhs)+sum(map(abs,parents.values()),F(0))>pivot:
            raise ValueError('actual auxiliary box not proved redundant')
    if rename is not None:
        values={rename(c):v for c,v in values.items()}
    return values,rhs


def roots_replaced(values,mapping,*,pool):
    pool.charge('c116_complete_root_substitution',64*len(values)+64)
    result={}
    for col,value in values.items():
        new,scale=mapping.get(col,(col,F(1)))
        result[new]=exact(result.get(new,F(0))+value*scale)
    return {c:v for c,v in result.items() if v}


def equal_equation(a,b,*,pool):
    pool.charge('c116_both_sides_complete_equation_comparison',32*(len(a[0])+len(b[0]))+64)
    if a!=b:raise ValueError('complete native equation or RHS identity differs')


def prove(old_aux,old_out,new_aux,new_out,base,kept_old_slots,aliases,*,pool,enabled=False):
    if not enabled:return None
    if type(base) is not int or base<0:raise ValueError('original continuous base required')
    old_n=base+len(old_aux);new_n=base+len(new_aux);kept=list(map(int,kept_old_slots))
    old_slots=[r['slot'] for r in old_aux];new_slots=[r['slot'] for r in new_aux]
    old_outputs=[r['slot'] for r in old_out];new_outputs=[r['slot'] for r in new_out]
    if (old_slots!=list(range(base,old_n)) or new_slots!=list(range(base,new_n))
            or kept!=sorted(set(kept)) or len(kept)!=len(new_aux) or set(kept)-set(old_slots)
            or len(set(old_outputs))!=len(old_outputs) or set(old_outputs)!=set(new_outputs)
            or len(set(new_outputs))!=len(new_outputs) or any(not 0<=s<base for s in old_outputs)):
        raise ValueError('complete original/candidate coordinate and output inventory required')
    old_to_new={s:base+i for i,s in enumerate(kept)}
    new_to_old={v:k for k,v in old_to_new.items()}
    rename=lambda c:new_to_old[c] if c>=base else c
    removed=set(old_slots)-set(kept)
    rootmap={int(a['old']):(int(a['representative']),exact(a['scale'])) for a in aliases}
    if len(rootmap)!=len(aliases) or set(rootmap)-removed:
        raise ValueError('duplicate or retained supplied root alias')
    for old,(rep,scale) in rootmap.items():
        if rep not in old_to_new or not 0<abs(scale)<=1:
            raise ValueError('kept representative and normalized exact root scale required')
    sinks=removed-set(rootmap);saved={};uses={s:[] for s in sinks}
    originals={r['slot']:r for r in [*old_aux,*old_out]}
    wanted=removed|{rep for rep,scale in rootmap.values()}
    # All old rows and all deleted-factor incidences are inspected independently.
    for row in [*old_aux,*old_out]:
        slot=row['slot'];values,rhs=checked(row,old_n,pool=pool,auxiliary=slot>=base)
        pool.charge('c116_complete_deleted_sink_incidence',32*len(values))
        for col in values:
            if col in sinks and col!=slot:
                if slot>=base:raise ValueError('deleted factor is not an output-only sink')
                uses[col].append(slot)
        if slot in wanted:saved[slot]=(values,rhs)
    for old,(rep,scale) in rootmap.items():
        av,ar=saved[old];bv,br=saved[rep];ap=av[old];bp=bv[rep]
        if (len(av)<2 or len(bv)<2 or any(c>=base and c!=old for c in av)
                or any(c>=base and c!=rep for c in bv)):
            raise ValueError('supplied alias is not a complete nonzero input-root definition')
        pool.charge('c116_every_root_alias_identity',64*(len(av)+len(bv)))
        a=({c:exact(v/ap) for c,v in av.items() if c!=old},exact(ar/ap))
        b=({c:exact(scale*v/bp) for c,v in bv.items() if c!=rep},exact(scale*br/bp))
        equal_equation(a,b,pool=pool)
    recipes={}
    for old in sorted(sinks):
        values,rhs=saved[old]
        if not uses[old] or not any(c>=base and c!=old for c in values):
            raise ValueError('complete live nonroot sink definition required')
        pivot=values[old];parents=roots_replaced({c:v for c,v in values.items() if c!=old},rootmap,pool=pool)
        if any(c>=base and c not in old_to_new for c in parents):
            raise ValueError('removed sink has unresolved or deleted parent')
        recipes[old]=({c:exact(-v/pivot) for c,v in parents.items()},exact(rhs/pivot))
    # Every retained new equation is checked, even if its packet bytes appear
    # unchanged. The unit factor-value map and actual positive gauges are explicit.
    for row,old in zip(new_aux,kept,strict=True):
        candidate=checked(row,new_n,pool=pool,rename=rename,auxiliary=True)
        values,rhs=checked(originals[old],old_n,pool=pool,auxiliary=True)
        expected=(roots_replaced(values,rootmap,pool=pool),rhs)
        equal_equation(expected,candidate,pool=pool)
    new_by_output={r['slot']:r for r in new_out}
    for row in old_out:
        values,rhs=checked(row,old_n,pool=pool)
        values=roots_replaced(values,rootmap,pool=pool)
        count=sum(len(recipes[c][0])+1 if c in recipes else 1 for c in values)
        pool.charge('c116_every_output_full_sink_composition',64*count+64)
        expected={}
        for col,value in values.items():
            if col in recipes:
                parents,offset=recipes[col];rhs=exact(rhs-value*offset)
                for parent,coefficient in parents.items():
                    expected[parent]=exact(expected.get(parent,F(0))+value*coefficient)
            else:expected[col]=exact(expected.get(col,F(0))+value)
        expected={c:v for c,v in expected.items() if v}
        candidate=checked(new_by_output[row['slot']],new_n,pool=pool,rename=rename)
        equal_equation((expected,rhs),candidate,pool=pool)
    # Store recipes solely in new coordinates. Even a root representative that
    # occurs later in the old numbering can be reconstructed in one step.
    inverse=[]
    for old in sorted(removed):
        if old in rootmap:
            rep,scale=rootmap[old];terms={rep:scale};offset=F(0)
        else:terms,offset=recipes[old]
        pool.charge('c116_complete_inverse_serialization',16*(3+3*len(terms)))
        inverse.append(dict(old_slot=old,constant=(offset.numerator,offset.denominator),
            new_coordinate_terms=[(old_to_new[c] if c>=base else c,v.numerator,v.denominator)
                                  for c,v in sorted(terms.items())]))
    before_nnz=sum(len(r['coefficients']) for r in [*old_aux,*old_out])
    after_nnz=sum(len(r['coefficients']) for r in [*new_aux,*new_out])
    evidence=dict(base=base,old_n_cont=old_n,new_n_cont=new_n,
        kept_old_slots=np.array(kept,np.int64),new_slots=np.arange(base,new_n,dtype=np.int64),
        inverse_recipes=inverse,complete_deleted_sink_output_uses={s:uses[s] for s in sorted(uses)})
    proof=dict(all_original_native_rows_checked=len(old_aux)+len(old_out),
        all_candidate_native_rows_checked=len(new_aux)+len(new_out),
        all_retained_defining_equations_equal=len(new_aux),all_original_output_equations_equal=len(old_out),
        all_root_alias_identities_proved=len(rootmap),all_deleted_sink_inverses_proved=len(sinks),
        all_original_and_new_auxiliary_boxes_proved=len(old_aux)+len(new_aux),
        complete_deleted_factor_incidence_proved=True,all_removed_factors_reconstructible=len(inverse),
        original_coordinate_map_is_identity=True,retained_factor_values_are_identical=True,
        every_actual_native_gauge_and_RHS_checked=True,original_nnz=before_nnz,new_nnz=after_nnz,
        strict_total_nnz_decrease=after_nnz<before_nnz,
        complete_original_HZ_and_binary_numeric_roots_required_separately=True,
        original_source_authentication_required_separately=True,source_runtime_LIVE_admitted=False,formal_gain=0)
    return proof,evidence


def reconstruct(new_point,evidence,*,pool):
    if len(new_point)!=evidence['new_n_cont']:raise ValueError('complete new point required')
    pool.charge('c116_complete_inverse_point',16*len(new_point))
    point=list(map(exact,new_point))
    if any(abs(v)>1 for v in point):raise ValueError('point outside normalized factor boxes')
    old=point[:evidence['base']]+[None]*(evidence['old_n_cont']-evidence['base'])
    for before,after in zip(evidence['kept_old_slots'],evidence['new_slots'],strict=True):old[int(before)]=point[int(after)]
    for recipe in evidence['inverse_recipes']:
        pool.charge('c116_each_exact_inverse_recipe',64*(len(recipe['new_coordinate_terms'])+1))
        value=F(*recipe['constant'])
        for col,numerator,denominator in recipe['new_coordinate_terms']:
            value=exact(value+F(numerator,denominator)*point[col])
        if abs(value)>1:raise ValueError('reconstructed factor outside normalized old box')
        old[recipe['old_slot']]=value
    if any(v is None for v in old):raise ValueError('incomplete original-coordinate inverse')
    return old
