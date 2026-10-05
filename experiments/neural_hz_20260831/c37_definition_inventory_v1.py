"""Complete source-bound anatomy of remaining MAIN definitions, read-only.

Local scalar/graph facts are not all-consumer or composed substitution proofs.
No matrix, map, binary, input or native execution is changed by this module.
"""
from collections import Counter
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables,checked_incidence
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import decode
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

LOW,HIGH=2.**-20,2.**40
TAGS={'alias':0,'splice':1,'row':2,'redirect':3}
DTYPE=np.dtype([('column','i4'),('tag','i1'),('definition','i4'),('degree','i4'),
    ('output_live','?'),('continuous_width','i4'),('binary_width','i4'),('direct','?'),
    ('pivot','f8'),('pivot_normal','?'),('rhs','f8'),('parent','i4'),
    ('ratio','f8'),('offset','f8'),('scalar_exact','i1'),('local_box','i1'),
    ('local_affine','?'),('parent_tag','i1'),('chain_depth','i4'),('chain_anchor','i4'),
    ('legacy_alias_children','i4'),('disjoint_nnz_delta','i8')])


def local_scalar(a,p,d,*,pool):
    pool.charge('inventory_exact_scalar_definition',160)
    result=dict(ratio=0.,offset=0.,scalar_exact=-1,local_box=-1)
    if not (math.isfinite(p) and LOW<=p<=HIGH and math.frexp(p)[0]==.5
            and math.isfinite(a) and (a==0. or LOW<=abs(a)<=HIGH)):
        result['scalar_exact']=0;return result
    if not math.isfinite(d):raise ValueError('bound definition RHS is not finite')
    power=math.frexp(p)[1]-1
    ratio=math.ldexp(-a,-power)
    try:offset=math.ldexp(d,-power)
    except OverflowError:
        result.update(ratio=ratio,scalar_exact=0,local_box=0);return result
    result.update(ratio=ratio,offset=offset,
        scalar_exact=int(math.isfinite(ratio) and math.isfinite(offset)
            and F(ratio)*F(p)==-F(a) and F(offset)*F(p)==F(d)),
        local_box=int(abs(F(a))+abs(F(d))<=F(p)))
    return result


def dependencies(table,roots,*,first,old_n_eq,pool):
    """Two complete topological metadata passes, no coefficient composition."""
    n=len(table);pool.charge('inventory_complete_affine_dependency_passes',48*n)
    if len(roots)!=old_n_eq+n:raise ValueError('incomplete dependency source map')
    for i in range(n):
        col=first+i
        if int(table['column'][i])!=col:raise ValueError('noncanonical complete MAIN table')
        if table['tag'][i]==TAGS['alias']:
            parent=-int(roots[old_n_eq+i])-1
            if not 0<=parent<col:raise ValueError('non-topological legacy witness alias')
            if first<=parent<first+n:table['legacy_alias_children'][parent-first]+=1
    for i in range(n):
        parent=int(table['parent'][i]);col=first+i
        if parent<0:continue
        if not parent<col:raise ValueError('current direct parent is not topological')
        if parent>=first:
            k=parent-first;kind=int(table['tag'][k]);table['parent_tag'][i]=kind
            if kind in (TAGS['alias'],TAGS['splice']):
                raise ValueError('surviving definition references an already removed MAIN factor')
        else:table['parent_tag'][i]=-2  # Protected original prefix.
        if table['local_affine'][i]:
            if parent>=first and table['local_affine'][parent-first]:
                table['chain_depth'][i]=table['chain_depth'][parent-first]+1
                table['chain_anchor'][i]=table['chain_anchor'][parent-first]
            else:table['chain_depth'][i]=1;table['chain_anchor'][i]=parent
    # Constants have no parent but are valid roots of a local affine chain.
    # Their initialization occurs in classify() BEFORE the topological pass.


def classify(state,final,tables,words,*,pool,observe=None):
    f,hz=state.original_fields,state.hz
    first,limit=f['old_n_cont'],f['logical_n_cont'];n=limit-first
    if words.shape!=(n,) or tables['definitions'].shape!=(n,):raise ValueError('incomplete actual MAIN population')
    pool.charge('inventory_complete_table_and_liveness',32*n+4*int(final.Gc.nnz+hz.Gc.nnz))
    table=np.zeros(n,DTYPE);table['column']=np.arange(first,limit,dtype=np.int32)
    for key in ('tag','definition','parent','parent_tag','chain_anchor','scalar_exact','local_box'):table[key]=-1
    live=set(map(int,final.Gc.indices));live.update(map(int,hz.Gc.indices))
    checked=0
    for i in range(n):
        col=first+i;tag=TAGS[decode(f['eq_roots'][f['old_n_eq']+i])[0]]
        table['tag'][i]=tag;table['degree'][i]=int(words[i])//RADIX;table['output_live'][i]=col in live
        if tag<2:
            if table['degree'][i] or table['output_live'][i]:raise ValueError('removed MAIN factor is still physically active')
            continue
        d=int(tables['definitions'][i]);table['definition'][i]=d
        if not 0<=d<hz.n_eq:raise ValueError('surviving MAIN definition is missing')
        pool.charge('inventory_all_surviving_definition_headers',64)
        a,b=map(int,hz.Ac.indptr[d:d+2]);width=b-a
        bw=int(hz.Ab.indptr[d+1]-hz.Ab.indptr[d])
        table['continuous_width'][i]=width;table['binary_width'][i]=bw
        rhs=float(hz.b[d]);table['rhs'][i]=rhs
        if not math.isfinite(rhs):raise ValueError('nonfinite actual RHS')
        direct=width>0 and int(hz.Ac.indices[b-1])==col;table['direct'][i]=direct
        if direct:
            if not table['degree'][i]:raise ValueError('direct definition absent from actual incidence')
            p=float(hz.Ac.data[b-1]);table['pivot'][i]=p
            table['pivot_normal'][i]=math.isfinite(p) and LOW<=p<=HIGH and math.frexp(p)[0]==.5
            if bw==0:
                m=width-1;k=int(table['degree'][i])
                table['disjoint_nnz_delta'][i]=-(m+1)+(k-1)*(m-1)
                if width<=2:
                    parent=int(hz.Ac.indices[a]) if width==2 else -1
                    table['parent'][i]=parent
                    coefficient=float(hz.Ac.data[a]) if width==2 else 0.
                    local=local_scalar(coefficient,p,rhs,pool=pool)
                    for key,value in local.items():table[key][i]=value
                    eligible=local['scalar_exact']==1 and local['local_box']==1 and not table['output_live'][i]
                    table['local_affine'][i]=eligible
                    if eligible and parent<0:table['chain_depth'][i]=1
        checked+=1
        if observe and checked%32768==0:
            observe(dict(event='complete_definition_header_progress',MAIN_through=i+1,
                surviving_definitions=checked,charged_work=pool.used,acceptance_published=False))
    dependencies(table,f['eq_roots'],first=first,old_n_eq=f['old_n_eq'],pool=pool)
    pool.charge('inventory_complete_summary_and_coverage',128*n)
    if np.any(table['tag']<0):raise ValueError('incomplete inventory cannot publish a subset')
    active=table['tag']>=2;direct=active & table['direct'];binary_free=direct & (table['binary_width']==0)
    non_two=active & (table['degree']!=2);local=table['local_affine'];zero=binary_free & (table['degree']==1)
    def hist(field,mask):
        counter=Counter(map(int,table[field][mask]));return {str(k):int(v) for k,v in sorted(counter.items())}
    groups=Counter()
    for row in table[active]:
        key=(int(row['degree']),int(row['continuous_width']),int(row['binary_width']),
             bool(row['direct']),bool(row['rhs']==0.),bool(row['pivot_normal']))
        groups[key]+=1
    shapes=[dict(degree=k[0],continuous_width=k[1],binary_width=k[2],direct=k[3],homogeneous=k[4],
        positive_normal_dyadic_pivot=k[5],count=v) for k,v in sorted(groups.items())]
    report=dict(all_MAIN_classified=n,already_legacy_aliases=int(np.count_nonzero(table['tag']==0)),
        already_unit_splices=int(np.count_nonzero(table['tag']==1)),surviving_definitions=int(active.sum()),
        redirected_definitions=int(np.count_nonzero(table['tag']==3)),surviving_output_live=int(np.count_nonzero(active & table['output_live'])),
        full_surviving_degree_histogram=hist('degree',active),non_degree_two_count=int(non_two.sum()),
        non_degree_two_degree_histogram=hist('degree',non_two),non_direct_definitions=int(np.count_nonzero(active & ~table['direct'])),
        direct_binary_definitions=int(np.count_nonzero(direct & (table['binary_width']!=0))),
        direct_binary_free_definitions=int(binary_free.sum()),actual_definition_shape_groups=shapes,
        direct_no_consumer_rows=int(zero.sum()),no_consumer_parent_width_histogram=hist('continuous_width',zero),
        no_consumer_local_box_proved=int(np.count_nonzero(zero & (table['local_box']==1))),
        no_consumer_box_unknown=int(np.count_nonzero(zero & (table['local_box']==-1))),
        local_affine_boxed_scalars=int(local.sum()),local_affine_degree_histogram=hist('degree',local),
        local_affine_homogeneous=int(np.count_nonzero(local & (table['rhs']==0.))),
        local_affine_nonhomogeneous=int(np.count_nonzero(local & (table['rhs']!=0.))),
        local_affine_constant_definitions=int(np.count_nonzero(local & (table['continuous_width']==1))),
        local_affine_one_parent_definitions=int(np.count_nonzero(local & (table['continuous_width']==2))),
        local_affine_chain_depth_histogram=hist('chain_depth',local),
        local_affine_parent_tag_histogram=hist('parent_tag',local),
        local_affine_with_legacy_alias_children=int(np.count_nonzero(local & (table['legacy_alias_children']>0))),
        legacy_aliases_depending_on_local_affine=int(table['legacy_alias_children'][local].sum()),
        local_affine_with_non_degree_two=int(np.count_nonzero(local & non_two)),
        disjoint_delta_negative_headers=int(np.count_nonzero(binary_free & (table['disjoint_nnz_delta']<0))),
        disjoint_delta_zero_headers=int(np.count_nonzero(binary_free & (table['disjoint_nnz_delta']==0))),
        disjoint_delta_positive_headers=int(np.count_nonzero(binary_free & (table['disjoint_nnz_delta']>0))),
        table_numeric_bytes=table.nbytes,table_numeric_records=len(table),
        actual_incident_substitution_or_overlap_proved=False,composed_chain_arithmetic_proved=False,
        actual_nnz_reduction_proved=False,wide_box_or_root_domain_totality_proved=False,
        runtime_payment_proved=False,new_lineage_proved=False,new_HZ_constructed=False,solver_executed=False,formal_gain=0)
    return report,table


def census(state,final,*,pool,branch=None,observe=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('actual bound SplicedState required')
    before=state.validate();final_before=source_digest(final)
    if (sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000
            or any(getattr(final,n) is not getattr(state.hz,n) for n in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('complete actual final source or entry cap mismatch')
    branch=pool if branch is None else branch
    if branch is not pool and getattr(branch,'whole',None) is not pool:raise ValueError('branch must charge the same whole pool')
    tables,uid=current_tables(state,pool=branch)
    if observe:observe(dict(event='complete_current_UID_partition',**uid,charged_work=pool.used))
    actual=checked_incidence(state,tables,pool=branch)
    if observe:observe(dict(event='complete_current_incidence_proved',all_MAIN_words=len(actual),charged_work=pool.used))
    report,table=classify(state,final,tables,actual,pool=pool,observe=observe)
    after=state.validate()
    if (before['complete_new_HZ_sha256']!=after['complete_new_HZ_sha256']
            or before['complete_semantic_lineage_sha256']!=after['complete_semantic_lineage_sha256']
            or source_digest(final)!=final_before):raise ValueError('read-only inventory changed source/final bytes')
    report.update(uid_partition=uid,complete_actual_incidence_equal=True,all_source_and_final_bytes_unchanged=True,
        complete_post_HZ_sha256=before['complete_new_HZ_sha256'],complete_final_HZ_sha256=final_before,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
    return report,table
