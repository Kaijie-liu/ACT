"""Complete current-state, exact joint-sum diagnostic; no native transformation."""
import numpy as np
from experiments.neural_hz_20260831.c36_exact_joint_sum_v1 import prove,GUARDS,REASONS,CODE
from experiments.neural_hz_20260831.c35_nonunit_exact_census_v1 import independent_mask
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables,checked_incidence
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import decode
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,unique_other
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

DTYPE=np.dtype([('column','i4'),('definition','i4'),('consumer','i4'),('inequality','?'),
    ('reason','i2'),('ratio','f8'),('parent_terms','i4'),('consumer_other_terms','i4'),
    ('joint_checked','i4'),('failed_term','i4'),('independent','?'),('consumer_head','?'),
    ('overlaps_seen','i4'),('cancellations_seen','i4'),('overlap_present','i1'),
    ('nnz_delta','i4'),('old_products_exact','i1'),*((g,'i1') for g in GUARDS)])


def classify(state,final,tables,words,*,pool,observe=None):
    f,hz=state.original_fields,state.hz
    first,limit=f['old_n_cont'],f['logical_n_cont'];n=limit-first
    if words.shape!=(n,) or tables['definitions'].shape!=(n,):raise ValueError('incomplete current population')
    pool.charge('joint_complete_table_and_liveness',32*n+4*int(final.Gc.nnz+hz.Gc.nnz))
    table=np.zeros(n,DTYPE);table['column']=np.arange(first,limit,dtype=np.int32)
    for key in ('definition','consumer','reason','failed_term','overlap_present','old_products_exact',*GUARDS):table[key]=-1
    live=set(map(int,final.Gc.indices));live.update(map(int,hz.Gc.indices))
    structural=0
    def reject(i,name):table['reason'][i]=CODE[name]
    for i in range(n):
        col=first+i;tag=decode(f['eq_roots'][f['old_n_eq']+i])[0]
        if tag=='alias':reject(i,'already_alias');continue
        if tag=='splice':reject(i,'already_unit_splice');continue
        if col in live:reject(i,'output_live');continue
        if int(words[i])//RADIX!=2:reject(i,'not_degree_two');continue
        d=int(tables['definitions'][i]);table['definition'][i]=d
        if d<0:reject(i,'missing_definition');continue
        pool.charge('joint_actual_pair_geometry',64)
        da,db=map(int,hz.Ac.indptr[d:d+2])
        if da==db or int(hz.Ac.indices[db-1])!=col:reject(i,'not_direct_definition');continue
        if hz.Ab.indptr[d]!=hz.Ab.indptr[d+1]:reject(i,'binary_definition');continue
        other=unique_other(int(words[i]),int(tables['eq'][d]))
        if other not in tables['lookup']:raise ValueError('actual unique consumer does not resolve')
        kind,c=tables['lookup'][other]
        if not kind and c==d:raise ValueError('consumer is the definition')
        table['consumer'][i]=c;table['inequality'][i]=kind
        matrix,rhs=(hz.Auc,hz.ub) if kind else (hz.Ac,hz.b)
        ca,cb=map(int,matrix.indptr[c:c+2]);tc,tv=matrix.indices[ca:cb],matrix.data[ca:cb]
        pool.charge('joint_consumer_pivot_search',8*max(1,len(tc).bit_length())+8)
        at=int(np.searchsorted(tc,col))
        if at==len(tc) or int(tc[at])!=col:raise ValueError('proved consumer lacks the pivot')
        structural+=1;table['consumer_head'][i]=at==0
        table['parent_terms'][i]=db-da-1;table['consumer_other_terms'][i]=cb-ca-1
        answer=prove(hz.Ac.indices[da:db-1],hz.Ac.data[da:db-1],tc,tv,at,
            float(hz.Ac.data[db-1]),float(tv[at]),float(hz.b[d]),float(rhs[c]),pool=pool)
        for key in ('reason','ratio','failed_term','overlap_present','nnz_delta','old_products_exact',*GUARDS):
            table[key][i]=answer[key]
        for dst,src in (('joint_checked','checked'),('overlaps_seen','overlaps'),('cancellations_seen','cancellations')):
            table[dst][i]=answer[src]
        if observe and structural%8192==0:
            observe(dict(event='complete_joint_population_progress',MAIN_through=i+1,structural_pairs=structural,
                charged_work=pool.used,acceptance_published=False))
    pool.charge('joint_complete_publication_summaries',128*n)
    if np.any(table['reason']<0):raise ValueError('incomplete joint census cannot publish a subset')
    conflicts=independent_mask(table,pool=pool)
    individual=table['reason']==CODE['individual_exact'];accepted=table['independent']
    if any(np.any(table[g][individual]!=1) for g in GUARDS) or np.any(table['nnz_delta'][individual]>=0):
        raise ValueError('unknown proof or nondecreasing nnz cannot admit a row')
    examined=table['ratio']!=0.;nonhead=examined & ~table['consumer_head']
    report=dict(all_MAIN_classified=n,binary_free_direct_single_consumer_pairs=structural,
        first_rejection_counts={r:int(np.count_nonzero(table['reason']==i)) for i,r in enumerate(REASONS)},
        guards={g:{'passed':int(np.count_nonzero(table[g]==1)),
                   'failed':int(np.count_nonzero(table[g]==0)),
                   'not_evaluated':int(np.count_nonzero(table[g]==-1))} for g in GUARDS},
        numeric_head_pairs=int(np.count_nonzero(examined & table['consumer_head'])),
        numeric_nonhead_pairs=int(nonhead.sum()),
        nonhead_overlap_present_proved=int(np.count_nonzero(nonhead & (table['overlap_present']==1))),
        nonhead_overlap_absence_proved=int(np.count_nonzero(nonhead & (table['overlap_present']==0))),
        nonhead_overlap_unknown=int(np.count_nonzero(nonhead & (table['overlap_present']==-1))),
        all_checked_joint_coefficients=int(table['joint_checked'].sum()),structural_parent_terms=int(table['parent_terms'].sum()),
        observed_overlaps=int(table['overlaps_seen'].sum()),observed_exact_cancellations=int(table['cancellations_seen'].sum()),
        individual_exact_pairs=int(individual.sum()),dependency_rejected_pairs=conflicts,
        simultaneous_independent_pairs=int(accepted.sum()),
        admitted_with_inexact_separate_products=int(np.count_nonzero(individual & (table['old_products_exact']==0))),
        independent_with_inexact_separate_products=int(np.count_nonzero(accepted & (table['old_products_exact']==0))),
        independent_parent_terms=int(table['parent_terms'][accepted].sum()),
        independent_overlaps=int(table['overlaps_seen'][accepted].sum()),
        independent_cancellations=int(table['cancellations_seen'][accepted].sum()),
        potential_predicate_nnz_delta=int(table['nnz_delta'][accepted].sum()),
        table_numeric_bytes=table.nbytes,table_numeric_records=len(table),
        runtime_payment_proved=False,new_joint_lineage_proved=False,new_HZ_constructed=False,solver_executed=False,formal_gain=0)
    return report,table


def census(state,final,*,pool,branch=None,observe=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('actual bound SplicedState required')
    before=state.validate();final_before=source_digest(final)
    if (sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000
            or any(getattr(final,n) is not getattr(state.hz,n) for n in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('complete final predicate source or entry cap mismatch')
    branch=pool if branch is None else branch
    if branch is not pool and getattr(branch,'whole',None) is not pool:
        raise ValueError('branch must charge the same whole pool')
    tables,uid=current_tables(state,pool=branch)
    if observe:observe(dict(event='complete_current_UID_partition',**uid,charged_work=pool.used))
    actual=checked_incidence(state,tables,pool=branch)
    if observe:observe(dict(event='complete_current_incidence_proved',all_MAIN_words=len(actual),charged_work=pool.used))
    report,table=classify(state,final,tables,actual,pool=pool,observe=observe)
    after=state.validate()
    if (before['complete_new_HZ_sha256']!=after['complete_new_HZ_sha256']
            or before['complete_semantic_lineage_sha256']!=after['complete_semantic_lineage_sha256']
            or source_digest(final)!=final_before):raise ValueError('joint census changed source/final bytes')
    report.update(uid_partition=uid,complete_actual_incidence_equal=True,all_source_and_final_bytes_unchanged=True,
        complete_post_HZ_sha256=before['complete_new_HZ_sha256'],complete_final_HZ_sha256=final_before,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
    return report,table
