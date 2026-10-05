"""Complete source-bound joint half-alias row-gauge arithmetic diagnostic.

No rewritten full matrix, native HZ, new lineage or solver is constructed.
One failed row rejects the entire algebraic cohort; full incidence continues.
"""
import numpy as np
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import claimed_stream,discover
from experiments.neural_hz_20260831.c39_half_alias_row_gauge_v1 import row_patch,RowRejected
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,UID_LIMIT
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

HALF=np.dtype([('column','i4'),('definition','i4'),('parent','i4'),('degree','i4'),
    ('sign','i1'),('seen','i4'),('definitions_seen','i4')])
ROWS=np.dtype([('uid','i4'),('inequality','?'),('row','i4'),('half_occurrences','i4'),
    ('decision','i1'),('source_continuous_nnz','i4'),('binary_nnz','i4'),('new_continuous_nnz','i4')])


def select_halves(cohort,*,pool):
    pool.charge('gauge_complete_algebraic_classification',64*len(cohort))
    homogeneous=cohort['offset']==0.;half=homogeneous & (np.abs(cohort['ratio'])==.5)
    chosen=cohort[half];pool.charge('gauge_half_cohort_metadata',96*len(chosen))
    table=np.zeros(len(chosen),HALF)
    for key in ('column','definition','parent','degree'):table[key]=chosen[key]
    table['sign']=np.sign(chosen['ratio']).astype(np.int8)
    if (len(set(map(int,table['column'])))!=len(table)
            or len(set(map(int,table['parent'])))!=len(table)
            or set(map(int,table['column'])) & set(map(int,table['parent']))):
        raise ValueError('whole half cohort must have independent parents')
    return table,dict(all_one_parent_factors_classified=len(cohort),
        complete_one_parent_algebraic_partition=dict(homogeneous_half=int(half.sum()),
            other_homogeneous=int((homogeneous & ~half).sum()),nonhomogeneous=int((~homogeneous).sum())),
        proposed_half_aliases=len(table),selection_uses_only_actual_definition_algebra=True)


def audit_rows(hz,tables,claimed,half,*,old_nc,logical_nc,pool,branch,observe=None):
    if not 0<=old_nc<=logical_nc<=hz.n_cont<=64_000_000:raise ValueError('invalid actual global frame')
    nnz=int(hz.Ac.nnz+hz.Auc.nnz);nrows=hz.n_eq+hz.n_ineq
    if nnz>64_000_000 or nrows>UID_LIMIT:raise MemoryError('unchanged incidence entry/UID cap')
    if claimed.shape!=(logical_nc-old_nc,):raise ValueError('incomplete actual MAIN ownership')
    branch.charge('independent_complete_incidence',7*nnz+12*nrows+4*hz.n_cont)
    pool.charge('gauge_global_diagnostic_lookup',4*hz.n_cont)
    pool.charge('gauge_complete_candidate_gather',4*nnz+16*nrows)
    words=np.zeros(hz.n_cont,np.int64);lookup=np.full(hz.n_cont,-1,np.int32)
    lookup[half['column']]=np.arange(len(half),dtype=np.int32)
    seen=set();journal=[];failure=None;numeric=0;passed=0;overlaps=0;cancellations=0;delta=0;physical=0
    touched_cont=0;touched_bin=0
    for kind,matrix,binary,rhs,uids in ((False,hz.Ac,hz.Ab,hz.b,tables['eq']),
                                      (True,hz.Auc,hz.Aub,hz.ub,tables['le'])):
        if (len(uids)!=matrix.shape[0] or not matrix.has_canonical_format
                or not np.isfinite(matrix.data).all() or np.any(matrix.data==0.)):
            raise ValueError('complete incidence requires canonical finite nonzero actual rows')
        for row,raw in enumerate(uids):
            uid=int(raw)
            if not 0<=uid<UID_LIMIT or uid in seen:raise ValueError('invalid/reused actual row UID')
            seen.add(uid);a,b=map(int,matrix.indptr[row:row+2]);cols=matrix.indices[a:b]
            words[cols]+=RADIX+uid
            ids=lookup[cols];aliases=[];excluded=False
            for pos in np.flatnonzero(ids>=0):
                i=int(ids[int(pos)]);pool.charge('gauge_every_half_occurrence',16)
                if not kind and row==int(half['definition'][i]):
                    half['definitions_seen'][i]+=1;excluded=True;continue
                half['seen'][i]+=1
                aliases.append((int(half['column'][i]),int(half['parent'][i]),int(half['sign'][i])))
            if aliases:
                if excluded:raise ValueError('selected defining row contains another half alias')
                pool.charge('gauge_complete_consumer_row_journal',32)
                ba,bb=map(int,binary.indptr[row:row+2]);decision=0;new_nnz=-1
                touched_cont+=b-a;touched_bin+=bb-ba
                if failure is None:
                    numeric+=1
                    try:
                        patch=row_patch(cols,matrix.data[a:b],binary.indices[ba:bb],binary.data[ba:bb],
                            float(rhs[row]),aliases,pool=pool,enabled=True)
                    except RowRejected as exc:
                        failure=dict(inequality=kind,row=row,uid=uid,**exc.record);decision=2
                    else:
                        decision=1;passed+=1;new_nnz=patch['new_continuous_nnz']
                        overlaps+=patch['parent_overlaps'];cancellations+=patch['exact_cancellations']
                        delta+=patch['consumer_nnz_delta']
                journal.append((uid,kind,row,len(aliases),decision,b-a,bb-ba,new_nnz))
            physical+=1
            if observe and physical%32768==0:
                observe(dict(event='complete_gauge_incidence_progress',physical_rows=physical,
                    charged_work=pool.used,arithmetic_failed=failure is not None,acceptance_published=False))
    branch.charge('compare_every_transferred_MAIN_word',8*len(claimed))
    if not np.array_equal(words[old_nc:logical_nc],claimed):raise ValueError('whole actual incidence differs from claimed ownership')
    pool.charge('gauge_complete_coverage_and_journal',128*len(half)+16*len(journal))
    if np.any(half['definitions_seen']!=1) or np.any(half['seen']!=half['degree']-1):
        raise ValueError('incomplete whole half-alias consumer coverage')
    rows=np.asarray(journal,ROWS)
    if int(rows['half_occurrences'].sum())!=int(half['seen'].sum()):raise ValueError('incomplete touched-row journal')
    success=failure is None
    if success and passed!=len(rows):raise ValueError('unexamined consumer cannot pass a transaction')
    return dict(complete_actual_incidence_equal=True,all_current_physical_rows_scanned=physical,
        all_continuous_predicate_nnz_scanned=nnz,all_half_defining_occurrences=int(half['definitions_seen'].sum()),
        all_half_consumer_occurrences=int(half['seen'].sum()),complete_touched_rows=len(rows),
        complete_touched_continuous_nnz=touched_cont,complete_touched_binary_nnz=touched_bin,
        numeric_rows_checked=numeric,numeric_rows_passed_before_whole_decision=passed,
        numeric_rows_unknown_after_first_failure=len(rows)-numeric,first_failure=failure,
        all_joint_row_arithmetic_proved=success,whole_cohort_rejected=not success,
        successful_subset_published=False,observed_parent_overlaps=overlaps,observed_cancellations=cancellations,
        hypothetical_full_predicate_nnz_delta=delta-2*len(half) if success else None,
        half_table_bytes=half.nbytes,complete_row_journal_bytes=rows.nbytes,
        complete_HZ_transformation_proved=False,old_alias_reconstruction_proved=False,
        source_transfer_or_new_lineage_proved=False,whole_LIVE_decrease_proved=False,
        runtime_payment_proved=False,new_HZ_constructed=False,solver_executed=False,formal_gain=0),rows


def census(state,final,*,pool,branch=None,observe=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('actual bound SplicedState required')
    before=state.validate();final_before=source_digest(final)
    if (sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000
            or any(getattr(final,n) is not getattr(state.hz,n) for n in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('complete actual final source or entry cap mismatch')
    branch=pool if branch is None else branch
    if branch is not pool and getattr(branch,'whole',None) is not pool:raise ValueError('branch must charge same whole pool')
    tables,uid=current_tables(state,pool=branch);claimed=claimed_stream(state,branch=branch)
    cohort,discovery=discover(state,final,tables,claimed,pool=pool)
    half,algebra=select_halves(cohort,pool=pool)
    if observe:observe(dict(event='complete_half_cohort_rediscovered_provisional',**algebra,
        charged_work=pool.used,acceptance_published=False))
    f=state.original_fields
    report,journal=audit_rows(state.hz,tables,claimed,half,old_nc=f['old_n_cont'],
        logical_nc=f['logical_n_cont'],pool=pool,branch=branch,observe=observe)
    after=state.validate()
    if (before['complete_new_HZ_sha256']!=after['complete_new_HZ_sha256']
            or before['complete_semantic_lineage_sha256']!=after['complete_semantic_lineage_sha256']
            or source_digest(final)!=final_before):raise ValueError('read-only gauge diagnostic changed source/final bytes')
    report.update(**discovery,**algebra,uid_partition=uid,all_source_and_final_bytes_unchanged=True,
        complete_post_HZ_sha256=before['complete_new_HZ_sha256'],complete_final_HZ_sha256=final_before,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
    return report,half,journal
