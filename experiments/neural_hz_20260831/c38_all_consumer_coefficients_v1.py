"""One complete actual incidence scan with early failing all-consumer guards.

Coefficient survivors are NOT projected HZs: RHS and lineage remain unproved.
Provisional ownership never authorizes publication before whole actual equality.
"""
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables
from experiments.neural_hz_20260831.c37_definition_inventory_v1 import local_scalar
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import decode
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,UID_LIMIT
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

LOW,HIGH=2.**-20,2.**40
REASONS={-1:'undecided',0:'all_consumer_coefficients_exact',1:'operand_window',2:'joint_inexact',3:'joint_window'}
DTYPE=np.dtype([('column','i4'),('definition','i4'),('parent','i4'),('degree','i4'),
    ('ratio','f8'),('offset','f8'),('seen','i4'),('definitions_seen','i4'),('checked','i4'),
    ('reason','i1'),('failed_inequality','?'),('failed_row','i4'),('failed_uid','i4'),
    ('failed_q','f8'),('failed_w','f8'),('failed_float_candidate','f8'),
    ('observed_overlaps','i4'),('exact_cancellations','i4')])


def claimed_stream(state,*,branch):
    f=state.original_fields;n=f['logical_n_cont']-f['old_n_cont']
    overlay=Overlay(f['owners'],state.events,state.old_uid_ceiling);overlay.validate()
    branch.charge('provisional_owner_word_materialization',2*n)
    words=np.empty(n,np.int64);count=0
    for i,value in enumerate(state.lineage.iter_words(overlay,pool=branch)):
        if i>=n:raise ValueError('oversized claimed MAIN stream')
        words[i]=value;count+=1
    if count!=n:raise ValueError('incomplete claimed MAIN stream')
    return words


def discover(state,final,tables,claimed,*,pool):
    f,hz=state.original_fields,state.hz;first,limit=f['old_n_cont'],f['logical_n_cont'];n=limit-first
    if (claimed.shape!=(n,) or tables['definitions'].shape!=(n,)
            or len(f['eq_roots'])!=f['old_n_eq']+n):raise ValueError('incomplete discovery population')
    pool.charge('allconsumer_complete_MAIN_discovery',32*n+4*int(hz.Gc.nnz+final.Gc.nnz))
    live=set(map(int,hz.Gc.indices));live.update(map(int,final.Gc.indices))
    records=[];counts=dict(already_removed=0,output_live=0,degree_below_three=0,
        shape_rejected=0,local_scalar_or_box_rejected=0,selected=0)
    for i in range(n):
        col=first+i;tag=decode(f['eq_roots'][f['old_n_eq']+i])[0]
        if tag in ('alias','splice'):
            if claimed[i] or col in live:raise ValueError('removed factor is still active')
            counts['already_removed']+=1;continue
        if col in live:counts['output_live']+=1;continue
        degree=int(claimed[i])//RADIX
        if degree<3:counts['degree_below_three']+=1;continue
        pool.charge('allconsumer_actual_definition_header',64)
        d=int(tables['definitions'][i])
        if not 0<=d<hz.n_eq:raise ValueError('missing surviving current definition')
        a,b=map(int,hz.Ac.indptr[d:d+2])
        if b-a!=2 or int(hz.Ac.indices[b-1])!=col or hz.Ab.indptr[d]!=hz.Ab.indptr[d+1]:
            counts['shape_rejected']+=1;continue
        parent=int(hz.Ac.indices[a])
        if not 0<=parent<col:raise ValueError('non-topological actual one-parent definition')
        facts=local_scalar(float(hz.Ac.data[a]),float(hz.Ac.data[b-1]),float(hz.b[d]),pool=pool)
        if facts['scalar_exact']!=1 or facts['local_box']!=1:
            counts['local_scalar_or_box_rejected']+=1;continue
        pool.charge('allconsumer_candidate_metadata_and_independence',96)
        records.append((col,d,parent,degree,facts['ratio'],facts['offset']))
        counts['selected']+=1
    columns={r[0] for r in records};parents={r[2] for r in records}
    if len(columns)!=len(records) or len(parents)!=len(records) or columns & parents:
        raise ValueError('entire proposed cohort has shared parents or internal dependencies')
    table=np.zeros(len(records),DTYPE)
    for i,record in enumerate(records):
        for name,value in zip(('column','definition','parent','degree','ratio','offset'),record):table[name][i]=value
    for name in ('reason','failed_row','failed_uid'):table[name]=-1
    if sum(counts.values())!=n:raise ValueError('incomplete MAIN discovery accounting')
    return table,dict(all_MAIN_classified=n,discovery_counts=counts,
        whole_cohort_parents_distinct=True,no_parent_inside_proposed_cohort=True,
        claimed_membership_provisional_until_actual_complete_audit=True)


def coefficient(q,w,ratio,*,pool):
    pool.charge('allconsumer_exact_joint_coefficient',128)
    if not (math.isfinite(q) and LOW<=abs(q)<=HIGH and math.isfinite(w)
            and (w==0. or LOW<=abs(w)<=HIGH) and math.isfinite(ratio)):
        return 1,0.,False
    value=F(w)+F(q)*F(ratio)
    try:combined=float(value)
    except OverflowError:return 2,0.,False
    if not math.isfinite(combined) or F(combined)!=value:return 2,combined,False
    if value and not LOW<=abs(combined)<=HIGH:return 3,combined,False
    return 0,combined,value==0


def audit_and_screen(hz,tables,claimed,candidates,*,old_nc,logical_nc,pool,branch,observe=None):
    """Complete C23 incidence oracle, augmented with a separately paid screen."""
    if not 0<=old_nc<=logical_nc<=hz.n_cont<=64_000_000:raise ValueError('invalid current frame')
    nnz=int(hz.Ac.nnz+hz.Auc.nnz);rows=hz.n_eq+hz.n_ineq
    if nnz>64_000_000 or rows>UID_LIMIT or len(candidates)>=UID_LIMIT:raise MemoryError('unchanged entry/UID cap')
    if claimed.shape!=(logical_nc-old_nc,):raise ValueError('incomplete provisional source ownership')
    # Original C23 actual-oracle price is retained in full, before all work.
    branch.charge('independent_complete_incidence',7*nnz+12*rows+4*hz.n_cont)
    pool.charge('allconsumer_global_diagnostic_lookup',4*hz.n_cont)
    pool.charge('allconsumer_complete_candidate_gather',4*nnz+16*rows)
    words=np.zeros(hz.n_cont,np.int64);lookup=np.full(hz.n_cont,-1,np.int32)
    lookup[candidates['column']]=np.arange(len(candidates),dtype=np.int32)
    seen_uids=set();journal=[];physical=0
    for kind,matrix,uids in ((False,hz.Ac,tables['eq']),(True,hz.Auc,tables['le'])):
        if (len(uids)!=matrix.shape[0] or not matrix.has_canonical_format
                or not np.isfinite(matrix.data).all() or np.any(matrix.data==0.)):
            raise ValueError('actual oracle requires complete canonical finite nonzero incidence')
        for row,raw in enumerate(uids):
            uid=int(raw)
            if not 0<=uid<UID_LIMIT or uid in seen_uids:raise ValueError('reused or invalid actual row UID')
            seen_uids.add(uid)
            a,b=map(int,matrix.indptr[row:row+2]);cols=matrix.indices[a:b]
            words[cols]+=RADIX+uid
            ids=lookup[cols]
            for rawpos in np.flatnonzero(ids>=0):
                pos=int(rawpos);i=int(ids[pos])
                pool.charge('allconsumer_every_occurrence_bookkeeping',16)
                if not kind and row==int(candidates['definition'][i]):
                    candidates['definitions_seen'][i]+=1;continue
                candidates['seen'][i]+=1
                if candidates['reason'][i]!=-1:continue
                parent=int(candidates['parent'][i])
                pool.charge('allconsumer_parent_range_test',8)
                matched=False;at=0
                if int(cols[0])<=parent<=int(cols[-1]):
                    pool.charge('allconsumer_parent_binary_search',12*max(1,len(cols).bit_length())+8)
                    at=int(np.searchsorted(cols,parent));matched=at<len(cols) and int(cols[at])==parent
                q=float(matrix.data[a+pos]);w=float(matrix.data[a+at]) if matched else 0.
                reason,value,zero=coefficient(q,w,float(candidates['ratio'][i]),pool=pool)
                candidates['checked'][i]+=1;candidates['observed_overlaps'][i]+=int(matched)
                if reason:
                    candidates['reason'][i]=reason;candidates['failed_inequality'][i]=kind
                    candidates['failed_row'][i]=row;candidates['failed_uid'][i]=uid
                    candidates['failed_q'][i]=q;candidates['failed_w'][i]=w
                    candidates['failed_float_candidate'][i]=value
                else:
                    candidates['exact_cancellations'][i]+=int(zero)
                    pool.charge('allconsumer_passed_UID_journal',16)
                    if len(journal)>=64_000_000:raise MemoryError('sparse journal entry cap')
                    journal.append((i<<20)|uid)
            physical+=1
            if observe and physical%32768==0:
                observe(dict(event='complete_actual_incidence_screen_progress',physical_rows=physical,
                    charged_work=pool.used,acceptance_published=False))
    branch.charge('compare_every_transferred_MAIN_word',8*len(claimed))
    if not np.array_equal(words[old_nc:logical_nc],claimed):
        raise ValueError('provisional ownership differs from COMPLETE actual incidence')
    pool.charge('allconsumer_complete_coverage_and_publication',128*len(candidates)+4*len(journal))
    if (np.any(candidates['definitions_seen']!=1)
            or np.any(candidates['seen']!=candidates['degree']-1)):
        raise ValueError('incomplete actual all-consumer/definition occurrence coverage')
    passed=candidates['reason']==-1
    if np.any(candidates['checked'][passed]!=candidates['seen'][passed]):
        raise ValueError('unexamined occurrence cannot pass a factor')
    candidates['reason'][passed]=0
    raw_journal=np.asarray(journal,np.uint64)
    keep=candidates['reason'][(raw_journal>>np.uint64(20)).astype(np.int64)]==0
    retained=raw_journal[keep]
    if len(retained)!=int(candidates['seen'][passed].sum()):raise ValueError('incomplete survivor UID journal')
    summary=dict(complete_actual_incidence_equal=True,all_current_physical_rows_scanned=physical,
        all_continuous_predicate_nnz_scanned=nnz,proposed_factors=len(candidates),
        all_consumer_occurrences=int(candidates['seen'].sum()),actual_defining_occurrences=int(candidates['definitions_seen'].sum()),
        numerical_occurrences_checked=int(candidates['checked'].sum()),
        unexamined_occurrences_after_necessary_failure=int((candidates['seen']-candidates['checked']).sum()),
        coefficient_decisions={name:int(np.count_nonzero(candidates['reason']==code)) for code,name in REASONS.items()},
        body_coefficient_survivors=int(passed.sum()),homogeneous_body_survivors=int(np.count_nonzero(passed & (candidates['offset']==0.))),
        nonhomogeneous_body_survivors=int(np.count_nonzero(passed & (candidates['offset']!=0.))),
        observed_overlaps=int(candidates['observed_overlaps'].sum()),observed_exact_cancellations=int(candidates['exact_cancellations'].sum()),
        all_actual_consumer_counts_match=True,survivor_UID_journal_entries=len(retained),
        candidate_table_bytes=candidates.nbytes,survivor_UID_journal_bytes=retained.nbytes,
        joint_RHS_arithmetic_executed=False,simultaneous_RHS_proved=False,
        new_lineage_or_reconstruction_proved=False,actual_nnz_reduction_proved=False,
        new_HZ_constructed=False,solver_executed=False,runtime_payment_proved=False,formal_gain=0)
    return summary,retained


def census(state,final,*,pool,branch=None,observe=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('actual bound SplicedState required')
    before=state.validate();final_before=source_digest(final)
    if (sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000
            or any(getattr(final,n) is not getattr(state.hz,n) for n in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('complete actual final source or entry cap mismatch')
    branch=pool if branch is None else branch
    if branch is not pool and getattr(branch,'whole',None) is not pool:raise ValueError('branch must charge the same whole pool')
    tables,uid=current_tables(state,pool=branch);claimed=claimed_stream(state,branch=branch)
    candidates,discovery=discover(state,final,tables,claimed,pool=pool)
    if observe:observe(dict(event='complete_structural_rediscovery_provisional',**discovery,
        proposed_factors=len(candidates),charged_work=pool.used,acceptance_published=False))
    f=state.original_fields
    report,journal=audit_and_screen(state.hz,tables,claimed,candidates,old_nc=f['old_n_cont'],
        logical_nc=f['logical_n_cont'],pool=pool,branch=branch,observe=observe)
    after=state.validate()
    if (before['complete_new_HZ_sha256']!=after['complete_new_HZ_sha256']
            or before['complete_semantic_lineage_sha256']!=after['complete_semantic_lineage_sha256']
            or source_digest(final)!=final_before):raise ValueError('read-only all-consumer screen changed source/final bytes')
    report.update(**discovery,uid_partition=uid,all_source_and_final_bytes_unchanged=True,
        complete_post_HZ_sha256=before['complete_new_HZ_sha256'],complete_final_HZ_sha256=final_before,
        diagnostic_work=pool.used,work_parts=dict(pool.parts))
    return report,candidates,journal
