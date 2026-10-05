"""Independent all-row circuit audit with portable complete sharing equality."""
import numpy as np
from types import SimpleNamespace
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import SCHEMA, row, _incidence
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays, operator_digest


def expression_key(expr,pool):
    """Canonical source/operator sharing, independent of decoder object IDs."""
    if expr is None:return None
    layout=numeric_layout(expr,pool)
    pool.charge('c98_complete_expression_semantic_binding',int(layout.resident_entries)+1024)
    sources={};operators={};terms=[]
    for term in expr.terms:
        if id(term.source) not in sources:
            sources[id(term.source)]=(len(sources),source_digest(term.source))
        ops=[]
        for op in term.operators:
            if id(op) not in operators:operators[id(op)]=(len(operators),operator_digest(op))
            ops.append(operators[id(op)])
        terms.append((sources[id(term.source)],tuple(ops)))
    return expr.frame_id,expr.n_out,digest_arrays(expr.bias),tuple(terms)


def audit(before,state,circuits,*,pool,enabled=False):
    """Every old/new row and all owner/map fields, from a qualified old source."""
    if not enabled:return None
    if state['schema']!=SCHEMA or state['native_or_LIVE_admission']:raise ValueError('explicit non-admitted circuit source required')
    fields=state['fields'];old=before['hz'];new=fields['hz'];aux=state['auxiliary_records'];routes=state['output_routes']
    pool.charge('c91_all_original_row_buffer_comparison',4*old.Ac.nnz+8*old.n_eq)
    if (new.frame_id!=old.frame_id or not new.exact or new.n_bin!=old.n_bin
        or new.n_cont!=old.n_cont+len(aux) or new.n_eq!=old.n_eq+len(aux)
        or not np.array_equal(new.c,old.c) or not np.array_equal(new.ub,old.ub)
        or not np.array_equal(new.b[:old.n_eq],old.b) or np.any(new.b[old.n_eq:])):
        raise ValueError('complete original frame/constant/binary partition differs')
    for name in ('Gc','Gb','Auc','Aub'):
        a,b=getattr(old,name),getattr(new,name)
        if any(not np.array_equal(getattr(a,k),getattr(b,k)) for k in ('data','indices','indptr')):
            raise ValueError('original value/inequality coefficients differ')
    if (not np.array_equal(old.Ab.data,new.Ab.data) or not np.array_equal(old.Ab.indices,new.Ab.indices)
        or not np.array_equal(old.Ab.indptr,new.Ab.indptr[:old.n_eq+1])
        or np.any(new.Ab.indptr[old.n_eq+1:]!=old.Ab.indptr[-1])):
        raise ValueError('original binary equations differ')
    pool.charge('c91_full_unchanged_maps_and_value_arrays',4*sum(before[k].size for k in
        ('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','uid_slabs','radix_gauges'))
        +4*sum(getattr(old,k).nnz+getattr(old,k).shape[0]+1 for k in ('Gc','Gb','Ab','Auc','Aub')))
    changed=np.sort(routes[:,0]);previous=0
    for physical in [*map(int,changed),old.n_eq]:
        oa,ob=int(old.Ac.indptr[previous]),int(old.Ac.indptr[physical])
        na,nb=int(new.Ac.indptr[previous]),int(new.Ac.indptr[physical])
        if (not np.array_equal(old.Ac.data[oa:ob],new.Ac.data[na:nb])
            or not np.array_equal(old.Ac.indices[oa:ob],new.Ac.indices[na:nb])
            or not np.array_equal(np.diff(old.Ac.indptr[previous:physical+1]),np.diff(new.Ac.indptr[previous:physical+1]))):
            raise ValueError('an unchanged original physical row differs')
        previous=physical+1
    eq_uids,_=closed_uid_tables(SimpleNamespace(**before));first=int(before['report']['radix_uid_base'])+len(before['def_rows'])
    expected_routes=[];expected_blocks=[];offset=0;written_nnz=0
    for packet in circuits:
        begin=len(expected_routes)
        n=packet['new_factors'];pool.charge('c91_every_written_circuit_literal',8*len(packet['native'])+64*len(packet['rhs']))
        for i,pivot in enumerate(packet['pivots']):
            a,b=map(int,packet['indptr'][i:i+2]);pivot=int(pivot)
            if i<n:
                physical=old.n_eq+offset+i
                p=int(np.flatnonzero(packet['columns'][a:b]==pivot)[0]);pv=float(packet['native'][a+p])
                record=aux[offset+i]
                if (record[0]!=physical or record[1]!=first+offset+i or record[2]!=pivot
                    or record[3]!=np.array(pv,np.float64).view(np.int64).item() or record[4]!=packet['gauges'][i]
                    or not np.array_equal(record[6:8].view(np.float64),np.array([-1.,1.]))):
                    raise ValueError('complete new inverse/UID/bound record differs')
            else:
                rank=before['old_n_eq']+pivot-before['old_n_cont'];physical=int(before['eq_roots'][rank])
                expected_routes.append((physical,pivot,int(packet['gauges'][i])))
                if fields['eq_scales'][rank]!=packet['gauges'][i]:raise ValueError('actual output row gauge differs')
            cols,vals=row(new.Ac,physical)
            if (not np.array_equal(cols,packet['columns'][a:b]) or not np.array_equal(vals,packet['native'][a:b])):
                raise ValueError('actual written circuit literal differs')
            written_nnz+=b-a
        expected_blocks.append((old.n_cont,old.n_cont+offset,n,begin,len(expected_routes),old.n_eq+offset,first+offset,0))
        offset+=n
    if not np.array_equal(routes,np.array(expected_routes,np.int64)):raise ValueError('complete circuit root routes differ')
    if not np.array_equal(state['block_records'],np.array(expected_blocks,np.int64)):raise ValueError('complete block routing differs')
    if (offset!=len(aux) or state['old_source_n_cont']!=old.n_cont or state['old_source_n_eq']!=old.n_eq
        or expression_key(fields['expression'],pool)!=expression_key(before['expression'],pool)):raise ValueError('complete original source identity differs')
    for name in ('keep','eq_roots','ineq_roots','ineq_scales','def_rows','uid_slabs','radix_gauges'):
        if not np.array_equal(before[name],fields[name]):raise ValueError('original source/inverse map changed: '+name)
    unchanged_scales=np.ones(len(before['eq_scales']),bool)
    unchanged_scales[before['old_n_eq']+routes[:,1]-before['old_n_cont']]=False
    if not np.array_equal(before['eq_scales'][unchanged_scales],fields['eq_scales'][unchanged_scales]):
        raise ValueError('old scalar inverse numerator/tag changed')
    old_words=_incidence(old.Ac,changed,eq_uids[changed],new.n_cont,pool=pool)
    added_rows=np.r_[changed,np.arange(old.n_eq,new.n_eq)].astype(np.int64)
    added_uids=np.r_[eq_uids[changed],np.arange(first,first+len(aux))].astype(np.int64)
    new_words=_incidence(new.Ac,added_rows,added_uids,new.n_cont,pool=pool)
    left,right=before['old_n_cont'],before['logical_n_cont']
    if (not np.array_equal(fields['owners'],before['owners']+new_words[left:right]-old_words[left:right])
        or not np.array_equal(aux[:,5],new_words[old.n_cont:])):
        raise ValueError('complete old/new owner vectors differ from independent incidence')
    if new.Ac.nnz>=old.Ac.nnz:raise ValueError('whole physical predicate nnz does not decrease')
    return dict(complete_original_rows_checked=old.n_eq,actual_original_rows_replaced=len(routes),
        actual_auxiliary_rows_checked=len(aux),actual_written_native_coefficients=written_nnz,
        complete_old_MAIN_owners_checked=len(fields['owners']),complete_new_owners_checked=len(aux),
        old_scalar_equations_retained=int(np.count_nonzero(fields['eq_roots']<0)),
        original_binary_factors=old.n_bin,all_original_maps_and_other_predicates_preserved=True,
        independent_complete_owner_delta_proved=True,whole_predicate_nnz_before=int(old.Ac.nnz+old.Ab.nnz+old.Auc.nnz+old.Aub.nnz),
        whole_predicate_nnz_after=int(new.Ac.nnz+new.Ab.nnz+new.Auc.nnz+new.Aub.nnz),
        source_and_box_theorem_requires_authenticated_C90=True,full_LIVE_admission=False,formal_gain=0)
