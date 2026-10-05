"""Complete current-state classification; no HZ construction or solver entry.

The arithmetic primitive returns necessary guard decisions, never a certificate
for untested guards. Every multiplier/row is derived from the actual matrices.
"""
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c10_predicate_census_v1 import exact_products
from experiments.neural_hz_20260831.c13_separated_affine_census_v1 import coefficient_l1_numerator, exact_box
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, unique_other
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import decode
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables, checked_incidence
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

LOW, HIGH, BATCH = 2.**-20, 2.**40, 8
GUARDS = ('pivot_ok','ratio_exact','operand_window','offset_exact','rhs_exact',
          'products_exact','result_window','redundant_box')
REASONS = ('individual_exact','already_alias','already_unit_splice','output_live',
    'not_degree_two','missing_definition','not_direct_definition','binary_definition',
    'consumer_not_head','pivot_window','ratio_inexact','operand_window',
    'offset_product_inexact','rhs_inexact','coefficient_product_inexact',
    'coefficient_result_window','nonredundant_box')
CODE = {name:i for i,name in enumerate(REASONS)}


def prove(values, pivot, q, d, c, *, pool):
    """One binary-free ordered pair, fixed normal window, exact float64 identity."""
    pool.charge('nonunit_pair_scalars', 160)
    out = {g:-1 for g in GUARDS}
    out.update(reason=CODE['pivot_window'], ratio=0., checked=0, failed_term=-1)
    if not math.isfinite(pivot) or not LOW <= pivot <= HIGH or math.frexp(pivot)[0] != .5:
        out['pivot_ok'] = 0; return out
    out['pivot_ok'] = 1
    if not math.isfinite(q) or not LOW <= abs(q) <= HIGH:
        out.update(reason=CODE['operand_window'],operand_window=0); return out
    ratio = math.ldexp(-q, -(math.frexp(pivot)[1]-1))
    if not math.isfinite(ratio) or F(ratio)*F(pivot) != -F(q):
        out.update(reason=CODE['ratio_inexact'],ratio_exact=0); return out
    out.update(ratio=ratio,ratio_exact=1)
    if not math.isfinite(d) or not math.isfinite(c):
        raise ValueError('bound actual RHS is not finite')
    change = 0.
    if d:
        pool.charge('nonunit_offset_products',64)
        precise, product = exact_products(np.array([d]),ratio)
        if not bool(precise[0]):
            out.update(reason=CODE['offset_product_inexact'],offset_exact=0); return out
        change = float(product[0])
    out['offset_exact'] = 1
    updated = c+change
    if not math.isfinite(updated) or F(updated) != F(c)+F(ratio)*F(d):
        out.update(reason=CODE['rhs_inexact'],rhs_exact=0); return out
    out['rhs_exact'] = 1
    for begin in range(0,len(values),BATCH):
        block = values[begin:begin+BATCH]
        pool.charge('nonunit_operand_windows',4*len(block))
        valid = np.isfinite(block) & (np.abs(block)>=LOW) & (np.abs(block)<=HIGH)
        if not valid.all():
            out.update(reason=CODE['operand_window'],operand_window=0,
                failed_term=begin+int(np.flatnonzero(~valid)[0])); return out
        pool.charge('nonunit_coefficient_products',64*len(block))
        precise, products = exact_products(block,ratio)
        window = (np.abs(products)>=LOW) & (np.abs(products)<=HIGH)
        out['checked'] += len(block)
        if not precise.all() or not window.all():
            if not precise.all(): out['products_exact'] = 0
            if not window.all(): out['result_window'] = 0
            failed = ~precise if not precise.all() else ~window
            out.update(reason=CODE['coefficient_product_inexact' if not precise.all() else 'coefficient_result_window'],
                failed_term=begin+int(np.flatnonzero(failed)[0])); return out
    out.update(operand_window=1,products_exact=1,result_window=1)
    pool.charge('nonunit_exact_box_norm',16*len(values))
    if not exact_box(coefficient_l1_numerator(values),d,pivot):
        out.update(reason=CODE['nonredundant_box'],redundant_box=0); return out
    out.update(reason=CODE['individual_exact'],redundant_box=1)
    return out


DTYPE = np.dtype([('column','i4'),('definition','i4'),('consumer','i4'),('inequality','?'),
    ('reason','i2'),('ratio','f8'),('parent_terms','i4'),('tail_terms','i4'),
    ('products_checked','i4'),('failed_term','i4'),('independent','?'),
    *((g,'i1') for g in GUARDS)])


def independent_mask(table, *, pool):
    selected = np.flatnonzero(table['reason']==CODE['individual_exact'])
    pool.charge('nonunit_complete_pair_dependencies',64*len(selected))
    definitions, consumers, columns = {}, {}, {}
    for raw in selected:
        i=int(raw)
        for mapping,key in ((definitions,int(table['definition'][i])),
                (consumers,(bool(table['inequality'][i]),int(table['consumer'][i]))),
                (columns,int(table['column'][i]))):
            mapping.setdefault(key,[]).append(i)
    bad=set()
    for mapping in (definitions,consumers,columns):
        for values in mapping.values():
            if len(values)>1:bad.update(values)
    for (kind,row),values in consumers.items():
        if not kind and row in definitions:
            bad.update(values);bad.update(definitions[row])
    for raw in selected:
        i=int(raw);table['independent'][i]=i not in bad
    return len(bad)


def classify(state, final, tables, words, *, pool, observe=None):
    """Internal diagnostic only; caller holds the complete checked transaction."""
    f,hz = state.original_fields,state.hz
    first,limit = f['old_n_cont'],f['logical_n_cont'];n=limit-first
    if words.shape != (n,) or tables['definitions'].shape != (n,):
        raise ValueError('classification requires complete current incidence/mapping')
    pool.charge('nonunit_complete_table_and_liveness',32*n+4*int(final.Gc.nnz+hz.Gc.nnz))
    table=np.zeros(n,DTYPE)
    table['column']=np.arange(first,limit,dtype=np.int32)
    for name in ('definition','consumer','reason','failed_term',*GUARDS):table[name]=-1
    live=set(map(int,final.Gc.indices));live.update(map(int,hz.Gc.indices))
    def reject(i,name):table['reason'][i]=CODE[name]
    structural=0
    for i in range(n):
        col=first+i;tag=decode(f['eq_roots'][f['old_n_eq']+i])[0]
        if tag=='alias':reject(i,'already_alias');continue
        if tag=='splice':reject(i,'already_unit_splice');continue
        if col in live:reject(i,'output_live');continue
        if int(words[i])//RADIX != 2:reject(i,'not_degree_two');continue
        d=int(tables['definitions'][i]);table['definition'][i]=d
        if d < 0:reject(i,'missing_definition');continue
        pool.charge('nonunit_actual_pair_geometry',64)
        da,db=map(int,hz.Ac.indptr[d:d+2])
        if da==db or int(hz.Ac.indices[db-1])!=col:
            reject(i,'not_direct_definition');continue
        if hz.Ab.indptr[d]!=hz.Ab.indptr[d+1]:reject(i,'binary_definition');continue
        other=unique_other(int(words[i]),int(tables['eq'][d]))
        if other not in tables['lookup']:raise ValueError('proved unique consumer UID does not resolve')
        kind,c=tables['lookup'][other]
        if not kind and c==d:raise ValueError('consumer is the defining row')
        table['consumer'][i]=c;table['inequality'][i]=kind
        matrix,rhs=(hz.Auc,hz.ub) if kind else (hz.Ac,hz.b)
        ca,cb=map(int,matrix.indptr[c:c+2])
        if ca==cb or int(matrix.indices[ca])!=col:
            reject(i,'consumer_not_head');continue
        structural+=1
        table['parent_terms'][i]=db-da-1;table['tail_terms'][i]=cb-ca-1
        answer=prove(hz.Ac.data[da:db-1],float(hz.Ac.data[db-1]),float(matrix.data[ca]),
            float(hz.b[d]),float(rhs[c]),pool=pool)
        table['reason'][i]=answer['reason'];table['ratio'][i]=answer['ratio']
        table['products_checked'][i]=answer['checked'];table['failed_term'][i]=answer['failed_term']
        for g in GUARDS:table[g][i]=answer[g]
        if observe and structural%4096==0:
            observe(dict(event='current_nonunit_numeric_progress',MAIN_through=i+1,
                structural_pairs=structural,charged_work=pool.used,acceptance_published=False))
    if np.any(table['reason']<0):raise ValueError('unfinished census cannot publish a subset')
    conflicts=independent_mask(table,pool=pool)
    accepted=table['independent'];individual=table['reason']==CODE['individual_exact']
    if any(np.any(table[g][individual]!=1) for g in GUARDS):
        raise ValueError('unknown numeric guard cannot certify a row')
    pool.charge('nonunit_complete_publication_summaries',96*n)
    report=dict(all_MAIN_classified=n,ordered_binary_free_single_consumer_pairs=structural,
        first_rejection_counts={r:int(np.count_nonzero(table['reason']==k)) for k,r in enumerate(REASONS)},
        guards={g:{'passed':int(np.count_nonzero(table[g]==1)),
                   'failed':int(np.count_nonzero(table[g]==0)),
                   'not_evaluated':int(np.count_nonzero(table[g]==-1))} for g in GUARDS},
        individual_exact_pairs=int(individual.sum()),dependency_rejected_pairs=conflicts,
        simultaneous_independent_pairs=int(accepted.sum()),
        nonunit_independent_pairs=int(np.count_nonzero(accepted & (np.abs(table['ratio'])!=1.))),
        structural_parent_terms=int(table['parent_terms'].sum()),
        all_checked_products=int(table['products_checked'].sum()),
        independent_parent_terms=int(table['parent_terms'][accepted].sum()),
        independent_tail_terms=int(table['tail_terms'][accepted].sum()),
        multiplier_payload_bytes_only_lower_bound=8*int(accepted.sum()),
        potential_predicate_nnz_delta=-2*int(accepted.sum()),
        table_numeric_bytes=table.nbytes,table_numeric_records=len(table),
        runtime_payment_proved=False,new_multiplier_lineage_proved=False,
        new_HZ_or_native_consumer_constructed=False,solver_executed=False,formal_gain=0)
    return report,table


def census(state, final, *, pool, branch=None, observe=None, enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('actual bound SplicedState required')
    before=state.validate();final_before=source_digest(final)
    if (sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000
            or any(getattr(final,n) is not getattr(state.hz,n) for n in ('Ac','Ab','Auc','Aub'))):
        raise ValueError('complete final predicate source or entry cap mismatch')
    branch=pool if branch is None else branch
    if branch is not pool and getattr(branch,'whole',None) is not pool:
        raise ValueError('branch charges must belong to the same whole diagnostic ledger')
    tables,uid=current_tables(state,pool=branch)
    if observe:observe(dict(event='complete_current_UID_partition',**uid,charged_work=pool.used))
    actual=checked_incidence(state,tables,pool=branch)
    if observe:observe(dict(event='complete_current_incidence_proved',all_MAIN_words=len(actual),charged_work=pool.used))
    report,table=classify(state,final,tables,actual,pool=pool,observe=observe)
    after=state.validate()
    if (before['complete_new_HZ_sha256']!=after['complete_new_HZ_sha256']
            or before['complete_semantic_lineage_sha256']!=after['complete_semantic_lineage_sha256']
            or source_digest(final)!=final_before):
        raise ValueError('read-only census changed actual source/final HZ')
    report.update(uid_partition=uid,complete_actual_incidence_equal=True,
        all_source_and_final_bytes_unchanged=True,complete_post_HZ_sha256=before['complete_new_HZ_sha256'],
        complete_final_HZ_sha256=final_before,diagnostic_work=pool.used,work_parts=dict(pool.parts))
    return report,table
