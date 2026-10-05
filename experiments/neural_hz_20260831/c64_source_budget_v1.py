"""Complete source-derived C64 construction bound; not runtime payment or a HZ."""
from collections import Counter
import math
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.c62_precision_plan_v1 import odd_significands
from experiments.neural_hz_20260831.c64_compact_boundary_dp_v1 import choose
from experiments.neural_hz_20260831.c64_gauged_products_v1 import products,gauge_definition,gauge_row
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_cost
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word

REPLACED=frozenset(('continuous_incidence_scan','product_operand_gathers','product_common',
    'product_general_exact','product_left_classification','rewrite_sort',
    'ownership_known_incidence_updates','ownership_verified_frontier_retirement'))


def bound(saved,legacy,reference,*,pool,enabled=False):
    if not enabled:return None
    hz=saved['hz'];old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);oe=int(saved['old_n_eq'])
    nodes=saved['definition_graph'];parents=reference['parents'];local=reference['local']
    pool.charge('c64_complete_source_budget_records',32*(hz.n_cont+hz.n_eq+len(parents))+1024)
    routed,route=route_rows(saved,np.array(sorted(parents),np.int64),pool=pool,enabled=True)
    raw=np.zeros(hz.n_cont,bool);raw[list(parents)]=True;own=np.zeros(hz.n_eq,bool)
    defining={v:int(saved['eq_roots'][oe+v-old]) for v in parents}
    own[list(defining.values())]=True;routed[own]=False
    dyadic=np.zeros(hz.n_eq,bool)
    for node in nodes:
        counts=Counter(node['parents'])
        if node['kind']=='sum' and all(n>0 and n&(n-1)==0 for n in counts.values()):
            slots=node['slots'][node['needed']]
            dyadic[saved['eq_roots'][oe+slots-old]]=True
    # All inherited prices survive except the exact obsolete operations above.
    model=WorkPool(256_000_000)
    for name,amount in legacy['report']['alias_quotient']['work_parts'].items():
        if name not in REPLACED:model.charge(name,int(amount))
    for node in nodes:model.charge('c64_birth_block_dispatch',32+8*len(node['parents']))
    model.charge('c62_distinct_local_equation_encoding',128*len(set(local.values())))
    maxima=np.zeros(hz.n_cont,np.uint64);external_rows=occurrences=dyadic_occurrences=0
    for r0 in np.flatnonzero(routed):
        r=int(r0);a,b=map(int,hz.Ac.indptr[r:r+2]);cc=hz.Ac.indices[a:b];cv=hz.Ac.data[a:b]
        model.charge('c63_routed_incidence_row',16+3*len(cc))
        positions=np.flatnonzero(raw[cc])
        if not len(positions):continue
        ids=cc[positions];external_rows+=1;occurrences+=len(ids)
        actual=odd_significands(cv[positions])
        if dyadic[r]:
            if np.any(actual!=1):raise ValueError('source sum power-of-two theorem differs from actual encoded coefficients')
            model.charge('c64_source_dyadic_maximum',32+8*len(cc)+2*len(ids));dyadic_occurrences+=len(ids)
        else:model.charge('c62_complete_external_maximum',32+8*len(cc)+16*len(ids))
        np.maximum.at(maxima,ids,actual)
    selected,roots,weights,stats=choose(parents,local,maxima,hz.n_cont,pool=model)
    if (not np.array_equal(selected,reference['selected']) or not np.array_equal(roots,reference['roots'])
            or weights!=reference['weights'] or external_rows!=reference['report']['external_rows']
            or occurrences!=reference['report']['external_occurrences']):raise ValueError('complete source-bound optimum or incidence differs')
    for v in np.flatnonzero(selected):
        r=defining[int(v)];a,b=map(int,hz.Ac.indptr[r:r+2]);cc=hz.Ac.indices[a:b]
        model.charge('ownership_known_incidence_updates',4*int(np.count_nonzero((cc>=old)&(cc<logical))))
    changes=total_active=general=0;gauges=Counter();audit=WorkPool(256_000_000)
    affected=reference['raw_hits']['Ac'] & ~reference['erased']
    for r0 in np.flatnonzero(affected):
        r=int(r0);a,b=map(int,hz.Ac.indptr[r:r+2]);cc=hz.Ac.indices[a:b];cv=hz.Ac.data[a:b]
        positions=np.flatnonzero(selected[cc])
        if not len(positions):continue
        pool.charge('c64_independent_complete_changed_row_budget_proof',32+8*len(cc)+16*len(positions))
        ratios=np.array([math.ldexp(float(weights[int(c)][0]),weights[int(c)][1]) for c in cc[positions]])
        if any(native_word(ratio)!=weights[int(c)] for c,ratio in zip(cc[positions],ratios)):
            raise ValueError('complete source-derived native multiplier differs')
        model.charge('product_operand_gathers',2*len(positions))
        changed,proof=products(cv[positions],ratios,pool=model)
        model.charge('rewrite_sort',sort_cost(len(cc)))
        prior=cc[positions];after=roots[prior]
        model.charge('ownership_known_incidence_updates',4*int(np.count_nonzero((prior>=old)&(prior<logical))))
        model.charge('ownership_known_incidence_updates',4*int(np.count_nonzero((after>=old)&(after<logical))))
        translated=cc.copy();values=cv.copy();translated[positions]=after;values[positions]=changed
        order=np.argsort(translated,kind='stable');translated=translated[order];values=values[order]
        if np.any(np.diff(translated)<=0):raise ValueError('coalescence outside complete precision-boundary class')
        ba,bb=map(int,hz.Ab.indptr[r:r+2]);bv=hz.Ab.data[ba:bb];rhs=float(hz.b[r])
        # Independently check the prerequisite, including all unchanged inputs.
        original_operands=np.r_[cv,hz.Ab.data[ba:bb],[rhs] if rhs else []]
        if not len(cv) or cv[-1]<=0 or np.max(np.abs(original_operands))>cv[-1]:
            raise ValueError('actual complete source/radix definition lacks dominant unchanged pivot')
        small=proof['minimum_absolute_product']
        expected=gauge_row(values,bv,rhs,pool=audit)
        actual=gauge_definition(values,bv,rhs,small,pool=model)
        if (expected[2:]!=actual[2:] or not np.array_equal(expected[0],actual[0])
                or not np.array_equal(expected[1],actual[1])):raise ValueError('complete row and dominant-pivot gauges disagree')
        changes+=1;total_active+=len(positions);general+=proof['general'];gauges[int(actual[3])]+=1
    pool.charge('c64_complete_independent_row_gauge_oracle',audit.used)
    whole=int(legacy['report']['whole_base_work'])+model.used
    branch=int(legacy['report']['branch_base_work'])+model.used
    old_nnz=int(sum(getattr(hz,k).nnz for k in ('Ac','Ab','Auc','Aub')))
    return dict(schema='c64_complete_source_construction_bound_v1',complete=True,
        whole_base_work=int(legacy['report']['whole_base_work']),branch_base_work=int(legacy['report']['branch_base_work']),
        coupled_extra_upper=model.used,whole_work_upper=whole,branch_work_upper=branch,work_parts=dict(model.parts),
        work_caps_fit=whole<=256_000_000 and branch<=200_000_000,
        whole_headroom=256_000_000-whole,branch_headroom=200_000_000-branch,
        raw_factors=len(parents),selected=int(selected.sum()),ancestor_states=stats['ancestor_states'],
        expected_identity_sha256=reference['report']['identity_sha256'],external_rows=external_rows,
        external_occurrences=occurrences,dyadic_source_occurrences=dyadic_occurrences,
        rewritten_rows=changes,rewritten_occurrences=total_active,general_products=general,row_gauges=dict(gauges),
        original_predicate_nnz=old_nnz,expected_predicate_nnz=old_nnz-2*int(selected.sum()),
        conservative_radix_and_old_prefix_routing=True,full_row_gauge_prerequisites_checked=True,
        no_new_physical_HZ=True,not_actual_generation_or_runtime_payment=True,
        full_physical_storage_publication_and_source_qualification_still_required=True,formal_gain=0)
