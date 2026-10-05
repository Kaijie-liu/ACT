"""Complete consumer-headed unit discovery in an authenticated read transaction.

The caller must bind Closed's complete source/box/UID proof and the actual
append-only post-HZ/Overlay incidence proof before calling. This default-off
primitive returns provisional plans, never native admission or a new HZ.
No complete MAIN stream, dense UID table, or full coefficient incidence scan.
"""

from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX,unique_other
from experiments.neural_hz_20260831.c22_uid_runs_v1 import uid_for_row,row_for_uid
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import resolve
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan


def discover(closed,post,overlay,*,pool,enabled=False):
    if not enabled:return None
    main=closed.logical_n_cont-closed.old_n_cont
    if (post.frame_id!=closed.hz.frame_id or not post.exact or main<0
            or len(overlay.base)!=main or len(closed.eq_roots)!=closed.old_n_eq+main
            or overlay.base is not closed.owners or post.n_eq<closed.hz.n_eq
            or post.n_ineq<closed.hz.n_ineq or post.n_cont<closed.hz.n_cont
            or post.n_bin<closed.hz.n_bin
            or overlay.old_uid_ceiling!=closed.report['radix_uid_base']+16384):
        raise ValueError('incomplete/mismatched authenticated append transaction')
    if any(int(m.nnz)>64_000_000 for m in (post.Ac,post.Auc,post.Gc)):
        raise MemoryError('unchanged coefficient entry ceiling exceeded')
    # Output incidence is already canonical/nonzero by the bound source proof.
    # A sparse set uses only actual outputs, not a full-global-slot mask.
    pool.charge('consumer_sparse_output_liveness',8*int(post.Gc.nnz))
    live=set(map(int,post.Gc.indices))
    stats={'all_physical_consumer_rows':0,'MAIN_head_rows':0,'base_degree_survivors':0,
        'defining_shape_survivors':0,'unit_pivot_survivors':0,'full_overlay_queries':0,
        'selected_old_consumers':0,'selected_new_consumers':0,'tail_terms':0}
    plans=[]
    for kind,matrix,rhs,old_rows in ((False,post.Ac,post.b,closed.hz.n_eq),
            (True,post.Auc,post.ub,closed.hz.n_ineq)):
        for row in range(matrix.shape[0]):
            pool.charge('consumer_row_header',8)
            stats['all_physical_consumer_rows']+=1
            a,b=int(matrix.indptr[row]),int(matrix.indptr[row+1])
            if a==b:continue
            col=int(matrix.indices[a])
            if not closed.old_n_cont<=col<closed.logical_n_cont:continue
            stats['MAIN_head_rows']+=1
            pool.charge('consumer_base_degree',8)
            i=col-closed.old_n_cont
            old=row<old_rows
            # Known D is OLD, C is a distinct actual row. Exact old incidence
            # must therefore have degree2 for old C, or degree1 for new C.
            # Append-only events cannot reduce it. This is a necessary filter,
            # NOT membership inferred from an arbitrary packed word.
            if int(overlay.base[i])//RADIX!=(2 if old else 1):continue
            stats['base_degree_survivors']+=1
            pool.charge('consumer_defining_shape',16)
            d=int(closed.eq_roots[closed.old_n_eq+i])
            if d<0:continue
            if not 0<=d<closed.hz.n_eq:raise ValueError('old MAIN definition outside bound prefix')
            if not kind and d==row:continue
            da,db=int(post.Ac.indptr[d]),int(post.Ac.indptr[d+1])
            if (da==db or int(post.Ac.indices[db-1])!=col
                    or post.Ab.indptr[d]!=post.Ab.indptr[d+1]):continue
            stats['defining_shape_survivors']+=1
            pool.charge('consumer_unit_pivot',16)
            pivot=float(post.Ac.data[db-1]);coefficient=float(matrix.data[a])
            if pivot<=0 or not math.isfinite(pivot) or math.frexp(pivot)[0]!=.5 or abs(coefficient)!=pivot:continue
            stats['unit_pivot_survivors']+=1
            pool.charge('consumer_output_filter',4)
            if col in live:continue
            u=uid_for_row(closed.uid_slabs,i,pool=pool)
            if u is None:raise ValueError('checked MAIN definition has no canonical UID')
            stats['full_overlay_queries']+=1
            packed=overlay.query(i,pool=pool)
            if packed//RADIX!=2:continue
            v=unique_other(packed,u)
            if v is None or v==u:raise ValueError('canonical unique consumer UID missing')
            ne=post.n_eq-closed.hz.n_eq
            if v>=overlay.old_uid_ceiling:
                pool.charge('consumer_new_phase_UID_resolution',12)
                at=v-overlay.old_uid_ceiling
                if at<ne:resolved=(False,closed.hz.n_eq+at)
                else:resolved=(True,closed.hz.n_ineq+at-ne)
                if at>=ne+post.n_ineq-closed.hz.n_ineq:raise ValueError('consumer UID outside actual appended rows')
                consumer_main=None
            else:
                resolved=resolve(closed,v,pool=pool)
                consumer_main=row_for_uid(closed.uid_slabs,v,pool=pool)
                consumer_main=None if consumer_main is None else closed.old_n_cont+consumer_main
            if resolved!=(kind,row):raise ValueError('actual consumer does not match proved unique incidence')
            pool.charge('consumer_exact_RHS_and_plan',160)
            sign=-1 if coefficient>0 else 1
            offset=float(post.b[d]);target=float(rhs[row]);updated=target+sign*offset
            if not all(map(math.isfinite,(offset,target,updated))) or F(updated)!=F(target)+sign*F(offset):
                raise ValueError('unit RHS update not exact')
            pool.charge('consumer_tail_read',4*(b-a-1))
            tail=tuple(map(int,matrix.indices[a+1:b]))
            plans.append(Plan(col,d,row,kind,u,v,pivot,sign,offset,consumer_main,tail))
            stats['selected_old_consumers' if old else 'selected_new_consumers']+=1
            stats['tail_terms']+=len(tail)
    n=len(plans)
    pool.charge('consumer_plan_sort_and_disjointness',4*n*max(1,(n-1).bit_length())+16*n)
    plans.sort(key=lambda p:p.column)
    definitions={p.definition for p in plans}
    if (len({p.column for p in plans})!=n or len(definitions)!=n
            or len({(p.inequality,p.consumer) for p in plans})!=n
            or any(not p.inequality and p.consumer in definitions for p in plans)):
        raise ValueError('selected unit plans are not a simultaneous independent population')
    stats.update(all_selected_plans=n,no_dense_UID_or_MAIN_oracle_built=True,
        all_original_coefficients_scanned=False,source_proof_required=True,
        live_admission_certificate=False,formal_gain=0)
    return plans,stats
