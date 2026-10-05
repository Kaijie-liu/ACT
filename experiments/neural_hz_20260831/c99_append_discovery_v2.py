"""C99: identical complete consumer rule with explicit circuit UID resolution.

The same complete source/box/incidence binding is mandatory. Block routing is
separately charged; all original C28 discovery prices and guards are retained.
"""

from fractions import Fraction as F
import math
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, unique_other
from experiments.neural_hz_20260831.c22_uid_runs_v1 import uid_for_row, row_for_uid
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import resolve
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan


def discover_append(closed, view, overlay, *, pool, enabled=False):
    if not enabled: return None
    view.validate_shape(pool)
    main = closed.logical_n_cont-closed.old_n_cont
    if (view.pre is not closed.hz or main < 0 or len(overlay.base) != main
            or len(closed.eq_roots) != closed.old_n_eq+main or overlay.base is not closed.owners
            or overlay.old_uid_ceiling != closed.report['radix_uid_base']+16384):
        raise ValueError('incomplete/mismatched authenticated append transaction')
    pool.charge('consumer_sparse_output_liveness',8*int(view.Gc.nnz))
    live = set(map(int,view.Gc.indices))
    stats = dict(all_physical_consumer_rows=0,MAIN_head_rows=0,base_degree_survivors=0,
        defining_shape_survivors=0,unit_pivot_survivors=0,full_overlay_queries=0,
        selected_old_consumers=0,selected_new_consumers=0,tail_terms=0)
    plans = []
    for kind in (False,True):
        old_rows = view.pre.n_ineq if kind else view.pre.n_eq
        offset = 0
        for matrix,rhs in zip(view.blocks('Auc' if kind else 'Ac'),view.blocks('ub' if kind else 'b')):
            for local in range(matrix.shape[0]):
                pool.charge('append_consumer_global_row_routing',4)
                r = offset+local
                pool.charge('consumer_row_header',8); stats['all_physical_consumer_rows'] += 1
                a,b = int(matrix.indptr[local]),int(matrix.indptr[local+1])
                if a == b: continue
                col = int(matrix.indices[a])
                if not closed.old_n_cont <= col < closed.logical_n_cont: continue
                stats['MAIN_head_rows'] += 1; pool.charge('consumer_base_degree',8)
                i = col-closed.old_n_cont; old = r < old_rows
                if int(overlay.base[i])//RADIX != (2 if old else 1): continue
                stats['base_degree_survivors'] += 1; pool.charge('consumer_defining_shape',16)
                d = int(closed.eq_roots[closed.old_n_eq+i])
                if d < 0: continue
                if not 0 <= d < view.pre.n_eq: raise ValueError('old MAIN definition outside bound prefix')
                if not kind and d == r: continue
                da,db = int(view.pre.Ac.indptr[d]),int(view.pre.Ac.indptr[d+1])
                if (da == db or int(view.pre.Ac.indices[db-1]) != col
                        or view.pre.Ab.indptr[d] != view.pre.Ab.indptr[d+1]): continue
                stats['defining_shape_survivors'] += 1; pool.charge('consumer_unit_pivot',16)
                pivot = float(view.pre.Ac.data[db-1]); coefficient = float(matrix.data[a])
                if pivot <= 0 or not math.isfinite(pivot) or math.frexp(pivot)[0] != .5 or abs(coefficient) != pivot: continue
                stats['unit_pivot_survivors'] += 1; pool.charge('consumer_output_filter',4)
                if col in live: continue
                u = uid_for_row(closed.uid_slabs,i,pool=pool)
                if u is None: raise ValueError('checked MAIN definition has no canonical UID')
                stats['full_overlay_queries'] += 1
                packed = overlay.query(i,pool=pool)
                if packed//RADIX != 2: continue
                v = unique_other(packed,u)
                if v is None or v == u: raise ValueError('canonical unique consumer UID missing')
                ne = view.eq_c.shape[0]
                if v >= overlay.old_uid_ceiling:
                    pool.charge('consumer_new_phase_UID_resolution',12)
                    at = v-overlay.old_uid_ceiling
                    resolved = (False,view.pre.n_eq+at) if at < ne else (True,view.pre.n_ineq+at-ne)
                    if at >= ne+view.le_c.shape[0]: raise ValueError('consumer UID outside appended rows')
                    consumer_main = None
                else:
                    resolved = resolve(closed,v,pool=pool)
                    consumer_main = row_for_uid(closed.uid_slabs,v,pool=pool)
                    consumer_main = None if consumer_main is None else closed.old_n_cont+consumer_main
                if resolved != (kind,r): raise ValueError('actual consumer differs from proved unique incidence')
                pool.charge('consumer_exact_RHS_and_plan',160)
                sign = -1 if coefficient > 0 else 1
                origin = float(view.pre.b[d]); target = float(rhs[local]); updated = target+sign*origin
                if not all(map(math.isfinite,(origin,target,updated))) or F(updated) != F(target)+sign*F(origin):
                    raise ValueError('unit RHS update not exact')
                pool.charge('consumer_tail_read',4*(b-a-1))
                tail = tuple(map(int,matrix.indices[a+1:b]))
                plans.append(Plan(col,d,r,kind,u,v,pivot,sign,origin,consumer_main,tail))
                stats['selected_old_consumers' if old else 'selected_new_consumers'] += 1
                stats['tail_terms'] += len(tail)
            offset += matrix.shape[0]
    n = len(plans)
    pool.charge('consumer_plan_sort_and_disjointness',4*n*max(1,(n-1).bit_length())+16*n)
    plans.sort(key=lambda p:p.column)
    definitions = {p.definition for p in plans}
    if (len({p.column for p in plans}) != n or len(definitions) != n
            or len({(p.inequality,p.consumer) for p in plans}) != n
            or any(not p.inequality and p.consumer in definitions for p in plans)):
        raise ValueError('selected unit plans are not a simultaneous independent population')
    stats.update(all_selected_plans=n,no_dense_UID_or_MAIN_oracle_built=True,
        all_original_coefficients_scanned=False,source_proof_required=True,
        live_admission_certificate=False,formal_gain=0,aggregate_post_CSR_built=False)
    return plans,stats
