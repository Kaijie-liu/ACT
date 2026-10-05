"""Full pre-quotient source census for a paid empty-span routing proposal.

Reads every original row and independently checks every proposed miss. It
does not rerun a failed generator, change a predicate or issue a new proof.
"""
import hashlib
import math
import struct
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c48_empty_alias_span_v1 import index
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def assess(hz,*,old_nc,logical_nc,old_eq,eq_roots,def_rows,output_slots,pool,enabled=False):
    if not enabled:return None
    if any(type(v) is not int or v<0 for v in (old_nc,logical_nc,old_eq)) or not old_nc<=logical_nc<=hz.n_cont:
        raise ValueError('complete original pre-quotient global frame required')
    matrices=(hz.Ac,hz.Ab,hz.Auc,hz.Aub)
    nnz=sum(int(m.nnz) for m in matrices);main=logical_nc-old_nc
    if nnz>64_000_000 or hz.n_cont>64_000_000:raise MemoryError('unchanged source entry cap')
    # Includes full numeric/canonical/frame qualification, original alias
    # algebra, and BOTH complete HZ content seals; no per-hit numeric rescue.
    pool.charge('c48_complete_original_geometry_and_two_source_seals',12*nnz+64*main+4096)
    if any(type(m) is not sp.csr_matrix or m.dtype!=np.float64 or m.indices.dtype!=np.int32
           or m.indptr.dtype!=np.int32 for m in (hz.Gc,hz.Gb,*matrices)):
        raise ValueError('complete f64/i32 original CSR source required')
    if (type(eq_roots) is not np.ndarray or eq_roots.dtype!=np.int64 or eq_roots.shape!=(old_eq+main,)
            or type(def_rows) is not np.ndarray or def_rows.dtype!=np.int64 or def_rows.ndim!=1
            or logical_nc+len(def_rows)!=hz.n_cont or not 0<=old_eq<=hz.n_eq):
        raise ValueError('complete original MAIN/radix mapping required')
    partition=np.r_[eq_roots,def_rows]
    if (len(partition)!=hz.n_eq or np.any(partition<0) or np.any(partition>=hz.n_eq)
            or np.any(np.bincount(partition,minlength=hz.n_eq)!=1)):
        raise ValueError('incomplete original row partition')
    output=np.asarray(output_slots)
    if (output.ndim!=1 or output.dtype.kind not in 'iu' or np.any(output<0) or np.any(output>=logical_nc)
            or not np.array_equal(np.unique(output),np.unique(hz.Gc.indices))):
        raise ValueError('complete original protected output slots required')
    for m,rows,cols in ((hz.Gc,hz.n_out,hz.n_cont),(hz.Gb,hz.n_out,hz.n_bin),
        (hz.Ac,hz.n_eq,hz.n_cont),(hz.Ab,hz.n_eq,hz.n_bin),
        (hz.Auc,hz.n_ineq,hz.n_cont),(hz.Aub,hz.n_ineq,hz.n_bin)):
        if (m.shape!=(rows,cols) or len(m.indptr)!=rows+1 or m.indptr[0]!=0
                or m.indptr[-1]!=len(m.data) or len(m.indices)!=len(m.data)
                or np.any(m.indptr[1:]<m.indptr[:-1]) or np.any(m.indices<0) or np.any(m.indices>=cols)
                or not np.isfinite(m.data).all() or np.any(m.data==0.)):
            raise ValueError('invalid actual original numeric CSR geometry')
        for row in range(rows):
            a,b=map(int,m.indptr[row:row+2])
            if b-a>1 and np.any(m.indices[a+1:b]<=m.indices[a:b-1]):
                raise ValueError('actual source rows are not canonical')
    before=source_digest(hz)
    protected=np.zeros(logical_nc,bool);protected[output]=True
    local=np.zeros(main,bool);ratios=np.zeros(main,np.float64)
    for i,physical in enumerate(eq_roots[old_eq:]):
        col=old_nc+i;a,b=map(int,hz.Ac.indptr[physical:physical+2])
        if (protected[col] or b-a!=2 or hz.Ab.indptr[physical]!=hz.Ab.indptr[physical+1]
                or hz.b[physical]!=0. or hz.Ac.indices[b-1]!=col):continue
        parent,pivot=int(hz.Ac.indices[a]),float(hz.Ac.data[b-1]);pm,pe=math.frexp(pivot)
        if not 0<=parent<col or pm!=.5:raise ValueError('non-topological/non-dyadic original MAIN pivot')
        ratio=math.ldexp(-float(hz.Ac.data[a]),1-pe)
        if math.ldexp(ratio,pe-1)!=-float(hz.Ac.data[a]):raise ValueError('nonreversible original scalar ratio')
        if 2.**-60<=abs(ratio)<=1.:local[i]=True;ratios[i]=ratio
    lookup=np.full(hz.n_cont,-1,np.int64);selected=np.flatnonzero(local)
    lookup[old_nc+selected]=selected
    own=np.zeros(hz.n_eq,bool);own[eq_roots[old_eq+selected]]=True
    names=('c48_successor_full_construction_seals_and_retirement','c48_successor_known_hit_scatter',
        'c48_uniform_row_width_dispatch','c48_exact_alias_span_query')
    initial_parts={n:pool.parts.get(n,0) for n in names}
    route=index(lookup,pool=pool,enabled=True)
    # Pay the complete independent reference scan, including zero-hit rows.
    cn=int(hz.Ac.nnz+hz.Auc.nnz)
    pool.charge('c48_independent_every_original_lookup_and_hit_scan',2*cn+8*(hz.n_eq+hz.n_ineq))
    reference=hashlib.sha256();candidate=hashlib.sha256();hits=hit_rows=kept_nnz=0
    per_kind=[]
    for kind,matrix in enumerate((hz.Ac,hz.Auc)):
        kind_misses=kind_skipped=kind_hits=0
        for row in range(matrix.shape[0]):
            a,b=map(int,matrix.indptr[row:row+2]);width=b-a
            stop=b-int(kind==0 and own[row]);columns=matrix.indices[a:stop]
            actual=lookup[columns];positions=np.flatnonzero(actual>=0)
            possible=route.may_hit(columns,original_width=width)
            if not possible:
                if len(positions):raise ValueError('range proof skipped a real local-alias occurrence')
                kind_misses+=1;kind_skipped+=width
            else:kept_nnz+=width
            if len(positions):
                n=len(positions);hits+=n;kind_hits+=n;hit_rows+=1
                pool.charge('c48_complete_unchanged_product_input_images',16*n+64)
                parts=[struct.pack('=bqq',kind,row,n),positions.astype(np.int64).tobytes(),
                    actual[positions].tobytes(),matrix.data[a:stop][positions].tobytes(),
                    ratios[actual[positions]].tobytes()]
                for part in parts:reference.update(part)
                if possible:
                    for part in parts:candidate.update(part)
        per_kind.append(dict(inequality=bool(kind),rows=matrix.shape[0],proved_misses=kind_misses,
            skipped_original_scan_work=kind_skipped,all_original_alias_hits=kind_hits))
    routing=route.finish()
    routing_parts={n:pool.parts.get(n,0)-initial_parts[n] for n in names}
    routing_work=sum(routing_parts.values())
    if reference.hexdigest()!=candidate.hexdigest() or source_digest(hz)!=before:
        raise ValueError('complete source or every original product-input image changed')
    baseline=cn;proposed=routing_work+kept_nnz
    return dict(completed=True,local_aliases=len(selected),all_original_hit_rows=hit_rows,all_original_hits=hits,
        all_original_product_input_images_equal=True,complete_product_input_sha256=reference.hexdigest(),
        complete_original_HZ_sha256=before,source_HZ_unchanged=True,all_rows_scanned=hz.n_eq+hz.n_ineq,
        original_continuous_predicate_nnz=cn,per_kind=per_kind,routing=routing,
        original_C31_incidence_scan_work=baseline,new_routing_plus_retained_scan_work=proposed,
        component_work_saving=baseline-proposed,strict_component_work_payment=proposed<baseline,
        new_routing_work=routing_work,new_routing_parts=routing_parts,
        generator_or_native_integration_executed=False,new_source_receipt_issued=False,
        actual_updated_C31_generation_report_proved=False,whole_LIVE_or_runtime_payment_proved=False,
        predicates_changed=False,solver_executed=False,formal_gain=0)
