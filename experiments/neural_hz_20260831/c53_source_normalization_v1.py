"""Read-only original logical-source census; no HZ writer or solver admission."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import math
import struct
import numpy as np
import scipy.sparse as sp
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c7_factored_hz_v1 import restricted_row, _source_row


def strict_envelope(values, powers=0):
    values=np.asarray(values,np.float64)
    if not 0<len(values)<=64_000_000 or np.any(values==0) or not np.isfinite(values).all():
        raise ValueError('complete finite nonzero source summands required')
    exponents=np.frexp(np.abs(values))[1].astype(np.int64)+powers
    floor=int(exponents.max())-26
    total=int(np.left_shift(np.ones(len(values),np.int64),np.maximum(exponents-floor,0)).sum(dtype=np.int64))
    unit=max(0,floor+(total-1).bit_length())
    if not 0<=unit<=1023:raise ValueError('unchanged finite source box window')
    return unit


def scalar_identity(value,power,unit):
    """Independent exact rational check, no rounded coefficient products."""
    value=float(value);mantissa,exponent=math.frexp(abs(value))
    if not math.isfinite(value) or not value or mantissa!=.5:return None
    shift=exponent-1+int(power)-int(unit);sign=1 if value>0 else -1
    exact=F(value)*F(2)**int(power)/F(2)**int(unit)
    if exact!=sign*F(2)**shift or not abs(exact)<1:
        raise ValueError('strict normalized singleton identity failed')
    return sign,shift


def assess(saved,*,pool,enabled=False,observe=None):
    if not enabled:return None
    nodes=saved['definition_graph'];expr=saved['expression'];hz=saved['hz']
    old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);root_id=saved['root']
    if not 0<=old<=logical<=64_000_000 or not 0<=root_id<len(nodes):raise ValueError('bounded original MAIN frame required')
    if hz.n_bin!=saved['old_n_bin'] or hz.frame_id!=expr.frame_id:raise ValueError('old nonconvex frame changed')
    pool.charge('c53_complete_MAIN_frame_paths_and_chain_metadata',12*logical+128*len(nodes)+1024)
    paths=[];frontier=old
    for index,node in enumerate(nodes):
        if node['kind'] not in ('source','op','sum') or any(not 0<=p<index for p in node['parents']):
            raise ValueError('complete original topological source graph required')
        width=node['width']
        if any(node[n].shape!=(width,) for n in ('slots','exponents','needed','support')):
            raise ValueError('complete graph coordinate maps required')
        if node['needed'].dtype!=np.dtype(bool) or np.any(node['needed'] & ~node['support']):raise ValueError('invalid original source liveness')
        rows=np.flatnonzero(node['needed']);slots=node['slots'][rows]
        if not np.array_equal(slots,np.arange(frontier,frontier+len(rows))):raise ValueError('logical birth order differs from original source')
        frontier+=len(rows)
        if node['kind']=='source':
            if node['parents']:raise ValueError('source unexpectedly has parents')
            expanded=[(id(node['source']),())]
        elif node['kind']=='op':
            if len(node['parents'])!=1:raise ValueError('operator arity changed')
            expanded=[(s,(*ops,id(node['op']))) for s,ops in paths[node['parents'][0]]]
        else:expanded=[term for p in node['parents'] for term in paths[p]]
        if len(expanded)>len(expr.terms):raise ValueError('unexpected source/operator paths')
        paths.append(expanded)
    if frontier!=logical or Counter(paths[root_id])!=Counter((id(t.source),tuple(id(op) for op in t.operators)) for t in expr.terms):
        raise ValueError('complete original expression binding differs')
    root=nodes[root_id];rows=np.flatnonzero(saved['keep'] & root['support'])
    if not np.array_equal(root['needed'],saved['keep'] & root['support']):raise ValueError('complete output consumers changed')
    slots=root['slots'][rows]
    expected=sp.csr_matrix((np.ldexp(np.ones(len(rows)),root['exponents'][rows]),(rows,slots)),shape=hz.Gc.shape)
    if (any(not np.array_equal(getattr(expected,n),getattr(hz.Gc,n)) for n in ('data','indices','indptr'))
            or hz.Gb.nnz or not np.array_equal(expr.bias,hz.c)):
        raise ValueError('complete original output map or bias differs')
    protected=np.zeros(logical,bool);protected[slots]=True
    roots=np.arange(logical,dtype=np.int64);shifts=np.zeros(logical,np.int64)
    signs=np.ones(logical,np.int8);depths=np.zeros(logical,np.int32)
    totals=Counter();ratio_hist=Counter();depth_hist=Counter();chain_shift_hist=Counter();per_node=[]
    digest=hashlib.sha256();edges=0
    for index,node in enumerate(nodes):
        counts=Counter();kind=node['kind']
        for coordinate in np.flatnonzero(node['needed']):
            slot=int(node['slots'][coordinate]);unit=int(node['exponents'][coordinate]);constant=0.
            bv=np.empty(0,np.float64);bc=np.empty(0,np.int64)
            if kind=='source':
                source=node['source'];(cc,cv),(bc,bv)=_source_row(source,coordinate)
                constant=float(source.c[coordinate]);powers=np.zeros(len(cv),np.int64)
                summands=np.concatenate((cv,bv,[constant] if constant else []));bound_powers=0
                scanned=int(source.Gc.indptr[coordinate+1]-source.Gc.indptr[coordinate]+source.Gb.indptr[coordinate+1]-source.Gb.indptr[coordinate])
            elif kind=='op':
                parent=nodes[node['parents'][0]];op=node['op']
                if type(op) is sp.csr_matrix:scanned=int(op.indptr[coordinate+1]-op.indptr[coordinate])
                elif type(op) is ImplicitConv2DOp:scanned=int(op._kernel[0].size)
                else:raise ValueError('unregistered original operator')
                pool.charge('c53_original_operator_geometry_scan',8*scanned)
                coords,cv=restricted_row(op,coordinate,parent['needed'])
                cc=parent['slots'][coords];powers=parent['exponents'][coords].astype(np.int64)
                summands=cv;bound_powers=powers
            else:
                terms=[(nodes[p]['slots'][coordinate],nodes[p]['exponents'][coordinate],n)
                    for p,n in sorted(Counter(node['parents']).items()) if nodes[p]['support'][coordinate]]
                cc=np.array([x[0] for x in terms],np.int64);cv=np.array([x[2] for x in terms],np.float64)
                powers=np.array([x[1] for x in terms],np.int64);summands=cv;bound_powers=powers;scanned=len(node['parents'])
            if kind!='op':pool.charge('c53_original_source_or_sum_scan',8*scanned)
            pool.charge('c53_complete_row_envelope_and_census',4*(len(cv)+len(bv))+64)
            if (np.any(cc<0) or np.any(cc>=slot) or np.any(np.diff(cc)<=0)
                    or len(powers)!=len(cc) or np.any(bc<0) or np.any(bc>=hz.n_bin)):
                raise ValueError('original row is not uniquely defined over older shared factors')
            if strict_envelope(summands,bound_powers)!=unit:raise ValueError('saved source normalization differs from strict envelope')
            counts['MAIN_rows']+=1;counts['strict_L1_below_pivot']+=1;counts['output_live']+=int(protected[slot])
            counts['binary_bearing']+=int(len(bv)>0);counts['nonzero_offset']+=int(constant!=0.)
            counts['multiple_continuous_parents']+=int(len(cc)>1)
            edges+=len(cv)+len(bv)
            if edges+logical>64_000_000:raise MemoryError('unchanged source entry cap')
            homogeneous=len(cc)==1 and not len(bv) and constant==0.
            if homogeneous:
                counts['homogeneous_singleton']+=1
                pool.charge('c53_exact_singleton_Fraction_identity_and_chain_record',128)
                identity=scalar_identity(cv[0],powers[0],unit)
                if identity is None:counts['non_power_two_singleton']+=1
                else:
                    counts['power_two_singleton']+=1
                    if protected[slot]:counts['power_two_output_live_kept']+=1
                    else:
                        counts['power_two_output_dead_candidate']+=1
                        sign,shift=identity;parent=int(cc[0])
                        roots[slot]=roots[parent];signs[slot]=sign*int(signs[parent]);shifts[slot]=shift+int(shifts[parent]);depths[slot]=1+int(depths[parent])
                        ratio_hist[str(shift)]+=1;depth_hist[str(int(depths[slot]))]+=1;chain_shift_hist[str(int(shifts[slot]))]+=1
                        digest.update(struct.pack('=8q',index,int(coordinate),slot,parent,sign,shift,int(roots[slot]),int(shifts[slot])))
            if homogeneous and not protected[slot]:counts['all_output_dead_singletons']+=1
        totals.update(counts)
        item=dict(node=index,kind=kind,operator_type=type(node['op']).__name__ if kind=='op' else None,counts=dict(counts))
        per_node.append(item)
        if observe:observe(dict(event='original_logical_node_censused',**item))
    if totals['MAIN_rows']!=logical-old:raise ValueError('a source MAIN row was omitted')
    return dict(schema='c53_complete_original_source_normalization_census_v1',totals=dict(totals),per_node=per_node,
        old_n_cont=old,logical_n_cont=logical,unchanged_n_binary=hz.n_bin,source_edges_scanned=edges,
        complete_original_source_operator_paths_checked=True,complete_output_maps_checked=True,
        all_MAIN_strict_L1_below_pivot=True,current_C52_signed_unit_population=0,
        current_C52_shared_coalescing_units=0,zero_population_reason='strict_envelope_prevents_first_signed_unit_merge',
        potential_scale_aware_singleton_only_lower_bound=totals['power_two_output_dead_candidate'],
        normalized_power_histogram=dict(ratio_hist),potential_chain_depth_histogram=dict(depth_hist),
        potential_chain_power_histogram=dict(chain_shift_hist),potential_identity_sha256=digest.hexdigest(),
        potential_map_numeric_bytes=roots.nbytes+shifts.nbytes+signs.nbytes+depths.nbytes,
        actual_forwarding_or_new_HZ_constructed=False,full_original_matrix_proof_recomputed=False,
        whole_request_LIVE_or_physical_reduction_proved=False,formal_gain=0)
