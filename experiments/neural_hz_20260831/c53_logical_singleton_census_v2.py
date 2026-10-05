"""Census all source-bound logical MAIN rows; never a source writer/adapter."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import math
import struct
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c9_radix_predicate_audit_v1 import row,recover_row,dyadic_power


def assess(saved,*,pool,enabled=False,observe=None):
    if not enabled:return None
    hz=saved['hz'];nodes=saved['definition_graph'];expr=saved['expression']
    old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);old_eq=int(saved['old_n_eq']);root_id=saved['root']
    main=logical-old;eq_roots=saved['eq_roots'];defs=saved['def_rows']
    if not 0<=old<=logical<=hz.n_cont<=64_000_000 or not 0<=root_id<len(nodes):raise ValueError('complete original frame required')
    if hz.n_bin!=saved['old_n_bin'] or hz.frame_id!=expr.frame_id or not hz.exact:raise ValueError('original binary/shared frame changed')
    nnz=sum(getattr(hz,n).nnz for n in ('Ac','Ab','Auc','Aub'))
    if nnz+logical>64_000_000:raise MemoryError('unchanged complete entry cap')
    pool.charge('c53v2_complete_stored_coefficients_frame_and_UID_scan',8*int(nnz)+12*logical+32*hz.n_eq+1024)
    if (eq_roots.shape!=(old_eq+main,) or saved['eq_scales'].shape!=eq_roots.shape
            or hz.n_cont!=logical+len(defs) or len(eq_roots)+len(defs)!=hz.n_eq
            or not np.array_equal(np.sort(np.r_[eq_roots,defs]),np.arange(hz.n_eq))
            or not np.array_equal(np.sort(saved['ineq_roots']),np.arange(hz.n_ineq))):
        raise ValueError('complete MAIN/radix/old-predicate partition changed')
    for matrix in (hz.Ac,hz.Ab,hz.Auc,hz.Aub):
        if (type(matrix) is not sp.csr_matrix or matrix.dtype!=np.dtype(np.float64)
                or not matrix.has_canonical_format or not np.isfinite(matrix.data).all()
                or np.any(matrix.data==0) or np.any(np.abs(matrix.data)<2.**-20)
                or np.any(np.abs(matrix.data)>2.**40)):
            raise ValueError('unchanged canonical stored coefficient window required')
    paths=[];frontier=old;source_positions=[]
    for index,node in enumerate(nodes):
        if node['kind'] not in ('source','op','sum') or any(not 0<=p<index for p in node['parents']):raise ValueError('original source birth order required')
        rows=np.flatnonzero(node['needed'])
        if (np.any(node['needed'] & ~node['support']) or not np.array_equal(node['slots'][rows],np.arange(frontier,frontier+len(rows)))):
            raise ValueError('original logical slot mapping differs')
        source_positions.append(rows);frontier+=len(rows)
        if node['kind']=='source':expanded=[(id(node['source']),())]
        elif node['kind']=='op':
            if len(node['parents'])!=1:raise ValueError('original operator arity differs')
            expanded=[(s,(*ops,id(node['op']))) for s,ops in paths[node['parents'][0]]]
        else:expanded=[term for p in node['parents'] for term in paths[p]]
        if len(expanded)>len(expr.terms):raise ValueError('extra original source path')
        paths.append(expanded)
    if frontier!=logical or Counter(paths[root_id])!=Counter((id(t.source),tuple(id(op) for op in t.operators)) for t in expr.terms):
        raise ValueError('complete source/operator identity differs')
    root=nodes[root_id];rows=np.flatnonzero(saved['keep'] & root['support'])
    if not np.array_equal(root['needed'],saved['keep'] & root['support']):raise ValueError('complete output liveness differs')
    slots=root['slots'][rows]
    expected=sp.csr_matrix((np.ldexp(np.ones(len(rows)),root['exponents'][rows]),(rows,slots)),shape=hz.Gc.shape)
    if (any(not np.array_equal(getattr(expected,n),getattr(hz.Gc,n)) for n in ('data','indices','indptr'))
            or hz.Gb.nnz or not np.array_equal(expr.bias,hz.c)):
        raise ValueError('original output map differs')
    protected=np.zeros(logical,bool);protected[slots]=True
    roots=np.arange(logical,dtype=np.int64);shifts=np.zeros(logical,np.int64);signs=np.ones(logical,np.int8);depths=np.zeros(logical,np.int32)
    holder=SimpleNamespace(original=SimpleNamespace(n_cont=logical,n_bin=hz.n_bin),hz=hz,
        eq_roots=eq_roots,eq_scales=saved['eq_scales'],def_rows=defs)
    counts=Counter();per_node=[];ratio_hist=Counter();depth_hist=Counter();chain_hist=Counter();sign_hist=Counter();digest=hashlib.sha256();recovered=set()
    for index,(node,coordinates) in enumerate(zip(nodes,source_positions)):
        local=Counter()
        for coordinate in coordinates:
            slot=int(node['slots'][coordinate]);logical_row=old_eq+slot-old;physical=int(eq_roots[logical_row])
            cc,cv=row(hz.Ac,physical);bc,bv=row(hz.Ab,physical);rhs=float(hz.b[physical])
            packed=bool(len(cc) and cc[-1]>=logical)
            if packed:
                # All packed logical MAIN definitions, not merely likely hits.
                pool.charge('c53v2_complete_packed_MAIN_recovery',16*(len(cc)+len(bc))+128)
                continuous,binary,rhs,reached=recover_row(holder,logical_row)
                if reached & recovered:raise ValueError('unregistered shared radix subtree')
                recovered.update(reached)
                cc=np.array(sorted(continuous),np.int64);cv=np.array([continuous[int(c)] for c in cc])
                bc=np.array(sorted(binary),np.int64);bv=np.array([binary[int(c)] for c in bc])
                pool.charge('c53v2_complete_recovered_logical_coefficients',16*(len(cc)+len(bc))+128*len(reached))
                local['packed_MAIN_rows_recovered']+=1
            if (not len(cc) or cc[-1]!=slot or np.any(cc[:-1]>=slot) or np.any(cc<0)
                    or np.any(np.diff(cc)<=0) or cv[-1]<=0):raise ValueError('logical MAIN definition is not canonical/topological')
            pivot_power=dyadic_power(cv[-1])
            # When packed, coefficients were fully unscaled. Otherwise use
            # the stored uniform row scale; it cancels in every ratio.
            expected_pivot=int(node['exponents'][coordinate])+(0 if packed else int(saved['eq_scales'][logical_row]))
            if pivot_power!=expected_pivot:raise ValueError('source exponent and logical pivot differ')
            pool.charge('c53v2_per_MAIN_classification',64)
            local['MAIN_rows']+=1;local['output_live']+=int(protected[slot]);local['binary_bearing']+=int(len(bc)>0)
            local['nonzero_offset']+=int(rhs!=0.);local['multiple_continuous_parents']+=int(len(cc)>2)
            homogeneous=len(cc)==2 and not len(bc) and rhs==0.
            if homogeneous:
                local['homogeneous_singleton']+=1;pool.charge('c53_exact_singleton_Fraction_identity_and_chain_record',128)
                ratio=-F(float(cv[0]))/F(float(cv[-1]))
                if not 0<abs(ratio)<=1:raise ValueError('singleton box not redundant')
                local['strictly_contracting_singleton']+=int(abs(ratio)<1)
                if protected[slot]:local['output_live_singleton_kept']+=1
                else:
                    local['all_output_dead_singletons']+=1
                    local['raw_C52_signed_unit']+=int(abs(ratio)==1)
                mantissa,exponent=math.frexp(abs(float(cv[0])))
                if mantissa!=.5:local['non_power_two_singleton']+=1
                else:
                    local['power_two_singleton']+=1
                    if protected[slot]:local['power_two_output_live_kept']+=1
                    else:
                        local['power_two_output_dead_candidate']+=1
                        sign=1 if ratio>0 else -1;shift=exponent-1-pivot_power;parent=int(cc[0])
                        if ratio!=sign*F(2)**shift:raise ValueError('exact exponent identity differs')
                        roots[slot]=roots[parent];shifts[slot]=shift+int(shifts[parent]);signs[slot]=sign*int(signs[parent]);depths[slot]=1+int(depths[parent])
                        ratio_hist[str(shift)]+=1;depth_hist[str(int(depths[slot]))]+=1;chain_hist[str(int(shifts[slot]))]+=1;sign_hist[str(sign)]+=1
                        digest.update(struct.pack('=8q',index,int(coordinate),slot,parent,sign,shift,int(roots[slot]),int(shifts[slot])))
        counts.update(local);item=dict(node=index,kind=node['kind'],operator_type=type(node['op']).__name__ if node['kind']=='op' else None,counts=dict(local));per_node.append(item)
        if observe:observe(dict(event='complete_logical_node_censused',**item))
    if counts['MAIN_rows']!=main:raise ValueError('incomplete original MAIN census')
    return dict(schema='c53_complete_source_bound_logical_singleton_census_v2',totals=dict(counts),per_node=per_node,
        old_n_cont=old,logical_n_cont=logical,unchanged_n_binary=hz.n_bin,
        current_C52_signed_unit_population=0 if counts['raw_C52_signed_unit']==0 else None,
        no_first_merge_proved=counts['raw_C52_signed_unit']==0,
        potential_scale_aware_singleton_only_lower_bound=counts['power_two_output_dead_candidate'],
        normalized_power_histogram=dict(ratio_hist),potential_chain_depth_histogram=dict(depth_hist),
        potential_chain_power_histogram=dict(chain_hist),potential_local_sign_histogram=dict(sign_hist),
        potential_identity_sha256=digest.hexdigest(),radix_definitions_recovered_for_MAIN=len(recovered),
        complete_original_source_paths_frame_and_output_maps_checked=True,
        original_coefficient_and_box_binding_inherited_not_recomputed=True,
        dense_original_operator_rows_or_normalization_recomputed=False,
        actual_forwarding_or_new_HZ_constructed=False,whole_request_LIVE_or_physical_reduction_proved=False,formal_gain=0)
