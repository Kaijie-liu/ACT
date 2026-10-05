"""Raw dense-source conditional plans without physical source-array access."""
from collections import Counter
import numpy as np
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c93_packed_mask_v1 import footprints,upper_bill
from experiments.neural_hz_20260831.c92_topology_first_v1 import select

_PATCH=tuple(sum(1<<(4*(i+a)+j+b) for a in range(3) for b in range(3))
             for i in range(2) for j in range(2))


def dense_direct(input_masks,output_masks,*,pool,enabled=False):
    if not enabled:return None
    a,o=np.asarray(input_masks),np.asarray(output_masks)
    if a.ndim!=1 or o.ndim!=1 or a.dtype!=np.uint16 or o.dtype!=np.uint8:
        raise ValueError('complete uint16/uint8 footprints required')
    pool.charge('c94_raw_direct_histogram_headers',128+12*(len(a)+len(o)))
    if np.any(o>15):raise ValueError('invalid output footprint')
    ac,oc=Counter(map(int,a)),Counter(map(int,o))
    pool.charge('c94_raw_direct_observed_class_count',16*(len(ac)+len(oc)))
    direct=sum(mask.bit_count()*n for mask,n in oc.items())
    for s,patch in enumerate(_PATCH):
        outputs=sum(n for mask,n in oc.items() if (mask>>s)&1)
        inputs=sum((mask&patch).bit_count()*n for mask,n in ac.items())
        direct+=outputs*inputs
    return int(direct)


def raw_plan(nodes,*,existing_aux,existing_work,pool,enabled=False):
    if not enabled:return None
    pool.charge('c93_complete_source_geometry_headers',64*len(nodes));records=[];all_masks={}
    for index,node in enumerate(nodes):
        if node['kind']!='op':continue
        op=node['op']
        if (type(op) is not ImplicitConv2DOp or op._stride!=(1,1) or op._dilation!=(1,1)
            or op._groups!=1 or op._kernel.shape[-2:]!=(3,3)
            or op.input_shape[0]!=1 or op.output_shape[0]!=1):continue
        pool.charge('c93_eligible_original_output_demand_read',int(node['width']))
        if not node['needed'].any():continue
        parent=nodes[node['parents'][0]]
        masks=footprints(parent['needed'].reshape(op.input_shape[1:]),node['needed'].reshape(op.output_shape[1:]),
            tuple(map(int,op._padding)),pool=pool,enabled=True);all_masks[index]=masks
        for t,(y,x) in enumerate(masks['positions'].tolist()):
            a,o=masks['input_masks'][t],masks['output_masks'][t]
            direct=dense_direct(a,o,pool=pool,enabled=True)
            cost=upper_bill(a,o,direct,pool=pool,enabled=True)
            records.append(dict(node=index,y=y,x=x,cost=cost,
                count_scope='conditional_raw_dense_source_not_yet_current_quotient'))
    plan=select(records,existing_aux=existing_aux,existing_work=existing_work,existing_entries=131072,pool=pool,enabled=True)
    return records,plan,all_masks


def selected_source_check(fields,maps,expected_direct,*,pool,enabled=False):
    if not enabled:return None
    ids,outs=maps['ids'],maps['outs'];parents,pivots=ids[ids>=0],outs[outs>=0]
    pool.charge('c94_selected_actual_source_postconditions',128+24*len(parents)+32*len(pivots))
    old,logical,ne=fields['old_n_cont'],fields['logical_n_cont'],fields['old_n_eq'];h=fields['hz']
    if (np.any(parents<old) or np.any(pivots<old) or np.any(parents>=logical) or np.any(pivots>=logical)):
        raise ValueError('selected original MAIN coordinate domain differs')
    pr=fields['eq_roots'][ne+parents-old];rr=fields['eq_roots'][ne+pivots-old]
    if np.any(pr<0) or np.any(rr<0):raise ValueError('selected source scalar quotient changes a coordinate')
    if np.any(h.b[rr]!=0) or np.any(h.Ab.indptr[rr+1]!=h.Ab.indptr[rr]):
        raise ValueError('selected original offset or binary predicate cannot be replaced')
    direct=int((h.Ac.indptr[rr+1]-h.Ac.indptr[rr]).sum())
    if direct!=expected_direct:raise ValueError('selected actual current source differs from raw dense count')
    return dict(all_selected_original_coordinates_survive=True,actual_current_direct_nnz=direct,
        original_zero_RHS_and_binary_rows_preserved=True,numeric_or_LIVE_admission=False)
