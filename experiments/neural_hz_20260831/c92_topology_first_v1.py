"""Conditional dense-circuit upper bill; no numerical or source proof receipt."""
import numpy as np
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T,R


def upper_bill(parent_ids,output_ids,direct_nnz,*,pool,enabled=False):
    if not enabled:return None
    ids,out=np.asarray(parent_ids),np.asarray(output_ids)
    if (ids.ndim!=3 or ids.shape[1:]!=(4,4) or out.ndim!=3 or out.shape[1:]!=(2,2)
        or ids.dtype.kind not in 'iu' or out.dtype.kind not in 'iu'):
        raise ValueError('complete integer input/output tile maps required')
    c,k=len(ids),len(out)
    pool.charge('c92_complete_topology_mask_degree_bound',1024+512*c+256*k)
    if (np.any(ids < -1) or np.any(out < -1) or type(direct_nnz) is not int
        or direct_nnz < int((out>=0).sum())):raise ValueError('invalid complete tile/direct count')
    parents,outputs=ids[ids>=0],out[out>=0]
    if (len(np.unique(parents))!=len(parents) or len(np.unique(outputs))!=len(outputs)
        or np.intersect1d(parents,outputs).size):
        return dict(topology_qualified=False,reason='source aliases require an independent count',numeric_admission=False)
    imap=(np.kron(T,T)!=0).astype(np.int64)
    omap=(np.kron(R,R)!=0).astype(np.int64)
    sizes=(ids.reshape(c,16)>=0).astype(np.int64)@imap.T
    want=(out.reshape(k,4)>=0).astype(np.int64)
    users=want@omap
    channels=np.count_nonzero(sizes,axis=0)
    keep_m=(users>1)&(channels[None,:]>1)
    uses=np.where(channels[None,:]>0,np.where(keep_m,1,users),0).sum(axis=0)
    keep_v=(sizes>1)&(uses[None,:]>1)
    widths=np.where(sizes>0,np.where(keep_v,1,sizes),0).sum(axis=0)
    nv,nm=int(keep_v.sum()),int(keep_m.sum())
    vnnz=int(np.where(keep_v,sizes+1,0).sum())
    mnnz=int(np.where(keep_m,widths[None,:]+1,0).sum())
    # Each demanded output has one pivot; the dense transform terms cannot
    # exceed this expansion even if exact word coalescence cancels some of them.
    onnz=int(want.sum()+(np.where(keep_m,1,widths[None,:])*users).sum())
    nnz=vnnz+mnnz+onnz;aux=nv+nm;outputs=int(want.sum());rows=aux+outputs
    old_bytes=12*direct_nnz+16*outputs+8
    new_bytes=12*nnz+88*aux+32*outputs+72
    delta=2*(nnz-direct_nnz)+13*aux+3*outputs+8
    bill=dict(new_factors=aux,rows=rows,output_rows=outputs,nnz_upper=nnz,
        direct_nnz=direct_nnz,nnz_saving_lower=direct_nnz-nnz,
        declared_old_bytes=old_bytes,declared_new_bytes_upper=new_bytes,
        byte_saving_lower=old_bytes-new_bytes,entry_delta_upper=delta,
        new_emission_work_upper=64*rows+16*(nnz+rows),kept_v=nv,kept_m=nm)
    return dict(topology_qualified=direct_nnz>nnz and old_bytes>new_bytes and delta<0,
        bill=bill,numeric_admission=False,dense_transform_precondition_unproved=True,
        auxiliary_count_conditional_exact=True,formal_gain=0)


def select(records,*,existing_aux,existing_work,existing_entries,pool,enabled=False):
    if not enabled:return None
    if not (0<=existing_aux<=16384 and 0<=existing_work<=16_000_000 and 0<=existing_entries<=131072):
        raise MemoryError('existing whole reserve exhausted')
    pool.charge('c92_complete_budget_order',64*len(records)*(1+max(1,len(records)).bit_length()))
    order=[i for i,r in enumerate(records) if r['cost'].get('topology_qualified')]
    order.sort(key=lambda i:(-records[i]['cost']['bill']['byte_saving_lower'],
                             records[i]['cost']['bill']['new_factors'],i))
    aux,work,entries=existing_aux,existing_work,existing_entries;chosen=[]
    for i in order:
        b=records[i]['cost']['bill']
        if (aux+b['new_factors']<=16384 and work+b['new_emission_work_upper']<=16_000_000
            and entries+max(0,b['entry_delta_upper'])<=131072):
            chosen.append(i);aux+=b['new_factors'];work+=b['new_emission_work_upper']
            entries+=max(0,b['entry_delta_upper'])
    return dict(selected_positions=chosen,ordered_positions=order,whole_auxiliary_reserve_used=aux,
        whole_emission_work_upper=work,whole_positive_entry_reserve_used=entries,
        requires_complete_fresh_dense_and_native_proof=True,numeric_admission=False,formal_gain=0)
