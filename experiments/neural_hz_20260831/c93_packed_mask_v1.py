"""Exact bounded source footprints and dense-circuit histogram cardinalities."""
from collections import Counter
import numpy as np
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T,R
from experiments.neural_hz_20260831.c92_topology_first_v1 import select

_INPUT=tuple(sum(1<<i for i,v in enumerate(row) if v) for row in np.kron(T,T))
_OUTPUT=tuple(sum(1<<i for i,v in enumerate(col) if v) for col in np.kron(R,R).T)


def _pack_rows(mask,*,pool):
    c,h,w=mask.shape
    pool.charge('c93_complete_boolean_row_packing',3*int(mask.size)+16*c*h)
    raw=np.zeros((c,h,8),np.uint8)
    raw[:,:,:((w+7)//8)]=np.packbits(mask,axis=-1,bitorder='little')
    return raw.view('<u8')[...,0]


def footprints(parent_mask,output_mask,padding,*,pool,enabled=False):
    if not enabled:return None
    a,o=np.asarray(parent_mask),np.asarray(output_mask)
    if (a.dtype!=np.dtype(bool) or o.dtype!=np.dtype(bool) or a.ndim!=3 or o.ndim!=3
        or any(not 1<=v<=64 for v in (*a.shape[1:],*o.shape[1:]))
        or len(padding)!=2 or any(type(v) is not int or v<0 for v in padding)):
        raise ValueError('complete ordinary bounded boolean spatial masks required')
    c,h,w=a.shape;k,oh,ow=o.shape;py,px=padding
    if oh!=h+2*py-2 or ow!=w+2*px-2:raise ValueError('complete original stride1 padded geometry differs')
    n=((oh+1)//2)*((ow+1)//2)
    pool.charge('c93_complete_packed_tile_extraction',n*(128+24*c+16*k))
    ar,orr=_pack_rows(a,pool=pool),_pack_rows(o,pool=pool)
    ins=np.zeros((n,c),np.uint16);outs=np.zeros((n,k),np.uint8);positions=[]
    for p,(y,x) in enumerate((y,x) for y in range(0,oh,2) for x in range(0,ow,2)):
        positions.append((y,x));start=max(0,x-px);stop=min(w,x-px+4)
        if start<stop:
            window=(1<<(stop-start))-1;horizontal=start-(x-px)
            for i in range(4):
                sy=y-py+i
                if 0<=sy<h:
                    word=((ar[:,sy]>>start)&window)<<(4*i+horizontal)
                    ins[p]|=word.astype(np.uint16)
        width=min(2,ow-x)
        for i in range(min(2,oh-y)):
            word=((orr[:,y+i]>>x)&((1<<width)-1))<<(2*i)
            outs[p]|=word.astype(np.uint8)
    return dict(input_masks=ins,output_masks=outs,positions=np.asarray(positions,np.int32))


def upper_bill(input_masks,output_masks,direct_nnz,*,pool,enabled=False):
    if not enabled:return None
    ins,outs=np.asarray(input_masks),np.asarray(output_masks)
    if ins.ndim!=1 or outs.ndim!=1 or ins.dtype!=np.uint16 or outs.dtype!=np.uint8:
        raise ValueError('complete uint16/uint8 tile footprints required')
    pool.charge('c93_complete_mask_histogram_headers',512+12*(len(ins)+len(outs)))
    if np.any(outs>15) or type(direct_nnz) is not int:raise ValueError('invalid exact output footprint/direct count')
    ic,oc=Counter(map(int,ins)),Counter(map(int,outs))
    pool.charge('c93_observed_mask_class_degrees',64*(len(ic)+len(oc)))
    output_rows=sum(mask.bit_count()*count for mask,count in oc.items())
    if direct_nnz<output_rows:raise ValueError('complete direct count omits original pivots')
    nv=nm=vnnz=mnnz=terms=0
    for pm,om in zip(_INPUT,_OUTPUT,strict=True):
        A=B=S=0
        for mask,count in ic.items():
            size=(mask&pm).bit_count();A+=count*(size>0);B+=count*(size>1);S+=count*size
        U=V=W=active=0
        for mask,count in oc.items():
            uses=(mask&om).bit_count();U+=count*uses;active+=count*(uses>0)
            V+=count*(uses>1);W+=count*uses*(uses>1)
        uses=active if A>1 else U if A else 0
        retained=uses>1
        nv+=B if retained else 0;vnnz+=S-A+2*B if retained else 0
        width=A if retained else S
        nm+=V if A>1 else 0;mnnz+=V*(width+1) if A>1 else 0
        terms+=width*U+(1-width)*W if A>1 else width*U
    nnz=vnnz+mnnz+terms+output_rows;aux=nv+nm;rows=aux+output_rows
    old_bytes=12*direct_nnz+16*output_rows+8;new_bytes=12*nnz+88*aux+32*output_rows+72
    delta=2*(nnz-direct_nnz)+13*aux+3*output_rows+8
    b=dict(new_factors=aux,rows=rows,output_rows=output_rows,nnz_upper=nnz,direct_nnz=direct_nnz,
        nnz_saving_lower=direct_nnz-nnz,declared_old_bytes=old_bytes,declared_new_bytes_upper=new_bytes,
        byte_saving_lower=old_bytes-new_bytes,entry_delta_upper=delta,
        new_emission_work_upper=64*rows+16*(nnz+rows),kept_v=nv,kept_m=nm)
    return dict(topology_qualified=direct_nnz>nnz and old_bytes>new_bytes and delta<0,bill=b,
        numeric_admission=False,dense_transform_precondition_unproved=True,
        auxiliary_count_conditional_exact=True,formal_gain=0)


def source_plan(nodes,fields,*,pool,enabled=False):
    if not enabled:return None
    pool.charge('c93_complete_source_geometry_headers',64*len(nodes))
    records=[];all_masks={};hz=fields['hz'];old=fields['old_n_cont'];ne=fields['old_n_eq']
    for index,node in enumerate(nodes):
        if node['kind']!='op':continue
        op=node['op']
        if (type(op) is not ImplicitConv2DOp or op._stride!=(1,1) or op._dilation!=(1,1)
            or op._groups!=1 or op._kernel.shape[-2:]!=(3,3) or op.input_shape[0]!=1
            or op.output_shape[0]!=1):continue
        pool.charge('c93_eligible_original_output_demand_read',int(node['width']))
        if not node['needed'].any():continue
        parent=nodes[node['parents'][0]];_,c,h,w=op.input_shape;_,k,oh,ow=op.output_shape
        if any(not 1<=v<=64 for v in (h,w,oh,ow)):raise ValueError('eligible actual spatial geometry outside packed ordinary domain')
        pool.charge('c93_complete_source_mask_counts',parent['width']+node['width'])
        pa_count=int(parent['needed'].sum());oa_count=int(node['needed'].sum())
        pool.charge('c93_complete_source_scalar_direct_routing',128+3*parent['width']+5*node['width']+16*pa_count+32*oa_count)
        pa=np.flatnonzero(parent['needed']);oa=np.flatnonzero(node['needed'])
        pi=parent['slots'][pa];oi=node['slots'][oa]
        if (np.any(pi<old) or np.any(oi<old) or np.any(pi>=fields['logical_n_cont'])
            or np.any(oi>=fields['logical_n_cont'])):raise ValueError('complete original MAIN coordinate domain differs')
        pr=fields['eq_roots'][ne+pi-old];rr=fields['eq_roots'][ne+oi-old]
        badp=np.zeros(parent['width'],bool);bado=np.zeros(node['width'],bool);direct=np.zeros(node['width'],np.int64)
        badp[pa]=pr<0;bado[oa]=rr<0;good=rr>=0;roots=rr[good];coords=oa[good]
        bado[coords]|=(hz.b[roots]!=0)|(hz.Ab.indptr[roots+1]!=hz.Ab.indptr[roots])
        direct[coords]=hz.Ac.indptr[roots+1]-hz.Ac.indptr[roots]
        badp=badp.reshape(c,h,w).any(axis=0);bado=bado.reshape(k,oh,ow).any(axis=0)
        direct=direct.reshape(k,oh,ow).sum(axis=0)
        masks=footprints(parent['needed'].reshape(c,h,w),node['needed'].reshape(k,oh,ow),
            tuple(map(int,op._padding)),pool=pool,enabled=True);all_masks[index]=masks;py,px=op._padding
        for t,(y,x) in enumerate(masks['positions'].tolist()):
            item=dict(node=index,y=y,x=x)
            alias=badp[max(0,y-py):min(h,y-py+4),max(0,x-px):min(w,x-px+4)].any()
            invalid=bado[y:y+2,x:x+2].any()
            if alias or invalid:
                item['cost']=dict(topology_qualified=False,reason='actual original source precondition not established')
            else:
                item['cost']=upper_bill(masks['input_masks'][t],masks['output_masks'][t],
                    int(direct[y:y+2,x:x+2].sum()),pool=pool,enabled=True)
            records.append(item)
    plan=select(records,existing_aux=len(fields['def_rows']),existing_work=fields['report']['actual_radix_work'],
        existing_entries=131072,pool=pool,enabled=True)
    return records,plan,all_masks
