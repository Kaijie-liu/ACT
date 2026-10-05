"""Exact ideal per-example operator incidence, NOT a current full HZ ledger."""
import numpy as np
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T,R


def count(weights,native,input_shape,output_shape,pads,*,pool):
    if (len(input_shape)!=4 or len(output_shape)!=4 or len(pads)!=4
            or min(*input_shape[1:],*output_shape[1:])<=0):
        return dict(geometry_known=False,complete_HZ_cost_proved=False)
    _,channels,height,width=input_shape;_,outputs,oh,ow=output_shape
    if weights.shape!=(outputs,channels,3,3):raise ValueError('original channel geometry mismatch')
    py,px,_,_=pads;tiles=((oh+1)//2)*((ow+1)//2)
    pool.charge('c85_complete_spatial_geometry_counts',32*oh*ow+256*tiles)
    wnz=np.count_nonzero(weights,axis=(0,1));direct=0
    for y in range(oh):
        for x in range(ow):
            for i in range(3):
                for j in range(3):
                    if 0<=y-py+i<height and 0<=x-px+j<width:direct+=int(wnz[i,j])
    nz=native!=0;v_used=nz.any(axis=0);m_used=nz.any(axis=1)
    weighted=nz.sum(axis=(0,1));vrows=mrows=vterms=mterms=outterms=0
    for y in range(0,oh,2):
        for x in range(0,ow,2):
            active_y=range(min(2,oh-y));active_x=range(min(2,ow-x))
            ys=np.any(R[list(active_y)]!=0,axis=0);xs=np.any(R[list(active_x)]!=0,axis=0)
            for a in range(4):
                for b in range(4):
                    if not ys[a] or not xs[b]:continue
                    parents=sum(1 for i in range(4) for j in range(4)
                        if T[a,i] and T[b,j] and 0<=y-py+i<height and 0<=x-px+j<width)
                    if not parents:continue
                    nv=int(v_used[:,a,b].sum());nm=int(m_used[:,a,b].sum())
                    vrows+=nv;vterms+=nv*(parents+1)
                    mrows+=nm;mterms+=int(weighted[a,b])+nm
                    consumers=sum(1 for i in active_y for j in active_x if R[i,a] and R[j,b])
                    outterms+=nm*consumers
    output_rows=outputs*oh*ow
    return dict(geometry_known=True,per_example=True,input_shape=input_shape,output_shape=output_shape,pads=pads,
        tiles=tiles,output_definition_rows=output_rows,input_transform_factors=vrows,
        channel_sum_factors=mrows,new_transform_factors=vrows+mrows,
        direct_definition_nnz=direct+output_rows,
        factored_definition_nnz=vterms+mterms+outterms+output_rows,
        input_transform_nnz=vterms,channel_sum_nnz=mterms,output_transform_nnz=outterms+output_rows,
        strict_ideal_operator_nnz_decrease=vterms+mterms+outterms<direct,
        independent_activation_coordinate_basis_assumed=True,
        source_center_scale_mask_inverse_and_cache_cost_included=False,
        full_HZ_physical_gate_proved=False,complete_HZ_cost_proved=False,
        new_transform_factor_count_not_assumed_exempt_from_existing_caps=True)
