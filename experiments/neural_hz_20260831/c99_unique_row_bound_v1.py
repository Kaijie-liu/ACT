"""Guaranteed final coefficient count for the existing exact dense tile program."""
from collections import Counter
import numpy as np
from experiments.neural_hz_20260831.c93_packed_mask_v1 import _INPUT,_OUTPUT


def minimum_final_nnz(input_masks,output_masks,*,pool,enabled=False):
    """Conditional on original unique slots and the existing full dense checks.

    Retained v rows have distinct original parents. Retained m rows combine
    disjoint input-channel coordinates or distinct fresh v slots. Output
    pivots and referenced retained m slots cannot collide with any inline
    term. Other output terms are deliberately omitted from the lower bound.
    No numerical success, selected identity, or observed final nnz is read.
    """
    if not enabled:return None
    a,o=np.asarray(input_masks),np.asarray(output_masks)
    if a.ndim!=1 or o.ndim!=1 or a.dtype!=np.uint16 or o.dtype!=np.uint8:
        raise ValueError('complete packed original input/output masks required')
    pool.charge('c99_guaranteed_unique_row_headers',512+12*(len(a)+len(o)))
    if np.any(o>15):raise ValueError('invalid ordinary output mask')
    ac,oc=Counter(map(int,a)),Counter(map(int,o))
    pool.charge('c99_guaranteed_unique_mask_classes',64*(len(ac)+len(oc)))
    outputs=sum(mask.bit_count()*n for mask,n in oc.items())
    vnnz=mnnz=out_m=0
    for pm,om in zip(_INPUT,_OUTPUT,strict=True):
        A=B=S=0
        for mask,n in ac.items():
            size=(mask&pm).bit_count();A+=n*(size>0);B+=n*(size>1);S+=n*size
        U=V=W=active=0
        for mask,n in oc.items():
            uses=(mask&om).bit_count();U+=n*uses;active+=n*(uses>0)
            V+=n*(uses>1);W+=n*uses*(uses>1)
        uses=active if A>1 else U if A else 0
        retained=uses>1
        vnnz+=S-A+2*B if retained else 0
        width=A if retained else S
        mnnz+=V*(width+1) if A>1 else 0
        out_m+=W if A>1 else 0
    return dict(auxiliary_final_nnz=vnnz+mnnz,output_pivots=outputs,
        distinct_output_m_terms=out_m,guaranteed_final_nnz=vnnz+mnnz+outputs+out_m,
        unique_original_slots_and_complete_dense_transform_required=True)
