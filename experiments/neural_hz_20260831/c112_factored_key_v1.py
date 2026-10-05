# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same exact V quotient with once-bound channel frames and mask templates."""
import numpy as np
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T
from experiments.neural_hz_20260831.c111_form_reuse_v1 import routes as literal_routes

# Fixed stencil constants, not a persistent input-dependent cache.
_POS=tuple(sum(1<<p for p,v in enumerate(row) if v>0) for row in np.kron(T,T))
_NEG=tuple(sum(1<<p for p,v in enumerate(row) if v<0) for row in np.kron(T,T))


def template(mask,*,pool):
    pool.charge('c112_exact_mask_templates',32+8*16)
    result=[]
    for positive,negative in zip(_POS,_NEG,strict=True):
        positive &= mask;negative &= mask
        active=positive|negative
        sign=-1 if active and negative & (active & -active) else 1
        result.append((negative,positive,sign) if sign<0 else (positive,negative,sign))
    return tuple(result)


def routes(forms,keep_v,ids,exponents,*,pool):
    """Internal C96 raw-form producer boundary; templates are not certificates."""
    ids,exponents=np.asarray(ids),np.asarray(exponents)
    if ids.ndim!=2 or ids.shape[1]!=16 or exponents.shape!=ids.shape:
        raise ValueError('complete original channel position maps required')
    ordered=sorted(keep_v);channels=sorted({c for c,t in ordered})
    pool.charge('c112_complete_channel_frame_binding',32*len(channels))
    pool.charge('c112_complete_channel_frame_binding',
        16*sum(int(np.count_nonzero(ids[c]>=0)) for c in channels))
    pool.charge('c112_all_form_route_binding',8*len(ordered))
    frame_ids={};frames={};column_owner={};ambiguous=set()
    for c in channels:
        active=[(p,int(ids[c,p]),int(exponents[c,p])) for p in range(16) if ids[c,p]>=0]
        if not active:raise ValueError('retained nonempty form has no original input')
        reference=min(active,key=lambda v:v[1])[2]
        key=tuple((p,col,power-reference) for p,col,power in active)
        frame=frame_ids.setdefault(key,len(frame_ids))
        mask=sum(1<<p for p,col,power in active)
        frames[c]=(frame,reference,mask)
        if len({col for p,col,power in active})!=len(active):ambiguous.add(frame)
        for p,col,power in active:
            prior=column_owner.setdefault(col,frame)
            if prior!=frame:ambiguous.update((prior,frame))
    slow={key for key in ordered if frames[key[0]][0] in ambiguous}
    mapping={};groups={};descriptors={};templates={}
    if slow:
        _,mapping=literal_routes(forms,slow,pool=pool)
    for key in ordered:
        if key in slow:continue
        c,t=key;frame,reference,mask=frames[c]
        if mask not in templates:templates[mask]=template(mask,pool=pool)
        positive,negative,sign=templates[mask][t]
        if (positive|negative).bit_count()<2:
            raise ValueError('retained multi-parent form/template binding differs')
        # Within one complete frame key the relative exponent of any leading
        # original column is fixed; its maximum therefore orders by reference.
        groups.setdefault((frame,positive,negative),[]).append(key)
        descriptors[key]=(sign,reference)
    for members in groups.values():
        representative=min(members,key=lambda k:(-descriptors[k][1],k))
        rep_sign,rep_power=descriptors[representative]
        for key in members:
            sign,power=descriptors[key]
            mapping[key]=(representative,sign*rep_sign,power-rep_power)
    earliest={}
    for key in ordered:
        representative=mapping[key][0]
        earliest.setdefault(representative,key)
    representatives=sorted(earliest,key=earliest.get)
    return representatives,mapping
