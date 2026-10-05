"""Independent native bound/slot audit, never assembles a second ReLU HZ."""
import numpy as np
from act.back_end.hybridz_tf import tf_mlp as mlp
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry


def verify(state,bounds):
    tf,layer=state['tf'],state['layer'];pre=state['views'][-1];new=state['lifted']
    plain_entry(tf)
    forced=bounds.ub.detach().cpu().numpy().reshape(-1)<=0.
    lower,upper=mlp._sparse_relu_bounds(pre,bounds,forced_stable_negative=forced)
    rows=np.flatnonzero((lower<0.)&(upper>0.));frame=int(pre.frame_id)
    old_nc,old_nb=state['entry_widths'][frame]
    nc,nb=max(old_nc,pre.n_cont),max(old_nb,pre.n_bin)
    mapping=dict(state['entry_slots'])
    for row in rows:
        key=(frame,int(layer.id),int(row))
        if key in mapping:raise ValueError('new selected phase reuses old phase slots')
        mapping[key]=(nc,nc+1,nb);nc+=2;nb+=1
    widths=dict(state['entry_widths']);widths[frame]=(nc,nb)
    if (tf._sparse_relu_slots!=mapping or tf._sparse_frame_widths!=widths
            or new.hz.n_cont!=nc or new.hz.n_bin!=nb or new.hz.frame_id!=frame
            or new.construction_report['new_phase_binaries']!=len(rows)):
        raise ValueError('actual new global phase slots/frame mismatch')
    proof=new.validate()
    return dict(old_phase_slots_unchanged=True,new_slots_disjoint_from_original_source=True,
        global_widths_exact=True,new_phase_binaries=len(rows),new_phase_continuous=2*len(rows),
        stable_negative=int(np.count_nonzero(upper<=0.)),stable_positive=int(np.count_nonzero(lower>=0.)),
        entire_actual_HZ_and_reconstruction_bound=proof,second_native_HZ_assembly_executed=False,
        diagnostic_bound_recomputation_executed=True,formal_gain=0)
