"""Remove only dead exact scalar-table entries after gauged carrier realization."""
import numpy as np
from experiments.neural_hz_20260831.c56_gauged_carrier_v1 import build as carrier_build,state_hash,layout,native,exact_binding,SHARED,digit_rows
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack,Pool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool


def build(exact,*,enabled=False,pool=None):
    if not enabled:return None
    whole=WorkPool(256_000_000) if pool is None else pool
    state=carrier_build(exact,enabled=True,pool=whole);prior=state['report']['lift_work']
    work=BranchPool(whole,cap=16_000_000-prior)
    values=unpack(state['scalars']);old=state['inverse']
    work.charge('c56v2_complete_inverse_liveness',64*(len(old)+len(values))+256)
    scalar_pool=Pool(work);inverse=np.empty_like(old)
    for i,word in enumerate(old):
        word=int(word);root=word&((1<<32)-1);sid=word>>32
        if sid>=len(values):raise ValueError('inverse scalar ID outside original pool')
        new=scalar_pool.intern(values[sid]);inverse[i]=np.uint64((new<<32)|root)
    state['scalars']=scalar_pool.pack();state['inverse']=inverse
    work.charge('c56v2_complete_inverse_pool_repack_and_binding',32*(len(inverse)+len(state['scalars']['limbs']))+256)
    state['report']['lift_work']=prior+work.used
    state['seal']=state_hash(state);layout(state)
    return state
