"""Complete semantic comparison with an independently proved C26 journal.

The caller authenticates the completed archive/result/exit hash chain first.
This comparison complements, and never replaces, the OLD source-image proof.
It grants no live/native admission and creates no new predicate matrix.
"""

import numpy as np
from experiments.neural_hz_20260831.c27_reversible_lineage_v1 import SPLICE,REDIRECT,MASK


def verify(candidate,reference,*,expected_reference_fingerprint,pool):
    candidate.validate(); reference.validate()
    if reference.fingerprint()!=expected_reference_fingerprint:
        raise ValueError('independently bound semantic reference changed')
    if (candidate.old_n_cont!=reference.old_n_cont or candidate.old_n_eq!=reference.old_n_eq
            or candidate.eq_roots.shape!=reference.eq_roots.shape):
        raise ValueError('semantic source geometry differs')
    pool.charge('complete_reversible_semantic_maps',16*len(candidate.eq_roots))
    for new,old in zip(candidate.eq_roots,reference.eq_roots):
        new=int(new)
        stripped=(new&(SPLICE|((1<<48)-1)) if new>=SPLICE else
            REDIRECT|(new&MASK) if new>=REDIRECT else new)
        if stripped!=int(old):raise ValueError('new row/alias/splice/redirect semantics differ')
    if not np.array_equal(candidate.eq_scales,reference.eq_scales):
        raise ValueError('new exact offset/scale bits differ')
    for key in ('columns','retired','tails'):
        pool.charge('complete_reversible_semantic_sparse_metadata',4*len(getattr(candidate,key)))
        if not np.array_equal(getattr(candidate,key),getattr(reference,key)):
            raise ValueError('new sparse selection/UID/tail semantics differ')
    return {'all_lineage_slots_compared':len(candidate.eq_roots),
        'all_selected_columns_compared':len(candidate.columns),
        'all_retired_UIDs_compared':len(candidate.retired),'all_MAIN_tails_compared':len(candidate.tails),
        'all_new_scale_and_offset_bits_equal':True,'new_native_HZ_constructed':False,
        'live_admission_certificate':False,'formal_gain':0}
