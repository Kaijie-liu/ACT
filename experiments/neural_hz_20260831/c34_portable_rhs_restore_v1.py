"""Restore proved live RHS sharing lost by NumPy view serialization.

Only in-memory decoded archives are edited. Every byte is authenticated first;
frozen files and the strict live terminal checker are never changed.
"""
import gc
import weakref
import numpy as np
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def restore(final,source,*,expected_final_sha256,expected_source_sha256,pool,enabled=False):
    if not enabled:return None
    pool.charge('portable_RHS_alias_binding_metadata',512)
    if source_digest(final)!=expected_final_sha256 or source_digest(source)!=expected_source_sha256:
        raise ValueError('complete serialized final/source bytes differ from independent proof')
    if any(getattr(final,n) is not getattr(source,n) for n in ('Ac','Ab','Auc','Aub')):
        raise ValueError('archive lost a predicate CSR identity, outside RHS view-only restoration')
    names=[];retired=[];retired_bytes=0
    for name in ('b','ub'):
        a,b=getattr(final,name),getattr(source,name)
        pool.charge('portable_complete_RHS_bit_and_layout_checks',8*a.size+128)
        if (a.ndim!=1 or a.shape!=b.shape or a.dtype!=b.dtype or a.strides!=b.strides
                or not np.array_equal(a.view(np.uint8),b.view(np.uint8))):
            raise ValueError('serialized RHS bytes/layout changed')
        if a.__array_interface__['data'][0]!=b.__array_interface__['data'][0]:
            names.append(name);retired.append(weakref.ref(a));retired_bytes+=a.nbytes
    for name in names:setattr(final,name,getattr(source,name))
    del a,b;gc.collect()
    if any(ref() is not None for ref in retired):raise ValueError('serialized duplicate RHS retained another archive owner')
    if source_digest(final)!=expected_final_sha256 or source_digest(source)!=expected_source_sha256:
        raise ValueError('RHS identity restoration changed mathematical content')
    return dict(restored_shared_RHS_fields=names,duplicate_RHS_field_payload_bytes=retired_bytes,
        duplicate_array_objects_physically_retired=True,all_HZ_bytes_unchanged=True,
        process_RSS_or_full_live_saving_claimed=False,
        decoded_memory_only=True,original_files_edited=False,solver_executed=False,
        source_hash_scans_separate_from_diagnostic_metadata_work=True,formal_gain=0)
