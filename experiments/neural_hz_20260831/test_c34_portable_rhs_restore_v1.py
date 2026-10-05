import pickle
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import sparse_hz_linear
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
from experiments.neural_hz_20260831.c34_portable_rhs_restore_v1 import restore
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def fixture():
    pre=source(6);final=sparse_hz_linear(pre,sp.csr_matrix(np.ones((2,6))/8),np.zeros(2))
    assert np.shares_memory(pre.b,final.b) and pre.b is not final.b
    raw=pickle.dumps((pre,final),protocol=5)
    return raw,source_digest(pre),source_digest(final)


def test_default_off_reads_nothing():
    assert restore(object(),object(),expected_final_sha256=object(),expected_source_sha256=object(),pool=object()) is None


def test_actual_numpy_pickle_view_loss_and_proved_alias_restoration():
    raw,pre_sha,final_sha=fixture();pre,final=pickle.loads(raw)
    assert not np.shares_memory(pre.b,final.b) and not np.shares_memory(pre.ub,final.ub)
    assert pre.Ac is final.Ac
    result=restore(final,pre,expected_final_sha256=final_sha,expected_source_sha256=pre_sha,
        pool=WorkPool(256_000_000),enabled=True)
    assert result['restored_shared_RHS_fields']==['b','ub'] and result['duplicate_array_objects_physically_retired']
    assert final.b is pre.b and final.ub is pre.ub
    assert source_digest(pre)==pre_sha and source_digest(final)==final_sha
    assert pickle.loads(raw)[0].b is not pre.b


@pytest.mark.parametrize('bad',['final_RHS','source_RHS','predicate_identity','final_hash','alias_owner','cap'])
def test_corruption_or_unaccounted_owner_rejects(monkeypatch,bad):
    raw,pre_sha,final_sha=fixture();pre,final=pickle.loads(raw);pool=WorkPool(256_000_000)
    if bad=='final_RHS':final.b[0]+=.125
    elif bad=='source_RHS':pre.ub[0]+=.125
    elif bad=='predicate_identity':final.Ac=final.Ac.copy()
    elif bad=='final_hash':final_sha='0'*64
    elif bad=='alias_owner':owner=final.b
    else:pool=WorkPool(0)
    with pytest.raises((ValueError,MemoryError)):
        restore(final,pre,expected_final_sha256=final_sha,expected_source_sha256=pre_sha,pool=pool,enabled=True)


def test_already_shared_runtime_state_is_noop():
    pre=source(6);final=sparse_hz_linear(pre,sp.eye(6,format='csr'))
    result=restore(final,pre,expected_final_sha256=source_digest(final),expected_source_sha256=source_digest(pre),
        pool=WorkPool(256_000_000),enabled=True)
    assert result['restored_shared_RHS_fields']==[] and result['duplicate_RHS_field_payload_bytes']==0
