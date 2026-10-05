"""Complete new source proof and graph-free physical archive on ordinary HZs."""
import gc
import hashlib
import pickle
import weakref
import numpy as np
import pytest
from experiments.neural_hz_20260831.c66_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c65_full_source_audit_v1 import audit
from experiments.neural_hz_20260831.c65_physical_archive_v1 import bind_proof,check_restored,metadata
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete,old_fields


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_complete_source_inverse_owner_proof_and_portable_physical_state(kind):
    _,saved=complete(kind);legacy,_=old_fields(saved)
    candidate=lift(saved['expression'],saved['keep'],enabled=True);pool=WorkPool(256_000_000)
    proof=audit(saved,legacy,candidate,pool=pool,enabled=True);fields=candidate['fields'];construction=candidate['construction']
    expected=actual_words(fields['hz'],fields['old_n_cont'],fields['logical_n_cont'],construction['eq_uids'],construction['ineq_uids'])
    assert np.array_equal(fields['owners'],expected)
    assert proof['original_EQ_checked']==saved['hz'].n_eq and proof['original_INEQ_checked']==saved['hz'].n_ineq
    assert not proof['new_native_or_LIVE_admission'] and proof['local_inverse_equations']>0
    refs=[weakref.ref(n[k]) for n in construction['nodes'] for k in ('support','needed','slots','exponents')]
    identity,raw=bind_proof(fields,proof,source_sha256='focused-original-source')
    del candidate,construction;gc.collect()
    assert all(ref() is None for ref in refs)
    payload=dict(schema='c65_graph_free_physical_archive_v1',fields=fields,proof_bytes=raw,
        proof_sha256=hashlib.sha256(raw).hexdigest(),native_or_LIVE_admission=False)
    restored=pickle.loads(pickle.dumps(payload,protocol=5));record=check_restored(restored)
    assert record['physical_identity']==identity
    complete_metadata=metadata(payload,pool=pool)
    assert complete_metadata['all_expression_and_operator_metadata_traversed'] and not complete_metadata['opaque_identity_cancellation_used']
    assert numeric_layout(payload,pool).resident_entries>0


def test_default_off_proof_does_not_read_inputs():
    assert audit(None,None,None,pool=None) is None
