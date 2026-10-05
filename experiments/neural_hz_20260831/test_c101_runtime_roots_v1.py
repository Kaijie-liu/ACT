# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full original C5 oracle on the actual new circuit/native custody shape."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.neural_hz_20260831.c5_live_roots_v2 import collect as original
from experiments.neural_hz_20260831.c101_runtime_roots_v1 import RuntimeRoots
from experiments.neural_hz_20260831.test_c100_circuit_terminal_v1 import native


@pytest.mark.parametrize('stage', ['source', 'native', 'terminal'])
def test_complete_circuit_roots_match_original_identity_and_storage(stage):
    n, inp = native()
    tf = SimpleNamespace(cache={0: inp, 1: n.hz}, variables=list(range(300, 10000)))
    extra = dict(source=n.source.circuit_state, input=inp)
    if stage != 'source':
        extra['actual_native'] = n.numeric_roots()
    if stage == 'terminal':
        extra.update(incoming=[dict(source=n.source.circuit_state)],
            proof_bytes=n.transfer_proof_bytes, source_proof=n.source.proof_bytes,
            shared_cache=[inp, n.hz], final_output=n.hz)
    expected = original(tf, extra)
    events = []
    traversal = RuntimeRoots(events.append, enabled=True)
    got = traversal.collect(tf, extra, boundary=stage)
    assert got.fingerprint == expected.fingerprint
    assert got.schema_counts == expected.schema_counts
    assert got.python_shallow_bytes == expected.python_shallow_bytes
    assert got.unique_objects == expected.unique_objects
    assert got.numeric.keys() == expected.numeric.keys()
    assert all(got.numeric[k] is value for k, value in expected.numeric.items())
    assert asdict(got.measure()) == asdict(expected.measure())
    assert events[-1]['measurement']['measured_transient_gate']
    assert events[-1]['unchanged_full_numeric_hashes_executed']
    assert not events[-1]['numeric_hash_traffic_in_token_pool']


def test_four_calls_share_one_pool_without_reset_and_preserve_input():
    tf = SimpleNamespace(variables=list(range(300, 1200)))
    traversal = RuntimeRoots(lambda event: None, enabled=True)
    prior = 0
    fingerprint = original(tf).fingerprint
    for label in ('entry', 'preservation', 'final', 'final_preservation'):
        got = traversal.collect(tf, {}, boundary=label)
        assert got.fingerprint == fingerprint
        assert traversal.pool.used > prior
        assert traversal.receipts[-1]['token_work'] == traversal.pool.used-prior
        prior = traversal.pool.used
    with pytest.raises(ValueError, match='four preregistered'):
        traversal.collect(tf, {}, boundary='unregistered')


def test_default_off_and_shared_budget_failure():
    traversal = RuntimeRoots(lambda event: None)
    assert traversal.collect(None, {}, boundary='disabled') is None
    assert traversal.calls == 0 and traversal.pool.used == 0
    traversal.enabled = True
    traversal.pool.charge('already_consumed_shared_budget', traversal.pool.cap)
    with pytest.raises(MemoryError):
        traversal.collect(SimpleNamespace(), {}, boundary='exhausted')


def test_complete_native_predicate_change_remains_visible():
    n, inp = native()
    extra = n.numeric_roots()
    traversal = RuntimeRoots(lambda event: None, enabled=True)
    tf = SimpleNamespace(input=inp)
    before = traversal.collect(tf, extra, boundary='native')
    n.hz.Auc.data[0] += .125
    after = traversal.collect(tf, extra, boundary='native_changed')
    assert after.fingerprint != before.fingerprint
    assert after.fingerprint == original(tf, extra).fingerprint
