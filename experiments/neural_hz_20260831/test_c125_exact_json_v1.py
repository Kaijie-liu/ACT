"""Exact byte identity, finite data, prepayment and exclusive result custody."""
import hashlib
import json

import pytest

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance


def test_exact_checked_bytes_are_published_without_indented_expansion(tmp_path):
    payload = dict(points=[[i, 128] for i in range(512)], label='complete')
    expected = (json.dumps(payload, sort_keys=True, separators=(',', ':'),
                           allow_nan=False, ensure_ascii=True)+'\n').encode()
    cap = len(expected)+1024
    assert len(json.dumps(payload, indent=2).encode())+1024 > cap
    pool = WorkPool(cap)
    allowance = JsonAllowance(pool, cap, 'complete_exact_JSON')
    receipt = allowance.write(tmp_path/'result.json', payload)
    assert pool.used == cap and (tmp_path/'result.json').read_bytes() == expected
    assert receipt['sha256'] == hashlib.sha256(expected).hexdigest()
    assert receipt['encoded_bytes']+receipt['overhead_bytes'] == receipt['prepaid_bytes']
    assert json.loads(expected) == payload and receipt['exact_checked_bytes_published']
    with pytest.raises(ValueError):
        allowance.write(tmp_path/'second.json', payload)
    assert not (tmp_path/'second.json').exists()


def test_unpaid_allowance_rejects_before_any_serialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unpaid serialization reached')
    monkeypatch.setattr(json, 'dumps', forbidden)
    pool = WorkPool(0)
    with pytest.raises(MemoryError):
        JsonAllowance(pool, 1024, 'unpaid')
    assert pool.used == 0


def test_oversize_fails_without_partial_evidence_or_refund(tmp_path):
    pool = WorkPool(1024)
    allowance = JsonAllowance(pool, 1024, 'too_small')
    with pytest.raises(MemoryError):
        allowance.write(tmp_path/'oversize.json', {'nonempty': [1, 2, 3]})
    assert pool.used == 1024 and not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        allowance.write(tmp_path/'retry.json', {})


def test_nonfinite_rejection_and_existing_artifact_preservation(tmp_path):
    pool = WorkPool(10000)
    with pytest.raises(ValueError):
        JsonAllowance(pool, 2048, 'nonfinite').write(tmp_path/'bad.json', {'value': float('nan')})
    assert not list(tmp_path.iterdir())
    path = tmp_path/'retained.json'
    JsonAllowance(pool, 2048, 'first').write(path, {'old': 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        JsonAllowance(pool, 2048, 'second').write(path, {'new': 2})
    assert path.read_bytes() == original and sorted(p.name for p in tmp_path.iterdir()) == ['retained.json']
