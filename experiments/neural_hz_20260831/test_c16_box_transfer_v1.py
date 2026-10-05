from fractions import Fraction as F
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from experiments.neural_hz_20260831 import c16_box_transfer_v1 as bound
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit
from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity
from experiments.neural_hz_20260831.test_c10_fused_emission_v1 import expr_fixture


def bytes_hash(value):
    raw = json.dumps(value, sort_keys=True).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def fixture(wide=False, main_binary=False, main_radix=False):
    expr = expr_fixture(wide)
    source = expr.terms[0].source
    if main_binary:
        gb = source.Gb.tolil()
        gb[0, 0] = .25
        source.Gb = gb.tocsr()
    if main_radix:
        gc = source.Gc.tolil()
        gc[0, 1] = 1e-40
        source.Gc = gc.tocsr()
    candidate = lift(expr, np.ones(2, bool), enabled=True)
    old = original_lift(expr, np.ones(2, bool), enabled=True)
    maps = {k: getattr(old, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    proof = {'completed': True, 'identity': audit(candidate, old.hz, maps),
        'checkpoint_sha256': 'toy-independent-checkpoint', 'hz_sha256': identity(candidate)['hz_sha256']}
    pb, ps = bytes_hash(proof)
    binding = {'all_rows_proved': True, 'identity': identity(candidate),
        'checkpoint_sha256': proof['checkpoint_sha256'], 'result_sha256': ps}
    bb, bs = bytes_hash(binding)
    receipt = bound.receipt_from_archive(candidate, pb, bb, proof_sha256=ps, binding_sha256=bs)
    return candidate, receipt, proof, binding


def test_default_off_does_not_read_proof_or_hz():
    assert bound.transfer(object(), object()) is None


@pytest.mark.parametrize('wide,main_binary,main_radix', [(False, False, False), (True, False, False),
    (False, True, False), (False, False, True)])
def test_complete_transfer_matches_fraction_norms_after_real_alias_quotient(wide, main_binary, main_radix):
    candidate, receipt, _, _ = fixture(wide, main_binary, main_radix)
    before = identity(candidate)
    report, codes = bound.transfer(candidate, receipt, enabled=True)
    assert report['complete_population'] == len(codes)
    assert report['already_erased_aliases'] and report['direct_main_boxes_transferred']
    hz = candidate.hz
    for i in np.flatnonzero(codes == 1):
        d = int(candidate.eq_roots[candidate.old_n_eq + i])
        a, b = hz.Ac.indptr[d:d + 2]
        ba, be = hz.Ab.indptr[d:d + 2]
        norm = sum((abs(F(float(v))) for v in hz.Ac.data[a:b - 1]), F(0))
        norm += sum((abs(F(float(v))) for v in hz.Ab.data[ba:be]), F(0)) + abs(F(float(hz.b[d])))
        assert norm <= F(float(hz.Ac.data[b - 1]))
    assert identity(candidate) == before
    assert not report['coefficient_norm_executed_by_transfer'] and report['receipt_numeric_arrays'] == 0
    # Box redundancy alone must not be mistaken for output-dead eliminability.
    output_slots = candidate.hz.Gc.indices
    assert any(codes[int(c) - candidate.old_n_cont] == 1 for c in output_slots)
    if main_binary:
        assert report['direct_binary_terms_not_norm_scanned'] > 0
    if main_radix:
        assert report['nondirect_rows_without_claim'] > 0


@pytest.mark.parametrize('corrupt', ['hash', 'box', 'count', 'quotient', 'lineage', 'completed', 'binding', 'checkpoint'])
def test_reject_incomplete_or_wrong_independent_proof(corrupt):
    candidate, receipt, proof, binding = fixture()
    if corrupt == 'box': proof['identity']['original_affine_proof']['all_redundant_main_and_radix_boxes_proved'] = False
    elif corrupt == 'count': proof['identity']['original_affine_proof']['all_main_defining_rows_checked'] -= 1
    elif corrupt == 'quotient': proof['identity']['quotient_proof']['redundant_boxes_proved'] = False
    elif corrupt == 'lineage': proof['identity']['tagged_physical_lineage_checked'] = False
    elif corrupt == 'completed': proof['completed'] = False
    elif corrupt == 'binding': binding['all_rows_proved'] = False
    elif corrupt == 'checkpoint': binding['checkpoint_sha256'] = 'different'
    pb, ps = bytes_hash(proof)
    binding['result_sha256'] = ps
    bb, bs = bytes_hash(binding)
    with pytest.raises(ValueError):
        bound.receipt_from_archive(candidate, pb, bb, proof_sha256='0' * 64 if corrupt == 'hash' else ps, binding_sha256=bs)


@pytest.mark.parametrize('corrupt', ['receipt', 'candidate_resealed', 'phase', 'source', 'hidden'])
def test_receipt_binding_rejects_mutation_even_after_candidate_reseal(corrupt):
    candidate, receipt, _, _ = fixture()
    if corrupt == 'receipt': receipt = replace(receipt, main_count=receipt.main_count + 1)
    elif corrupt == 'candidate_resealed':
        candidate.hz.b[0] += .125
        candidate.seal = candidate.fingerprint()
    elif corrupt == 'phase': candidate.hz.Ab.data[0] += .25
    elif corrupt == 'source': candidate.expression.bias[0] += .125
    else: object.__setattr__(receipt, 'extra', np.zeros(2))
    with pytest.raises(ValueError):
        bound.transfer(candidate, receipt, enabled=True)


def test_charge_before_authentication_and_query(monkeypatch):
    candidate, receipt, _, _ = fixture()
    def forbidden(*args):
        raise AssertionError('authentication executed before capacity acceptance')
    monkeypatch.setattr(bound, 'identity', forbidden)
    with pytest.raises(MemoryError):
        bound.transfer(candidate, receipt, enabled=True, max_work=0)
    with pytest.raises(ValueError):
        bound.transfer(candidate, receipt, enabled=True, max_work=256_000_001)


def test_mid_transaction_mutation_does_not_publish_a_partial_population(monkeypatch):
    candidate, receipt, _, _ = fixture()
    original = bound.classify_direct_main
    def corrupt(c, index):
        result = original(c, index)
        if index == 0:
            c.hz.b[0] += .125
            c.seal = c.fingerprint()
        return result
    monkeypatch.setattr(bound, 'classify_direct_main', corrupt)
    with pytest.raises(ValueError):
        bound.transfer(candidate, receipt, enabled=True)


@pytest.mark.parametrize('index', [-1, 100000, 1.5])
def test_query_outside_main_rejected(index):
    candidate, _, _, _ = fixture()
    with pytest.raises(ValueError):
        bound.classify_direct_main(candidate, index)


@pytest.mark.parametrize('ratio', [F(1), -F(1), F(3, 4), -F(3, 4), F(1, 8)])
@pytest.mark.parametrize('collision', [F(1, 4), -F(1, 4)])
def test_compositional_l1_bound_with_alias_collision_and_rhs(ratio, collision):
    # Independent rational theorem fixture: replacing a*x_alias by a*r*x_parent
    # and combining a sibling term cannot increase L1; RHS and binaries remain.
    a, h, binary = F(1, 4), F(1, 8), F(1, 16)
    before = abs(a) + abs(collision) + abs(h) + abs(binary)
    after = abs(a * ratio + collision) + abs(h) + abs(binary)
    assert after <= before
    for scale in (F(1, 8), F(1), F(32)):
        assert scale * after <= scale * before
