import copy

import numpy as np
import pytest
import torch

from act.back_end.core import Fact, ConSet
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.solver.test_neural_hz_compact_relu import _phase_selective_fixture, _phase_selective_interval_output
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import (
    plain_entry, stable_fingerprint, verify_phase, verify_precomputed, verify_consumed,
)


def fixture():
    _, expr, bounds, layer, tf = _phase_selective_fixture()
    entry = plain_entry(tf)
    before = stable_fingerprint(tf, {'expr': expr, 'bounds': bounds})
    probe_mask = np.ones(expr.n_out, dtype=bool)
    probe_mask[0] = probe_mask[9] = False
    probe = cnn._lazy_materialize(expr, probe_mask, 20_000)
    candidate = cnn._try_phase_selective_exact_relu(expr, bounds, tf, layer)
    return expr, bounds, layer, tf, entry, before, probe, candidate


def test_independent_core_slots_paths_and_actual_consumption():
    expr, bounds, layer, tf, entry, before, probe, candidate = fixture()
    proof = verify_phase(expr, probe, bounds, layer, candidate, tf, entry)
    assert proof['new_binary_slots'] == 1
    assert stable_fingerprint(tf, {'expr': expr, 'bounds': bounds}) == before
    tf._sparse_precomputed_relu[layer.id] = (candidate.core, bounds.lb.clone(), bounds.ub.clone(), candidate.expression, candidate.output_bounds)
    assert verify_precomputed(tf, layer.id, candidate, bounds)['all_candidate_identities']
    interval = Fact(_phase_selective_interval_output(bounds), ConSet())
    out = tf._propagate_sparse_hz(layer, bounds, interval)
    assert verify_consumed(tf, layer.id, candidate, interval, out)['expression_installed_by_identity']


@pytest.mark.parametrize('mutation', ['predicate', 'positive_source', 'bias', 'slot', 'output_bounds'])
def test_candidate_corruption_rejects(mutation):
    expr, bounds, layer, tf, entry, before, probe, candidate = fixture()
    if mutation == 'predicate':
        candidate.core.b[0] = np.nextafter(candidate.core.b[0], np.inf)
    elif mutation == 'positive_source':
        t = candidate.expression.terms[0]
        altered = cnn.SparseHZAffineTerm(copy.deepcopy(t.source), t.operators)
        object.__setattr__(candidate.expression, 'terms', (altered, *candidate.expression.terms[1:]))
    elif mutation == 'bias':
        candidate.expression.bias[1] += .25
    elif mutation == 'slot':
        tf._sparse_relu_slots[(59, layer.id, 10)] = (0, 1, 0)
    else:
        candidate.output_bounds.ub[0, 1] += .25
    with pytest.raises(ValueError):
        verify_phase(expr, probe, bounds, layer, candidate, tf, entry)


def test_plain_mode_guard_and_non_phase_state_drift():
    expr, bounds, layer, tf, entry, before, probe, candidate = fixture()
    tf._neural_hz_compact_relu = True
    with pytest.raises(ValueError, match='plain'):
        plain_entry(tf)
    assert stable_fingerprint(tf, {'expr': expr, 'bounds': bounds}) != before


def test_incorrect_tuple_and_missing_consumption_reject():
    expr, bounds, layer, tf, entry, before, probe, candidate = fixture()
    tf._sparse_precomputed_relu[layer.id] = (candidate.core, bounds.lb.clone(), bounds.ub.clone(), candidate.expression, None)
    with pytest.raises(ValueError, match='replaced'):
        verify_precomputed(tf, layer.id, candidate, bounds)
    interval = Fact(_phase_selective_interval_output(bounds), ConSet())
    with pytest.raises(ValueError, match='not consumed'):
        verify_consumed(tf, layer.id, candidate, interval, interval)
