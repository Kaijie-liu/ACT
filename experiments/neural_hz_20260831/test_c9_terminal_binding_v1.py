from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c9_terminal_binding_v1 import bind
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c9_live_runtime_v1 import frame
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture


@pytest.mark.parametrize('mutation', ['none', 'coefficient', 'old_slot', 'new_slot', 'slot_permutation', 'width', 'gate', 'proof'])
def test_complete_post_content_and_phase_binding(monkeypatch, mutation):
    expr, _ = fixture()
    tf, retained = frame(), []
    bounds = Bounds(torch.full((6,), -2., dtype=torch.float64), torch.full((6,), 2., dtype=torch.float64))
    layer = SimpleNamespace(id=123, kind='RELU')
    def applied(self, selected, *args):
        handled, hz, separated, reason = cnn.sparse_hz_apply_affine_expr_layer(selected, expr, bounds, None, self)
        assert handled and hz is not None and separated is None
        self._sparse_hz_cache[selected.id] = hz
        return hz
    monkeypatch.setattr(HybridzTF, 'apply', applied)
    with runtime.installed(enabled=True, ready=retained.append):
        actual = HybridzTF.apply(tf, layer)
    state = retained[0]
    proof = {'passed': True, 'status': 'LIVE_RELU_QUALIFIED', 'actual_hz_sha256': source_digest(actual),
        'post_relu': {'new_phase_binaries': actual.n_bin - state['lifted'].hz.n_bin}}
    if mutation == 'coefficient':
        actual.Gc.data[0] += .1
    elif mutation == 'old_slot':
        tf._sparse_relu_slots[(7, -8, 3)] = (0, 1, 0)
    elif mutation == 'new_slot':
        key = next(key for key in tf._sparse_relu_slots if key[1] == 123)
        tf._sparse_relu_slots[key] = (0, 1, 0)
    elif mutation == 'slot_permutation':
        a, b = [key for key in tf._sparse_relu_slots if key[1] == 123][:2]
        tf._sparse_relu_slots[a], tf._sparse_relu_slots[b] = tf._sparse_relu_slots[b], tf._sparse_relu_slots[a]
    elif mutation == 'width':
        tf._sparse_frame_widths[7] = (1, 1)
    elif mutation == 'gate':
        state['consumer_construction']['measured_transient_gate'] = False
    elif mutation == 'proof':
        proof['passed'] = False
    if mutation == 'none':
        assert bind(state, proof)['all_hz_content_matches_prior_audit']
    else:
        with pytest.raises(ValueError):
            bind(state, proof)
