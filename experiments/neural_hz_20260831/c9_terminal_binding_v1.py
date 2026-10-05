"""Content-bound reuse of a fully checked live post-ReLU certificate."""

from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry


def bind(state, proof):
    tf, layer, lifted = state['tf'], state['layer'], state['lifted']
    plain_entry(tf)
    lifted.validate()
    if not proof.get('passed') or proof.get('status') != 'LIVE_RELU_QUALIFIED':
        raise ValueError('terminal requires a passed live prerequisite')
    actual = tf._sparse_hz_cache.get(layer.id)
    if actual is None or source_digest(actual) != proof['actual_hz_sha256']:
        raise ValueError('fresh post-ReLU HZ differs from complete audited content')
    if layer.id in tf._sparse_affine_expr_cache:
        raise ValueError('bound native post-HZ has a conflicting expression')
    if not state['construction']['measured_transient_gate'] or not state['consumer_construction']['measured_transient_gate']:
        raise ValueError('bound native construction gate failed')
    old_slots, slots = state['entry_slots'], tf._sparse_relu_slots
    if any(slots.get(key) != value for key, value in old_slots.items()):
        raise ValueError('bound post-HZ changed old slots')
    fresh = {key: value for key, value in slots.items() if key not in old_slots}
    if len(fresh) != proof['post_relu']['new_phase_binaries']:
        raise ValueError('bound post-HZ fresh phase count changed')
    for index, (key, (a, b, z)) in enumerate(sorted(fresh.items())):
        if key[:2] != (actual.frame_id, layer.id) or a < lifted.hz.n_cont or b < lifted.hz.n_cont or z < lifted.hz.n_bin:
            raise ValueError('bound post-HZ slots overlap or belong to another frame/layer')
        expected = (lifted.hz.n_cont + 2 * index, lifted.hz.n_cont + 2 * index + 1, lifted.hz.n_bin + index)
        if (a, b, z) != expected or not 0 <= key[2] < actual.n_out or actual.Gc[key[2], b] == 0.:
            raise ValueError('bound post-HZ fresh slots differ from native ordered allocation')
    expected_widths = dict(state['entry_widths'])
    expected_widths[actual.frame_id] = (actual.n_cont, actual.n_bin)
    if tf._sparse_frame_widths != expected_widths:
        raise ValueError('bound post-HZ global frame changed')
    return {'post_hz_sha256': source_digest(actual), 'all_hz_content_matches_prior_audit': True,
        'old_slots_preserved': True, 'fresh_slots_disjoint': True, 'global_widths_checked': True}
