"""Independent slot assembly and unchanged extended-ReLU consumer oracle."""

import numpy as np

from act.back_end.hybridz_tf import tf_mlp as mlp
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz


def verify_plain(preactivation, bounds, layer, actual, tf, entry_widths, entry_slots):
    plain_entry(tf)  # Reject any quotient/sharing/compact alternate semantics.
    forced = bounds.ub.detach().cpu().numpy().reshape(-1) <= 0.
    lower, upper = mlp._sparse_relu_bounds(preactivation, bounds, forced_stable_negative=forced)
    rows = np.flatnonzero((lower < 0.) & (upper > 0.))
    frame = int(preactivation.frame_id)
    old_nc, old_nb = entry_widths[frame]
    nc, nb = max(old_nc, preactivation.n_cont), max(old_nb, preactivation.n_bin)
    slots, mapping = [], dict(entry_slots)
    for row in rows:
        key = (frame, int(layer.id), int(row))
        if key in mapping:
            raise ValueError('new reducer ReLU unexpectedly reuses an old phase slot')
        mapping[key] = (nc, nc + 1, nb)
        slots.append(mapping[key])
        nc, nb = nc + 2, nb + 1
    widths = dict(entry_widths)
    widths[frame] = (nc, nb)
    if tf._sparse_relu_slots != mapping or tf._sparse_frame_widths != widths:
        raise ValueError('live C9 phase-slot or global-width mismatch')
    if any(a < preactivation.n_cont or b < preactivation.n_cont or z < preactivation.n_bin
           for a, b, z in slots):
        raise ValueError('ReLU slots overlap reserved C9 factors')
    expected = mlp.sparse_hz_apply_relu_exact(preactivation, lower, upper, slots, nc, nb)
    checks = compare_hz(actual, expected)
    if not all(checks.values()):
        raise ValueError('live C9 post-ReLU HZ differs from independently assembled native graph')
    return {'all_hz_fields': checks, 'old_phase_slots_unchanged': True,
        'new_slots_disjoint_from_c9': True, 'global_widths_exact': True,
        'original_frame_preserved': actual.frame_id == frame,
        'new_phase_binaries': len(slots), 'new_phase_continuous': 2 * len(slots),
        'stable_negative': int(np.count_nonzero(upper <= 0.)),
        'stable_positive': int(np.count_nonzero(lower >= 0.)),
        'n_cont': actual.n_cont, 'n_bin': actual.n_bin,
        'n_eq': actual.n_eq, 'n_ineq': actual.n_ineq}
