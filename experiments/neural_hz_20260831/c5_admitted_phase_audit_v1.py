"""Independent plain-core/slot assembly and exact P-path/publication checks."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn, tf_mlp as mlp
from act.back_end.solver.solver_hz import sparse_hz_linear, hz_tighten_bounds
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz, equal_payload

PLAIN_DISABLED = ('_neural_hz_compact_relu', '_neural_hz_duplicate_census',
    '_neural_hz_proportional_census', '_neural_hz_signed_proportional_census',
    '_neural_hz_signed_cancellation_census', '_neural_hz_share_relu',
    '_neural_hz_share_signed_relu', '_neural_hz_share_signed_compact_relu',
    '_neural_hz_signed_cancellation')
PHASE_WRITES = frozenset(('_sparse_frame_widths', '_sparse_relu_slots',
    '_neural_hz_phase_selective_relus', '_neural_hz_phase_selective_profile'))


def plain_entry(tf):
    if any(bool(getattr(tf, name, False)) for name in PLAIN_DISABLED):
        raise ValueError('admitted audit requires unchanged plain extended-ReLU mode')
    return dict(tf._sparse_frame_widths), dict(tf._sparse_relu_slots)


def stable_fingerprint(tf, extra):
    proxy = SimpleNamespace(**{key: value for key, value in vars(tf).items() if key not in PHASE_WRITES})
    return collect(proxy, extra).fingerprint


def expected_core(probe, bounds, layer, widths, previous_slots):
    # Build the U projection and synthetic interval explicitly, independently
    # of the candidate phase-selector and its native slot mutation.
    lower = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    upper = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    if lower.size != probe.n_out or upper.size != probe.n_out or np.any(lower > upper):
        raise ValueError('invalid authoritative phase interval')
    negative = upper <= 0
    positive = (~negative) & (lower >= 0)
    unstable = ~(negative | positive)
    matrix = sp.diags(unstable.astype(np.float64), format='csr')
    core_input = sparse_hz_linear(probe, matrix)
    mask = torch.as_tensor(unstable, device=bounds.lb.device).reshape_as(bounds.lb)
    synthetic = Bounds(torch.where(mask, bounds.lb, torch.zeros_like(bounds.lb)),
                       torch.where(mask, bounds.ub, torch.zeros_like(bounds.ub)))
    lb, ub = mlp._sparse_relu_bounds(core_input, synthetic)
    rows = np.flatnonzero((lb < 0) & (ub > 0))
    frame = probe.frame_id
    nc, nb = widths.get(frame, (probe.n_cont, probe.n_bin))
    nc, nb = max(nc, probe.n_cont), max(nb, probe.n_bin)
    slot_map, slots = dict(previous_slots), []
    for row in rows:
        key = (int(frame), int(layer.id), int(row))
        if key not in slot_map:
            slot_map[key] = (nc, nc + 1, nb)
            nc, nb = nc + 2, nb + 1
        slots.append(slot_map[key])
    expected_widths = dict(widths)
    expected_widths[frame] = (nc, nb)
    core = mlp.sparse_hz_apply_relu_exact(core_input, lb, ub, slots, nc, nb)
    return core, expected_widths, slot_map, positive, unstable


def verify_phase(expr, oracle_probe, bounds, layer, candidate, tf, entry):
    if candidate is None:
        raise ValueError('registered admitted phase unexpectedly rejected')
    core, widths, slots, positive, unstable = expected_core(oracle_probe, bounds, layer, *entry)
    checks = compare_hz(candidate.core, core)
    if not all(checks.values()):
        raise ValueError('admitted exact core/predicate mismatch')
    if tf._sparse_frame_widths != widths or tf._sparse_relu_slots != slots:
        raise ValueError('admitted global slot/frame mismatch')
    out = candidate.expression
    if out.frame_id != expr.frame_id or out.n_out != expr.n_out or len(out.terms) != len(expr.terms) + 1:
        raise ValueError('admitted positive/core expression geometry mismatch')
    p = sp.diags(positive.astype(np.float64), format='csr')
    for source_term, term in zip(expr.terms, out.terms[:-1]):
        if term.source is not source_term.source or len(term.operators) != len(source_term.operators) + 1:
            raise ValueError('admitted positive source identity changed')
        if any(a is not b for a, b in zip(source_term.operators, term.operators[:-1])):
            raise ValueError('admitted positive operator identity changed')
        if not equal_payload(term.operators[-1], p):
            raise ValueError('admitted positive row mask mismatch')
    final = out.terms[-1]
    if final.source is not candidate.core or final.operators:
        raise ValueError('admitted core not represented as same-frame identity')
    # Native linear append adds a zero vector; preserve its signed-zero order.
    bias = (p @ expr.bias + np.zeros(expr.n_out)) + np.zeros(expr.n_out)
    if not equal_payload(out.bias, bias):
        raise ValueError('admitted full bias mismatch')
    outside = ~unstable
    if np.any(core.c[outside] != 0) or core.Gc[outside].nnz or core.Gb[outside].nnz:
        raise ValueError('admitted core outside unstable rows')
    probed = np.zeros(expr.n_out, dtype=bool)
    probed[np.flatnonzero(positive)[:8]] = True
    expected_bounds = cnn._phase_selective_output_bounds(bounds, core, oracle_probe, unstable, probed)
    if not torch.equal(candidate.output_bounds.lb, expected_bounds.lb) or not torch.equal(candidate.output_bounds.ub, expected_bounds.ub):
        raise ValueError('admitted output-bound hint mismatch')
    return {'all_core_fields': checks, 'positive_paths_identity_preserved': True,
        'global_slots_equal': True, 'global_widths_equal': True, 'full_bias_equal': True,
        'output_bounds_equal': True, 'positive_rows': int(positive.sum()), 'unstable_rows': int(unstable.sum()),
        'new_binary_slots': len(slots) - len(entry[1]), 'expression_terms': len(out.terms),
        'core_n_cont': core.n_cont, 'core_n_bin': core.n_bin, 'core_n_eq': core.n_eq, 'core_n_ineq': core.n_ineq}


def verify_precomputed(tf, layer_id, candidate, bounds):
    prepared = tf._sparse_precomputed_relu.get(layer_id)
    if prepared is None or len(prepared) != 5:
        raise ValueError('native five-field publication absent')
    if prepared[0] is not candidate.core or prepared[3] is not candidate.expression or prepared[4] is not candidate.output_bounds:
        raise ValueError('native publication replaced candidate object')
    if not torch.equal(prepared[1], bounds.lb.detach().cpu()) or not torch.equal(prepared[2], bounds.ub.detach().cpu()):
        raise ValueError('native publication authoritative bounds mismatch')
    return {'five_field_tuple': True, 'all_candidate_identities': True, 'authoritative_bounds_equal': True}


def verify_consumed(tf, layer_id, candidate, interval_result, actual):
    if layer_id in tf._sparse_precomputed_relu or layer_id in tf._sparse_hz_cache:
        raise ValueError('native phase tuple not consumed or conflicting HZ retained')
    if tf._sparse_affine_expr_cache.get(layer_id) is not candidate.expression:
        raise ValueError('native phase expression not installed by identity')
    expected = hz_tighten_bounds(interval_result.bounds, candidate.output_bounds)
    if actual.cons is not interval_result.cons or not torch.equal(actual.bounds.lb, expected.lb) or not torch.equal(actual.bounds.ub, expected.ub):
        raise ValueError('native publication fact/bounds mismatch')
    return {'precomputed_consumed': True, 'expression_installed_by_identity': True,
        'no_conflicting_sparse_hz': True, 'fact_constraints_same_identity': True, 'fact_bounds_equal': True}
