# ===- act/back_end/fchz_tf/forward_global_fal.py - LP-witness FAL =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Forward LP-witness FAL detection: when LP/MILP finds a feasible point
#   in the unsafe polytope, extract the input portion x* as candidate
#   counterexample. Then ORT-replay x* through actual network to verify
#   the spec violation independently.
#
#   PRINCIPLES (advisor 2026-06-09):
#     - LP witness + strict ORT replay infrastructure (advisor's explicit allowance)
#     - No PGD / no random search / no backward
#     - Open-source HiGHS for LP/MILP
#     - Witness must be verified via independent ORT replay (sound by construction)
#
# ===---------------------------------------------------------------------===#
"""Forward LP-witness Falsification detection."""
import numpy as np
from typing import Optional, Dict, Tuple

try:
    from scipy.optimize import milp, LinearConstraint, Bounds as LPBounds
    from scipy.optimize import linprog
    HAS_MILP = True
except ImportError:
    HAS_MILP = False


def extract_input_from_lp(lp_data: Dict, input_layer_id: int) -> Optional[np.ndarray]:
    """Given an LP solution, extract just the input portion x*."""
    input_idx = lp_data['layer_var_idx'].get(input_layer_id)
    if input_idx is None: return None
    start, dim = input_idx
    sol = lp_data.get('_last_solution')
    if sol is None or len(sol) < start + dim: return None
    return sol[start:start+dim]


def solve_lp_or_milp_for_witness(lp_data: Dict, output_layer_id: int, C_row: np.ndarray,
                                                       t_val: float, time_limit_s: float = 8.0,
                                                       use_milp: bool = True) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Solve: minimize C_row @ y subject to LP/MILP constraints.

    For UNSAFE_LINEAR spec (C @ y <= t is unsafe):
      - If min C@y <= t, then there's a point in LP feasible region in unsafe polytope
      - The solution's x* is a candidate counterexample

    Returns (min_obj_value, x_star_input). Both None if infeasible/error.
    """
    if not HAS_MILP: return None, None
    output_idx = lp_data['layer_var_idx'].get(output_layer_id)
    if output_idx is None: return None, None
    out_start, out_dim = output_idx
    n_total = lp_data['var_count']
    if C_row.shape[0] != out_dim: return None, None

    obj = np.zeros(n_total)
    obj[out_start:out_start + out_dim] = C_row.astype(np.float64)

    constraints = []
    if lp_data.get('A_ub') is not None:
        constraints.append(LinearConstraint(lp_data['A_ub'], ub=lp_data['b_ub']))
    if lp_data.get('A_eq') is not None:
        constraints.append(LinearConstraint(lp_data['A_eq'], lb=lp_data['b_eq'], ub=lp_data['b_eq']))

    lbs = np.array([b[0] for b in lp_data['var_bounds']])
    ubs = np.array([b[1] for b in lp_data['var_bounds']])
    bounds_obj = LPBounds(lb=lbs, ub=ubs)

    try:
        if use_milp and 'integrality' in lp_data and lp_data['integrality'].sum() > 0:
            result = milp(
                c=obj,
                constraints=constraints if constraints else None,
                integrality=lp_data['integrality'],
                bounds=bounds_obj,
                options={'time_limit': time_limit_s, 'disp': False})
        else:
            # Fall back to plain LP
            integrality = lp_data.get('integrality', np.zeros(n_total))
            result = linprog(
                c=obj, A_ub=lp_data.get('A_ub'), b_ub=lp_data.get('b_ub'),
                A_eq=lp_data.get('A_eq'), b_eq=lp_data.get('b_eq'),
                bounds=lp_data['var_bounds'], method='highs',
                options={'presolve': True, 'time_limit': time_limit_s})
        if result.success:
            return float(result.fun), result.x
        return None, None
    except Exception:
        return None, None


def try_lp_witness_fal(net, lb_in: np.ndarray, ub_in: np.ndarray,
                                  pre_bounds: Dict,
                                  C: np.ndarray, t: np.ndarray, canon_kind: str,
                                  onnx_path: str, in_shape: tuple,
                                  K_per_layer: int = 20,
                                  solver_time_s: float = 8.0) -> Tuple[bool, Optional[np.ndarray], Dict]:
    """Try to find sound FAL witness via LP/MILP + ORT replay.

    Args:
      net, lb_in, ub_in, pre_bounds: standard FCHZ inputs
      C, t: spec matrix (rows × n_out) and threshold
      canon_kind: 'UNSAFE_LINEAR' / 'LINEAR_LE' / 'TOP1_ROBUST'
      onnx_path: path to ONNX file for replay
      in_shape: network input shape
      K_per_layer: MILP top-K
      solver_time_s: per-LP time limit

    Returns:
      (is_falsified, witness_x, info)
    """
    from act.back_end.fchz_tf.forward_global_milp import build_forward_milp

    info = {'mechanism': 'LP-witness + strict ORT replay'}
    if canon_kind != 'UNSAFE_LINEAR':
        # For LINEAR_LE/TOP1_ROBUST, FAL is when any C@y > t
        # Approach: maximize C[r] @ y over each row, then check
        pass

    # Find INPUT and ASSERT layer
    input_layer = next((L for L in net.layers if L.kind == 'INPUT'), None)
    if input_layer is None: return False, None, {'fail': 'no_input_layer'}
    assert_layer = next((L for L in reversed(net.layers) if L.kind == 'ASSERT'), None)
    if assert_layer is None: return False, None, {'fail': 'no_assert'}
    pre_assert_id = (net.preds.get(assert_layer.id, [None]) or [None])[0]
    if pre_assert_id is None: return False, None, {'fail': 'no_pre_assert'}

    # Build MILP (will use pre_bounds and add Tjeng top-K)
    # For FAL: solve min C[r] @ y to find candidate y_lp; extract x* from solution
    # We need d_objective for top-K selection — use C[0] as proxy
    milp_data = build_forward_milp(net, lb_in, ub_in, pre_bounds=pre_bounds,
                                                   K_per_layer=K_per_layer,
                                                   d_objective=C[0] if C is not None else None,
                                                   output_layer_id=pre_assert_id)
    if milp_data is None: return False, None, {'fail': 'milp_build'}

    # Try each row: find input x* where C[r] @ y_lp <= t[r]
    candidates = []
    for r in range(C.shape[0]):
        d_row = C[r]
        if canon_kind == 'UNSAFE_LINEAR':
            # Minimize C[r] @ y; if min <= t[r] then unsafe region intersects LP feasible set
            obj_val, x_full = solve_lp_or_milp_for_witness(milp_data, pre_assert_id, d_row,
                                                                                  float(t[r]),
                                                                                  time_limit_s=solver_time_s)
            if obj_val is not None and x_full is not None and obj_val <= float(t[r]) + 1e-6:
                x_star = extract_input_from_lp_data(milp_data, input_layer.id, x_full)
                if x_star is not None:
                    candidates.append({'row': r, 'lp_obj': obj_val, 't': float(t[r]),
                                              'x_star': x_star.tolist()})

    if not candidates:
        return False, None, {'fail': 'no_candidates', **info}

    # ORT replay each candidate
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        in_name = sess.get_inputs()[0].name
    except Exception as e:
        return False, None, {'fail': f'ort_setup: {str(e)[:80]}', **info}

    for cand in candidates:
        x_star = np.asarray(cand['x_star'], dtype=np.float32)
        # Clip to input box (numerical safety)
        x_star = np.clip(x_star, lb_in.astype(np.float32), ub_in.astype(np.float32))
        # Reshape to ONNX input shape
        try:
            x_reshape = x_star.reshape((1,) + tuple(in_shape[1:])) if len(in_shape) > 1 else x_star.reshape(1, -1)
            y_actual = sess.run(None, {in_name: x_reshape.astype(np.float32)})[0].flatten()
        except Exception as e:
            continue
        # Check spec violation (AND-polytope CORRECT semantic for UNSAFE_LINEAR)
        cy = C @ y_actual
        if canon_kind == 'UNSAFE_LINEAR':
            # Unsafe iff ALL rows satisfy C @ y <= t (AND-polytope)
            if (cy <= t.astype(np.float64) + 1e-5).all():
                return True, x_star, {**info, 'ort_y': y_actual.tolist()[:10],
                                              'ort_cy': cy.tolist()[:10],
                                              'cand_row': cand['row']}
        elif canon_kind in ('LINEAR_LE', 'TOP1_ROBUST'):
            # Unsafe iff ANY row > t
            if (cy > t.astype(np.float64) + 1e-5).any():
                return True, x_star, {**info, 'ort_y': y_actual.tolist()[:10],
                                              'ort_cy': cy.tolist()[:10],
                                              'cand_row': cand['row']}
    return False, None, {'fail': 'ort_no_violation', **info, 'n_candidates': len(candidates)}


def extract_input_from_lp_data(milp_data: Dict, input_layer_id: int,
                                              x_full: np.ndarray) -> Optional[np.ndarray]:
    """Extract input portion x* from full LP solution vector."""
    input_idx = milp_data['layer_var_idx'].get(input_layer_id)
    if input_idx is None: return None
    start, dim = input_idx
    if x_full is None or len(x_full) < start + dim: return None
    return np.asarray(x_full[start:start + dim])
