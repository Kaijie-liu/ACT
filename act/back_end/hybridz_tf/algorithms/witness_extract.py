"""WitnessExtract: for SpecAwareLP `unknown` results, extract an x_in
candidate from the LP and try to confirm it is a true counterexample
via ONNX runtime replay.

Three escalating strategies in order:
  S1. LP corner: use the x_in from the feasibility LP solution directly.
  S2. Max-margin objective: re-solve LP with objective
        max t  s.t.  c_k · y ≤ d_k − t  for all unsafe rows k
      so the witness sits as deep in the unsafe set as the relaxation allows.
  S3. Per-row max-violation: for each unsafe row k, re-solve LP with
        max (c_k · y − d_k)   subject to other unsafe rows holding,
      gives a witness biased toward violating row k specifically.

For each candidate x_in, replay through ORT, check whether ALL unsafe
rows of THAT disjunct are violated by NN(x_in). If yes, the instance
is falsified. Returns ('falsified', x_in, y_ort) or ('unknown', None, None).

Soundness: the replay uses the *actual* ONNX runtime output, so a
reported 'falsified' is a true counterexample (input within the box
that produces an output satisfying the unsafe spec).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from act.back_end.hybridz_tf.algorithms import global_triangle_lp as base
from act.back_end.hybridz_tf.algorithms.global_triangle_lp import (
    extract_layers, parse_vnnlib, _lazy_imports,
    _add_rows_to_lp, _solve_one_obj,
)


def _build_and_refine_lp(sub_const, layers, output_layer, lb_x, ub_x, unsafe_rows,
                           time_limit_per_lp=15.0, max_refinement_passes=3):
    """Build the full SpecAware LP (including refinement) and return the
    HiGHS handle + variable offsets, ready for objective changes."""
    _lazy_imports()
    hp = base._hp
    n_in = lb_x.shape[0]
    n_out = output_layer[1].shape[0]
    n_l = [b.shape[0] for W, b in layers]
    var_offsets = {}; cur = 0
    var_offsets['x_in'] = (cur, cur + n_in); cur += n_in
    for li in range(len(layers)):
        var_offsets[f'pre_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
        var_offsets[f'post_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
    var_offsets['y'] = (cur, cur + n_out); cur += n_out
    nvars = cur

    lb_arr = np.full(nvars, -hp.kHighsInf); ub_arr = np.full(nvars, hp.kHighsInf)
    s, e = var_offsets['x_in']; lb_arr[s:e] = lb_x; ub_arr[s:e] = ub_x

    h = hp.Highs(); h.silent()
    h.setOptionValue("time_limit", float(time_limit_per_lp))
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    lp = hp.HighsLp(); lp.num_col_ = nvars; lp.num_row_ = 0
    lp.col_cost_ = [0.0] * nvars
    lp.col_lower_ = lb_arr.tolist(); lp.col_upper_ = ub_arr.tolist()
    lp.row_lower_ = []; lp.row_upper_ = []
    lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [0] * (nvars + 1)
    lp.a_matrix_.index_ = []; lp.a_matrix_.value_ = []
    h.passModel(lp)

    # Layer 0 affine
    W0, b0 = layers[0]
    in_to_out = (W0.shape[0] == n_in)
    s_pre0, _ = var_offsets['pre_0']; s_xin, _ = var_offsets['x_in']
    if sub_const is not None:
        b_eff0 = b0 - (sub_const @ W0 if in_to_out else W0 @ sub_const)
    else:
        b_eff0 = b0
    rows = []
    for j in range(n_l[0]):
        coefs = {s_pre0 + j: 1.0}
        for k in range(n_in):
            wkj = W0[k, j] if in_to_out else W0[j, k]
            if wkj != 0: coefs[s_xin + k] = -wkj
        rows.append(('eq', float(b_eff0[j]), list(coefs.items())))
    _add_rows_to_lp(h, rows)

    # Build hidden layers with LP-tight bounds + triangles
    layer_pre_bounds = []
    for li in range(len(layers)):
        n_li = n_l[li]
        s_pre, _ = var_offsets[f'pre_{li}']; s_post, _ = var_offsets[f'post_{li}']
        lb_pre = np.zeros(n_li); ub_pre = np.zeros(n_li)
        for j in range(n_li):
            st_lb, lb_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='min',
                                           time_limit=time_limit_per_lp)
            st_ub, ub_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='max',
                                           time_limit=time_limit_per_lp)
            if st_lb != 'ok' or st_ub != 'ok':
                return None  # fail
            lb_pre[j] = lb_j; ub_pre[j] = ub_j
        layer_pre_bounds.append((lb_pre, ub_pre))

        new_lb_post = np.maximum(0, lb_pre); new_ub_post = np.maximum(0, ub_pre)
        for j in range(n_li):
            h.changeColBounds(s_post + j, float(new_lb_post[j]), float(new_ub_post[j]))
        rows = []
        for j in range(n_li):
            l, u = float(lb_pre[j]), float(ub_pre[j])
            if l >= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0), (s_pre + j, -1.0)]))
            elif u <= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0)]))
            else:
                rows.append(('le', 0.0, [(s_pre + j, 1.0), (s_post + j, -1.0)]))
                lam = u / (u - l); rhs = -lam * l
                rows.append(('le', float(rhs), [(s_post + j, 1.0), (s_pre + j, -lam)]))
        _add_rows_to_lp(h, rows)

        if li + 1 < len(layers):
            W_n, b_n = layers[li + 1]; n_next = n_l[li + 1]
            in_to_out_n = (W_n.shape[0] == n_li)
            s_pre_n, _ = var_offsets[f'pre_{li + 1}']
            rows_n = []
            for j in range(n_next):
                coefs = {s_pre_n + j: 1.0}
                for k in range(n_li):
                    wkj = W_n[k, j] if in_to_out_n else W_n[j, k]
                    if wkj != 0: coefs[s_post + k] = -wkj
                rows_n.append(('eq', float(b_n[j]), list(coefs.items())))
            _add_rows_to_lp(h, rows_n)
        else:
            s_y, _ = var_offsets['y']
            s_xin_eff, _ = var_offsets['x_in']
            from act.back_end.hybridz_tf.algorithms.global_triangle_lp import output_affine_rows
            rows_y = output_affine_rows(output_layer, n_li, n_in, n_out,
                                          s_post=s_post, s_xin=s_xin_eff, s_y=s_y)
            _add_rows_to_lp(h, rows_y)

    # Spec rows (track first index so we can change row bounds for max-margin)
    s_y, _ = var_offsets['y']
    spec_first_row = h.getNumRow()
    for c_vec, d in unsafe_rows:
        entries = [(s_y + k, float(c_vec[k])) for k in range(n_out) if c_vec[k] != 0]
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        h.addRow(-hp.kHighsInf, float(d), len(cols), cols, vals)

    return (h, var_offsets, nvars, layer_pre_bounds, spec_first_row, n_in, n_out)


def _ort_replay(onnx_path: Path, x_in: np.ndarray, unsafe_rows):
    """Run ORT inference, return (y, is_real_counterex)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    in_meta = sess.get_inputs()[0]
    # ONNX shapes can contain symbolic dims (str) or unknown (0). Use 1 for both.
    in_shape = tuple(d if (isinstance(d, int) and d > 0) else 1 for d in in_meta.shape)
    x_arr = None
    y = None
    # Try the declared shape first, then common fallbacks
    candidate_shapes = [in_shape, (1, len(x_in)), (1, len(x_in), 1, 1), (len(x_in),), (1, 1, 1, len(x_in))]
    for shp in candidate_shapes:
        try:
            x_arr = np.asarray(x_in, dtype=np.float32).reshape(shp)
            y = sess.run(None, {in_meta.name: x_arr})[0].flatten()
            break
        except Exception:
            continue
    if y is None:
        return None, False
    # All unsafe rows must hold for a true counterexample
    eps = 1e-6
    for c_vec, d in unsafe_rows:
        if float(np.dot(c_vec, y)) > d + eps:
            return y, False
    return y, True


def try_falsify_disjunct(onnx_path, sub_const, layers, output_layer,
                          lb_x, ub_x, unsafe_rows,
                          time_limit_per_lp=15.0):
    """Run S1/S2/S3 in turn; return ('falsified', x, y) on first success, else ('unknown', None, None)."""
    _lazy_imports(); hp = base._hp

    built = _build_and_refine_lp(sub_const, layers, output_layer, lb_x, ub_x, unsafe_rows,
                                   time_limit_per_lp=time_limit_per_lp)
    if built is None:
        return ('unknown', None, None)
    h, var_offsets, nvars, layer_pre_bounds, spec_first_row, n_in, n_out = built

    # Strategy S1: solve as feasibility (zero obj), grab x_in
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), [0.0] * nvars)
    h.run()
    sm = h.getModelStatus()
    if sm == hp.HighsModelStatus.kInfeasible:
        # Verified already — shouldn't happen if we were called on an unknown
        return ('verified', None, None)
    if sm != hp.HighsModelStatus.kOptimal:
        return ('unknown', None, None)
    sol = h.getSolution()
    cv = np.asarray(sol.col_value, dtype=np.float64)
    s_xin, e_xin = var_offsets['x_in']
    x1 = cv[s_xin:e_xin].copy()
    y_ort, real = _ort_replay(Path(onnx_path), x1, unsafe_rows)
    if real:
        return ('falsified', x1, y_ort)

    # Strategy S2: per-row maximize the LHS of each unsafe row, see if any
    # individual maximizer corresponds to a real counterexample
    s_y, _ = var_offsets['y']
    for c_vec, d in unsafe_rows:
        obj = {}
        for k in range(n_out):
            if c_vec[k] != 0:
                obj[s_y + k] = float(c_vec[k])
        st, val = _solve_one_obj(h, nvars, obj, sense='max', time_limit=time_limit_per_lp)
        if st != 'ok':
            continue
        sol = h.getSolution()
        cv = np.asarray(sol.col_value, dtype=np.float64)
        x_cand = cv[s_xin:e_xin].copy()
        y_ort, real = _ort_replay(Path(onnx_path), x_cand, unsafe_rows)
        if real:
            return ('falsified', x_cand, y_ort)

    # Strategy S3: small random perturbations around each LP candidate
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Perturb x1 within box
        delta = rng.uniform(-1, 1, size=len(x1)) * 1e-3 * (ub_x - lb_x)
        x_pert = np.clip(x1 + delta, lb_x, ub_x)
        y_ort, real = _ort_replay(Path(onnx_path), x_pert, unsafe_rows)
        if real:
            return ('falsified', x_pert, y_ort)

    return ('unknown', None, None)


def verify_with_falsification(onnx_path, vnnlib_path,
                                time_limit_per_lp: float = 15.0,
                                max_refinement_passes: int = 3,
                                return_witness: bool = False):
    """SpecAwareLP then witness extraction on unknowns.

    Returns ('verified' | 'falsified' | 'unknown', elapsed_s) by default.
    If return_witness=True, returns
        ('falsified', x_witness, y_ort, elapsed_s) on falsification
        ('verified' | 'unknown', None, None, elapsed_s) otherwise.
    """
    import SpecAwareLP as sa
    _lazy_imports()
    t0 = time.time()
    op = Path(onnx_path); vp = Path(vnnlib_path)

    # First try standard SpecAware verification
    st, sa_elapsed = sa.verify(op, vp,
                                 time_limit_per_lp=time_limit_per_lp,
                                 max_refinement_passes=max_refinement_passes)
    if st == 'verified':
        if return_witness:
            return ('verified', None, None, time.time() - t0)
        return ('verified', time.time() - t0)

    # Unknown — try to falsify each disjunct
    sub_const, layers, output_layer = extract_layers(op)
    if sub_const is None:
        W0 = layers[0][0]; b0_dim = layers[0][1].shape[0]
        n_in = W0.shape[0] if W0.shape[1] == b0_dim else W0.shape[1]
    else:
        n_in = sub_const.shape[0]
    n_out = output_layer[1].shape[0]
    disjuncts = parse_vnnlib(vp, n_in, n_out)
    for lb_d, ub_d, unsafe_rows in disjuncts:
        if not (np.isfinite(lb_d).all() and np.isfinite(ub_d).all()):
            continue
        st2, x_w, y_w = try_falsify_disjunct(op, sub_const, layers, output_layer,
                                                lb_d, ub_d, unsafe_rows,
                                                time_limit_per_lp=time_limit_per_lp)
        if st2 == 'falsified':
            if return_witness:
                return ('falsified', x_w, y_w, time.time() - t0)
            return ('falsified', time.time() - t0)
    if return_witness:
        return ('unknown', None, None, time.time() - t0)
    return ('unknown', time.time() - t0)
