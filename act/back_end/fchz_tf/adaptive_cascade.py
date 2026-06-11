# ===- act/back_end/fchz_tf/adaptive_cascade.py - Adaptive Cascade Verifier ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Adaptive cascade verification policy. ONE deterministic strategy applied
#   uniformly to every instance:
#
#     Stage 0: FCHZ closed-form propagation       (always run, fast)
#     Stage 1: K=20 MILP default                  (60s budget)
#     Stage 2: K=20 + K_eq_last=99 on last ReLU    (120s budget)
#     Stage 3: K=20 + analytical chord K1          (100s budget)
#     Stage 4: K=20 + K_eq_last=99 + chord K2      (150s, only for
#                                                   Sigmoid/Tanh models)
#
#   Total budget: configurable, default 480s (VNN-COMP standard).
#
#   This is NOT per-benchmark tuning. The cascade order and budgets are
#   FIXED. The cascade decisions are based on:
#     - Verification result (CERT/FAL → early exit)
#     - Time remaining (skip stage if not enough budget)
#     - Model architecture (has_sigmoid_or_tanh → enables conditional Stage 4)
#
#   All these decisions are deterministic per (model, spec), independent
#   of benchmark name. Same model + same spec → same cascade path.
#
# ===---------------------------------------------------------------------===#
"""Adaptive cascade verifier: single deterministic policy for full dataset."""
import os
import time
import hashlib
import numpy as np
from typing import Optional, Tuple, Dict, Any


def has_sigmoid_or_tanh(net) -> bool:
    """Check if model has Sigmoid or Tanh layers."""
    for L in net.layers:
        if L.kind in ('SIGMOID', 'TANH'):
            return True
    return False


def verify_adaptive_cascade(net, lb, ub, queries, n_out, pre_bounds, pre_assert_id,
                                              input_layer_id: int,
                                              total_timeout: float = 480.0,
                                              session_factory=None,
                                              instances_path=None,
                                              input_shape=None) -> Dict[str, Any]:
    """Verify with adaptive cascade policy. Returns dict with verdict.

    Cascade stages applied uniformly to all instances:
      Stage 0: FCHZ closed-form  (always run)
      Stage 1: K=20 MILP default  (~60s)
      Stage 2: K=20 + K_eq_last=99 on last ReLU (~120s)
      Stage 3: K=20 + analytical chord K1 (~100s)
      Stage 4: K=20 + K_eq_last=99 + chord K2
               (~150s, only if model has Sigmoid/Tanh)

    Args:
      net: ACT Net
      lb, ub: input box bounds (flat)
      queries: parsed VNN-LIB queries
      n_out: output dimension
      pre_bounds: per-layer FCHZ pre-activation bounds
      pre_assert_id: pre-spec layer id
      input_layer_id: input layer id (for x* extraction)
      total_timeout: VNN-COMP timeout (default 480s)
      session_factory: callable returning ORT session for replay (for FAL)
      instances_path: optional path for logging
      input_shape: optional original ONNX input shape used for strict ORT
                   replay. If absent, FAL replay falls back to flat input.

    Returns:
      dict with 'verdict' (CERT/FAL/UNK), 'witness_info', 'mechanism',
      'cascade_stage_reached', 'elapsed_s'.
    """
    from act.back_end.fchz_tf.spec_canonicalize import canonicalize_queries
    from act.back_end.fchz_tf.forward_global_milp import (
        build_forward_milp, solve_forward_milp_lb_with_receipt)
    from act.back_end.fchz_tf.forward_global_fal import (
        solve_lp_or_milp_for_witness_with_receipt, extract_input_from_lp_data)
    from act.back_end.fchz_tf.verifier_fchz import verify_once_fchz
    from act.back_end.transfer_functions import get_transfer_function

    result = {'verdict': 'UNK', 'cascade_stage_reached': 0, 'elapsed_s': 0.0,
                'mechanism': None, 'witness_info': None}
    t_start = time.time()
    remaining = total_timeout

    # ─── Stage 0: FCHZ closed-form ───
    t0 = time.time()
    tf = get_transfer_function()
    fchz_result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)
    stage0_t = time.time() - t0
    remaining -= stage0_t
    result['cascade_stage_reached'] = 0
    if fchz_result['verdict'] == 'CERTIFIED':
        result['verdict'] = 'CERT'
        result['mechanism'] = 'FCHZ closed-form'
        # ─── Closed-form CERT receipt ───
        # Stage 0 uses no solver — the FCHZ interval bound itself proves
        # the spec. The receipt records the FCHZ propagation outcome and
        # the input box digest so an auditor can re-derive the bound.
        try:
            lb_sha = hashlib.sha256(np.asarray(lb, dtype=np.float64).tobytes()).hexdigest()
            ub_sha = hashlib.sha256(np.asarray(ub, dtype=np.float64).tobytes()).hexdigest()
        except Exception:
            lb_sha = ub_sha = None
        result['cert_receipt'] = {
            'mechanism': 'FCHZ closed-form (Stage 0)',
            'solver': 'none (interval bound)',
            'fchz_verdict': fchz_result.get('verdict'),
            'fchz_margin': fchz_result.get('margin'),
            'input_lb_sha256': lb_sha,
            'input_ub_sha256': ub_sha,
            'input_dim': int(np.asarray(lb).size),
            'output_n_out': int(n_out),
            'topk_select': os.environ.get('ACT_HZ_TOPK_SELECT', 'width'),
            'stage0_elapsed_s': float(stage0_t),
        }
        result['elapsed_s'] = time.time() - t_start
        return result

    # Parse spec
    canon = canonicalize_queries(queries, n_out)
    canon_kind = canon.get('kind', '?')
    if canon_kind == 'MULTI_QUERY':
        qs_list = []
        for in_s, out_s in canon.get('queries', []):
            C_q = (out_s.c.detach().cpu().numpy() if hasattr(out_s.c, 'detach')
                       else np.asarray(out_s.c)).astype(np.float64)
            if C_q.ndim == 1: C_q = C_q.reshape(1, -1)
            t_q = (out_s.d.detach().cpu().numpy() if hasattr(out_s.d, 'detach')
                       else np.asarray(out_s.d)).astype(np.float64).reshape(-1)
            if t_q.shape[0] == 1 and C_q.shape[0] > 1:
                t_q = np.broadcast_to(t_q, (C_q.shape[0],))
            qs_list.append((out_s.kind, C_q, t_q))
    else:
        qs_list = [(canon_kind, canon.get('C'), canon.get('t'))]
    d_for_select = qs_list[0][1][0] if qs_list and qs_list[0][1] is not None else None

    # Stage budgets — proportional to total_timeout but always honor the actual
    # remaining budget. The fractions are tuned for VNN-COMP 480 s but the same
    # cascade is correct for any timeout: each stage clamps to remaining-εoverhead
    # and the per-stage MILP/LP solver gets its own time_limit. For very short
    # official timeouts (e.g. safenlp_2024=20 s, cora_2024=30 s) the cascade
    # collapses to "Stage 0 + best-effort Stage 1" rather than UNK-bailing.
    stage_configs = [
        ('K=20 default',         20, None, None, '0',  0.125,  60.0),
        ('K=99 eq_layers',       20, 1,    99,   '0',  0.25,  120.0),
        ('K=20 chord K1',        20, None, None, '1',  0.21,  100.0),
    ]
    if has_sigmoid_or_tanh(net):
        stage_configs.append(
            ('K=99 eq + chord K2',  20, 1,    99,   'K2', 0.31,  150.0))

    # Reserve a small overhead for the final ORT replay path; never go below 1 s.
    OVERHEAD_S = 1.0
    for stage_idx, (name, K, eq_layers, K_eq, chord_mode, frac, max_budget) \
            in enumerate(stage_configs, start=1):
        # Always allocate from the CURRENT remaining (not the pre-computed
        # snapshot). Stop only when truly nothing left.
        if remaining <= OVERHEAD_S:
            break
        budget = min(remaining * frac, max_budget, remaining - OVERHEAD_S)
        if budget < 0.5:   # too small to do anything useful
            continue
        result['cascade_stage_reached'] = stage_idx

        # Set env vars for this stage
        os.environ['ACT_HZ_SIGMOID_CHORD'] = chord_mode
        if eq_layers is not None:
            os.environ['ACT_HZ_EQ_LAYERS'] = str(eq_layers)
            os.environ['ACT_HZ_K_EQ_LAST'] = str(K_eq)
        else:
            os.environ.pop('ACT_HZ_EQ_LAYERS', None)
            os.environ.pop('ACT_HZ_K_EQ_LAST', None)

        t_stage = time.time()

        # Build MILP for this stage
        milp = build_forward_milp(net, lb, ub, pre_bounds=pre_bounds,
                                                K_per_layer=K, d_objective=d_for_select,
                                                output_layer_id=pre_assert_id)
        if milp is None:
            continue

        # Try CERT first (cheaper if it works).
        #
        # SOUNDNESS NOTE (2026-06-11 fix):
        #   For MULTI_QUERY canonicalized from vnnlib `(assert (or Q0 Q1 ...))`,
        #   the spec is SAFE iff ALL sub-queries individually CERT
        #   (canon_verdict in spec_canonicalize.py implements this for Stage 0).
        #   The prior implementation here did `if cert: break` after a single
        #   sub-query CERT — that was unsound for OR-disjunction specs and
        #   produced false CERTs on collins_rul_cnn_2022 / cersyve /
        #   tllverifybench at official_run on 2026-06-10. The corrected loop
        #   below tracks per-sub-query CERT and only declares the overall
        #   verdict CERT once every sub-query has been individually certified.
        #   Single-query canon (TOP1_ROBUST / LINEAR_LE / UNSAFE_LINEAR with
        #   1 query) reduces to "all of 1" so the semantics are unchanged for
        #   those cases.
        cert_per_query: list = [False] * len(qs_list)
        cert_receipts_per_query: list = [None] * len(qs_list)
        cert_obj_per_query: list = [None] * len(qs_list)
        cert_threshold_per_query: list = [None] * len(qs_list)
        cert_row_per_query: list = [None] * len(qs_list)
        cert_kind_per_query: list = [None] * len(qs_list)
        # last_rcpt holds the most recent solver receipt for diagnostic use
        # if every query proves CERT
        last_rcpt: Optional[Dict[str, Any]] = None
        for qi, (q_kind, C_q, t_q) in enumerate(qs_list):
            if C_q is None: continue
            C_q = np.asarray(C_q).astype(np.float64)
            if C_q.ndim == 1: C_q = C_q.reshape(1, -1)
            t_q = np.asarray(t_q).astype(np.float64).reshape(-1)
            time_per_row = max(5.0, min(15.0, (budget - (time.time() - t_stage)) / max(1, C_q.shape[0])))
            if q_kind == 'UNSAFE_LINEAR':
                # UNSAFE_LINEAR (AND-of-rows convention): safe iff ANY single row
                # has lower-bound > t. We can stop at the first such row.
                for r in range(C_q.shape[0]):
                    if time.time() - t_stage > budget * 0.6: break
                    v, rcpt = solve_forward_milp_lb_with_receipt(milp, pre_assert_id, C_q[r], time_limit_s=time_per_row)
                    last_rcpt = rcpt
                    if v is not None and v > float(t_q[r]) + 1e-9:
                        cert_per_query[qi] = True
                        cert_receipts_per_query[qi] = rcpt
                        cert_obj_per_query[qi] = float(v)
                        cert_threshold_per_query[qi] = float(t_q[r])
                        cert_row_per_query[qi] = r
                        cert_kind_per_query[qi] = q_kind
                        break
            elif q_kind in ('LINEAR_LE', 'TOP1_ROBUST'):
                # LINEAR_LE / TOP1_ROBUST: safe iff EVERY row's upper bound < t.
                ok = True
                row_receipts: list = []
                for r in range(C_q.shape[0]):
                    if time.time() - t_stage > budget * 0.6:
                        ok = False; break
                    vn, rcpt = solve_forward_milp_lb_with_receipt(milp, pre_assert_id, -C_q[r], time_limit_s=time_per_row)
                    last_rcpt = rcpt
                    if vn is None: ok = False; break
                    if not (-vn < float(t_q[r]) - 1e-9): ok = False; break
                    row_receipts.append({'row': int(r), 'obj_value': float(-vn), 'threshold': float(t_q[r]),
                                                'solver_status': rcpt.get('status')})
                if ok:
                    cert_per_query[qi] = True
                    cert_receipts_per_query[qi] = {
                        'solver': last_rcpt.get('solver') if last_rcpt else 'scipy.optimize.milp (HiGHS)',
                        'rows_proved': len(row_receipts),
                        'row_details': row_receipts,
                        'var_count': last_rcpt.get('var_count') if last_rcpt else None,
                        'n_binary': last_rcpt.get('n_binary') if last_rcpt else None,
                        'n_ub_rows': last_rcpt.get('n_ub_rows') if last_rcpt else None,
                        'n_eq_rows': last_rcpt.get('n_eq_rows') if last_rcpt else None,
                    }
                    cert_kind_per_query[qi] = q_kind
            else:
                # Unknown query kind — cannot CERT. Mark as failed so the AND
                # over per-query CERT will reject this stage.
                cert_per_query[qi] = False
            # Early exit: if this sub-query cannot be CERTed, no point trying
            # the rest for this stage — the overall AND-of-sub-queries fails.
            if not cert_per_query[qi]:
                break

        # Overall CERT iff ALL sub-queries individually CERT'd (and we actually
        # attempted all of them — a partial loop with cert_per_query rolled
        # forward but qi < len-1 means we bailed early because some sub-query
        # failed; that already sets `cert` overall to False).
        cert = all(cert_per_query) and len(cert_per_query) > 0
        # For the receipt block, use the first CERT'd sub-query as the
        # representative witness (back-compat with single-query receipt schema).
        if cert:
            for qi_i in range(len(cert_per_query)):
                if cert_receipts_per_query[qi_i] is not None:
                    cert_receipt = cert_receipts_per_query[qi_i]
                    cert_qi = qi_i
                    cert_q_kind = cert_kind_per_query[qi_i]
                    cert_obj_val = cert_obj_per_query[qi_i]
                    cert_threshold = cert_threshold_per_query[qi_i]
                    cert_row = cert_row_per_query[qi_i]
                    break
        else:
            cert_receipt = None
            cert_qi = cert_row = None
            cert_q_kind = None
            cert_obj_val = None
            cert_threshold = None

        if cert:
            result['verdict'] = 'CERT'
            result['mechanism'] = f'MILP CERT (stage {stage_idx}: {name})'
            result['K_per_layer'] = K
            result['elapsed_s'] = time.time() - t_start
            # ─── CERT receipt ───
            result['cert_receipt'] = {
                'spec_kind': cert_q_kind,
                'spec_qi': cert_qi,
                'witness_row': cert_row,
                'objective_value': cert_obj_val,
                'spec_threshold': cert_threshold,
                'margin': (cert_obj_val - cert_threshold) if (cert_obj_val is not None and cert_threshold is not None) else None,
                'solver': cert_receipt.get('solver') if cert_receipt else None,
                'solver_status': cert_receipt.get('status') if cert_receipt else None,
                'solver_success': cert_receipt.get('success') if cert_receipt else None,
                'mip_gap': cert_receipt.get('mip_gap') if cert_receipt else None,
                'mip_dual_bound': cert_receipt.get('mip_dual_bound') if cert_receipt else None,
                'mip_node_count': cert_receipt.get('mip_node_count') if cert_receipt else None,
                'var_count': cert_receipt.get('var_count') if cert_receipt else None,
                'n_binary': cert_receipt.get('n_binary') if cert_receipt else None,
                'n_ub_rows': cert_receipt.get('n_ub_rows') if cert_receipt else None,
                'n_eq_rows': cert_receipt.get('n_eq_rows') if cert_receipt else None,
                'rows_proved': cert_receipt.get('rows_proved') if cert_receipt else None,
                'time_limit_s_per_row': cert_receipt.get('time_limit_s') if cert_receipt else None,
                'topk_select': os.environ.get('ACT_HZ_TOPK_SELECT', 'width'),
            }
            return result

        # Try LP-witness FAL with remaining stage budget
        if session_factory is not None:
            sess, in_name = session_factory()
            # Capture ORT provider list for the receipt
            try:
                ort_providers = list(sess.get_providers())
            except Exception:
                ort_providers = ['unknown']
            t_fal_budget = budget - (time.time() - t_stage)
            if t_fal_budget < 5:
                remaining -= time.time() - t_stage
                continue
            t_per_row = max(3.0, min(10.0, t_fal_budget / max(1, qs_list[0][1].shape[0] if qs_list[0][1] is not None else 1)))
            for qi, (q_kind, C_q, t_q) in enumerate(qs_list):
                if C_q is None: continue
                C_q = np.asarray(C_q).astype(np.float64)
                if C_q.ndim == 1: C_q = C_q.reshape(1, -1)
                t_q = np.asarray(t_q).astype(np.float64).reshape(-1)
                replay_shape = tuple(input_shape) if input_shape is not None else None
                if q_kind == 'UNSAFE_LINEAR':
                    for r in range(C_q.shape[0]):
                        if time.time() - t_stage > budget: break
                        obj, x_full, lp_rcpt = solve_lp_or_milp_for_witness_with_receipt(
                            milp, pre_assert_id, C_q[r], float(t_q[r]),
                            time_limit_s=t_per_row, use_milp=True)
                        if obj is None or x_full is None or obj > float(t_q[r]) + 1e-6: continue
                        x_star = extract_input_from_lp_data(milp, input_layer_id, x_full)
                        if x_star is None: continue
                        x_star = np.asarray(x_star).astype(np.float32)
                        box_ok = bool((x_star >= lb.astype(np.float32) - 1e-7).all() and
                                              (x_star <= ub.astype(np.float32) + 1e-7).all())
                        if not box_ok: continue
                        try:
                            x_re = x_star.reshape(replay_shape) if replay_shape is not None else x_star.reshape(1, -1)
                            y_act = sess.run(None, {in_name: x_re.astype(np.float32)})[0].flatten()
                        except Exception as e:
                            continue
                        cy = (C_q @ y_act).astype(np.float64)
                        margins = (t_q.astype(np.float64) - cy)
                        if (cy <= t_q.astype(np.float64)).all():
                            result['verdict'] = 'FAL'
                            result['mechanism'] = f'LP-witness FAL (stage {stage_idx}: {name})'
                            result['witness_info'] = {'qi': qi, 'row': int(r),
                                                              'lp_obj': float(obj), 't': float(t_q[r]),
                                                              'ort_cy': cy.tolist()[:5]}
                            result['fal_receipt'] = _build_fal_receipt(
                                q_kind=q_kind, qi=qi, row=int(r),
                                lp_obj=float(obj), spec_threshold=float(t_q[r]),
                                lp_rcpt=lp_rcpt, x_star=x_star, lb=lb, ub=ub,
                                box_ok=box_ok, replay_shape=replay_shape,
                                y_act=y_act, cy=cy, t_q=t_q,
                                margins=margins, ort_providers=ort_providers,
                                topk_select=os.environ.get('ACT_HZ_TOPK_SELECT', 'width'))
                            result['elapsed_s'] = time.time() - t_start
                            return result
                elif q_kind in ('LINEAR_LE', 'TOP1_ROBUST'):
                    for r in range(C_q.shape[0]):
                        if time.time() - t_stage > budget: break
                        obj_neg, x_full, lp_rcpt = solve_lp_or_milp_for_witness_with_receipt(
                            milp, pre_assert_id, -C_q[r], -float(t_q[r]),
                            time_limit_s=t_per_row, use_milp=True)
                        if obj_neg is None or x_full is None: continue
                        obj = -obj_neg
                        if obj < float(t_q[r]) - 1e-6: continue
                        x_star = extract_input_from_lp_data(milp, input_layer_id, x_full)
                        if x_star is None: continue
                        x_star = np.asarray(x_star).astype(np.float32)
                        box_ok = bool((x_star >= lb.astype(np.float32) - 1e-7).all() and
                                              (x_star <= ub.astype(np.float32) + 1e-7).all())
                        if not box_ok: continue
                        try:
                            x_re = x_star.reshape(replay_shape) if replay_shape is not None else x_star.reshape(1, -1)
                            y_act = sess.run(None, {in_name: x_re.astype(np.float32)})[0].flatten()
                        except Exception:
                            continue
                        cy = (C_q @ y_act).astype(np.float64)
                        margins = (cy - t_q.astype(np.float64))
                        if (cy > t_q.astype(np.float64)).any():
                            result['verdict'] = 'FAL'
                            result['mechanism'] = f'LP-witness FAL (stage {stage_idx}: {name})'
                            result['witness_info'] = {'qi': qi, 'row': int(r),
                                                              'lp_obj': float(obj), 't': float(t_q[r]),
                                                              'ort_cy': cy.tolist()[:5]}
                            result['fal_receipt'] = _build_fal_receipt(
                                q_kind=q_kind, qi=qi, row=int(r),
                                lp_obj=float(obj), spec_threshold=float(t_q[r]),
                                lp_rcpt=lp_rcpt, x_star=x_star, lb=lb, ub=ub,
                                box_ok=box_ok, replay_shape=replay_shape,
                                y_act=y_act, cy=cy, t_q=t_q,
                                margins=margins, ort_providers=ort_providers,
                                topk_select=os.environ.get('ACT_HZ_TOPK_SELECT', 'width'))
                            result['elapsed_s'] = time.time() - t_start
                            return result

        # Update remaining budget
        remaining -= (time.time() - t_stage)

    result['elapsed_s'] = time.time() - t_start
    return result


def _build_fal_receipt(*, q_kind, qi, row, lp_obj, spec_threshold,
                                      lp_rcpt, x_star, lb, ub, box_ok, replay_shape,
                                      y_act, cy, t_q, margins, ort_providers,
                                      topk_select) -> Dict[str, Any]:
    """Build the audit-grade FAL receipt.

    Captures everything an independent auditor needs to re-replay the witness:
      - Solver receipt (objective + status + MIP gap + var/constraint counts)
      - Input witness hash (sha256 of x_star bytes) for tamper detection
      - Box containment check (lb ≤ x_star ≤ ub, with margins)
      - ORT replay output hash (first 32 chars) and full numeric output digest
      - Worst-case violation margin (how strongly the spec is violated)
      - no_clip = True flag (we never clip a solver-returned candidate; box-OOB
        candidates are rejected outright, per strict witness contract)
      - ORT provider list (CPUExecutionProvider for this run)
      - Selection policy used (ACT_HZ_TOPK_SELECT)
    """
    x_star_arr = np.asarray(x_star, dtype=np.float32)
    y_act_arr = np.asarray(y_act, dtype=np.float64).reshape(-1)
    x_hash = hashlib.sha256(x_star_arr.tobytes()).hexdigest()
    y_hash = hashlib.sha256(y_act_arr.tobytes()).hexdigest()
    lb_f = lb.astype(np.float32); ub_f = ub.astype(np.float32)
    box_lb_slack = float((x_star_arr - lb_f).min()) if x_star_arr.size else None
    box_ub_slack = float((ub_f - x_star_arr).min()) if x_star_arr.size else None
    violation_margin = (
        float(margins.min()) if q_kind == 'UNSAFE_LINEAR' and margins.size else
        float(margins.max()) if margins.size else None)
    return {
        'spec_kind': q_kind,
        'spec_qi': qi,
        'witness_row': row,
        'lp_objective': lp_obj,
        'spec_threshold': spec_threshold,
        'no_clip': True,
        'box_check': {
            'in_box': bool(box_ok),
            'min_lb_slack': box_lb_slack,
            'min_ub_slack': box_ub_slack,
            'tol': 1e-7,
        },
        'witness_x_sha256': x_hash,
        'witness_x_dim': int(x_star_arr.size),
        'witness_x_replay_shape': list(replay_shape) if replay_shape else None,
        'ort': {
            'providers': list(ort_providers),
            'y_sha256': y_hash,
            'y_dim': int(y_act_arr.size),
            'y_head': y_act_arr[:8].tolist(),
        },
        'violation_margin': violation_margin,
        'all_row_margins_head': margins.tolist()[:8] if margins.size else [],
        'solver': lp_rcpt.get('solver') if lp_rcpt else None,
        'solver_kind': lp_rcpt.get('solver_kind') if lp_rcpt else None,
        'solver_status': lp_rcpt.get('status') if lp_rcpt else None,
        'solver_success': lp_rcpt.get('success') if lp_rcpt else None,
        'mip_gap': lp_rcpt.get('mip_gap') if lp_rcpt else None,
        'var_count': lp_rcpt.get('var_count') if lp_rcpt else None,
        'n_binary': lp_rcpt.get('n_binary') if lp_rcpt else None,
        'n_ub_rows': lp_rcpt.get('n_ub_rows') if lp_rcpt else None,
        'n_eq_rows': lp_rcpt.get('n_eq_rows') if lp_rcpt else None,
        'time_limit_s_per_row': lp_rcpt.get('time_limit_s') if lp_rcpt else None,
        'topk_select': topk_select,
    }


__all__ = ['verify_adaptive_cascade', 'has_sigmoid_or_tanh']
