"""Acasxu CERT audit V2 — with correct field naming + decisive analysis.

Per advisor 2026-06-08:
  - field overall_cert_via_m4_lp WAS MISLEADING
  - replace with:
      cf_under_correct_unsafelinear_semantics: bool  (CF can CERT under correct conjunctive UNSAFE_LINEAR)
      m4_lp_decisive: bool                                    (M4 LP made the difference vs CF)
      m4_lp_marginal_improvement: float                  (how much LP improved bound)
  - This was the canon-fix correction, NOT M4 LP, for iids 107 and 132.
  - For iid 102, M4 LP IS decisive (cf can't CERT but LP can).
"""
import sys, os, json, hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()


def audit_one(iid):
    from act.front_end.vnnlib_loader.onnx_converter import (
        convert_onnx_to_pytorch, get_onnx_input_shape)
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.front_end.verifiable_model import (
        VerifiableModel, InputLayer, InputSpecLayer, OutputSpecLayer)
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.transfer_functions import (
        set_transfer_function_mode, get_transfer_function)
    from act.back_end.core import Bounds
    from act.back_end.fchz_tf.verifier_fchz import (
        fchz_upper_bound, fchz_lower_bound, verify_once_fchz)
    from act.back_end.fchz_tf.spec_canonicalize import canonicalize_queries
    from research.fchz.m4_full_lp import solve_full_lp, is_dense_only_chain
    import onnx

    os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
    os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

    base = Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023')
    with open(base / 'instances.csv') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    onnx_p = base / rows[iid][0]; vnn_p = base / rows[iid][1]

    in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
    pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))
    labeled = LabeledInputTensor(tensor=torch.zeros(in_shape, dtype=torch.float32),
                                              label=torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
    in_spec0, out_spec0 = queries[0]
    if in_spec0.kind == 'BOX':
        lb = in_spec0.lb.detach().cpu().numpy().reshape(-1)
        ub = in_spec0.ub.detach().cpu().numpy().reshape(-1)
    else:
        center = in_spec0.center.detach().cpu().numpy().reshape(-1)
        eps = float(in_spec0.eps); lb = center - eps; ub = center + eps

    verifiable = VerifiableModel(
        input_layer=InputLayer(labeled_input=labeled, shape=in_shape, dtype=torch.float32),
        input_spec=InputSpecLayer(in_spec0), model=pytorch_model, output_spec=OutputSpecLayer(out_spec0))
    net = TorchToACT(verifiable).run()
    set_transfer_function_mode("fchz")
    tf = get_transfer_function()
    input_bounds = Bounds(lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                                  ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
    before, after = {}, {}
    for L in net.layers:
        in_b = input_bounds if L.id == 0 or not net.preds.get(L.id) else after[net.preds[L.id][0]].bounds
        after[L.id] = tf.apply(L, in_b, net, before, after)

    m = onnx.load(str(onnx_p))
    od = [d.dim_value if d.dim_value > 0 else 1
             for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_out = int(np.prod(od[1:])) if len(od) > 1 else od[0]

    pre_assert = net.preds.get(next(L for L in reversed(net.layers) if L.kind == 'ASSERT').id, [None])[0]
    state = tf._state_cache.get(pre_assert)

    # Production verify_once_fchz (uses CORRECTED canonicalize, no M4 LP)
    cf_result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)
    cf_corrected_cert = (cf_result['verdict'] == 'CERTIFIED')

    # Per-query audit using UNSAFE_LINEAR semantics directly
    canon = canonicalize_queries(queries, n_out)
    query_evidence = []
    for q_i, (q_in, q_out) in enumerate(queries):
        if q_out.kind == 'UNSAFE_LINEAR':
            C = q_out.c.detach().cpu().numpy().astype(np.float64)
            t = q_out.d.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if C.ndim == 1: C = C.reshape(1, -1)
            rows_evidence = []
            any_row_cf_certs = False
            any_row_lp_certs = False
            for i in range(C.shape[0]):
                d = C[i]
                cf_lb = float(fchz_lower_bound(state, d.reshape(1, -1))[0])
                lp_lb, diag_lb = solve_full_lp(net, tf, d, sense='min')
                threshold = float(t[i] if i < len(t) else t[0])
                # UNSAFE_LINEAR conjunctive: any row LB > t → unreachable → CERT
                row_cf_unreach = cf_lb > threshold
                row_lp_unreach = lp_lb > threshold if lp_lb is not None else False
                if row_cf_unreach: any_row_cf_certs = True
                if row_lp_unreach: any_row_lp_certs = True
                rows_evidence.append({
                    'row_idx': i,
                    'threshold': threshold,
                    'cf_lb': cf_lb,
                    'lp_lb': lp_lb,
                    'lp_improvement_over_cf': (lp_lb - cf_lb) if lp_lb is not None else 0.0,
                    'row_cf_certifies': row_cf_unreach,
                    'row_lp_certifies': row_lp_unreach,
                    'lb_eq_resid': diag_lb.get('eq_resid', 0),
                    'lb_lp_status': diag_lb.get('lp_status'),
                })
            query_evidence.append({
                'query_idx': q_i, 'kind': 'UNSAFE_LINEAR',
                'n_rows': C.shape[0], 'rows': rows_evidence,
                'cf_certifies_query': any_row_cf_certs,
                'lp_certifies_query': any_row_lp_certs,
            })

    cf_under_correct = all(q.get('cf_certifies_query', False) for q in query_evidence)
    lp_under_correct = all(q.get('lp_certifies_query', False) for q in query_evidence)
    m4_lp_decisive = lp_under_correct and not cf_under_correct

    return {
        'bench': 'acasxu_2023', 'iid': iid,
        'onnx_path': str(onnx_p), 'onnx_sha256': sha256_file(onnx_p),
        'vnnlib_path': str(vnn_p), 'vnnlib_sha256': sha256_file(vnn_p),
        'n_queries': len(queries),
        'dense_only': is_dense_only_chain(net),
        'state_K': int(state.G.shape[1]) if state.G is not None else 0,
        'state_n': int(state.c.shape[0]),
        'queries': query_evidence,
        'verify_once_fchz_verdict': cf_result['verdict'],
        'cf_under_correct_unsafelinear_semantics': cf_under_correct,
        'lp_under_correct_unsafelinear_semantics': lp_under_correct,
        'm4_lp_decisive': m4_lp_decisive,
        'source_attribution': (
            'canonicalize_semantics_fix' if cf_under_correct and not m4_lp_decisive
            else ('m4_lp_refinement' if m4_lp_decisive
                     else ('not_certified' if not lp_under_correct else 'unknown_source'))
        ),
    }


def main():
    out_dir = Path('/data1/Kane/ACT/audit_results/m4_acasxu_cert_audit_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'audit_iids_102_107_132_v2.json'
    audits = {}
    for iid in (102, 107, 132):
        print(f"Auditing acasxu/{iid}...")
        audits[iid] = audit_one(iid)
        a = audits[iid]
        print(f"  → CF correct: {a['cf_under_correct_unsafelinear_semantics']}, "
                  f"LP: {a['lp_under_correct_unsafelinear_semantics']}, "
                  f"M4 LP decisive: {a['m4_lp_decisive']}, "
                  f"source: {a['source_attribution']}")
    with open(out_path, 'w') as f:
        json.dump(audits, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Final attribution ===")
    canon_fix_count = sum(1 for a in audits.values() if a['source_attribution'] == 'canonicalize_semantics_fix')
    m4_lp_count = sum(1 for a in audits.values() if a['source_attribution'] == 'm4_lp_refinement')
    print(f"  Canon fix: {canon_fix_count} V (semantic correction)")
    print(f"  M4 LP:     {m4_lp_count} V (LP refinement)")


if __name__ == '__main__':
    main()
