"""Audit the 3 acasxu CERT-flips from M4 smoke (iid 102, 107, 132).

Per advisor 2026-06-08: before counting these as production V, capture:
  - per-row LP UB and LB
  - solver status, residuals
  - n_queries, kind per query
  - vnnlib path + sha256
  - input box bounds
Save to JSON for reproducibility.
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
        fchz_upper_bound, fchz_lower_bound)
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

    # Per-query audit
    query_evidence = []
    for q_i, (q_in, q_out) in enumerate(queries):
        if q_out.kind == 'UNSAFE_LINEAR':
            C = q_out.c.detach().cpu().numpy().astype(np.float64)
            t = q_out.d.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if C.ndim == 1: C = C.reshape(1, -1)
            rows_evidence = []
            for i in range(C.shape[0]):
                d = C[i]
                cf_lb = float(fchz_lower_bound(state, d.reshape(1, -1))[0])
                cf_ub = float(fchz_upper_bound(state, d.reshape(1, -1))[0])
                lp_lb, diag_lb = solve_full_lp(net, tf, d, sense='min')
                lp_ub, diag_ub = solve_full_lp(net, tf, d, sense='max')
                eff_lb = max(cf_lb, lp_lb if lp_lb is not None else float('-inf'))
                eff_ub = min(cf_ub, lp_ub if lp_ub is not None else float('inf'))
                threshold = float(t[i] if i < len(t) else t[0])
                row_safe = eff_lb > threshold   # UNSAFE_LINEAR: any row unreachable
                rows_evidence.append({
                    'row_idx': i,
                    'C_row_nz': [(int(k), float(d[k])) for k in np.where(np.abs(d) > 1e-9)[0]],
                    'threshold': threshold,
                    'cf_lb': cf_lb, 'cf_ub': cf_ub,
                    'lp_lb': lp_lb, 'lp_ub': lp_ub,
                    'eff_lb': eff_lb, 'eff_ub': eff_ub,
                    'lb_eq_resid': diag_lb.get('eq_resid'),
                    'ub_eq_resid': diag_ub.get('eq_resid'),
                    'lb_lp_status': diag_lb.get('lp_status'),
                    'ub_lp_status': diag_ub.get('lp_status'),
                    'row_certifies_polytope_unreachable': row_safe,
                    'lp_n_vars': diag_lb.get('n_vars'),
                    'lp_n_eq': diag_lb.get('n_eq'),
                    'lp_n_ub': diag_lb.get('n_ub'),
                })
            query_safe = any(r['row_certifies_polytope_unreachable'] for r in rows_evidence)
            query_evidence.append({
                'query_idx': q_i, 'kind': 'UNSAFE_LINEAR',
                'n_rows': C.shape[0], 'rows': rows_evidence,
                'query_certifies_safe': query_safe,
            })
        else:
            query_evidence.append({'query_idx': q_i, 'kind': q_out.kind,
                                              'note': 'audit only UNSAFE_LINEAR for now'})

    overall_safe = all(q.get('query_certifies_safe', False) for q in query_evidence)
    return {
        'bench': 'acasxu_2023', 'iid': iid,
        'onnx_path': str(onnx_p), 'onnx_sha256': sha256_file(onnx_p),
        'vnnlib_path': str(vnn_p), 'vnnlib_sha256': sha256_file(vnn_p),
        'input_dim': len(lb), 'input_lb': lb.tolist(), 'input_ub': ub.tolist(),
        'n_queries': len(queries),
        'dense_only': is_dense_only_chain(net),
        'state_K': int(state.G.shape[1]) if state.G is not None else 0,
        'state_n': int(state.c.shape[0]),
        'queries': query_evidence,
        'overall_cert_via_m4_lp': overall_safe,
    }


def main():
    out_dir = Path('/data1/Kane/ACT/audit_results/m4_acasxu_cert_audit')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'audit_iids_102_107_132.json'
    audits = {}
    for iid in (102, 107, 132):
        print(f"Auditing acasxu/{iid}...")
        audits[iid] = audit_one(iid)
        print(f"  → overall_cert={audits[iid]['overall_cert_via_m4_lp']}, "
                  f"dense_only={audits[iid]['dense_only']}, state_K={audits[iid]['state_K']}, "
                  f"n_queries={audits[iid]['n_queries']}")
    with open(out_path, 'w') as f:
        json.dump(audits, f, indent=2)
    print(f"\nWrote {out_path}")

    # Brief summary
    for iid, a in audits.items():
        for q in a['queries']:
            if q.get('kind') == 'UNSAFE_LINEAR':
                certifying_rows = [r for r in q['rows'] if r['row_certifies_polytope_unreachable']]
                print(f"\nacasxu/{iid} query 0 ({q['n_rows']} rows):")
                for r in certifying_rows:
                    print(f"  ✓ row[{r['row_idx']}]: lp_lb={r['lp_lb']:.4f} > t={r['threshold']:.4f}, "
                              f"cf_lb={r['cf_lb']:.4f} (LP improved by {r['lp_lb']-r['cf_lb']:.4f}), "
                              f"eq_resid={r.get('lb_eq_resid', 0):.2e}")


if __name__ == '__main__':
    main()
