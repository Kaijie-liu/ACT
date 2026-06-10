# ===- research/fchz/m4b_linearizenn_scout.py - M4B linearizenn sentinel ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08 P3:
#   10 linearizenn sentinels (small + medium models).
#   Gate:
#     >=3 NEW V → run full 60
#     1-2 NEW V → sidecar only
#     0 NEW V → close M4B
# ===---------------------------------------------------------------------===#

"""M4B linearizenn sentinel scout."""

import sys, os, json, signal, time, argparse, hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import torch


SENTINELS = [0, 6, 10, 12, 14, 25, 40, 55, 30, 50]


def sha256_short(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()[:16]


def run_one(iid):
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
        verify_once_fchz, fchz_upper_bound, fchz_lower_bound)
    from research.fchz.m4b_affine_head import (
        is_dense_body_with_affine_head, extract_affine_head)
    from research.fchz.m4b_lp_with_head import (
        solve_lp_with_head, m4b_refine_lb, m4b_refine_ub)
    import onnx

    os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
    os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

    base = Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/linearizenn_2024')
    with open(base / 'instances.csv') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    onnx_p = base / rows[iid][0]; vnn_p = base / rows[iid][1]

    in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
    pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))
    labeled = LabeledInputTensor(tensor=torch.zeros(in_shape, dtype=torch.float32),
                                              label=torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
    in_spec0, out_spec0 = queries[0]
    if in_spec0.kind == 'LINF_BALL':
        center = in_spec0.center.detach().cpu().numpy().reshape(-1)
        eps = float(in_spec0.eps); lb = center - eps; ub = center + eps
    elif in_spec0.kind == 'BOX':
        lb = in_spec0.lb.detach().cpu().numpy().reshape(-1)
        ub = in_spec0.ub.detach().cpu().numpy().reshape(-1)
    else: return {'iid': iid, 'verdict': f'UNSUPPORTED_INSPEC'}

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
    od = [d.dim_value if d.dim_value > 0 else 1 for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_out = int(np.prod(od[1:])) if len(od) > 1 else od[0]

    cf_result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)

    has_head, last_relu_id = is_dense_body_with_affine_head(net)
    if not has_head:
        return {'iid': iid, 'cf_verdict': cf_result['verdict'], 'm4b_verdict': 'NOT_AFFINE_HEAD'}

    # M4B: per-query AND with LP-with-head
    pre_assert = net.preds.get(next(L for L in reversed(net.layers) if L.kind == 'ASSERT').id, [None])[0]
    cf_state = tf._state_cache.get(pre_assert)

    # Walk each query independently (AND)
    all_query_safe = True
    cf_query_safe_count = 0
    lp_query_safe_count = 0
    for q_in, q_out in queries:
        if q_out.kind != 'UNSAFE_LINEAR':
            all_query_safe = False; break
        C = q_out.c.detach().cpu().numpy().astype(np.float64)
        t = q_out.d.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if C.ndim == 1: C = C.reshape(1, -1)
        # UNSAFE_LINEAR: any row LB > t → CERT for this query
        cf_safe = False
        lp_safe = False
        for i in range(C.shape[0]):
            d = C[i]
            cf_lb = float(fchz_lower_bound(cf_state, d.reshape(1, -1))[0])
            lp_lb, _diag = solve_lp_with_head(net, tf, d, sense='min')
            t_i = float(t[i] if i < len(t) else t[0])
            if cf_lb > t_i: cf_safe = True
            if lp_lb is not None and lp_lb > t_i: lp_safe = True
        if cf_safe: cf_query_safe_count += 1
        if lp_safe: lp_query_safe_count += 1
        if not lp_safe:
            all_query_safe = False

    m4b_verdict = 'CERTIFIED' if all_query_safe else 'UNKNOWN'
    return {
        'iid': iid,
        'cf_verdict': cf_result['verdict'],
        'm4b_verdict': m4b_verdict,
        'flip': cf_result['verdict'] != 'CERTIFIED' and m4b_verdict == 'CERTIFIED',
        'n_queries': len(queries),
        'cf_queries_safe': cf_query_safe_count,
        'lp_queries_safe': lp_query_safe_count,
        'vnnlib_sha256': sha256_short(vnn_p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    results = []
    t_start = time.time()
    for iid in SENTINELS:
        try:
            signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(120)
            r = run_one(iid)
            signal.alarm(0)
            results.append(r)
            print(f"  linearizenn/{iid}: cf={r.get('cf_verdict')} → m4b={r.get('m4b_verdict')} "
                      f"flip={r.get('flip', False)} cf_qs={r.get('cf_queries_safe')}/{r.get('n_queries')} "
                      f"lp_qs={r.get('lp_queries_safe')}/{r.get('n_queries')}", flush=True)
        except TimeoutError:
            results.append({'iid': iid, 'verdict': 'TIMEOUT'})
        except Exception as e:
            results.append({'iid': iid, 'verdict': f'ERROR:{type(e).__name__}', 'error': str(e)[:120]})
            print(f"  linearizenn/{iid}: ERROR {str(e)[:80]}", flush=True)
        finally:
            signal.alarm(0)

    flips = sum(1 for r in results if r.get('flip'))
    print("\n" + "=" * 60)
    print(f"M4B linearizenn SCOUT RESULT:")
    print(f"  total sentinels: {len(results)}")
    print(f"  NEW V from M4B LP (cf UNK → m4b CERT): {flips}")
    if flips >= 3:
        print(f"  ✅ GATE PASS (>=3 NEW V): run full 60")
    elif flips >= 1:
        print(f"  ⚠️ Sidecar only ({flips} NEW V)")
    else:
        print(f"  ✗ Gate FAIL (0 NEW V): close M4B")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'wall_s': time.time()-t_start,
                       'sentinels': SENTINELS, 'flips': flips,
                       'results': results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
