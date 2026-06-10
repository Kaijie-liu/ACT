# ===- research/fchz/m4_relu_diag.py - Per-ReLU diagnostic dump ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08:
#   Productionized version of M4 scout phase 1. For each (bench, iid), dumps
#   per-ReLU statistics so we can decide between F1 LP (last layer only),
#   2-layer LP, or full-network LP.
#
# Output: JSON per iid + aggregate CSV.
# ===---------------------------------------------------------------------===#

"""Per-ReLU diagnostic for FCHZ small-dense networks."""

import sys, os, json, signal, time, argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import torch


def read_official_row(bench, iid):
    base = Path(f'/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/{bench}')
    with open(base / 'instances.csv') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    if iid >= len(rows): return None, None
    return base / rows[iid][0], base / rows[iid][1]


def diag_one(bench, iid):
    """Run FchzTF forward propagation, capture per-ReLU stats. Returns dict."""
    onnx_p, vnn_p = read_official_row(bench, iid)
    if onnx_p is None: return None

    os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
    os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

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
    from act.back_end.fchz_tf.spec_canonicalize import canonicalize_queries
    from act.back_end.fchz_tf.verifier_fchz import (
        fchz_upper_bound, fchz_lower_bound)
    import onnx

    in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
    pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))
    labeled = LabeledInputTensor(
        tensor=torch.zeros(in_shape, dtype=torch.float32), label=torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
    if not queries: return None
    in_spec0, out_spec0 = queries[0]
    if in_spec0.kind == 'LINF_BALL':
        center = in_spec0.center.detach().cpu().numpy().reshape(-1)
        eps = float(in_spec0.eps); lb = center - eps; ub = center + eps
    elif in_spec0.kind == 'BOX':
        lb = in_spec0.lb.detach().cpu().numpy().reshape(-1)
        ub = in_spec0.ub.detach().cpu().numpy().reshape(-1)
    else:
        return {'bench': bench, 'iid': iid, 'verdict': f'UNSUPPORTED_INSPEC:{in_spec0.kind}'}

    verifiable = VerifiableModel(
        input_layer=InputLayer(labeled_input=labeled, shape=in_shape, dtype=torch.float32),
        input_spec=InputSpecLayer(in_spec0), model=pytorch_model, output_spec=OutputSpecLayer(out_spec0))
    net = TorchToACT(verifiable).run()
    set_transfer_function_mode("fchz")
    tf = get_transfer_function()
    input_bounds = Bounds(
        lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
        ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
    before, after = {}, {}
    for L in net.layers:
        in_b = (input_bounds if L.id == 0 or not net.preds.get(L.id)
                    else after[net.preds[L.id][0]].bounds)
        after[L.id] = tf.apply(L, in_b, net, before, after)

    # Per-ReLU stats
    relu_stats = []
    for L in net.layers:
        if L.kind != 'RELU': continue
        pred_id = net.preds.get(L.id, [None])[0]
        state_pre = tf._state_cache.get(pred_id)
        if state_pre is None:
            relu_stats.append({'layer_id': L.id, 'state_ok': False})
            continue
        c = state_pre.c
        G = state_pre.G
        tail = state_pre.tail_radius
        G_l1 = np.abs(G).sum(axis=1) if G is not None and G.size > 0 else np.zeros(c.shape[0])
        rad = G_l1 + (tail if tail is not None else 0.0)
        l = c - rad; u = c + rad
        relu_stats.append({
            'layer_id': int(L.id),
            'state_ok': True,
            'n_pre': int(c.shape[0]),
            'K': int(G.shape[1]) if G is not None else 0,
            'tail_max': float(tail.max()) if tail is not None else 0.0,
            'unstable': int(((l < 0) & (u > 0)).sum()),
            'stable_act': int((l >= 0).sum()),
            'stable_inact': int((u <= 0).sum()),
            'max_width': float((u - l).max()),
            'c_max': float(np.abs(c).max()),
        })

    # Final state + closed-form bound
    assert_layer = next((L for L in reversed(net.layers) if L.kind == 'ASSERT'), None)
    pre_assert = net.preds.get(assert_layer.id, [None])[0]
    state_final = tf._state_cache.get(pre_assert)
    n_out = state_final.c.shape[0] if state_final else 0
    canon = canonicalize_queries(queries, n_out)
    cf_bound = None
    if state_final is not None and canon.get('C') is not None:
        C = canon['C']
        if canon['kind'] in ('TOP1_ROBUST', 'LINEAR_LE'):
            cf_bound = float(fchz_upper_bound(state_final, C).max())
        elif canon['kind'] == 'UNSAFE_LINEAR':
            cf_bound = float(fchz_lower_bound(state_final, C).min())

    return {
        'bench': bench,
        'iid': iid,
        'n_relu_layers': len(relu_stats),
        'final_n': int(state_final.c.shape[0]) if state_final is not None else None,
        'final_K': int(state_final.G.shape[1]) if state_final is not None and state_final.G is not None else 0,
        'final_c_max': float(np.abs(state_final.c).max()) if state_final is not None else None,
        'closed_form_bound': cf_bound,
        'canon_kind': canon.get('kind'),
        'y_true': canon.get('y_true'),
        'relu_stats': relu_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True, help='Output JSON path')
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--config', type=str, default='m4_default',
                       help='m4_default | m4a_scout | full_groupD')
    args = ap.parse_args()

    configs = {
        'm4_default': {
            'acasxu_2023':         [0, 1, 50, 100, 180],
            'linearizenn_2024':    [0, 10, 25, 40, 55],
            'sat_relu':                [0, 20, 40, 60, 80],
            'tllverifybench_2023': [0, 6, 14, 22, 30],
            'lsnc_relu':                [0, 20, 40, 60, 75],
        },
        'm4a_scout': {
            'acasxu_2023':         [0, 1, 50, 100, 180],
            'linearizenn_2024':    [0, 10, 25, 40, 55],
            'lsnc_relu':                [0, 20, 40, 60, 75],
            'sat_relu':                [0, 20, 40, 60, 80],
            'tllverifybench_2023': [0, 6],
        },
    }
    sentinel_set = configs.get(args.config, configs['m4_default'])

    results = []
    t_start = time.time()
    for bench, iids in sentinel_set.items():
        for iid in iids:
            try:
                signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(args.timeout)
                d = diag_one(bench, iid)
                signal.alarm(0)
                if d is not None: results.append(d)
            except TimeoutError:
                results.append({'bench': bench, 'iid': iid, 'verdict': 'TIMEOUT'})
            except Exception as e:
                results.append({'bench': bench, 'iid': iid, 'verdict': f'ERROR:{type(e).__name__}',
                                      'error': str(e)[:120]})
            finally:
                signal.alarm(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'config': args.config, 'sentinels': sentinel_set,
                       'wall_s': time.time() - t_start, 'results': results}, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"  {len(results)} records, wall {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
