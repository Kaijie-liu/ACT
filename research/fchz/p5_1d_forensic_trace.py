# ===- research/fchz/p5_1d_forensic_trace.py - mid/big_simple forensic ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor P5-1d 2026-06-08 (time-box 4h):
#   For mid/big_simple NO_STATE iids (6, 7, 10, 11, 12, 14):
#     - first malignant STATE_LOSS layer
#     - layer kind, preds
#     - pred state present/missing
#     - previous layer (kind, output dim)
#     - shape before/after
#     - if dense_input_dim_mismatch: expected/actual dim
# ===---------------------------------------------------------------------===#

"""P5-1d forensic trace — find root cause for mid/big_simple NO_STATE."""

import sys, os, json, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import torch


SENTINELS = [6, 7, 10, 11, 12, 14]


def trace_one(iid):
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

    os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
    os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

    base = Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/nn4sys')
    with open(base / 'instances.csv') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    onnx_p = base / rows[iid][0]; vnn_p = base / rows[iid][1]
    onnx_name = rows[iid][0].split('/')[-1]

    try:
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
        if hasattr(tf, '_state_loss_log'):
            tf._state_loss_log = []

        input_bounds = Bounds(lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                                      ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
        before, after = {}, {}
        for L in net.layers:
            in_b = input_bounds if L.id == 0 or not net.preds.get(L.id) else after[net.preds[L.id][0]].bounds
            after[L.id] = tf.apply(L, in_b, net, before, after)

        # Find first malignant STATE_LOSS
        state_loss = list(getattr(tf, '_state_loss_log', []))
        BENIGN = {'multi_pred_resolved_to_data'}
        first_malignant = None
        for e in state_loss:
            if e.get('reason') not in BENIGN:
                first_malignant = e
                break

        # Build per-layer dict (id → layer info)
        layer_info = {L.id: L for L in net.layers}

        result = {
            'iid': iid, 'onnx': onnx_name,
            'n_layers': len(net.layers),
            'total_state_loss': len(state_loss),
            'malignant_count': sum(1 for e in state_loss if e.get('reason') not in BENIGN),
            'first_malignant': None,
        }

        if first_malignant is not None:
            L_id = first_malignant['layer_id']
            L = layer_info[L_id]
            preds = net.preds.get(L_id, [])
            pred_info = []
            for p in preds:
                pL = layer_info.get(p)
                in_cache = p in tf._state_cache
                cached_dim = None
                if in_cache:
                    s = tf._state_cache[p]
                    cached_dim = int(s.c.shape[0])
                pred_info.append({
                    'pred_id': p,
                    'pred_kind': pL.kind if pL else 'UNKNOWN',
                    'state_in_cache': in_cache,
                    'cached_state_c_dim': cached_dim,
                })
            # Extra: for DENSE shape mismatch, show weight shape
            extra = {}
            if L.kind == 'DENSE':
                W = L.params.get('weight')
                if W is not None:
                    if hasattr(W, 'shape'): extra['W_shape'] = list(W.shape)
                    extra['in_features'] = L.params.get('in_features')
                    extra['out_features'] = L.params.get('out_features')
                extra['input_shape_decl'] = L.params.get('input_shape')
                extra['output_shape_decl'] = L.params.get('output_shape')

            result['first_malignant'] = {
                'state_loss_entry': first_malignant,
                'layer_id': L_id,
                'kind': L.kind,
                'preds': pred_info,
                'extra': extra,
            }

        # All malignant entries summary
        result['all_malignant'] = [
            {'layer_id': e.get('layer_id'),
              'kind': e.get('kind'),
              'reason': e.get('reason'),
              'expected_dim': e.get('expected_dim'),
              'actual_dim': e.get('actual_dim'),
              'W_in_features': e.get('W_in_features'),
              'state_c_dim': e.get('state_c_dim'),
            }
            for e in state_loss if e.get('reason') not in BENIGN
        ]
        return result
    except Exception as e:
        return {'iid': iid, 'top_level_exception': f'{type(e).__name__}: {str(e)[:100]}'}


def main():
    out_dir = Path('/data1/Kane/ACT/audit_results/p5_1d_forensic')
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out_path = out_dir / f'forensic_{stamp}.json'
    results = []
    for iid in SENTINELS:
        print(f"Tracing nn4sys/{iid}...")
        r = trace_one(iid)
        results.append(r)
        if r.get('first_malignant'):
            fm = r['first_malignant']
            print(f"  Layer count: {r['n_layers']}, total state_loss: {r['total_state_loss']}, malignant: {r['malignant_count']}")
            print(f"  FIRST MALIGNANT at L{fm['layer_id']} {fm['kind']}: reason={fm['state_loss_entry'].get('reason')}")
            for p in fm['preds']:
                print(f"    pred L{p['pred_id']} {p['pred_kind']}: in_cache={p['state_in_cache']}, c_dim={p['cached_state_c_dim']}")
            if fm['extra']:
                print(f"    extra: {fm['extra']}")
        elif r.get('top_level_exception'):
            print(f"  TOP-LEVEL EXC: {r['top_level_exception']}")
        else:
            print(f"  No malignant found ({r['total_state_loss']} total state_loss entries)")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
