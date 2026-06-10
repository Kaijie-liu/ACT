# ===- research/fchz/p5_1c_parallel_forensic.py - parallel KeyError trace =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor P5-1c 2026-06-08:
#   For parallel pensieve KeyError iids (1,4,8,9,17,19):
#     - capture exact KeyError layer/kind/preds
#     - identify missing pred state
#     - identify which FCHZ op semantics is missing
#     - DON'T bypass multi-pred with first-pred hack
# ===---------------------------------------------------------------------===#

"""P5-1c parallel pensieve KeyError forensic."""

import sys, os, json, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import torch


SENTINELS = [1, 4, 8, 9, 17, 19]


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

    result = {'iid': iid, 'onnx': onnx_name}
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
        result['n_layers'] = len(net.layers)
        result['layer_kinds_summary'] = {}
        from collections import Counter
        kinds = Counter(L.kind for L in net.layers)
        result['layer_kinds_summary'] = dict(kinds)

        set_transfer_function_mode("fchz")
        tf = get_transfer_function()
        if hasattr(tf, '_state_loss_log'):
            tf._state_loss_log = []
        input_bounds = Bounds(lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                                      ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
        before, after = {}, {}
        for L in net.layers:
            # Use only id-based lookup to avoid hidden KeyError
            in_b = input_bounds
            if L.id != 0 and net.preds.get(L.id):
                pred_id = net.preds[L.id][0]
                if pred_id in after:
                    in_b = after[pred_id].bounds
                else:
                    # KeyError site
                    result['keyerror_site'] = {
                        'layer_id': L.id, 'kind': L.kind,
                        'preds': list(net.preds.get(L.id, [])),
                        'pred_in_after': [p in after for p in net.preds.get(L.id, [])],
                        'pred_kinds': [next((LL.kind for LL in net.layers if LL.id == p), 'NOT_FOUND')
                                              for p in net.preds.get(L.id, [])],
                        'pred_id_missing_from_after': pred_id,
                        'pred_kind_missing': next((LL.kind for LL in net.layers if LL.id == pred_id), 'NOT_FOUND'),
                    }
                    result['first_failing_layer_full_context'] = {
                        'layer_id': L.id, 'kind': L.kind,
                        'params_keys': list(L.params.keys()),
                    }
                    return result
            try:
                after[L.id] = tf.apply(L, in_b, net, before, after)
            except Exception as e:
                result['propagation_exception'] = {
                    'layer_id': L.id, 'kind': L.kind,
                    'exception': f'{type(e).__name__}: {str(e)[:200]}',
                    'preds': list(net.preds.get(L.id, [])),
                    'pred_kinds': [next((LL.kind for LL in net.layers if LL.id == p), '?')
                                          for p in net.preds.get(L.id, [])],
                }
                # Capture state_loss up to this point
                state_loss = list(getattr(tf, '_state_loss_log', []))
                from collections import Counter
                BENIGN = {'multi_pred_resolved_to_data'}
                reasons = Counter(e.get('reason', '?') for e in state_loss)
                result['state_loss_count_at_fail'] = len(state_loss)
                result['state_loss_reasons_at_fail'] = dict(reasons)
                # Last few state loss entries for context
                result['state_loss_last_5'] = state_loss[-5:]
                return result
        result['propagation_complete'] = True
        result['total_state_loss'] = len(getattr(tf, '_state_loss_log', []))
        return result
    except Exception as e:
        result['top_level_exception'] = f'{type(e).__name__}: {str(e)[:200]}'
        result['tb'] = traceback.format_exc()[-1500:]
        return result


def main():
    out_dir = Path('/data1/Kane/ACT/audit_results/p5_1c_parallel_forensic')
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out_path = out_dir / f'forensic_{stamp}.json'

    results = []
    for iid in SENTINELS:
        print(f"Tracing nn4sys/{iid}...")
        r = trace_one(iid)
        results.append(r)
        if r.get('keyerror_site'):
            site = r['keyerror_site']
            print(f"  KEYERROR at L{site['layer_id']} {site['kind']}: preds={site['preds']}, pred_kinds={site['pred_kinds']}")
            print(f"    pred {site['pred_id_missing_from_after']} ({site['pred_kind_missing']}) missing from `after` dict")
        elif r.get('propagation_exception'):
            exc = r['propagation_exception']
            print(f"  PROP EXC at L{exc['layer_id']} {exc['kind']}: {exc['exception']}")
            print(f"    pred_kinds: {exc['pred_kinds']}")
            print(f"    state_loss reasons: {r.get('state_loss_reasons_at_fail', {})}")
        elif r.get('top_level_exception'):
            print(f"  TOP-LEVEL: {r['top_level_exception']}")
        else:
            print(f"  PROPAGATED ({r.get('total_state_loss', 0)} state_loss entries)")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
