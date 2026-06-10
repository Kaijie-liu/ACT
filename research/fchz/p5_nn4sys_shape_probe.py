# ===- research/fchz/p5_nn4sys_shape_probe.py - real nn4sys probe ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08 P5-1a.4:
#   Probe real nn4sys ACT graphs for first failing layer + STATE_LOSS log.
#   Output JSON per sentinel (no production change).
#
# Sentinels:
#   simple:   iid 0, 6, 7   (pensieve_small_simple, pensieve_big_simple, pensieve_mid)
#   parallel: iid 1, 4       (pensieve_big_parallel, pensieve_small_parallel)
#
# For each: first failing layer kind + reason + state_loss_log.
# Gate (per advisor):
#   simple sentinels' first failing point must move from "DENSE shape mismatch" to
#   either "GATHER shape correctly propagated" or a NEW failure point (no regression).
# ===---------------------------------------------------------------------===#

"""P5 nn4sys real-net shape probe."""

import sys, os, json, argparse, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import torch


SENTINELS = {
    'simple_0_pensieve_small': 0,
    'simple_6_pensieve_mid':   6,
    'simple_7_pensieve_big':   7,
    'parallel_1_big':            1,
    'parallel_4_small':          4,
}


def probe_one(iid):
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
        # Reset trace
        if hasattr(tf, '_state_loss_log'):
            tf._state_loss_log = []
        input_bounds = Bounds(lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                                      ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
        before, after = {}, {}
        first_fail = None
        last_ok = None
        for L in net.layers:
            in_b = input_bounds if L.id == 0 or not net.preds.get(L.id) else after[net.preds[L.id][0]].bounds
            try:
                after[L.id] = tf.apply(L, in_b, net, before, after)
                last_ok = L.id
            except Exception as e:
                first_fail = {
                    'layer_id': L.id, 'kind': L.kind,
                    'preds': net.preds.get(L.id, []),
                    'params_keys': list(L.params.keys())[:10],
                    'exception': f'{type(e).__name__}: {str(e)[:100]}',
                }
                break
        state_loss = list(getattr(tf, '_state_loss_log', []))
        # Categorize STATE_LOSS by reason (advisor 2026-06-08)
        from collections import Counter
        reason_hist = Counter(e.get('reason', '?') for e in state_loss)
        BENIGN = {'multi_pred_resolved_to_data'}    # shape-param separation, not correlation loss
        MALIGNANT = {'pred_state_missing', 'multi_pred',
                            'multi_pred_not_whitelisted',
                            'shape_contract_violation', 'G_c_dim_mismatch',
                            'dense_input_dim_mismatch',
                            'tf_gather_exception', 'tf_slice_exception',
                            'no_preds_in_graph', 'multi_pred_data_pred_not_found',
                            'missing_params'}
        benign_count = sum(reason_hist.get(r, 0) for r in BENIGN)
        malignant_count = sum(reason_hist.get(r, 0) for r in MALIGNANT)

        # Try verify_once_fchz to see final verdict
        verify_verdict = None
        verify_reason = None
        if first_fail is None:
            from act.back_end.fchz_tf.verifier_fchz import verify_once_fchz
            try:
                import onnx
                m = onnx.load(str(onnx_p))
                od = [d.dim_value if d.dim_value > 0 else 1
                         for d in m.graph.output[0].type.tensor_type.shape.dim]
                n_out = int(np.prod(od[1:])) if len(od) > 1 else od[0]
                result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)
                verify_verdict = result.get('verdict')
                verify_reason = result.get('reason') or result.get('canon_reason')
            except Exception as e:
                verify_verdict = f'ERROR:{type(e).__name__}'
                verify_reason = str(e)[:80]

        return {
            'iid': iid, 'onnx': onnx_name,
            'n_layers': len(net.layers),
            'last_ok_layer_id': last_ok,
            'first_fail': first_fail,
            'state_loss_count': len(state_loss),
            'state_loss_benign': benign_count,
            'state_loss_malignant': malignant_count,
            'state_loss_reason_histogram': dict(reason_hist),
            'state_loss_log_sample': state_loss[:5],
            'propagation_complete': first_fail is None,
            'verify_once_fchz_verdict': verify_verdict,
            'verify_once_fchz_reason': verify_reason,
        }
    except Exception as e:
        return {'iid': iid, 'onnx': onnx_name,
                    'top_level_exception': f'{type(e).__name__}: {str(e)[:120]}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for name, iid in SENTINELS.items():
        print(f"Probing nn4sys/{iid} ({name})...")
        results[name] = probe_one(iid)
        r = results[name]
        if 'top_level_exception' in r:
            print(f"  TOP-LEVEL EXCEPTION: {r['top_level_exception']}")
        elif r.get('propagation_complete'):
            print(f"  PROPAGATED COMPLETELY ({r['n_layers']} layers)")
            print(f"    STATE_LOSS: benign={r['state_loss_benign']}, malignant={r['state_loss_malignant']}")
            print(f"    Reason histogram: {r['state_loss_reason_histogram']}")
            print(f"    verify_once_fchz: {r['verify_once_fchz_verdict']} ({r.get('verify_once_fchz_reason', '')})")
        else:
            ff = r['first_fail']
            print(f"  FIRST FAIL at L{ff['layer_id']} {ff['kind']}: {ff['exception']}")
            print(f"  preds={ff['preds']}, state_loss entries={len(r['state_loss_log'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
