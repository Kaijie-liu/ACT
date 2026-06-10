# ===- research/fchz/m4_smoke.py - M4 LP smoke on acasxu/tllverify/sat_relu ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08 M4A smoke (post-audit, post-multi-query fix):
#   - acasxu_2023 FULL
#   - tllverifybench_2023 FULL
#   - sat_relu 10 sentinels (confirm UNK, no false flip expected)
#
#   Gate: >=5 NEW V on acasxu+tllverify → productionize
#         <5 + high bound drop → M4B affine-op extension
#         <5 + no boundary cases → close
#
# Output: JSON per bench + aggregated summary.
# ===---------------------------------------------------------------------===#

"""M4 LP smoke runner on acasxu/tllverify/sat_relu."""

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
    return base / rows[iid][0], base / rows[iid][1], len(rows)


def run_one(bench, iid):
    onnx_p, vnn_p, _ = read_official_row(bench, iid)
    if onnx_p is None: return {'bench': bench, 'iid': iid, 'verdict': 'NO_INSTANCE'}

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
    from act.back_end.fchz_tf.verifier_fchz import verify_once_fchz
    from research.fchz.m4_verdict import m4_verdict_for_queries
    from research.fchz.m4_full_lp import is_dense_only_chain
    import onnx

    try:
        in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
        pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))
        labeled = LabeledInputTensor(
            tensor=torch.zeros(in_shape, dtype=torch.float32), label=torch.tensor([0]))
        queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
        if not queries:
            return {'bench': bench, 'iid': iid, 'verdict': 'NO_QUERY'}
        in_spec0, out_spec0 = queries[0]
        if in_spec0.kind == 'LINF_BALL':
            center = in_spec0.center.detach().cpu().numpy().reshape(-1)
            eps = float(in_spec0.eps); lb = center - eps; ub = center + eps
        elif in_spec0.kind == 'BOX':
            lb = in_spec0.lb.detach().cpu().numpy().reshape(-1)
            ub = in_spec0.ub.detach().cpu().numpy().reshape(-1)
        else:
            return {'bench': bench, 'iid': iid, 'verdict': f'UNSUPPORTED:{in_spec0.kind}'}

        verifiable = VerifiableModel(
            input_layer=InputLayer(labeled_input=labeled, shape=in_shape, dtype=torch.float32),
            input_spec=InputSpecLayer(in_spec0),
            model=pytorch_model, output_spec=OutputSpecLayer(out_spec0))
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

        # Baseline closed-form via verify_once_fchz (existing canon)
        t0 = time.time()
        cf_result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)
        t_cf = time.time() - t0

        # M4 LP via per-query AND
        pre_assert = net.preds.get(next(L for L in reversed(net.layers) if L.kind == 'ASSERT').id, [None])[0]
        cf_state = tf._state_cache.get(pre_assert)
        t0 = time.time()
        m4_result = m4_verdict_for_queries(net, tf, queries, n_out, cf_state=cf_state)
        t_m4 = time.time() - t0

        return {
            'bench': bench, 'iid': iid,
            'cf_verdict': cf_result['verdict'],
            'm4_verdict': m4_result['verdict'],
            'm4_reason': m4_result.get('reason'),
            'flip': cf_result['verdict'] != 'CERTIFIED' and m4_result['verdict'] == 'CERTIFIED',
            'n_queries': len(queries),
            'dense_only': is_dense_only_chain(net),
            'cf_kind': cf_result.get('kind'),
            'cf_wall_s': round(t_cf, 3),
            'm4_wall_s': round(t_m4, 3),
        }
    except Exception as e:
        return {'bench': bench, 'iid': iid,
                    'verdict': f'ERROR:{type(e).__name__}',
                    'error': str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()

    benches = {
        'acasxu_2023':         None,    # full
        'tllverifybench_2023': None,    # full
        'sat_relu':                list(range(0, 100, 10)),    # 10 sentinels
    }

    results = {}
    t_start = time.time()
    for bench, iids in benches.items():
        if iids is None:
            _, _, total = read_official_row(bench, 0)
            iids = list(range(total))
        print(f"\n=== {bench} ({len(iids)} iids) ===", flush=True)
        bench_results = []
        for iid in iids:
            try:
                signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(args.timeout)
                r = run_one(bench, iid)
                signal.alarm(0)
                bench_results.append(r)
                if iid % 20 == 0 or r.get('flip') or r.get('verdict', '').startswith('ERROR'):
                    print(f"  {iid}: cf={r.get('cf_verdict')} → m4={r.get('m4_verdict')} "
                              f"flip={r.get('flip', False)} dense_only={r.get('dense_only')} "
                              f"cf_wall={r.get('cf_wall_s')}s m4_wall={r.get('m4_wall_s')}s",
                              flush=True)
            except TimeoutError:
                bench_results.append({'bench': bench, 'iid': iid, 'verdict': 'TIMEOUT'})
            except Exception as e:
                bench_results.append({'bench': bench, 'iid': iid,
                                              'verdict': f'ERROR:{type(e).__name__}',
                                              'error': str(e)[:120]})
            finally:
                signal.alarm(0)
        results[bench] = bench_results
        cf_v = sum(1 for r in bench_results if r.get('cf_verdict') == 'CERTIFIED')
        m4_v = sum(1 for r in bench_results if r.get('m4_verdict') == 'CERTIFIED')
        flips = sum(1 for r in bench_results if r.get('flip'))
        print(f"  Summary: cf_V={cf_v}, m4_V={m4_v}, flips={flips}", flush=True)

    # Aggregate
    print("\n" + "=" * 70)
    print("M4 SMOKE AGGREGATE")
    print("=" * 70)
    print(f"{'Bench':<32} {'cf_V':>5} {'m4_V':>5} {'flips':>6} {'n':>4}")
    print("-" * 70)
    total_cf = 0; total_m4 = 0; total_flips = 0
    for b, rs in results.items():
        cf_v = sum(1 for r in rs if r.get('cf_verdict') == 'CERTIFIED')
        m4_v = sum(1 for r in rs if r.get('m4_verdict') == 'CERTIFIED')
        flips = sum(1 for r in rs if r.get('flip'))
        total_cf += cf_v; total_m4 += m4_v; total_flips += flips
        print(f"{b:<32} {cf_v:>5} {m4_v:>5} {flips:>6} {len(rs):>4}")
    print("-" * 70)
    print(f"{'TOTAL':<32} {total_cf:>5} {total_m4:>5} {total_flips:>6}")
    print(f"\nNEW V from M4 LP: {total_flips}")
    if total_flips >= 5:
        print(f"✅ GATE PASS (>=5 NEW V): productionize M4 LP")
    elif total_flips > 0:
        print(f"⚠️ Marginal: {total_flips} flips < 5 (consider M4B affine-op extension)")
    else:
        print(f"✗ Gate FAIL: 0 flips on smoke → close M4A branch")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'wall_s': time.time() - t_start,
            'total_cf': total_cf, 'total_m4': total_m4, 'total_flips': total_flips,
            'results': results,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
