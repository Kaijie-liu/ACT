# ===- research/fchz/m4a_scout.py - M4A small-dense full LP scout --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08:
#   Run small-dense full-network LP on 22 UNK sentinels.
#   Gate: ≥5 V flip OR ≥40% UB drop avg → continue
#          <10% UB drop + 0 V → close
#
# Output: JSON report with baseline / LP-refined / verdict-flip per sentinel.
# ===---------------------------------------------------------------------===#

"""M4A scout: small-dense full-network LP — gate decision tool."""

import sys, os, json, signal, time, argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import torch


SENTINELS = {
    'acasxu_2023':         [0, 1, 50, 100, 180],
    'linearizenn_2024':    [0, 10, 25, 40, 55],
    'lsnc_relu':                [0, 20, 40, 60, 75],
    'sat_relu':                [0, 20, 40, 60, 80],
    'tllverifybench_2023': [0, 6],
}


def read_official_row(bench, iid):
    base = Path(f'/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/{bench}')
    with open(base / 'instances.csv') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    return base / rows[iid][0], base / rows[iid][1]


def run_one(bench, iid):
    onnx_p, vnn_p = read_official_row(bench, iid)
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
    from research.fchz.m4_full_lp import m4_full_lp_refine, is_dense_only_chain
    import onnx

    in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
    pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))
    labeled = LabeledInputTensor(
        tensor=torch.zeros(in_shape, dtype=torch.float32), label=torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
    if not queries: return {'bench': bench, 'iid': iid, 'verdict': 'NO_QUERY'}
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
    canon = canonicalize_queries(queries, n_out)

    pre_assert = net.preds.get(next(L for L in reversed(net.layers) if L.kind == 'ASSERT').id, [None])[0]
    state = tf._state_cache.get(pre_assert)
    if state is None: return {'bench': bench, 'iid': iid, 'verdict': 'NO_STATE'}

    kind = canon.get('kind')
    # Handle MULTI_QUERY: evaluate first query as representative
    if kind == 'MULTI_QUERY':
        q_in, q_out = canon['queries'][0]
        if q_out.kind == 'UNSAFE_LINEAR':
            C = q_out.c.detach().cpu().numpy().astype(np.float64) if hasattr(q_out.c, 'detach') else np.asarray(q_out.c, dtype=np.float64)
            t = q_out.d.detach().cpu().numpy().astype(np.float64).reshape(-1) if hasattr(q_out.d, 'detach') else np.asarray(q_out.d, dtype=np.float64).reshape(-1)
            if C.ndim == 1: C = C.reshape(1, -1)
            kind = 'UNSAFE_LINEAR'   # use single-query path
        else:
            return {'bench': bench, 'iid': iid, 'verdict': 'MULTI_KIND_UNSUPPORTED', 'kind': canon['kind']}
    else:
        C = canon.get('C'); t = canon.get('t')
        if C is None: return {'bench': bench, 'iid': iid, 'verdict': 'NO_C', 'kind': kind}

    # Baseline closed-form
    if kind in ('TOP1_ROBUST', 'LINEAR_LE'):
        cf_ub = fchz_upper_bound(state, C)
        cf_excess = cf_ub - t
        cf_verdict = 'CERTIFIED' if (cf_excess < 0).all() else 'UNKNOWN'
    elif kind == 'UNSAFE_LINEAR':
        cf_lb = fchz_lower_bound(state, C)
        cf_excess = cf_lb - t   # > 0 means unreachable
        cf_verdict = 'CERTIFIED' if (cf_excess > 0).any() else 'UNKNOWN'
    elif kind == 'MULTI_QUERY':
        cf_verdict = 'UNKNOWN'   # treat as worst case for scout
        cf_excess = None
    else:
        cf_verdict = 'UNKNOWN'; cf_excess = None

    # Check net compatibility
    dense_only = is_dense_only_chain(net)
    if not dense_only:
        return {'bench': bench, 'iid': iid, 'cf_verdict': cf_verdict,
                    'lp_verdict': 'NOT_DENSE_ONLY', 'dense_only': False}

    # M4 LP refinement
    t_lp_start = time.time()
    if kind in ('TOP1_ROBUST', 'LINEAR_LE'):
        lp_ub = m4_full_lp_refine(net, tf, C, t)
        if lp_ub is None:
            return {'bench': bench, 'iid': iid, 'cf_verdict': cf_verdict, 'lp_verdict': 'LP_NONE'}
        eff_ub = np.minimum(cf_ub, lp_ub)
        eff_excess = eff_ub - t
        lp_verdict = 'CERTIFIED' if (eff_excess < 0).all() else 'UNKNOWN'
        cf_ub_max = float(cf_ub.max()); lp_ub_max = float(lp_ub.max()); eff_ub_max = float(eff_ub.max())
    elif kind == 'UNSAFE_LINEAR':
        lp_neg = m4_full_lp_refine(net, tf, -C, -t)
        if lp_neg is None:
            return {'bench': bench, 'iid': iid, 'cf_verdict': cf_verdict, 'lp_verdict': 'LP_NONE'}
        lp_lb = -lp_neg
        eff_lb = np.maximum(cf_lb, lp_lb)
        eff_excess = eff_lb - t
        lp_verdict = 'CERTIFIED' if (eff_excess > 0).any() else 'UNKNOWN'
        cf_ub_max = float(cf_lb.min()); lp_ub_max = float(lp_lb.min()); eff_ub_max = float(eff_lb.min())
    else:
        return {'bench': bench, 'iid': iid, 'cf_verdict': cf_verdict, 'lp_verdict': 'KIND_UNSUPPORTED'}
    lp_wall = time.time() - t_lp_start

    return {
        'bench': bench, 'iid': iid,
        'kind': kind, 'cf_verdict': cf_verdict, 'lp_verdict': lp_verdict,
        'cf_bound': cf_ub_max, 'lp_bound': lp_ub_max, 'eff_bound': eff_ub_max,
        'lp_wall_s': round(lp_wall, 3),
        'dense_only': dense_only,
        'flip': cf_verdict != lp_verdict and lp_verdict == 'CERTIFIED',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()

    print(f"M4A scout: small-dense full-network LP refinement")
    print(f"Sentinels: {SENTINELS}")
    print(f"Gate: >=5 V flip OR >=40% UB drop → continue; <10% UB + 0 V → close\n")

    results = []
    t_start = time.time()
    for bench, iids in SENTINELS.items():
        for iid in iids:
            try:
                signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(args.timeout)
                r = run_one(bench, iid)
                signal.alarm(0)
                results.append(r)
                print(f"  {bench}/{iid}: cf={r.get('cf_verdict')} → lp={r.get('lp_verdict')} "
                          f"cf_bound={r.get('cf_bound', '?')!s:.20s} lp_bound={r.get('lp_bound', '?')!s:.20s} "
                          f"flip={r.get('flip', False)} dense_only={r.get('dense_only', '?')} "
                          f"wall={r.get('lp_wall_s', 0)}s")
            except TimeoutError:
                r = {'bench': bench, 'iid': iid, 'verdict': 'TIMEOUT'}
                results.append(r)
                print(f"  {bench}/{iid}: TIMEOUT")
            except Exception as e:
                r = {'bench': bench, 'iid': iid, 'verdict': f'ERROR:{type(e).__name__}',
                        'error': str(e)[:120]}
                results.append(r)
                print(f"  {bench}/{iid}: ERROR {str(e)[:80]}")
            finally:
                signal.alarm(0)

    # Gate evaluation
    flips = [r for r in results if r.get('flip')]
    valid_uls = [r for r in results
                       if r.get('cf_verdict') == 'UNKNOWN' and r.get('cf_bound') is not None
                       and r.get('lp_bound') is not None and r.get('dense_only')]
    if valid_uls:
        improvements = []
        for r in valid_uls:
            cf = abs(r['cf_bound']); lp = abs(r['lp_bound'])
            if cf > 1e-9:
                improvements.append((cf - lp) / cf)
        avg_drop = float(np.mean(improvements)) if improvements else 0.0
    else:
        avg_drop = 0.0

    print("\n" + "=" * 60)
    print(f"M4A SCOUT RESULT:")
    print(f"  total sentinels: {len(results)}")
    print(f"  V flips (cf UNK → lp CERT): {len(flips)}")
    print(f"  Avg UB drop (UNK cases only): {avg_drop*100:.1f}%")
    print()
    if len(flips) >= 5:
        print(f"  ✅ GATE PASS (>=5 flips): productionize M4 LP")
    elif avg_drop >= 0.4:
        print(f"  ✅ GATE PASS (>=40% UB drop): productionize M4 LP")
    elif avg_drop < 0.1 and len(flips) == 0:
        print(f"  ✗ GATE FAIL (<10% UB drop + 0 flips): CLOSE M4 LP")
    else:
        print(f"  ⚠️ Marginal: re-evaluate or expand sentinel set")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'sentinels': SENTINELS,
            'wall_s': time.time() - t_start,
            'gate_flips': len(flips),
            'gate_avg_drop': avg_drop,
            'results': results,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
