# ===- act/back_end/fchz_tf/group_runner.py - Single-backend group runner ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Production runner for FCHZ single-backend verification on official
#   VNN-COMP 2025 benchmarks. Reads instances.csv directly, uses
#   verify_once_fchz(queries=...) for canonicalized check, outputs
#   per-iid JSON with full provenance.
#
#   Per advisor 2026-06-08: don't run full mixed sweep. Use groups:
#     A) cifar100_2024 + tinyimagenet_2024 (headline win, 399 V target)
#     B) yolo/traffic/cctsdb (FCHZ conv shape fix later)
#     C) cgan/ml4acopf (input shape fix later)
#     D) small dense (F1 LP migration later)
#
# ===---------------------------------------------------------------------===#

"""FCHZ single-backend group runner — official pipeline + canonicalize."""

import sys, os, json, time, signal, hashlib
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch


# Default benchmarks data root (resolves regardless of CWD).
DEFAULT_BENCH_ROOT = Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')


# Group definitions per advisor (don't run "all at once" mess)
GROUPS = {
    'A': [   # CIFAR/TINY headline win (399 V target via production path)
        ('cifar100_2024', 200, 128),
        ('tinyimagenet_2024', 200, 128),
    ],
    'small_dense': [   # Group D candidates (need F1 LP later)
        ('test', 5, None),
        ('cersyve', 12, None),
        ('tllverifybench_2023', 32, None),
        ('linearizenn_2024', 60, None),
        ('collins_rul_cnn_2022', 62, None),
        ('dist_shift_2023', 72, None),
        ('lsnc_relu', 80, None),
        ('sat_relu', 100, None),
        ('malbeware', 150, None),
        ('cora_2024', 180, None),
        ('acasxu_2023', 186, None),
        ('nn4sys', 194, None),
        ('relusplitter', 220, None),
        ('safenlp_2024', 1080, None),
    ],
    'cnn_lift': [   # cifar/tiny + clean small benches that worked in M2 diagnostic
        ('cifar100_2024', 200, 128),
        ('tinyimagenet_2024', 200, 128),
        ('metaroom_2023', 100, 128),
        ('malbeware', 150, None),
        ('safenlp_2024', 1080, None),
    ],
    'safenlp_only': [
        ('safenlp_2024', 1080, None),
    ],
    'nn4sys_only': [('nn4sys', 194, None)],
    # P5: only run simple pensieve variants (iids 0-19 are typically simple variants)
    'nn4sys_simple_sentinel': [('nn4sys', 20, None)],
    # P6: metaroom re-inclusion (advisor 2026-06-08; historical ~92 V)
    'acasxu_only': [('acasxu_2023', 186, None)],
    'acasxu_sentinel': [('acasxu_2023', 20, None)],
    'linearizenn_only': [('linearizenn_2024', 60, None)],
    'tll_only': [('tllverifybench_2023', 32, None)],
    'metaroom_only': [('metaroom_2023', 100, 128)],
    'metaroom_sentinel': [('metaroom_2023', 10, 128)],
    'A_mini': [   # 10 cifar + 10 tiny mini-regression (advisor 2026-06-08)
        ('cifar100_2024', 10, 128),
        ('tinyimagenet_2024', 10, 128),
    ],
}


TIMEOUT = 60


class _TimeoutErr(Exception): pass
def _handler(signum, frame): raise _TimeoutErr("worker timeout")


def _sha256_short(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()[:16]


def _read_official_row(bench, row_iid):
    base = DEFAULT_BENCH_ROOT / bench
    csv_p = base / 'instances.csv'
    if not csv_p.exists(): return None, None
    with open(csv_p) as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    if row_iid >= len(rows): return None, None
    return base / rows[row_iid][0], base / rows[row_iid][1]


def _run_one(args):
    bench, iid, g_max = args
    t0 = time.time()
    try:
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(TIMEOUT)

            os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
            if g_max:
                os.environ['HYZOR_FCHZ_G_MAX_COLS'] = str(g_max)
            else:
                os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

            onnx_p, vnn_p = _read_official_row(bench, iid)
            if onnx_p is None:
                return {'bench': bench, 'iid': iid, 'verdict': 'NO_INSTANCE',
                            'elapsed_s': round(time.time()-t0, 3)}

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
            import onnx

            in_shape = tuple(get_onnx_input_shape(Path(onnx_p)))
            pytorch_model = convert_onnx_to_pytorch(Path(onnx_p))

            labeled = LabeledInputTensor(
                tensor=torch.zeros(in_shape, dtype=torch.float32),
                label=torch.tensor([0]))
            queries = parse_vnnlib_queries(Path(vnn_p), labeled_tensor=labeled)
            if not queries:
                return {'bench': bench, 'iid': iid, 'verdict': 'NO_QUERY',
                            'elapsed_s': round(time.time()-t0, 3)}

            m = onnx.load(str(onnx_p))
            od = [d.dim_value if d.dim_value > 0 else 1
                     for d in m.graph.output[0].type.tensor_type.shape.dim]
            n_out = int(np.prod(od[1:])) if len(od) > 1 else od[0]

            # Build VerifiableModel using first query for net topology
            in_spec0, out_spec0 = queries[0]
            if in_spec0.kind == 'LINF_BALL':
                center = in_spec0.center.detach().cpu().numpy().reshape(-1)
                eps = float(in_spec0.eps)
                lb = center - eps; ub = center + eps
            elif in_spec0.kind == 'BOX':
                lb = in_spec0.lb.detach().cpu().numpy().reshape(-1)
                ub = in_spec0.ub.detach().cpu().numpy().reshape(-1)
            else:
                return {'bench': bench, 'iid': iid,
                            'verdict': f'UNSUPPORTED_INSPEC:{in_spec0.kind}',
                            'elapsed_s': round(time.time()-t0, 3)}

            verifiable = VerifiableModel(
                input_layer=InputLayer(labeled_input=labeled, shape=in_shape,
                                                  dtype=torch.float32),
                input_spec=InputSpecLayer(in_spec0),
                model=pytorch_model,
                output_spec=OutputSpecLayer(out_spec0))
            net = TorchToACT(verifiable).run()
            set_transfer_function_mode("fchz")
            tf = get_transfer_function()
            input_bounds = Bounds(
                lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))
            # Topological sort (advisor P5-1c 2026-06-08): use net.preds as-is,
            # but for backward references (pred_id > self.id) detected in parallel
            # architectures, resolve via var-graph (in_vars/out_vars relationship).
            # This minimizes scope: simple nets (no backward refs) preserve original
            # behavior; only parallel nets get var-pred fix.
            id_to_layer = {L.id: L for L in net.layers}
            # Build var → producer map for backward-ref resolution
            var_producer = {}
            for L in net.layers:
                for v in getattr(L, 'out_vars', []) or []:
                    var_producer[v] = L.id
            # Patch net.preds: for each backward ref, replace with var-pred resolution
            patched_preds = {}
            for L in net.layers:
                original = list(net.preds.get(L.id, []))
                if any(p > L.id for p in original):
                    # Compute var-pred for this layer
                    var_pred = []
                    seen = set()
                    for v in getattr(L, 'in_vars', []) or []:
                        p = var_producer.get(v)
                        if p is not None and p != L.id and p not in seen:
                            var_pred.append(p); seen.add(p)
                    # Replace backward refs with var-pred result + keep forward refs
                    if var_pred:
                        # Replace only backward part; keep forward (CONSTANT) preds
                        forward_preds = [p for p in original if p <= L.id]
                        # var_pred provides data; CONSTANT params remain
                        merged = list(var_pred)
                        for p in forward_preds:
                            if p not in merged: merged.append(p)
                        patched_preds[L.id] = merged
                    else:
                        patched_preds[L.id] = original
                else:
                    patched_preds[L.id] = original
            # Use patched preds for topo
            remaining_preds = {L.id: list(patched_preds.get(L.id, [])) for L in net.layers}
            ready = [L.id for L in net.layers if not remaining_preds[L.id]]
            topo = []
            while ready:
                L_id = ready.pop(0)
                topo.append(L_id)
                for other_id in remaining_preds:
                    if L_id in remaining_preds[other_id]:
                        remaining_preds[other_id].remove(L_id)
                        if not remaining_preds[other_id] and other_id not in topo and other_id not in ready:
                            ready.append(other_id)
            # If there is a cycle or unresolved dependency, fail closed.  Do not
            # continue in declaration order: using a placeholder input state for a
            # missing predecessor would silently verify the wrong computation.
            if len(topo) < len(net.layers):
                missing = [L.id for L in net.layers if L.id not in topo]
                return {
                    'bench': bench, 'iid': iid,
                    'verdict': 'NO_STATE',
                    'error': f'topological_sort_incomplete missing={missing[:10]}',
                    'elapsed_s': round(time.time()-t0, 3),
                }

            # Override net.preds with patched_preds so propagation +
            # _get_input_fchz + verify_once_fchz see corrected backward refs.
            net.preds = patched_preds

            before, after = {}, {}
            for L_id in topo:
                L = id_to_layer[L_id]
                preds = patched_preds.get(L.id, [])
                if L.id == 0 or not preds:
                    in_b = input_bounds
                else:
                    pred0 = preds[0]
                    if pred0 not in after:
                        return {
                            'bench': bench, 'iid': iid,
                            'verdict': 'NO_STATE',
                            'error': f'missing_primary_pred_state layer={L.id} pred={pred0}',
                            'elapsed_s': round(time.time()-t0, 3),
                        }
                    in_b = after[pred0].bounds
                after[L.id] = tf.apply(L, in_b, net, before, after)

            # PRODUCTION verify_once_fchz with queries
            result = verify_once_fchz(net, tf, queries=queries, n_out=n_out)

            mech = (f'FCHZ_TF_official_canon_K{g_max}' if g_max
                       else 'FCHZ_TF_official_canon_full')

            return {
                'bench': bench, 'iid': iid,
                'verdict': result['verdict'],
                'mechanism': mech,
                'canon_kind': result.get('kind'),
                'y_true': result.get('y_true'),
                'n_input_queries': result.get('n_input_queries'),
                'canonicalized': result.get('canonicalized'),
                'onnx_sha256': _sha256_short(onnx_p),
                'vnnlib_sha256': _sha256_short(vnn_p),
                'backend': 'act.back_end.fchz_tf.FchzTF',
                'verifier': 'verify_once_fchz(queries=...)',
                'principle_clean': True,
                'elapsed_s': round(time.time()-t0, 3),
            }
        except _TimeoutErr:
            signal.alarm(0)
            return {'bench': bench, 'iid': iid, 'verdict': 'TIMEOUT',
                        'elapsed_s': round(time.time()-t0, 3)}
        except Exception as e:
            signal.alarm(0)
            return {'bench': bench, 'iid': iid,
                        'verdict': f'ERROR:{type(e).__name__}',
                        'error': str(e)[:150],
                        'elapsed_s': round(time.time()-t0, 3)}
        finally:
            signal.alarm(0)
    except BaseException as e:
        return {'bench': bench, 'iid': iid,
                    'verdict': f'FATAL:{type(e).__name__}',
                    'error': str(e)[:150],
                    'elapsed_s': round(time.time()-t0, 3)}


def run_group(group_name: str, out_dir: Path, n_workers: int = 4):
    """Run a benchmark group via single-backend FchzTF + canonicalize."""
    benches = GROUPS.get(group_name)
    if benches is None:
        raise ValueError(f"Unknown group {group_name!r}; choices: {sorted(GROUPS.keys())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for bench, max_iid, g_max in benches:
        for iid in range(max_iid):
            jobs.append((bench, iid, g_max))

    print(f"Group: {group_name}")
    print(f"Benches: {[b[0] for b in benches]}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Workers: {n_workers}")
    print(f"Backend: act.back_end.fchz_tf.FchzTF + verify_once_fchz(queries=)")
    print(f"Loader: instances.csv (official VNN-COMP 2025)", flush=True)

    files = {b[0]: open(out_dir / f'{b[0]}.jsonl', 'w') for b in benches}
    t0 = time.time()
    done = 0; last = 0
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(_run_one, jobs, chunksize=2):
            files[r['bench']].write(json.dumps(r) + '\n')
            files[r['bench']].flush()
            done += 1
            if done - last >= 50 or done == len(jobs):
                el = time.time() - t0
                rate = done / el
                eta = (len(jobs) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(jobs)}] {done*100/len(jobs):.1f}% "
                          f"rate={rate:.2f}/s ETA={eta/60:.1f}min", flush=True)
                last = done
    for f in files.values(): f.close()

    # Aggregate
    print("\n" + "=" * 75)
    print(f"GROUP {group_name}: act.back_end.fchz_tf.FchzTF + canonicalize")
    print("=" * 75)
    print(f"{'Bench':<32} {'V':>4} {'U':>4} {'ERR':>4} {'TO':>4}")
    print("-" * 75)
    totals = {'V': 0, 'U': 0, 'ERR': 0, 'TO': 0}
    # Per advisor P5 2026-06-08: NO_STATE/NO_QUERY/MIXED are SOUND fail-closed
    # (not crashes), so they count as UNKNOWN, not ERR. Real ERR is only:
    # ERROR:<exc>, FATAL:<exc>, NO_INSTANCE, UNSUPPORTED_INSPEC:*.
    UNKNOWN_LIKE = {'UNKNOWN', 'NO_STATE', 'NO_QUERY', 'UNSUPPORTED_QUERY', 'MIXED'}
    for bench, _, _ in benches:
        f = out_dir / f'{bench}.jsonl'
        if not f.exists(): continue
        records = [json.loads(l) for l in open(f)]
        c = {'V': 0, 'U': 0, 'ERR': 0, 'TO': 0}
        for r in records:
            v = r['verdict']
            if v == 'CERTIFIED': c['V'] += 1
            elif v in UNKNOWN_LIKE: c['U'] += 1
            elif v == 'TIMEOUT': c['TO'] += 1
            else: c['ERR'] += 1
        for k in totals: totals[k] += c[k]
        print(f"{bench:<32} {c['V']:>4} {c['U']:>4} {c['ERR']:>4} {c['TO']:>4}")
    print("-" * 75)
    print(f"{'TOTAL':<32} {totals['V']:>4} {totals['U']:>4} {totals['ERR']:>4} {totals['TO']:>4}")
    print(f"\nV = {totals['V']}")
    print(f"Wall: {(time.time()-t0)/60:.1f} min")
    return totals


if __name__ == '__main__':
    group = sys.argv[1] if len(sys.argv) > 1 else 'A'
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = Path(f'/tmp/fchz_group_{group}_{time.strftime("%Y%m%d_%H%M%S")}')
    print(f"Output: {out}")
    run_group(group, out, workers)
