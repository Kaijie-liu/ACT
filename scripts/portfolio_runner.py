"""HyZor Portfolio Runner — strict P1-P5 forward HZ portfolio.

Per advisor 2026-06-07: single command that runs multiple in-principle profiles
per iid, records provenance, and aggregates to a paper-grade V/A count.

Profiles (all forward, all P1-P5 clean):
  P1: raw walker regular        (G_max=None)         — small dense, MLP
  P2: raw walker sparse-slack   (G_max=128)          — cifar/tiny ResNet
  P3: raw walker sparse-slack   (G_max=64)           — vggnet, heavy conv
  P4: raw walker hz_only        (G_max=None)         — memory-tight ReLU
  P5: raw walker hz_only        (G_max=128)          — deep + memory-tight

Verdict per iid: first CERT/FAL wins. UNKNOWN only if ALL profiles UNK.

Output JSON per iid:
  bench, iid, verdict, mechanism, hz_excess, profile_order, sha256, principle_clean
"""
import sys, os, json, time, signal, hashlib
from pathlib import Path
from multiprocessing import Pool

OUT_DIR = Path(f'/tmp/hyzor_portfolio_{time.strftime("%Y%m%d_%H%M%S")}')
OUT_DIR.mkdir(exist_ok=True)
print(f"Output: {OUT_DIR}", flush=True)

# Profile order: tries small-G first (cheap), then heavier
# Each: (label, hz_only, G_max_cols, mechanism_tag)
PROFILES = [
    ('regular_K_none',     False,  None, 'FCHZ_walker_HZ_closed_form'),
    ('sparse_K128',         False,  128,  'FCHZ_walker_HZ_closed_form_sparse_slack_K128'),
    ('sparse_K64',          False,  64,   'FCHZ_walker_HZ_closed_form_sparse_slack_K64'),
    ('hz_only_K_none',     True,   None, 'FCHZ_walker_hz_only_tail_radius_sound'),
    ('hz_only_K128',       True,   128,  'FCHZ_walker_hz_only_tail_radius_K128'),
]

# (bench, max_iid, lane)
# lane: 'small' (8w), 'medium' (4w), 'large' (1w)
BENCHES = [
    ('test', 5, 'small'),
    ('collins_aerospace_benchmark', 6, 'large'),  # 572GB OOM, skip but try
    ('cersyve', 12, 'small'),
    ('vggnet16_2022', 18, 'large'),  # 169GB MemoryError
    ('cgan_2023', 21, 'medium'),
    ('tllverifybench_2023', 32, 'small'),
    ('cctsdb_yolo_2023', 39, 'medium'),
    ('traffic_signs_recognition_2023', 45, 'medium'),
    ('soundnessbench', 50, 'small'),
    ('linearizenn_2024', 60, 'small'),
    ('collins_rul_cnn_2022', 62, 'small'),
    ('ml4acopf_2024', 69, 'medium'),
    ('dist_shift_2023', 72, 'small'),
    ('yolo_2023', 72, 'large'),
    ('lsnc_relu', 80, 'small'),
    ('metaroom_2023', 100, 'medium'),
    ('sat_relu', 100, 'small'),
    ('malbeware', 150, 'small'),
    ('cora_2024', 180, 'small'),
    ('acasxu_2023', 186, 'small'),
    ('nn4sys', 194, 'small'),
    ('vit_2023', 200, 'large'),
    ('tinyimagenet_2024', 200, 'medium'),
    ('cifar100_2024', 200, 'medium'),
    ('relusplitter', 220, 'small'),
    ('safenlp_2024', 1080, 'small'),
]

# Profile selection per bench (lean — 1 profile most benches, 2 max for fallback)
# Single profile for high-mem benches to avoid OOM.
BENCH_PROFILES = {
    # Large CNN: single sparse profile only (OOM-safe)
    'cifar100_2024':       ['sparse_K128'],
    'tinyimagenet_2024':   ['sparse_K128'],
    'vggnet16_2022':       ['sparse_K64'],
    'vit_2023':            ['sparse_K128'],
    'yolo_2023':           ['sparse_K128'],
    'cctsdb_yolo_2023':    ['sparse_K128'],
    'metaroom_2023':       ['sparse_K128'],
    'traffic_signs_recognition_2023': ['sparse_K128'],
    'cgan_2023':           ['sparse_K128'],
    'collins_aerospace_benchmark': ['sparse_K64'],
    # Small benches: 2 profiles for fallback (regular + hz_only)
    'safenlp_2024':        ['regular_K_none', 'hz_only_K_none'],
    'acasxu_2023':         ['regular_K_none', 'hz_only_K_none'],
    'malbeware':           ['regular_K_none', 'hz_only_K_none'],
    'dist_shift_2023':     ['regular_K_none'],
    'cora_2024':           ['regular_K_none', 'hz_only_K_none'],
    'nn4sys':              ['regular_K_none'],
    'relusplitter':        ['regular_K_none', 'hz_only_K_none'],
    'collins_rul_cnn_2022': ['regular_K_none'],
    'tllverifybench_2023': ['regular_K_none'],
    'linearizenn_2024':    ['regular_K_none'],
    'lsnc_relu':           ['regular_K_none'],
    'ml4acopf_2024':       ['regular_K_none'],
    'sat_relu':            ['regular_K_none'],
    'cersyve':             ['regular_K_none'],
    'test':                ['regular_K_none'],
    'soundnessbench':      ['regular_K_none'],
}

TIMEOUT = 30


class TimeoutError_(Exception): pass
def _handler(signum, frame): raise TimeoutError_("worker timeout")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()


def run_profile(bench, iid, profile, onnx_p, lb_np, ub_np, unsafe, n_in, n_cls):
    """Run one profile on one iid. Returns (verdict, mechanism, excess, error)."""
    label, hz_only, g_max, mech = profile
    try:
        signal.alarm(TIMEOUT)
        # Plain imports — no re-import (was leaking memory)
        from research.sc_hz.fchz_walker import forward_fchz
        from research.sc_hz.fc_hz_state import hz_closed_form_ub
        import numpy as np
        import gc

        if g_max:
            os.environ['HYZOR_FCHZ_G_MAX_COLS'] = str(g_max)
        else:
            os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)
        os.environ['HYZOR_FCHZ_USE_CUDA'] = '1'

        result = forward_fchz(str(onnx_p), lb_np, ub_np, hz_only=hz_only, G_max_cols=g_max)
        if not unsafe:
            signal.alarm(0)
            del result
            gc.collect()
            return ('UNKNOWN', mech, None, 'no unsafe')

        max_excess = max(hz_closed_form_ub(result.state, d) - float(t)
                                 for d, t, _ in unsafe)
        verdict = 'CERTIFIED' if max_excess < 0 else 'UNKNOWN'
        signal.alarm(0)
        # Free walker state before next profile (cifar/tiny state ~10GB)
        del result
        gc.collect()
        return (verdict, mech, float(max_excess), None)
    except TimeoutError_:
        return ('TIMEOUT', mech, None, 'profile timeout')
    except Exception as e:
        return ('ERROR', mech, None, f'{type(e).__name__}: {str(e)[:100]}')
    finally:
        signal.alarm(0)


def run_iid(args):
    bench, iid = args
    t0 = time.time()
    try:
        sys.path.insert(0, '/data1/Kane/ACT')
        sys.path.insert(0, '/data1/Kane/ACT/research/sc_hz')
        signal.signal(signal.SIGALRM, _handler)
        import numpy as np
        import onnx

        from research.canonical_provenance import load_instance
        from research.sc_hz.vnnlib_parse import parse_vnnlib

        try:
            onnx_p, vnn_p = load_instance(bench, iid)
        except Exception as e:
            return {'bench': bench, 'iid': iid, 'verdict': 'NO_INSTANCE',
                        'error': str(e)[:80], 'elapsed_s': time.time() - t0}

        m = onnx.load(str(onnx_p))
        init = {x.name for x in m.graph.initializer}
        din = [x for x in m.graph.input if x.name not in init][0]
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in din.type.tensor_type.shape.dim]
        n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))
        od = [d.dim_value if d.dim_value > 0 else 1 for d in m.graph.output[0].type.tensor_type.shape.dim]
        n_cls = int(np.prod(od[1:])) if len(od) > 1 else od[0]
        lb_np, ub_np, unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)

        # Get profiles for this bench
        profile_labels = BENCH_PROFILES.get(bench, ['regular_K_none'])
        profile_map = {p[0]: p for p in PROFILES}
        profiles_to_run = [profile_map[lbl] for lbl in profile_labels if lbl in profile_map]

        # Try each profile in order, first CERT wins
        profile_results = []
        winning = None
        for profile in profiles_to_run:
            verdict, mech, excess, err = run_profile(
                bench, iid, profile, onnx_p, lb_np, ub_np, unsafe, n_in, n_cls)
            profile_results.append({
                'profile': profile[0], 'verdict': verdict,
                'mechanism': mech, 'excess': excess, 'error': err,
            })
            if verdict == 'CERTIFIED':
                winning = profile_results[-1]
                break
            elif verdict == 'FALSIFIED':  # not used in walker, but reserve
                winning = profile_results[-1]
                break

        if winning is None:
            # No profile gave V/A. Take first non-error or UNK.
            for r in profile_results:
                if r['verdict'] in ('UNKNOWN',):
                    winning = r; break
            if winning is None and profile_results:
                winning = profile_results[0]
            elif winning is None:
                winning = {'verdict': 'NO_PROFILE', 'mechanism': '?', 'excess': None}

        return {
            'bench': bench, 'iid': iid,
            'verdict': winning['verdict'],
            'mechanism': winning.get('mechanism', '?'),
            'hz_excess': winning.get('excess'),
            'profile_order': [r['profile'] for r in profile_results],
            'profile_results': profile_results,
            'onnx_sha256': sha256_file(onnx_p)[:16],
            'vnnlib_sha256': sha256_file(vnn_p)[:16],
            'principle_clean': True,
            'elapsed_s': round(time.time() - t0, 3),
        }
    except BaseException as e:
        return {'bench': bench, 'iid': iid, 'verdict': f'FATAL:{type(e).__name__}',
                    'error': str(e)[:200], 'elapsed_s': time.time() - t0}


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # Build jobs
    jobs = []
    for bench, max_iid, lane in BENCHES:
        for iid in range(max_iid):
            jobs.append((bench, iid))

    print(f"Total jobs: {len(jobs)}")
    print(f"Workers: {n_workers}")
    print(f"Profiles: {[p[0] for p in PROFILES]}\n", flush=True)

    files = {b: open(OUT_DIR / f'{b}.jsonl', 'w') for b, _, _ in BENCHES}

    t0 = time.time()
    done = 0; last = 0
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(run_iid, jobs, chunksize=2):
            bench = r['bench']
            files[bench].write(json.dumps(r) + '\n')
            files[bench].flush()
            done += 1
            if done - last >= 200 or done == len(jobs):
                el = time.time() - t0
                rate = done / el
                eta = (len(jobs) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(jobs)}] {done*100/len(jobs):.1f}% rate={rate:.2f}/s ETA={eta/60:.1f}min", flush=True)
                last = done

    for f in files.values(): f.close()

    # Aggregate
    print("\n" + "=" * 75)
    print("HYZOR PORTFOLIO AGGREGATE")
    print("=" * 75)
    print(f"{'Bench':<32} {'V':>4} {'A':>3} {'U':>4} {'ERR':>4}")
    print("-" * 75)
    totals = {'V': 0, 'A': 0, 'U': 0, 'ERR': 0}
    for bench, _, _ in BENCHES:
        f = OUT_DIR / f'{bench}.jsonl'
        if not f.exists(): continue
        records = [json.loads(l) for l in open(f)]
        c = {'V': 0, 'A': 0, 'U': 0, 'ERR': 0}
        for r in records:
            v = r['verdict']
            if v == 'CERTIFIED': c['V'] += 1
            elif v == 'FALSIFIED': c['A'] += 1
            elif v == 'UNKNOWN': c['U'] += 1
            else: c['ERR'] += 1
        for k in totals: totals[k] += c[k]
        print(f"{bench:<32} {c['V']:>4} {c['A']:>3} {c['U']:>4} {c['ERR']:>4}")
    print("-" * 75)
    print(f"{'TOTAL':<32} {totals['V']:>4} {totals['A']:>3} {totals['U']:>4} {totals['ERR']:>4}")
    print(f"\nV+A = {totals['V'] + totals['A']}")
    print(f"Total time: {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    main()
