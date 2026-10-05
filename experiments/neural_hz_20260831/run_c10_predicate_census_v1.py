"""Freeze, test and retain one read-only census of the actual C9 final HZ."""

import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c10_predicate_census_20260908_v1'
FINAL = EXP / 'results/c9_first_terminal_20260908_v1/final_hz.pickle'
LIVE = EXP / 'results/c9_live_relu_20260906_v1/relu78.pickle'


def worker():
    from dataclasses import asdict
    import numpy as np
    import torch
    from experiments.neural_hz_20260831.c10_predicate_census_v1 import census
    from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c10_predicate_census_v1', 'formal_gain': 0, 'completed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    started = time.monotonic()
    try:
        if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
            raise ValueError('read-only census source/artifact freeze drift')
        if (_sha256(FINAL) != '841af01fb74ffa8cfdb7ac434a4f8da632739d866983f2b44ed0f34c353ed0b0'
                or _sha256(LIVE) != '5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65'):
            raise ValueError('original checkpoint hash drift before unpickling')
        with FINAL.open('rb') as stream:
            final_saved = pickle.load(stream)
        with LIVE.open('rb') as stream:
            live_saved = pickle.load(stream)
        if final_saved['schema'] != 'neural_hz_checkpoint_v1' or live_saved['schema'] != 'c9_live_relu_checkpoint_v1':
            raise ValueError('unexpected original checkpoint schema')
        hz, pre, post = final_saved['final_hz'], live_saved['preactivation_hz'], live_saved['post_relu_hz']
        if (source_digest(hz) != 'b2024dfbe2af20f7c9e729bf79c835d5cf7b2050ab57113cc33bebe78f9801c8'
                or source_digest(post) != '8549d44ed7254710e86d94dd9ce8be3d9bc279246fe78f94f9a56cba25b46fe7'):
            raise ValueError('actual final/post-ReLU content drift')
        for name in ('Ac', 'Ab', 'Auc', 'Aub'):
            matrix, original = getattr(hz, name), getattr(pre, name)
            if (not equal_payload(matrix[:original.shape[0], :original.shape[1]], original)
                    or matrix[:original.shape[0], original.shape[1]:].nnz):
                raise ValueError('original C9 predicate prefix not preserved in final HZ')
        if not equal_payload(hz.b[:pre.n_eq], pre.b) or not equal_payload(hz.ub[:pre.n_ineq], pre.ub):
            raise ValueError('original C9 predicate right hand side drift')
        original_hashes = [source_digest(v) for v in (hz, pre, post, final_saved['input_hz'])]
        maps = live_saved['numeric_roots']
        old_nc, logical_nc = int(live_saved['old_n_cont']), int(live_saved['logical_n_cont'])
        old_eq = pre.n_eq - (logical_nc - old_nc) - maps['def_rows'].size
        with (DIRECTORY / 'events.jsonl').open('x') as stream:
            def observe(event):
                stream.write(json.dumps({'elapsed_s': time.monotonic() - started, **event}) + '\n')
                stream.flush()
                print(json.dumps(event), flush=True)
            (report, table), measured = measured_build(lambda: census(hz, old_n_cont=old_nc,
                logical_n_cont=logical_nc, old_n_eq=int(old_eq), eq_roots=maps['eq_roots'],
                def_rows=maps['def_rows'], observe=observe))
        if [source_digest(v) for v in (hz, pre, post, final_saved['input_hz'])] != original_hashes:
            raise ValueError('census mutated a source HZ')
        table_path = DIRECTORY / 'main_factor_table.npz'
        with table_path.open('xb') as stream:
            np.savez(stream, **table)
            stream.flush()
            os.fsync(stream.fileno())
        record.update(completed=True, report=report, construction=measured,
            source_hz_unchanged=True, predicate_prefix_verified=True,
            factor_table_sha256=_sha256(table_path), factor_table_bytes=table_path.stat().st_size,
            input_checkpoint_sha256=_sha256(FINAL), live_checkpoint_sha256=_sha256(LIVE))
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        record.update(wall_s=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY / 'result.json', record)
        print(json.dumps({'completed': record['completed'], 'failure': record.get('failure'), 'formal_gain': 0}), flush=True)
    if not record['completed']:
        raise SystemExit(1)


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prior_dir = EXP / 'results/c9_first_terminal_20260908_v1'
    if _sha256(prior_dir / 'result.json') != 'b432becfa9e6dd88149587b9c594e9a3c6e5794515abeb3b898d338c558017d9':
        raise ValueError('closed C9 outcome drift')
    prior = json.loads((prior_dir / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prerequisite source drift')
    names = [Path(__file__).name, 'c10_predicate_census_v1.py', 'test_c10_predicate_census_v1.py',
        'C10_PREDICATE_CENSUS_PREREG_20260908.md', 'C9_FIRST_TERMINAL_AUDIT_20260908.md',
        'CHECKPOINT_C9_TERMINAL_20260908_SHA256SUMS', str(FINAL), str(LIVE),
        'results/c9_first_terminal_20260908_v1/result.json', 'results/c9_first_terminal_20260908_v1/exit.json']
    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c10_predicate_census_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--worker']
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'entry_cap': 64_000_000, 'construction_cap_bytes': 1024**3,
        'solver_authorized': False, 'transformation_authorized': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = tests_run.returncode
        if tests_run.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / name) != sha for name, sha in hashes.items()),
            provenance_drift=baseline._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    worker() if len(sys.argv) == 2 and sys.argv[1] == '--worker' else main()
