"""One exclusive, automatically archived C10 quotient component transaction."""

import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c10_alias_quotient_20260908_v1'
FINAL = EXP / 'results/c9_first_terminal_20260908_v1/final_hz.pickle'
LIVE = EXP / 'results/c9_live_relu_20260906_v1/relu78.pickle'


def worker():
    import torch
    from experiments.neural_hz_20260831.c10_alias_quotient_v1 import quotient, ARRAYS
    from experiments.neural_hz_20260831.c10_alias_quotient_audit_v1 import audit
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
    from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
    from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c10_alias_quotient_component_v1', 'completed': False, 'formal_gain': 0,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance'],
        'solver_executed': False, 'live_integration_executed': False}
    started = time.monotonic()
    source_hz, original_digest = None, None
    try:
        if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
            raise ValueError('frozen quotient source/artifact drift')
        if (_sha256(FINAL) != '841af01fb74ffa8cfdb7ac434a4f8da632739d866983f2b44ed0f34c353ed0b0'
                or _sha256(LIVE) != '5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65'):
            raise ValueError('sealed checkpoint drift before unpickling')
        with FINAL.open('rb') as stream:
            final_saved = pickle.load(stream)
        with LIVE.open('rb') as stream:
            live_saved = pickle.load(stream)
        if final_saved['schema'] != 'neural_hz_checkpoint_v1' or live_saved['schema'] != 'c9_live_relu_checkpoint_v1':
            raise ValueError('unexpected checkpoint schema')
        source_hz = final_saved['final_hz']
        original_digest = source_digest(source_hz)
        if original_digest != 'b2024dfbe2af20f7c9e729bf79c835d5cf7b2050ab57113cc33bebe78f9801c8':
            raise ValueError('wrong original final HZ content')
        old_nc = int(live_saved['old_n_cont'])
        logical_nc = int(live_saved['logical_n_cont'])
        maps = live_saved['numeric_roots']
        old_eq = int(live_saved['preactivation_hz'].n_eq - (logical_nc - old_nc) - maps['def_rows'].size)
        with (DIRECTORY / 'events.jsonl').open('x') as stream:
            def observe(event):
                stream.write(json.dumps({'elapsed_s': time.monotonic() - started, **event}) + '\n')
                stream.flush()
                print(json.dumps(event), flush=True)
            candidate, construction = measured_build(lambda: quotient(source_hz, enabled=True,
                old_n_cont=old_nc, logical_n_cont=logical_nc, old_n_eq=old_eq,
                eq_roots=maps['eq_roots'], def_rows=maps['def_rows'], observe=observe))
            record.update(construction=construction, quotient=candidate.report)
            observe({'event': 'independent_audit_started'})
            audit_started = time.monotonic()
            proof = audit(source_hz, candidate)
            proof['elapsed_s'] = time.monotonic() - audit_started
            record['proof'] = proof
            observe({'event': 'independent_audit_passed', **proof})
            scalar_metadata = {key: value for key, value in vars(candidate).items() if key not in {'hz', *ARRAYS}}
            roots = collect(SimpleNamespace(), {'component': candidate.numeric_roots(), 'metadata': scalar_metadata})
            record['storage_diagnostics'] = {
                'numeric_root_count': len(roots.numeric),
                'python_shallow_bytes': roots.python_shallow_bytes + sys.getsizeof(candidate),
                'python_allocator_in_numeric_gate': False,
                'full_original_and_live_checkpoints_retained_for_offline_audit': True,
                'whole_live_storage_gain_claimed': False}
            native = inspect(candidate.hz)
            record['native_ingestion'] = native
            if not native['passed'] or native['lowered_n_bin'] != source_hz.n_bin:
                raise ValueError('native coefficient/integrality retention failed')
            observe({'event': 'native_ingestion_passed', 'matrix_nnz': native['retained_matrix_nnz'],
                     'lowered_n_cont': native['lowered_n_cont'], 'lowered_n_bin': native['lowered_n_bin']})
        if source_digest(source_hz) != original_digest:
            raise ValueError('original HZ changed')
        path = DIRECTORY / 'quotient.pickle'
        with path.open('xb') as stream:
            pickle.dump({'schema': 'c10_alias_quotient_checkpoint_v1', 'quotient': candidate,
                'provenance': freeze['provenance'], 'source_sha256': freeze['source_sha256'],
                'proof': proof, 'formal_gain': 0}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        record.update(completed=True, checkpoint_sha256=_sha256(path), checkpoint_bytes=path.stat().st_size,
            quotient_hz_sha256=source_digest(candidate.hz))
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        if original_digest is not None:
            record['original_hz_unchanged'] = source_digest(source_hz) == original_digest
        record.update(wall_s=time.monotonic() - started,
            max_rss_kib_including_original_checkpoints_oracles_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY / 'result.json', record)
        print(json.dumps({'completed': record['completed'], 'failure': record.get('failure'), 'formal_gain': 0}), flush=True)
    if not record['completed']:
        raise SystemExit(1)


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prior_dir = EXP / 'results/c10_predicate_census_20260908_v1'
    if _sha256(prior_dir / 'result.json') != 'd3b470ce9eb923ce013f2a75236a5507c39a75d398c7db83010ef823ba824e9a':
        raise ValueError('sealed census result drift')
    prior = json.loads((prior_dir / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('inherited source drift')
    names = [Path(__file__).name, 'c10_alias_quotient_v1.py', 'c10_alias_quotient_audit_v1.py',
        'test_c10_alias_quotient_v1.py', 'C10_ALIAS_QUOTIENT_PREREG_20260908.md',
        'C10_ALIAS_DEVELOPMENT_NOTE_20260908.md', 'CHECKPOINT_C10_CENSUS_20260908_SHA256SUMS',
        'results/c10_predicate_census_20260908_v1/result.json',
        'results/c10_predicate_census_20260908_v1/exit.json']
    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c10_alias_quotient_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--worker']
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes,
        'provenance': provenance, 'tests': tests, 'command': command,
        'wall_cap_s': 240, 'memory_gb': 16, 'work_cap': 256_000_000,
        'entry_cap': 64_000_000, 'construction_cap_bytes': 1024**3,
        'transformation_authorized': True, 'native_ingestion_authorized_after_proof': True,
        'solver_authorized': False, 'live_integration_authorized': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = run.returncode
        if run.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            run = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = run.returncode
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
