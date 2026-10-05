"""One sealed read-only census and independent ordinary model-lowering timing."""

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
DIRECTORY = EXP / 'results/c11_single_use_census_20260910_v1'
FINAL = EXP / 'results/c10_first_terminal_20260908_v1/final_hz.pickle'
LIVE = EXP / 'results/c10_live_relu_20260908_v1/relu78.pickle'


def worker():
    import numpy as np
    import torch
    from act.back_end.solver.solver_hz import _lower_hz_milp
    from experiments.neural_hz_20260831.c11_single_use_census_v1 import census
    from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c11_single_use_census_v1', 'formal_gain': 0, 'completed': False,
        'solver_executed': False, 'candidate_transformation_constructed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    started = time.monotonic()
    with (DIRECTORY / 'events.jsonl').open('x') as stream:
        def observe(event):
            stream.write(json.dumps({'elapsed_s': time.monotonic() - started, **event}) + '\n')
            stream.flush()
            print(json.dumps(event), flush=True)
        try:
            if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
                raise ValueError('source/artifact freeze drift')
            if (_sha256(FINAL) != '192f3a5a95637933bbbed8b57b10fd20f7d237b4c6f2066573e5dfa14e6b1971'
                    or _sha256(LIVE) != '1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d'):
                raise ValueError('original checkpoint hash drift before unpickling')
            with FINAL.open('rb') as handle:
                final_saved = pickle.load(handle)
            with LIVE.open('rb') as handle:
                live_saved = pickle.load(handle)
            if final_saved['schema'] != 'neural_hz_checkpoint_v1' or live_saved['schema'] != 'c10_live_relu_checkpoint_v1':
                raise ValueError('unexpected original checkpoint schema')
            # Both full source checkpoints remain reachable throughout these
            # read-only diagnostics; no omitted runtime root or rewritten HZ.
            hz, pre, post = final_saved['final_hz'], live_saved['preactivation_hz'], live_saved['post_relu_hz']
            if (source_digest(hz) != '7d32e48d3c9b24360e83ae35c72e27fc29ace03b04f1f5763f006899585ba8f6'
                    or source_digest(pre) != '41a3bb791a7887da1088d69678952668a08433b1efeb5cdf9fc182f9afe2adb3'
                    or source_digest(post) != '82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367'):
                raise ValueError('sealed HZ content drift')
            for name in ('Ac', 'Ab', 'Auc', 'Aub'):
                matrix, original = getattr(hz, name), getattr(pre, name)
                if (not equal_payload(matrix[:original.shape[0], :original.shape[1]], original)
                        or matrix[:original.shape[0], original.shape[1]:].nnz):
                    raise ValueError('original tagged C10 predicate prefix not preserved')
            if not equal_payload(hz.b[:pre.n_eq], pre.b) or not equal_payload(hz.ub[:pre.n_ineq], pre.ub):
                raise ValueError('original C10 predicate right hand side drift')
            original_hashes = [source_digest(v) for v in (hz, pre, post, final_saved['input_hz'])]
            maps = live_saved['numeric_roots']
            old_nc, logical_nc = int(live_saved['old_n_cont']), int(live_saved['logical_n_cont'])
            old_eq = int(maps['eq_roots'].size) - (logical_nc - old_nc)
            observe({'event': 'sealed_source_loaded', 'old_n_cont': old_nc, 'logical_n_cont': logical_nc,
                'old_n_eq': old_eq, 'full_source_checkpoints_retained': True})
            census_record = {'completed': False, 'formal_gain': 0}
            try:
                (report, table), measured = measured_build(lambda: census(hz, old_n_cont=old_nc,
                    logical_n_cont=logical_nc, old_n_eq=old_eq, eq_roots=maps['eq_roots'],
                    eq_scales=maps['eq_scales'], def_rows=maps['def_rows'], observe=observe))
                table_path = DIRECTORY / 'single_use_factor_table.npz'
                with table_path.open('xb') as handle:
                    np.savez(handle, **table)
                    handle.flush()
                    os.fsync(handle.fileno())
                census_record.update(completed=True, report=report, construction=measured,
                    factor_table_sha256=_sha256(table_path), factor_table_bytes=table_path.stat().st_size)
            except MemoryError as exc:
                census_record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
            _atomic_exclusive_json(DIRECTORY / 'census.json', census_record)
            record['census'] = census_record
            if [source_digest(v) for v in (hz, pre, post, final_saved['input_hz'])] != original_hashes:
                raise ValueError('census mutated a source HZ')
            observe({'event': 'census_saved', 'census_completed': census_record['completed'], 'formal_gain': 0})
            # This separate diagnostic does not accept or retry a failed census.
            observe({'event': 'ordinary_lowering_start', 'optimizer_called': False})
            model, measured = measured_build(lambda: _lower_hz_milp(hz, prune_unused=True,
                coalesce_rows=True, project_inactive_cont=False, fix_implied_binary=False))
            model_arrays = {key: getattr(model, key) for key in
                ('cont_source', 'bin_source', 'var_lb', 'var_ub', 'row_lb', 'row_ub', 'integrality')}
            model_arrays.update(A_data=model.A.data, A_indices=model.A.indices, A_indptr=model.A.indptr)
            # Save source maps/bounds, not another 132MB matrix copy; matrix
            # payload hashes are exact and final_hz is already preserved.
            import hashlib
            array_hashes = {key: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()}
                for key, v in model_arrays.items()}
            source_path = DIRECTORY / 'lowering_sources.npz'
            with source_path.open('xb') as handle:
                np.savez(handle, cont_source=model.cont_source, bin_source=model.bin_source)
                handle.flush()
                os.fsync(handle.fileno())
            lowered = {'completed': True, 'construction': measured, 'n_cont': model.n_cont, 'n_bin': model.n_bin,
                'n_var': model.n_var, 'rows': model.A.shape[0], 'matrix_nnz': model.A.nnz,
                'array_sha256': array_hashes, 'source_map_sha256': _sha256(source_path),
                'optimizer_called': False, 'presolve_called': False, 'native_pass_model_called': False,
                'projection_enabled': False, 'phase_fix_enabled': False, 'formal_gain': 0,
                'prior_timeout_location_proved': False}
            _atomic_exclusive_json(DIRECTORY / 'lowering.json', lowered)
            observe({'event': 'ordinary_lowering_saved', 'elapsed_s_lowering': measured['elapsed_s'],
                'n_cont': model.n_cont, 'n_bin': model.n_bin, 'matrix_nnz': model.A.nnz})
            if [source_digest(v) for v in (hz, pre, post, final_saved['input_hz'])] != original_hashes:
                raise ValueError('diagnostics mutated a source HZ')
            record.update(completed=census_record['completed'], lowering=lowered,
                diagnostics_finished=True, source_hz_unchanged=True, predicate_prefix_verified=True)
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
    prerequisite = EXP / 'results/c10_first_terminal_20260908_v1'
    if (_sha256(prerequisite / 'exit.json') != 'a2486ac5391992a0c20f62d79f8dec877e9d1097d1f14e15bfcc859a4309b437'
            or _sha256(prerequisite / 'terminal_gate.json') != 'bebaf0e4477cf65a52dde6bc18a1bad12d419debd1eccbe131529af38fb22333'):
        raise ValueError('closed C10 terminal artifacts drift')
    prior = json.loads((prerequisite / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c11_single_use_census_v1.py', 'test_c11_single_use_census_v1.py',
        'C11_SINGLE_USE_PREREG_20260910.md', 'C11_SINGLE_USE_DEVELOPMENT_20260910.md',
        'CHECKPOINT_C10_TERMINAL_20260910_SHA256SUMS', 'C10_FIRST_TERMINAL_AUDIT_20260910.md',
        str(FINAL), str(LIVE), 'results/c10_first_terminal_20260908_v1/exit.json',
        'results/c10_first_terminal_20260908_v1/terminal_gate.json']
    hashes.update({name: _sha256(EXP / name) for name in names})
    tests = ['test_c11_single_use_census_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--worker']
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'entry_cap': 64_000_000, 'construction_cap_bytes': 1024**3,
        'solver_authorized': False, 'transformation_authorized': False, 'formal_gain': 0,
        'lowering_diagnostic_independent_of_census_acceptance': True})
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
