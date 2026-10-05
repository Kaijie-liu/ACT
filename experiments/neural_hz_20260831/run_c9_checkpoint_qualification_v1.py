"""Fresh checkpoint audit and native SciPy-HiGHS model loading, never solving."""

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
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline_worker

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c9_checkpoint_qualification_20260905_v1'
PREVIOUS = EXP / 'results/c9_integrated_suffix_20260905_v1'


def worker():
    from dataclasses import asdict
    from types import SimpleNamespace
    from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import Integrated, expression_binding
    from experiments.neural_hz_20260831.c9_integrated_suffix_audit_v1 import audit
    from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
    from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
    from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('source/library/artifact freeze drift')
    checkpoint = PREVIOUS / 'lifted_hz.pickle'
    snapshot = EXP / 'results/c5_first_terminal_20260905_v1/layer75.pickle'
    if (_sha256(checkpoint) != '616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
            or _sha256(snapshot) != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'):
        raise ValueError('checkpoint/original snapshot seal drift')
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    with snapshot.open('rb') as stream:
        original_native = pickle.load(stream)
    if saved['schema'] != 'c9_integrated_suffix_checkpoint_v1':
        raise ValueError('unexpected candidate checkpoint schema')
    expected_fields = {'schema', 'hz', 'original_prefix_hz_cache', 'definition_graph', 'expression',
        'root', 'old_n_cont', 'old_n_bin', 'old_n_eq', 'keep', 'report', 'identity_audit',
        'provenance', 'logical_n_cont', 'eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales',
        'def_rows', 'origin_snapshot_sha256', 'formal_gain'}
    if set(saved) != expected_fields or saved['origin_snapshot_sha256'] != _sha256(snapshot):
        raise ValueError('unknown/missing candidate checkpoint payload')
    lifted = Integrated(saved['expression'], expression_binding(saved['expression']), saved['hz'],
        saved['definition_graph'], saved['root'], saved['old_n_cont'], saved['old_n_bin'],
        saved['old_n_eq'], saved['logical_n_cont'], saved['keep'], saved['report'],
        saved['eq_roots'], saved['eq_scales'], saved['ineq_roots'], saved['ineq_scales'], saved['def_rows'])
    lifted.seal = lifted.fingerprint()
    started = time.monotonic()
    identity = audit(lifted)
    identity['stage_elapsed_s'] = time.monotonic() - started
    print(json.dumps({'event': 'complete_checkpoint_identity', **identity}), flush=True)
    full = collect(SimpleNamespace(), {'original_native': original_native, 'checkpoint': saved,
        **lifted.numeric_roots()})
    candidate = full.measure()
    print(json.dumps({'event': 'complete_duplicate_owner_union', 'roots': len(full.numeric),
        'bytes': candidate.resident_bytes, 'entries': candidate.resident_entries}), flush=True)
    witness, reference = reference_subset(full, original_native['net'])
    lower = reference['reference_lower_bound']
    physical = candidate.resident_bytes < lower['resident_bytes'] and candidate.resident_entries < lower['resident_entries']
    if collect(SimpleNamespace(), {'original_native': original_native, 'checkpoint': saved,
            **lifted.numeric_roots()}).fingerprint != full.fingerprint:
        raise ValueError('fresh physical audit mutated original/candidate state')
    print(json.dumps({'event': 'fresh_reference_qualification', 'physical': physical}), flush=True)
    # No measured construction follows this call. No HZ was rebuilt here.
    native = inspect(lifted.hz)
    passed = physical and native['passed']
    record = {'schema': 'c9_checkpoint_qualification_v1', 'formal_gain': 0,
        'status': 'CHECKPOINT_QUALIFIED' if passed else 'CHECKPOINT_REJECTED', 'passed': passed,
        'identity': identity, 'native_ingestion': native,
        'complete_reloaded_numeric_roots': len(full.numeric), 'complete_reloaded_state': asdict(candidate),
        'reference': reference, 'strict_physical_decrease_with_duplicate_owners': physical,
        'original_snapshot_and_checkpoint_both_retained': True, 'equal_payloads_coalesced': False,
        'candidate_constructed': False, 'failed_transaction_v1_reclassified': False,
        'live_relu78_executed': False, 'terminal_solve_executed': False,
        'native_ingestion_after_reference': True,
        'provenance': freeze['provenance'], 'source_sha256': freeze['source_sha256'],
        'checkpoint_sha256': _sha256(checkpoint), 'snapshot_sha256': _sha256(snapshot),
        'max_rss_kib_including_oracles': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    _atomic_exclusive_json(DIRECTORY / 'result.json', record)
    print(json.dumps({'status': record['status'], 'physical': physical,
        'native_differences': native['different_coefficients'], 'formal_gain': 0}), flush=True)
    if not passed:
        raise ValueError('checkpoint qualification failed')


def main():
    import scipy.optimize._highspy._core as core
    import scipy.optimize._highspy._highs_wrapper as wrapper
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    previous_exit = json.loads((PREVIOUS / 'exit.json').read_text())
    events = [json.loads(line) for line in (PREVIOUS / 'factor_events.jsonl').read_text().splitlines()]
    stages = {event['event']: event for event in events}
    if (previous_exit.get('worker_exit_code') != 1 or previous_exit.get('tests_exit_code') != 0
            or previous_exit['source_drift'] or previous_exit['provenance_drift']
            or not stages['construction_complete']['construction']['measured_transient_gate']
            or not stages['native_ingestion']['passed']
            or stages['complete_identity_audit']['all_main_defining_rows_checked'] != 243162):
        raise ValueError('closed transaction prerequisite changed or incomplete')
    prior = json.loads((PREVIOUS / 'preregistered.json').read_text())
    hashes = prior['source_sha256']
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prior source drift')
    for path in (Path(__file__), EXP / 'C9_INTEGRATED_V1_CLOSED_CHECKPOINT_PREREG_20260905.md',
                 PREVIOUS / 'factor_events.jsonl', PREVIOUS / 'worker.log', PREVIOUS / 'exit.json', PREVIOUS / 'lifted_hz.pickle', Path(core.__file__), Path(wrapper.__file__)):
        hashes[str(path)] = _sha256(path)
    provenance = baseline_worker._provenance(ROOT)
    command = [sys.executable, str(Path(__file__)), '--worker']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'command': command, 'wall_cap_s': 240, 'memory_gb': 16, 'solve_authorized': False, 'candidate_construction_authorized': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in prior['tests'])], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = tests.returncode
        if tests.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started, source_drift=any(_sha256(EXP / name) != sha for name, sha in hashes.items()),
            provenance_drift=baseline_worker._provenance(ROOT) != provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    worker() if len(sys.argv) == 2 and sys.argv[1] == '--worker' else main()


