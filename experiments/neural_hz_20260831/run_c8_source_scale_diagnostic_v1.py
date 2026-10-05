"""Exclusive read-only diagnosis; no lift, operator expansion or solver call."""

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
DIRECTORY = EXP / 'results/c8_source_scale_diagnostic_20260905_v1'
PREVIOUS = EXP / 'results/c8_dyadic_balance_20260905_v1'
CHECKPOINT = EXP / 'results/c7_factored_hz_20260905_v1/lifted_hz.pickle'


def worker():
    import numpy as np
    from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import box_exponent, scaled_exact
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('source/checkpoint freeze drift')
    if _sha256(CHECKPOINT) != '6cbe92faf0d6eb5e4113f3e6e0b324a82a00d2d1a6019a73f2c2fd0f522e82df':
        raise ValueError('sealed checkpoint drift')
    with CHECKPOINT.open('rb') as handle:
        saved = pickle.load(handle)
    nodes = saved['definition_graph']
    old_counts = saved['report']['node_counts']
    preflight = json.loads((PREVIOUS / 'factor_events.jsonl').read_text().splitlines()[0])
    for old, current in zip(old_counts, preflight['node_counts'], strict=True):
        for key in ('kind', 'width', 'auxiliaries', 'continuous_edges', 'binary_edges', 'center_edges'):
            if old[key] != current[key]:
                raise ValueError('source topology differs from actual C8 target')

    def row_range(values, constant, exponent):
        normalized = scaled_exact(values, -exponent)
        normalized_rhs = scaled_exact([constant], -exponent)[0]
        magnitudes = np.abs(np.concatenate((normalized, [1.])))
        low, high = float(magnitudes.min()), float(magnitudes.max())
        shift = max(0, -20 - (int(np.frexp(low)[1]) - 1))
        scaled_high = float(np.ldexp(high, shift))
        reversible = True
        try:
            scaled_exact(normalized, shift)
            scaled_exact([normalized_rhs, 1.], shift)
        except (ValueError, FloatingPointError):
            reversible = False
        return {'exponent': exponent, 'coefficient_min_before_row_scale': low,
            'coefficient_max_before_row_scale': high, 'row_shift': shift,
            'coefficient_max_after_row_scale': scaled_high,
            'exactly_reversible': reversible, 'fixed_window_passed': reversible and scaled_high <= 2.**40}

    summaries, rejected, visited, total_rows = [], [], 0, 0
    for node_index, node in enumerate(nodes):
        if node['kind'] != 'source':
            continue
        source = node['source']
        digest = source_digest(source)
        rows = np.flatnonzero(node['needed'])
        report = {'node': node_index, 'source_sha256': digest, 'required_rows': int(rows.size),
            'floored_rejected': 0, 'unfloored_rejected': 0, 'raw_exponent_min': None,
            'raw_exponent_max': None, 'visited_source_coefficients': 0}
        for row in rows:
            pieces = []
            for matrix in (source.Gc, source.Gb):
                start, stop = matrix.indptr[row:row + 2]
                values = matrix.data[start:stop]
                pieces.append(values[values != 0.])
            values = np.concatenate(pieces)
            constant = float(source.c[row])
            summands = np.concatenate((values, [constant] if constant != 0. else []))
            visited += int(summands.size)
            report['visited_source_coefficients'] += int(summands.size)
            if visited > 64_000_000:
                raise MemoryError('source entry inspection ceiling exceeded')
            exponents = np.frexp(np.abs(summands))[1]
            floor = int(exponents.max()) - 26
            total = sum(1 << max(int(e) - floor, 0) for e in exponents)
            raw = floor + (total - 1).bit_length()
            if box_exponent(summands) != max(0, raw):
                raise ValueError('independent sum envelope differs from C8 rule')
            fixed, unfloored = row_range(values, constant, max(0, raw)), row_range(values, constant, raw)
            report['floored_rejected'] += int(not fixed['fixed_window_passed'])
            report['unfloored_rejected'] += int(not unfloored['fixed_window_passed'])
            report['raw_exponent_min'] = raw if report['raw_exponent_min'] is None else min(raw, report['raw_exponent_min'])
            report['raw_exponent_max'] = raw if report['raw_exponent_max'] is None else max(raw, report['raw_exponent_max'])
            if not fixed['fixed_window_passed']:
                nonzero = np.abs(values)
                rejected.append({'node': node_index, 'coordinate': int(row), 'coefficient_count': int(values.size),
                    'original_nonzero_coefficient_min': float(nonzero.min()) if nonzero.size else None,
                    'original_nonzero_coefficient_max': float(nonzero.max()) if nonzero.size else None,
                    'original_center': constant, 'floored': fixed, 'unfloored_diagnostic_only': unfloored})
            total_rows += 1
        if source_digest(source) != digest:
            raise ValueError('diagnostic mutated an original source')
        summaries.append(report)
    hz, old_predicates = saved['hz'], {}
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        matrix = getattr(hz, name)
        data = matrix.data[:matrix.indptr[saved['old_n_eq']]] if name in ('Ac', 'Ab') else matrix.data
        magnitude = np.abs(data[data != 0.])
        old_predicates[name] = {'nonzero': int(magnitude.size),
            'min_nonzero_abs': float(magnitude.min()) if magnitude.size else None,
            'max_abs': float(magnitude.max()) if magnitude.size else None,
            'at_or_below_native_small': int(np.count_nonzero(magnitude <= 1e-9)),
            'at_or_above_native_large': int(np.count_nonzero(magnitude >= 1e15))}
    record = {'schema': 'c8_source_scale_diagnostic_v1', 'formal_gain': 0, 'status': 'DIAGNOSTIC_COMPLETE',
        'source_nodes': summaries, 'all_rejected_rows': rejected, 'inspected_rows': total_rows,
        'inspected_source_coefficients': visited, 'old_predicates': old_predicates,
        'actual_target_topology_matched': True, 'lift_executed': False, 'solve_executed': False,
        'operator_rows_executed': False, 'unfloored_full_candidate_qualification_claimed': False,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    _atomic_exclusive_json(DIRECTORY / 'result.json', record)
    print(json.dumps({'status': record['status'], 'inspected_rows': total_rows,
        'floored_rejected': len(rejected), 'unfloored_rejected': sum(s['unfloored_rejected'] for s in summaries),
        'formal_gain': 0}), flush=True)


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    hashes = json.loads((PREVIOUS / 'preregistered.json').read_text())['source_sha256']
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('previous freeze drift')
    for path in (Path(__file__), EXP / 'C8_CLOSED_AND_SOURCE_SCALE_AUDIT_PREREG_20260905.md',
                 PREVIOUS / 'exit.json', PREVIOUS / 'factor_events.jsonl', PREVIOUS / 'worker.log', CHECKPOINT):
        hashes[str(path)] = _sha256(path)
    provenance = baseline_worker._provenance(ROOT)
    command = [sys.executable, str(Path(__file__)), '--worker']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'command': command, 'wall_cap_s': 240, 'memory_gb': 16, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'worker.log').open('x') as stream:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = done.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / name) != sha for name, sha in hashes.items()),
            provenance_drift=baseline_worker._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    worker() if len(sys.argv) == 2 and sys.argv[1] == '--worker' else main()
