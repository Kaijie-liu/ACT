"""One full-population compositional box-transfer diagnostic; no solver."""

from dataclasses import asdict
from fractions import Fraction as F
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
DIRECTORY = EXP / 'results/c16_box_transfer_20260910_v1'
PRIOR = EXP / 'results/c15_unit_row_splice_20260910_v1'
FUSED = EXP / 'results/c10_fused_emission_20260908_v1/fused_hz.pickle'
PROOF = FUSED.parent / 'result.json'
BINDING = EXP / 'results/c10_live_relu_20260908_v1/proof_binding.json'
SPLICED = PRIOR / 'spliced_hz.pickle'
INPUT_HASHES = {
    FUSED: '24b22d8a06c0c4d51525dd9d5ee0902941c40f20cc6a711e6231673e3efa9d08',
    PROOF: '315a152e3910b8340f5971be8346434a8d81e5106cd5f67321deea8def29fd5b',
    BINDING: '7358034b062d0561ea77d08e086aa99e6af421f0e27062605df7d1dd7368723a',
    SPLICED: '6b93e5f929be286d762f9c0779c0b8d505440325e9b786f943df73ff0022dc30'}


def worker():
    import numpy as np
    import torch
    from experiments.neural_hz_20260831.c16_box_transfer_v1 import receipt_from_archive, transfer
    from experiments.neural_hz_20260831.c10_fused_emission_v1 import FusedIntegrated
    from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
    from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity
    from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c16_box_transfer_actual_v1', 'formal_gain': 0, 'completed': False,
        'candidate_transformation_constructed': False, 'whole_live_path_proved': False,
        'solver_executed': False, 'native_ingestion_executed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    started = time.monotonic()
    with (DIRECTORY / 'events.jsonl').open('x') as stream:
        def observe(event):
            # Use a distinct key; stage durations must never overwrite timeline.
            event = {**event, 'worker_elapsed_s': time.monotonic() - started}
            stream.write(json.dumps(event) + '\n')
            stream.flush()
            print(json.dumps(event), flush=True)
        try:
            if any(_sha256(EXP / n) != sha for n, sha in freeze['source_sha256'].items()):
                raise ValueError('source/artifact freeze drift')
            if any(_sha256(p) != sha for p, sha in INPUT_HASHES.items()):
                raise ValueError('sealed independent input hash drift before unpickling')
            with FUSED.open('rb') as handle:
                fused_saved = pickle.load(handle)
            with SPLICED.open('rb') as handle:
                splice_saved = pickle.load(handle)
            if (fused_saved['schema'] != 'c10_fused_emission_checkpoint_v1'
                    or splice_saved['schema'] != 'c15_isolated_component_checkpoint_v1'):
                raise ValueError('wrong input checkpoint schema')
            fields = fused_saved['fields']
            candidate = FusedIntegrated(**fields, origin_binding=expression_binding(fields['expression']))
            candidate.seal = candidate.fingerprint()
            spliced = splice_saved['candidate']
            spliced.validate()
            original_identity = identity(candidate)
            original_splice_seal = spliced.seal
            observe({'event': 'full_source_checkpoints_loaded', 'full_saved_dictionaries_retained': True})
            def build():
                receipt = receipt_from_archive(candidate, PROOF.read_bytes(), BINDING.read_bytes(),
                    proof_sha256=INPUT_HASHES[PROOF], binding_sha256=INPUT_HASHES[BINDING])
                report, codes = transfer(candidate, receipt, enabled=True, observe=observe)
                return receipt, report, codes
            (receipt, report, codes), construction = measured_build(build)
            _atomic_exclusive_json(DIRECTORY / 'transfer.json', {'report': report, 'construction': construction,
                'receipt': asdict(receipt), 'formal_gain': 0})
            with (DIRECTORY / 'complete_main_codes.npz').open('xb') as handle:
                np.savez(handle, codes=codes)
                handle.flush()
                os.fsync(handle.fileno())
            observe({'event': 'complete_transfer_saved', 'construction': construction})
            # Complete independent Fraction norm audit, never used by transfer.
            audit_started = time.monotonic()
            checked = coefficients = binary_coefficients = 0
            hz = candidate.hz
            for i in np.flatnonzero(codes == 1):
                d = int(candidate.eq_roots[candidate.old_n_eq + i])
                start, stop = hz.Ac.indptr[d:d + 2]
                bstart, bstop = hz.Ab.indptr[d:d + 2]
                norm = sum((abs(F(float(v))) for v in hz.Ac.data[start:stop - 1]), F(0))
                norm += sum((abs(F(float(v))) for v in hz.Ab.data[bstart:bstop]), F(0))
                norm += abs(F(float(hz.b[d])))
                if norm > F(float(hz.Ac.data[stop - 1])):
                    raise ValueError(f'compositional bound contradicted by exact norm at MAIN {int(i)}')
                checked += 1
                coefficients += int(stop - start - 1)
                binary_coefficients += int(bstop - bstart)
            if checked != report['direct_main_boxes_transferred']:
                raise ValueError('incomplete independent direct-row audit')
            c15_checked = 0
            for column, unused_target, pivot, offset, unused_sign in spliced.decoded():
                i = column - candidate.old_n_cont
                if not 0 <= i < len(codes) or codes[i] != 1:
                    raise ValueError('existing C15 population not covered by compositional theorem')
                d = int(candidate.eq_roots[candidate.old_n_eq + i])
                stop = hz.Ac.indptr[d + 1]
                if float(hz.Ac.data[stop - 1]) != pivot or float(hz.b[d]) != offset:
                    raise ValueError('C15 original row pivot/RHS differs from bound source')
                c15_checked += 1
            if c15_checked != 268:
                raise ValueError('C15 complete coverage count changed')
            if identity(candidate) != original_identity or spliced.seal != original_splice_seal:
                raise ValueError('proof diagnostic mutated a source')
            spliced.validate()
            oracle = {'all_direct_rows_fraction_checked': checked,
                'continuous_parent_coefficients_fraction_checked': coefficients,
                'binary_coefficients_fraction_checked': binary_coefficients,
                'complete_c15_pairs_covered': c15_checked,
                'c15_pivots_and_rhs_match': True, 'source_identities_unchanged': True,
                'audit_elapsed_s': time.monotonic() - audit_started,
                'independent_fraction_oracle_passed': True, 'formal_gain': 0}
            _atomic_exclusive_json(DIRECTORY / 'independent_oracle.json', oracle)
            observe({'event': 'independent_full_population_oracle_saved', **oracle})
            record.update(completed=True, report=report, construction=construction,
                independent_oracle=oracle, full_source_dictionaries_retained=True)
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
    if (_sha256(PRIOR / 'exit.json') != '002d9f5b1fac9999484639befa523e59d8e14085fc8627422e5e3a64ca32faff'
            or _sha256(PRIOR / 'result.json') != '1284c9a1c645387d1285f1be39967ed592597995f10c5aac74aed2a2d7e1f9cd'):
        raise ValueError('completed C15 artifacts drift')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c16_box_transfer_v1.py', 'test_c16_box_transfer_v1.py',
        'C16_BOX_TRANSFER_PREREG_20260910.md', 'C16_BOX_TRANSFER_DEVELOPMENT_20260910.md',
        'CHECKPOINT_C15_UNIT_20260910_SHA256SUMS', 'C15_UNIT_SPLICE_AUDIT_20260910.md',
        'C15_GENERATION_INTEGRATION_DESIGN_20260910.md',
        *(str(p) for p in INPUT_HASHES), str(PRIOR / 'exit.json'), str(PRIOR / 'result.json')]
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c16_box_transfer_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--worker']
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'entry_cap': 64_000_000, 'construction_cap_bytes': 1024**3,
        'solver_authorized': False, 'transformation_authorized': False, 'native_ingestion_authorized': False,
        'whole_live_path_authorized': False, 'partial_acceptance_allowed': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / n) for n in tests)], cwd=ROOT, env=environment,
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
            source_drift=any(_sha256(EXP / n) != sha for n, sha in hashes.items()),
            provenance_drift=baseline._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    worker() if len(sys.argv) == 2 and sys.argv[1] == '--worker' else main()
