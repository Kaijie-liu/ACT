"""Fresh fused C10 native ReLU, scalar proof binding and full live roots."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as c5_installed
from experiments.neural_hz_20260831 import c10_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c10_portable_binding_v1 import verify
from experiments.neural_hz_20260831.c9_live_relu_audit_v1 import verify_plain
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent


class QualifiedStop(BaseException):
    """Stop at the audited live boundary before even a terminal API call."""


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != EXP / 'results/c10_live_relu_20260908_v1':
        raise ValueError('not preregistered isolated directory')
    freeze = json.loads((directory / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('source/library freeze drift')
    proof_binding = json.loads((directory / 'proof_binding.json').read_text())
    started = time.monotonic()
    incoming, retained = {}, []
    record = {'schema': 'c10_live_relu_qualification_v1', 'formal_gain': 0,
        'passed': False, 'terminal_solve_executed': False, 'fresh_original_network': True,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance'], 'islands': []}
    try:
        with (directory / 'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps({'elapsed_s': time.monotonic() - started, **event}) + '\n')
                stream.flush()
                if event['event'] in {'c9_live_constructed', 'c9_native_relu_constructed', 'c9_live_value_view'}:
                    print(json.dumps(event), flush=True)

            def before(state):
                tf = state['tf']
                plain_entry(tf)
                extra = {**caller_roots(), 'apply_args': state['apply_args'],
                    'apply_kwargs': state['apply_kwargs'], 'current_expression': state['expression'], 'scalar_proof_binding': proof_binding}
                roots = collect(tf, extra)
                incoming[id(state)] = (extra, roots.fingerprint)
                entry = roots.measure()
                emit({'event': 'c9_live_entry', 'layer': state['layer'].id,
                    'roots': len(roots.numeric), 'bytes': entry.resident_bytes, 'entries': entry.resident_entries})

            def ready(state):
                extra, fingerprint = incoming[id(state)]
                identity = verify(state['lifted'], proof_binding)
                if collect(state['tf'], extra).fingerprint != fingerprint:
                    raise ValueError('C9 construction/audit changed incoming live roots')
                retained.append(state)
                record['islands'].append({'layer': state['layer'].id, 'identity': identity,
                    'construction': state['construction'], 'report': state['lifted'].report,
                    'incoming_roots_unchanged': True})
                emit({'event': 'c9_all_definitions_verified', 'layer': state['layer'].id, **identity})

            def consumed(state, fact):
                tf, layer = state['tf'], state['layer']
                if len(retained) != 1 or layer.id != 78 or layer.kind != 'RELU':
                    raise ValueError('registered diagnostic target not the unique selected ReLU')
                actual = tf._sparse_hz_cache.get(layer.id)
                if actual is None or layer.id in tf._sparse_affine_expr_cache:
                    raise ValueError('native post-ReLU cache publication missing or conflicted')
                if state['consumer_construction'] is None or not state['views']:
                    raise ValueError('native selected consumer not measured')
                bounds = state['apply_args'][0] if state['apply_args'] else state['apply_kwargs']['input_bounds']
                check, oracle_build = measured_build(lambda: verify_plain(state['views'][-1], bounds,
                    layer, actual, tf, state['entry_widths'], state['entry_slots']))
                record['post_relu'] = check
                record['post_relu']['oracle_construction'] = oracle_build
                record['post_relu']['actual_construction'] = state['consumer_construction']
                emit({'event': 'c9_live_consumer_exact', **check})
                extra = {**caller_roots(), 'applied_fact': fact, 'scalar_proof_binding': proof_binding,
                    'retained_runtime': [runtime.numeric_roots(item) for item in retained],
                    'incoming_numeric_states': [item[0] for item in incoming.values()]}
                roots = collect(tf, extra)
                candidate = roots.measure()
                record['whole_live_state'] = asdict(candidate)
                record['whole_live_numeric_roots'] = len(roots.numeric)
                record['python_shallow_bytes'] = roots.python_shallow_bytes
                emit({'event': 'c9_complete_live_state', 'roots': len(roots.numeric),
                    'bytes': candidate.resident_bytes, 'entries': candidate.resident_entries})
                witness, reference = reference_subset(roots, tf._net)
                lower = reference['reference_lower_bound']
                physical = candidate.resident_bytes < lower['resident_bytes'] and candidate.resident_entries < lower['resident_entries']
                record['reference'], record['physical_decrease'] = reference, physical
                if not physical or collect(tf, extra).fingerprint != roots.fingerprint:
                    raise ValueError('live post-ReLU physical gate or fingerprint failed')
                emit({'event': 'c9_live_physical_pass', 'bytes': candidate.resident_bytes,
                    'reference_lower_bytes': lower['resident_bytes']})
                # All measured construction is complete before this native load.
                native = inspect(actual)
                record['native_ingestion'] = native
                if not native['passed']:
                    raise ValueError('post-ReLU native coefficient fidelity rejected')
                payload = {'schema': 'c10_live_relu_checkpoint_v1', 'formal_gain': 0,
                    'post_relu_hz': actual, 'preactivation_hz': state['lifted'].hz,
                    'definition_graph': state['lifted'].nodes, 'root': state['lifted'].root,
                    'old_n_cont': state['lifted'].old_n_cont, 'old_n_bin': state['lifted'].old_n_bin,
                    'logical_n_cont': state['lifted'].logical_n_cont,
                    'numeric_roots': runtime.numeric_roots(state), 'net': tf._net,
                    'hz_cache': tf._sparse_hz_cache, 'expr_cache': tf._sparse_affine_expr_cache,
                    'frame_widths': tf._sparse_frame_widths, 'relu_slots': tf._sparse_relu_slots,
                    'aux_slots': tf._sparse_aux_slots, 'applied_fact': fact,
                    'provenance': freeze['provenance'], 'post_relu_identity': check,
                    'scalar_proof_binding': proof_binding}
                checkpoint = directory / 'relu78.pickle'
                with checkpoint.open('xb') as handle:
                    pickle.dump(payload, handle, protocol=5)
                    handle.flush()
                    os.fsync(handle.fileno())
                record.update(passed=True, status='LIVE_RELU_QUALIFIED',
                    actual_cache_publication=True, checkpoint_sha256=_sha256(checkpoint),
                    checkpoint_bytes=checkpoint.stat().st_size, actual_hz_sha256=source_digest(actual))
                emit({'event': 'c9_live_qualified', 'native_nnz': native['retained_matrix_nnz'], 'formal_gain': 0})
                raise QualifiedStop()

            with c5_installed(enabled=True, emit=emit):
                with runtime.installed(enabled=True, before=before, ready=ready, consumed=consumed, emit=emit):
                    prefix.main()
        if not record['passed']:
            raise ValueError('registered live target did not qualify')
    except QualifiedStop:
        pass
    except (runtime.SelectedRejected, Exception) as exc:
        record.update(passed=False, status='LIVE_RELU_REJECTED', failure={'type': type(exc).__name__, 'reason': str(exc)})
    finally:
        record.update(wall_s=time.monotonic() - started,
            max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory / 'qualification.json', record)
        print(json.dumps({'status': record.get('status'), 'passed': record['passed'], 'failure': record.get('failure')}), flush=True)
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
