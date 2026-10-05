"""Fresh original-network Closed HZ, actual native phase and complete live gate."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as c5_installed
from experiments.neural_hz_20260831 import c25_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c24_closed_state_v1 import export
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool, incidence_oracle, verify_all_and_discover
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
DIRECTORY = EXP / 'results/c25_live_relu_20260911_v1'
PROOF_SHA = '99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'
POST_SHA = '82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367'
TABLE = EXP / 'results/c14_early_rejection_census_20260910_v1/single_use_factor_table.npz'


class QualifiedStop(BaseException):
    """Stop before any terminal API: C25 does not change the C10 matrix."""


def audit_actual_phase(state):
    """Independent complete actual incidence, with no graph or sealed selection."""
    candidate = state['lifted']
    overlay = runtime.phase_overlay(state)
    post = state['phase_ownership']['post_hz']
    first = overlay.old_uid_ceiling
    ne, nl = post.n_eq - candidate.hz.n_eq, post.n_ineq - candidate.hz.n_ineq
    diagnostic = WorkPool(256_000_000)
    main = candidate.logical_n_cont - candidate.old_n_cont
    diagnostic.charge('complete_closed_UID_metadata', 32 * (main + post.n_eq + post.n_ineq))
    eq, le = closed_uid_tables(candidate)
    eq = np.r_[eq, np.arange(first, first + ne, dtype=np.int64)]
    le = np.r_[le, np.arange(first + ne, first + ne + nl, dtype=np.int64)]
    actual = incidence_oracle(post, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=diagnostic)
    columns, report = verify_all_and_discover(candidate, post, overlay, actual, eq, le,
        whole=diagnostic, branch=BranchPool(diagnostic))
    runtime.phase_overlay(state)
    report.update(independent_diagnostic_work=diagnostic.used,
        independent_diagnostic_work_parts=dict(diagnostic.parts),
        graph_free_phase_queries_proved=True, new_phase_executed=True,
        actual_native_rows_used=True, complete_post_HZ_sha256=source_digest(post),
        unit_splice_executed=False, formal_gain=0)
    return columns, report


def checkpoint_fields(state):
    """Portable data only; no opaque runtime receipt or construction graph."""
    fields, raw = export(state['lifted'])
    runtime.phase_overlay(state)
    owned = state['phase_ownership']
    return {'closed_fields': fields, 'closed_proof_bytes': raw,
        'closed_proof_sha256': hashlib.sha256(raw).hexdigest(),
        'closed_binding': state['closed_binding'],
        'phase_ownership': {k:v for k,v in owned.items() if k != 'receipt'},
        'numeric_roots': runtime.numeric_roots(state)}


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != DIRECTORY:
        raise ValueError('not preregistered isolated directory')
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((directory / 'preregistered.json').read_text())
    started = time.monotonic()
    incoming, retained = {}, []
    record = {'schema': 'c25_live_relu_qualification_v1', 'formal_gain': 0,
        'passed': False, 'terminal_solve_executed': False, 'fresh_original_network': True,
        'archived_HZ_loaded_or_substituted': False, 'unit_splice_executed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance'], 'islands': []}
    try:
        if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
            raise ValueError('source/library/artifact freeze drift')
        raw = (directory / 'closed_proof.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != PROOF_SHA or freeze['closed_proof_sha256'] != PROOF_SHA:
            raise ValueError('independently anchored closed proof drift')
        with (directory / 'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps({'elapsed_s': time.monotonic() - started, **event}) + '\n')
                stream.flush()
                if event['event'].startswith('c25_') or event['event'] in {'c9_live_constructed', 'c9_native_relu_constructed'}:
                    print(json.dumps(event), flush=True)

            def before(state):
                tf = state['tf']
                plain_entry(tf)
                extra = {**caller_roots(), 'apply_args': state['apply_args'],
                    'apply_kwargs': state['apply_kwargs'], 'current_expression': state['expression'],
                    'independently_anchored_proof_bytes': raw}
                roots = collect(tf, extra)
                incoming[id(state)] = (extra, roots.fingerprint)
                entry = roots.measure()
                emit({'event': 'c25_live_entry', 'layer': state['layer'].id,
                    'roots': len(roots.numeric), 'bytes': entry.resident_bytes, 'entries': entry.resident_entries})

            def ready(state):
                extra, fingerprint = incoming[id(state)]
                state['lifted'].validate()
                if collect(state['tf'], extra).fingerprint != fingerprint:
                    raise ValueError('closed construction/binding changed incoming live roots')
                retained.append(state)
                record['islands'].append({'layer': state['layer'].id, 'binding': state['closed_binding'],
                    'construction': state['construction'], 'report': state['lifted'].report,
                    'incoming_roots_unchanged': True})

            def consumed(state, fact):
                tf, layer = state['tf'], state['layer']
                if len(retained) != 1 or layer.id != 78 or layer.kind != 'RELU':
                    raise ValueError('registered diagnostic target not the unique selected ReLU')
                actual = tf._sparse_hz_cache.get(layer.id)
                if actual is None or layer.id in tf._sparse_affine_expr_cache:
                    raise ValueError('native post-ReLU cache publication missing or conflicted')
                if state['consumer_construction'] is None or not state['views']:
                    raise ValueError('native selected consumer not measured')
                if state['phase_ownership']['post_hz'] is not actual:
                    raise ValueError('ownership does not bind actual cache publication')
                bounds = state['apply_args'][0] if state['apply_args'] else state['apply_kwargs']['input_bounds']
                check, oracle_build = measured_build(lambda: verify_plain(state['views'][-1], bounds,
                    layer, actual, tf, state['entry_widths'], state['entry_slots']))
                if source_digest(actual) != POST_SHA:
                    raise ValueError('actual native result differs from sealed same-target C10 matrix')
                check.update(oracle_construction=oracle_build,
                    actual_construction=state['consumer_construction'],
                    same_C10_matrix_not_new_terminal_problem=True)
                record['post_relu'] = check
                emit({'event': 'c25_live_consumer_exact', **check})
                phase_started = time.monotonic()
                unit_columns, phase = audit_actual_phase(state)
                # All columns are discovered from actual rows BEFORE table access.
                with np.load(TABLE, allow_pickle=False) as table:
                    expected = table['column'][table['individually_admissible']]
                    if len(expected) != 268 or not np.array_equal(unit_columns, expected):
                        raise ValueError('actual phase did not recover ALL 268 independent unit pairs')
                del expected
                phase.update(elapsed_s=time.monotonic() - phase_started, all_268_C15_columns_match=True,
                    actual_event_construction=state['phase_ownership']['construction'],
                    actual_event_report=state['phase_ownership']['report'])
                record['phase_event_audit'] = phase
                _atomic_exclusive_json(directory / 'phase_event_audit.json', phase)
                emit({'event': 'c25_actual_phase_complete_proof', **phase})
                extra = {**caller_roots(), 'applied_fact': fact,
                    'independently_anchored_proof_bytes': raw, 'unit_columns': unit_columns,
                    'retained_runtime': [runtime.numeric_roots(item) for item in retained],
                    'incoming_numeric_states': [item[0] for item in incoming.values()]}
                roots = collect(tf, extra)
                candidate = roots.measure()
                record.update(whole_live_state=asdict(candidate), whole_live_numeric_roots=len(roots.numeric),
                    python_shallow_bytes=roots.python_shallow_bytes)
                emit({'event': 'c25_complete_live_state', 'roots': len(roots.numeric),
                    'bytes': candidate.resident_bytes, 'entries': candidate.resident_entries})
                witness, reference = reference_subset(roots, tf._net)
                lower = reference['reference_lower_bound']
                if (lower['resident_bytes'], lower['resident_entries']) != (629346312, 52428800):
                    raise ValueError('frozen two-leaf reference lower bound changed')
                physical = candidate.resident_bytes < lower['resident_bytes'] and candidate.resident_entries < lower['resident_entries']
                record['reference'], record['physical_decrease'] = reference, physical
                if not physical or collect(tf, extra).fingerprint != roots.fingerprint:
                    raise ValueError('complete live post-ReLU storage or fingerprint gate failed')
                emit({'event': 'c25_live_physical_pass', 'bytes': candidate.resident_bytes,
                    'entries': candidate.resident_entries, 'reference_lower_bound': lower})
                native = inspect(actual)
                record['native_ingestion'] = native
                if not native['passed']:
                    raise ValueError('actual post-ReLU native coefficient fidelity rejected')
                payload = {'schema': 'c25_live_relu_checkpoint_v1', 'formal_gain': 0,
                    **checkpoint_fields(state), 'post_relu_hz': actual,
                    'net': tf._net, 'hz_cache': tf._sparse_hz_cache,
                    'expr_cache': tf._sparse_affine_expr_cache,
                    'frame_widths': tf._sparse_frame_widths, 'relu_slots': tf._sparse_relu_slots,
                    'aux_slots': tf._sparse_aux_slots, 'applied_fact': fact,
                    'provenance': freeze['provenance'], 'source_sha256': freeze['source_sha256'],
                    'post_relu_identity': check, 'phase_event_audit': phase, 'unit_columns': unit_columns,
                    'whole_live_path_proved': True, 'terminal_solve_executed': False}
                checkpoint = directory / 'relu78.pickle'
                with checkpoint.open('xb') as handle:
                    pickle.dump(payload, handle, protocol=5)
                    handle.flush()
                    os.fsync(handle.fileno())
                record.update(passed=True, status='LIVE_CLOSED_RELU_QUALIFIED',
                    actual_cache_publication=True, checkpoint_sha256=_sha256(checkpoint),
                    checkpoint_bytes=checkpoint.stat().st_size, actual_hz_sha256=source_digest(actual))
                emit({'event': 'c25_live_qualified', 'native_nnz': native['retained_matrix_nnz'], 'formal_gain': 0})
                raise QualifiedStop()

            with c5_installed(enabled=True, emit=emit):
                with runtime.installed(enabled=True, proof_bytes=raw, expected_proof_sha256=PROOF_SHA,
                        before=before, ready=ready, consumed=consumed, emit=emit):
                    prefix.main()
        if not record['passed']:
            raise ValueError('registered live target did not qualify')
    except QualifiedStop:
        pass
    except (runtime.SelectedRejected, Exception) as exc:
        record.update(passed=False, status='LIVE_CLOSED_RELU_REJECTED',
            failure={'type': type(exc).__name__, 'reason': str(exc)})
    finally:
        record.update(wall_s=time.monotonic() - started,
            max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory / 'qualification.json', record)
        print(json.dumps({'status': record.get('status'), 'passed': record['passed'], 'failure': record.get('failure')}), flush=True)
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
