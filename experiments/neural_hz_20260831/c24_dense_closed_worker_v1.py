"""C24 fresh dense-owned generation, full checker closure and sparse event state."""

from dataclasses import asdict
import json
import gc
import hashlib
import weakref
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831.c24_dense_emission_v1 import lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import close, export
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool as DiagnosticPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool, check_append, incidence_oracle, verify_all_and_discover
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c24_dense_closed_20260911_v1'
SNAPSHOT = EXP / 'results/c5_first_terminal_20260905_v1/layer75.pickle'
LIVE = EXP / 'results/c9_live_relu_20260906_v1/relu78.pickle'
PHASE = EXP / 'results/c10_live_relu_20260908_v1/relu78.pickle'
TABLE = EXP / 'results/c14_early_rejection_census_20260910_v1/single_use_factor_table.npz'
MAPS = ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')


def load_oracle():
    # External file deserialization is not a mutation/deletion of live native
    # caches. Retain the COMPLETE declared pre-HZ and maps in the measured union.
    with LIVE.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['schema'] != 'c9_live_relu_checkpoint_v1':
        raise ValueError('unexpected sealed live checkpoint schema')
    return saved['preactivation_hz'], {key: saved['numeric_roots'][key] for key in MAPS}


def load_phase_oracle():
    with PHASE.open('rb') as handle:
        saved = pickle.load(handle)
    if saved['schema'] != 'c10_live_relu_checkpoint_v1':
        raise ValueError('unexpected sealed phase checkpoint schema')
    return saved['post_relu_hz']



def phase_audit(candidate, post):
    """Same complete external phase diagnostic, without graph or owner copy."""
    candidate.validate()
    before = candidate.fingerprint()
    check_append(candidate.hz, post)
    pre = candidate.hz
    ne, nl = post.n_eq - pre.n_eq, post.n_ineq - pre.n_ineq
    first = candidate.report['radix_uid_base'] + 16384
    construction = WorkPool(candidate.report['total_work_upper'], candidate.report['largest_branch_work_upper'])
    new_nnz = int(post.Ac.nnz - pre.Ac.nnz + post.Auc.nnz - pre.Auc.nnz)
    construction.charge('diagnostic_appended_CSR_slices', 8 * new_nnz + 4 * (ne + nl))
    overlay, report = build_overlay(candidate,
        [(post.Ac[pre.n_eq:], first), (post.Auc[pre.n_ineq:], first + ne)],
        pool=construction, enabled=True)
    event_sha = hashlib.sha256(overlay.events.tobytes()).hexdigest()
    diagnostic = DiagnosticPool(256_000_000)
    main = candidate.logical_n_cont - candidate.old_n_cont
    diagnostic.charge('complete_closed_UID_metadata', 32 * (main + post.n_eq + post.n_ineq))
    eq, le = closed_uid_tables(candidate)
    eq = np.r_[eq, np.arange(first, first + ne, dtype=np.int64)]
    le = np.r_[le, np.arange(first + ne, first + ne + nl, dtype=np.int64)]
    actual = incidence_oracle(post, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=diagnostic)
    columns, checked = verify_all_and_discover(candidate, post, overlay, actual, eq, le,
        whole=diagnostic, branch=BranchPool(diagnostic))
    overlay.validate()
    candidate.validate()
    if candidate.fingerprint() != before or hashlib.sha256(overlay.events.tobytes()).hexdigest() != event_sha:
        raise ValueError('closed base or append events mutated during proof')
    report.update(checked, new_EQ_rows=ne, new_INEQ_rows=nl,
        event_words_sha256=event_sha, base_precondition='independent_full_source_owner_UID_checker',
        full_base_validation_repeated=False, event_construction_work=construction.used,
        event_construction_work_parts=dict(construction.parts),
        generation_plus_event_whole_work=construction.whole_base + construction.used,
        generation_plus_event_branch_work=construction.branch_base + construction.used,
        independent_diagnostic_work=diagnostic.used, independent_diagnostic_work_parts=dict(diagnostic.parts),
        graph_free_phase_queries_proved=True, new_native_relu_executed=False,
        complete_post_HZ_sha256=source_digest(post), whole_live_path_proved=False, formal_gain=0)
    return report, columns, overlay

def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c24_dense_closed_actual_v1', 'completed': False, 'formal_gain': 0,
        'live_publication_executed': False, 'terminal_solve_executed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    started, initial, original_sha = time.monotonic(), None, None
    try:
        if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
            raise ValueError('source/artifact freeze drift')
        if (_sha256(SNAPSHOT) != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
                or _sha256(LIVE) != '5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65'
                or _sha256(PHASE) != '1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d'
                or _sha256(TABLE) != '0208ecdb11a35896c1ecb86a6fabb1c1e9f3c49bf7b47d6248a18d52c6e3328e'):
            raise ValueError('sealed input checkpoint drift before unpickling')
        original_hz, maps = load_oracle()
        original_sha = source_digest(original_hz)
        post_oracle = load_phase_oracle()
        if source_digest(post_oracle) != '82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367':
            raise ValueError('post-ReLU content drift')
        with SNAPSHOT.open('rb') as stream:
            saved = pickle.load(stream)
        original, net = saved['expr_cache'][75], saved['net']
        dense = net.by_id[77]
        if (dense.kind != 'DENSE' or net.by_id[76].kind not in {'FLATTEN', 'RESHAPE'}
                or net.preds[77] != [76] or net.preds[76] != [75]):
            raise ValueError('original registered native affine tail changed')
        weight = sp.csr_matrix(dense.params['weight'].detach().cpu().double().numpy())
        bias = dense.params.get('bias')
        bias = None if bias is None else bias.detach().cpu().double().numpy().reshape(-1)
        expr = cnn._lazy_append_linear(original, weight, bias, 64_000_000)
        journal_path = SNAPSHOT.parent / 'composition_events.jsonl'
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        expected = [e for e in journal if e['event'] == 'materialization_start' and e['layer'] == 78]
        sources, terms = {}, []
        for term in expr.terms:
            s = term.source
            if id(s) not in sources:
                sources[id(s)] = {'source_index': len(sources), 'n_out': s.n_out, 'n_cont': s.n_cont,
                    'n_bin': s.n_bin, 'live_value_rows': int(live_value_rows(s).sum())}
            terms.append({'source_index': sources[id(s)]['source_index'],
                'operators': [{'type': type(op).__name__, 'shape': list(op.shape)} for op in term.operators]})
        if (len(expected) != 1 or terms != expected[0]['terms'] or list(sources.values()) != expected[0]['sources']
                or expr.n_out != expected[0]['n_out']):
            raise ValueError('suffix expression differs from original native schema')
        initial = collect(SimpleNamespace(), {'saved': saved, 'expr': expr,
            'oracle_hz': original_hz, 'oracle_maps': maps, 'post_oracle': post_oracle}).fingerprint
        same_frame = [hz for hz in saved['hz_cache'].values() if hz.frame_id == expr.frame_id]
        widths = (max(hz.n_cont for hz in same_frame), max(hz.n_bin for hz in same_frame))
        with (DIRECTORY / 'events.jsonl').open('x') as stream:
            def emit(name, payload):
                event = {'event': name, **payload, 'worker_elapsed_s': time.monotonic() - started}
                stream.write(json.dumps(event) + '\n')
                stream.flush()
                print(json.dumps(event), flush=True)
            draft, construction = measured_build(lambda: lift(expr, np.ones(expr.n_out, dtype=bool),
                enabled=True, frame_widths=widths, observe=emit))
            record.update(report=draft.report, construction=construction)
            emit('construction_complete', {'construction': construction, 'report': draft.report})
            draft_roots = draft.numeric_roots()
            old_roots = {**{k:v for k,v in draft_roots.items() if k not in {'owners','uid_slabs'}},
                'hz': original_hz, **maps}
            old_component = collect(SimpleNamespace(), old_roots).measure()
            retired = [weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
            old_draft_ref = weakref.ref(draft)
            proof_start = time.monotonic()
            candidate, proof = close(draft, original_hz, maps, enabled=True)
            record['closed_proof_elapsed_s'] = time.monotonic() - proof_start
            record['identity'] = proof
            unused_fields, proof_bytes = export(candidate)
            with (DIRECTORY / 'closed_proof.json').open('xb') as handle:
                handle.write(proof_bytes); handle.flush(); os.fsync(handle.fileno())
            # Retire only this newly built, unpublished construction draft,
            # after the FULL independent source/owner/UID proof. All original
            # snapshot/oracle roots remain. Registry holds text, never arrays.
            del draft, draft_roots, old_roots, unused_fields
            gc.collect()
            if old_draft_ref() is not None or any(ref() is not None for ref in retired):
                raise ValueError('closed state still retains construction graph fields')
            record['all_new_graph_fields_physically_retired'] = True
            emit('full_source_owner_UID_proof_and_closure_passed', {
                'elapsed_s': record['closed_proof_elapsed_s'], 'uid_slabs': len(candidate.uid_slabs),
                'all_MAIN_checked': proof['all_MAIN_ownership_checked'],
                'all_reserved_UIDs_checked': proof['all_reserved_MAIN_UIDs_checked'],
                'all_new_graph_fields_physically_retired': True})
            phase_start = time.monotonic()
            phase, unit_columns, overlay = phase_audit(candidate, post_oracle)
            with np.load(TABLE, allow_pickle=False) as table:
                expected_columns = table['column'][table['individually_admissible']]
                if len(expected_columns) != 268 or not np.array_equal(unit_columns, expected_columns):
                    raise ValueError('complete closed phase ownership did not recover ALL C15 unit pairs')
            phase['elapsed_s'] = time.monotonic() - phase_start
            phase['all_268_C15_columns_match'] = True
            record['phase_event_audit'] = phase
            _atomic_exclusive_json(DIRECTORY / 'phase_event_audit.json', phase)
            emit('complete_closed_phase_ownership_proved', phase)
            new_roots = candidate.numeric_roots()
            new_component = collect(SimpleNamespace(), new_roots).measure()
            component_decrease = (new_component.resident_bytes < old_component.resident_bytes
                and new_component.resident_entries < old_component.resident_entries)
            record['component'] = {'original_graph_hz_lineage_bytes': old_component.resident_bytes,
                'new_closed_hz_lineage_owner_bytes': new_component.resident_bytes,
                'original_graph_hz_lineage_entries': old_component.resident_entries,
                'new_closed_hz_lineage_owner_entries': new_component.resident_entries,
                'strict_bytes_and_entries_decrease': component_decrease}
            if not component_decrease:
                raise ValueError('complete closed component failed strict storage gate')
            full = collect(SimpleNamespace(), {'saved': saved, 'expr': expr,
                'oracle_hz': original_hz, 'oracle_maps': maps, 'post_oracle': post_oracle,
                'phase_events': overlay.events, 'unit_columns': unit_columns, **new_roots})
            union = full.measure()
            record.update(complete_offline_union=asdict(union), complete_offline_root_count=len(full.numeric),
                complete_external_oracle_and_original_native_snapshot_retained=True,
                python_shallow_bytes=full.python_shallow_bytes)
            emit('complete_offline_union', {'bytes': union.resident_bytes, 'entries': union.resident_entries,
                                          'numeric_roots': len(full.numeric)})
            witness, reference = reference_subset(full, net)
            record['reference'] = reference
            lower = reference['reference_lower_bound']
            physical = union.resident_bytes < lower['resident_bytes'] and union.resident_entries < lower['resident_entries']
            record['strict_complete_offline_reference_decrease'] = physical
            if not physical:
                raise ValueError('complete offline union failed frozen reference gate')
            emit('reference_qualified', {'physical': physical, 'reference_lower_bound': lower})
            native = inspect(candidate.hz)
            record['native_ingestion'] = native
            if not native['passed']:
                raise ValueError('native matrix/bound/integrality loss')
            emit('native_ingestion_passed', {'matrix_nnz': native['retained_matrix_nnz'],
                'lowered_n_cont': native['lowered_n_cont'], 'lowered_n_bin': native['lowered_n_bin']})
            if collect(SimpleNamespace(), {'saved': saved, 'expr': expr,
                    'oracle_hz': original_hz, 'oracle_maps': maps, 'post_oracle': post_oracle}).fingerprint != initial:
                raise ValueError('original snapshot/expression/oracle mutated')
            candidate.validate()
            fields, proof_bytes = export(candidate)
            payload = {'schema': 'c24_dense_closed_checkpoint_v1', 'fields': fields, 'identity': proof,
                'provenance': freeze['provenance'], 'source_sha256': freeze['source_sha256'],
                'origin_snapshot_sha256': _sha256(SNAPSHOT), 'formal_gain': 0,
                'proof_bytes': proof_bytes, 'closed_proof_sha256': _sha256(DIRECTORY / 'closed_proof.json'),
                'phase_events': overlay.events, 'unit_columns': unit_columns, 'phase_event_audit': phase,
                'post_oracle_file_sha256': _sha256(PHASE), 'whole_live_path_proved': False}
            path = DIRECTORY / 'closed_hz.pickle'
            with path.open('xb') as handle:
                pickle.dump(payload, handle, protocol=5)
                handle.flush()
                os.fsync(handle.fileno())
            record.update(completed=True, checkpoint_sha256=_sha256(path), checkpoint_bytes=path.stat().st_size,
                hz_sha256=source_digest(candidate.hz), original_snapshot_and_oracle_unchanged=True)
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        if initial is not None:
            record['original_snapshot_and_oracle_unchanged'] = collect(SimpleNamespace(),
                {'saved': saved, 'expr': expr, 'oracle_hz': original_hz, 'oracle_maps': maps, 'post_oracle': post_oracle}).fingerprint == initial
        if original_sha is not None:
            record['external_original_hz_unchanged'] = source_digest(original_hz) == original_sha
        record.update(wall_s=time.monotonic() - started,
            max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY / 'result.json', record)
        print(json.dumps({'completed': record['completed'], 'failure': record.get('failure'), 'formal_gain': 0}), flush=True)
    if not record['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
