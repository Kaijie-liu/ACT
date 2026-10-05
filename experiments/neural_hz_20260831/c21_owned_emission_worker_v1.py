"""One fresh owned suffix generation plus complete incidence/phase-event audits."""

from dataclasses import asdict
import json
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
from experiments.neural_hz_20260831.c21_owned_emission_v1 import lift
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import audit as ownership_audit, phase_event_audit
from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c21_owned_emission_20260911_v1'
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


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c21_owned_emission_actual_v1', 'completed': False, 'formal_gain': 0,
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
            candidate, construction = measured_build(lambda: lift(expr, np.ones(expr.n_out, dtype=bool),
                enabled=True, frame_widths=widths, observe=emit))
            record.update(report=candidate.report, construction=construction)
            emit('construction_complete', {'construction': construction, 'report': candidate.report})
            audit_start = time.monotonic()
            proof = audit(candidate, original_hz, maps)
            proof['elapsed_s'] = time.monotonic() - audit_start
            record['identity'] = proof
            emit('full_identity_passed', {'elapsed_s': proof['elapsed_s'],
                'selected_aliases': proof['quotient_proof']['all_definitions_checked']})
            ownership_start = time.monotonic()
            incidence = ownership_audit(candidate)
            incidence['elapsed_s'] = time.monotonic() - ownership_start
            record['ownership_audit'] = incidence
            _atomic_exclusive_json(DIRECTORY / 'ownership_audit.json', incidence)
            emit('all_generated_ownership_proved', incidence)
            phase_start = time.monotonic()
            phase, unit_columns, phase_owners = phase_event_audit(candidate, post_oracle, exact_proof=proof)
            with np.load(TABLE, allow_pickle=False) as table:
                expected_columns = table['column'][table['individually_admissible']]
                if len(expected_columns) != 268 or not np.array_equal(unit_columns, expected_columns):
                    raise ValueError('complete owned phase event did not recover ALL C15 unit pairs')
            phase['elapsed_s'] = time.monotonic() - phase_start
            phase['all_268_C15_columns_match'] = True
            record['phase_event_audit'] = phase
            _atomic_exclusive_json(DIRECTORY / 'phase_event_audit.json', phase)
            emit('complete_phase_ownership_proved', phase)
            new_roots = candidate.numeric_roots()
            # The unquotiented comparator does NOT carry the new ownership array.
            old_roots = {**{k: v for k, v in new_roots.items() if k != 'owners'}, 'hz': original_hz, **maps}
            old_component = collect(SimpleNamespace(), old_roots).measure()
            new_component = collect(SimpleNamespace(), new_roots).measure()
            component_decrease = (new_component.resident_bytes < old_component.resident_bytes
                and new_component.resident_entries < old_component.resident_entries)
            record['component'] = {'original_graph_hz_lineage_bytes': old_component.resident_bytes,
                'new_graph_hz_lineage_bytes': new_component.resident_bytes,
                'original_graph_hz_lineage_entries': old_component.resident_entries,
                'new_graph_hz_lineage_entries': new_component.resident_entries,
                'strict_bytes_and_entries_decrease': component_decrease}
            if not component_decrease:
                raise ValueError('complete component including tagged lineage failed strict storage gate')
            full = collect(SimpleNamespace(), {'saved': saved, 'expr': expr,
                'oracle_hz': original_hz, 'oracle_maps': maps, 'post_oracle': post_oracle,
                'phase_owners': phase_owners, 'unit_columns': unit_columns, **new_roots})
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
            fields = {k: v for k, v in vars(candidate).items() if k not in {'seal', 'origin_binding'}}
            payload = {'schema': 'c21_owned_emission_checkpoint_v1', 'fields': fields, 'identity': proof,
                'provenance': freeze['provenance'], 'source_sha256': freeze['source_sha256'],
                'origin_snapshot_sha256': _sha256(SNAPSHOT), 'formal_gain': 0,
                'phase_owners': phase_owners, 'unit_columns': unit_columns, 'phase_event_audit': phase,
                'post_oracle_file_sha256': _sha256(PHASE), 'whole_live_path_proved': False}
            path = DIRECTORY / 'owned_hz.pickle'
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
