"""Offline complete suffix qualification; never invokes a verifier or solver."""

import json
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
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import lift
from experiments.neural_hz_20260831.c8_dyadic_balance_audit_v1 import audit, reference_subset
from dataclasses import asdict
import os
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
PREVIOUS = EXPERIMENT / 'results/c5_first_terminal_20260905_v1'


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != EXPERIMENT / 'results/c8_dyadic_balance_20260905_v1':
        raise ValueError('not the exclusive preregistered result directory')
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((directory / 'preregistered.json').read_text())
    if any(_sha256(EXPERIMENT / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('source freeze drift before unpickling')
    snapshot = PREVIOUS / 'layer75.pickle'
    if _sha256(snapshot) != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed':
        raise ValueError('actual snapshot drift')
    events = PREVIOUS / 'composition_events.jsonl'
    old_exit = json.loads((PREVIOUS / 'exit.json').read_text())
    if _sha256(events) != old_exit['artifacts'][events.name]:
        raise ValueError('native journal drift')
    with snapshot.open('rb') as stream:
        saved = pickle.load(stream)
    original = saved['expr_cache'][75]
    net = saved['net']
    dense = net.by_id[77]
    if dense.kind != 'DENSE' or net.by_id[76].kind not in {'FLATTEN', 'RESHAPE'}:
        raise ValueError('actual native affine tail changed')
    if net.preds[77] != [76] or net.preds[76] != [75]:
        raise ValueError('actual native tail graph changed')
    weight = sp.csr_matrix(dense.params['weight'].detach().cpu().double().numpy())
    bias = dense.params.get('bias')
    bias = None if bias is None else bias.detach().cpu().double().numpy().reshape(-1)
    expr = cnn._lazy_append_linear(original, weight, bias, 64_000_000)
    journal = [json.loads(line) for line in events.read_text().splitlines()]
    expected = [entry for entry in journal if entry['event'] == 'materialization_start' and entry['layer'] == 78]
    if len(expected) != 1:
        raise ValueError('actual ReLU78 invocation is not unique')
    sources, terms = {}, []
    for term in expr.terms:
        s = term.source
        if id(s) not in sources:
            sources[id(s)] = {'source_index': len(sources), 'n_out': s.n_out, 'n_cont': s.n_cont,
                'n_bin': s.n_bin, 'live_value_rows': int(live_value_rows(s).sum())}
        terms.append({'source_index': sources[id(s)]['source_index'],
            'operators': [{'type': type(op).__name__, 'shape': list(op.shape)} for op in term.operators]})
    if terms != expected[0]['terms'] or list(sources.values()) != expected[0]['sources'] or expr.n_out != expected[0]['n_out']:
        raise ValueError('complete offline expression differs from actual native schema')
    before = collect(SimpleNamespace(), {'saved': saved, 'expr': expr}).fingerprint
    started = time.monotonic()
    with (directory / 'factor_events.jsonl').open('x') as stream:
        def emit(name, payload):
            stream.write(json.dumps({'event': name, 'elapsed_s': time.monotonic() - started, **payload}) + '\n')
            stream.flush()
        same_frame = [hz for hz in saved['hz_cache'].values() if hz.frame_id == expr.frame_id]
        frame_widths = (max(hz.n_cont for hz in same_frame), max(hz.n_bin for hz in same_frame))
        lifted, construction = measured_build(lambda: lift(expr, np.ones(expr.n_out, dtype=bool),
            enabled=True, frame_widths=frame_widths, observe=emit))
        emit('construction_complete', {'construction': construction, 'report': lifted.report})
        audit_started = time.monotonic()
        identity = audit(lifted)
        identity['stage_elapsed_s'] = time.monotonic() - audit_started
        emit('complete_identity_audit', identity)
        if collect(SimpleNamespace(), {'saved': saved, 'expr': expr}).fingerprint != before:
            raise ValueError('loaded native state changed')
        checkpoint_payload = {'schema': 'c8_dyadic_balance_checkpoint_v1', 'hz': lifted.hz,
            'original_prefix_hz_cache': saved['hz_cache'], 'definition_graph': lifted.nodes,
            'expression': expr, 'root': lifted.root, 'old_n_cont': lifted.old_n_cont,
            'old_n_bin': lifted.old_n_bin, 'old_n_eq': lifted.old_n_eq, 'keep': lifted.keep,
            'report': lifted.report, 'identity_audit': identity, 'provenance': freeze['provenance'],
            'origin_snapshot_sha256': _sha256(snapshot), 'formal_gain': 0}
        checkpoint = directory / 'lifted_hz.pickle'
        with checkpoint.open('xb') as handle:
            pickle.dump(checkpoint_payload, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        emit('checkpoint_saved', {'sha256': _sha256(checkpoint), 'bytes': checkpoint.stat().st_size})
        full = collect(SimpleNamespace(), {'saved': saved, 'checkpoint': checkpoint_payload,
            **lifted.numeric_roots()})
        candidate = full.measure()
        emit('complete_offline_numeric_roots', {'roots': len(full.numeric),
            'resident_bytes': candidate.resident_bytes, 'resident_entries': candidate.resident_entries})
        witness, reference = reference_subset(full, net)
        lower = reference['reference_lower_bound']
        physical = candidate.resident_bytes < lower['resident_bytes'] and candidate.resident_entries < lower['resident_entries']
        if collect(SimpleNamespace(), {'saved': saved, 'expr': expr}).fingerprint != before:
            raise ValueError('reference construction changed the loaded native state')
        lifted.validate()
        record = {'schema': 'c8_dyadic_balance_actual_v1', 'formal_gain': 0,
            'status': 'OFFLINE_IDENTITY_AND_STORAGE_QUALIFIED' if physical else 'OFFLINE_PHYSICAL_GATE_REJECTED',
            'offline_registered_gates_passed': physical, 'live_publication_executed': False,
            'terminal_solve_executed': False, 'quarter_work_proved': False,
            'report': lifted.report, 'construction': construction, 'identity': identity,
            'complete_offline_numeric_roots': len(full.numeric), 'candidate': asdict(candidate),
            'reference': reference, 'strict_complete_offline_physical_decrease': physical,
            'incoming_snapshot_unchanged': True, 'actual_native_schema_matched': True,
            'checkpoint_sha256': _sha256(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
            'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance'],
            'snapshot_sha256': _sha256(snapshot),
            'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        _atomic_exclusive_json(directory / 'result.json', record)
        emit('completed', {'status': record['status'], 'work_upper': lifted.report['total_work_upper'],
            'n_cont': lifted.hz.n_cont, 'n_bin': lifted.hz.n_bin, 'physical': physical})
        if not physical:
            raise MemoryError('complete offline physical gate rejected')
    print(json.dumps({'status': record['status'], 'formal_gain': 0,
        'work_upper': lifted.report['total_work_upper'], 'physical_decrease': physical}), flush=True)


if __name__ == '__main__':
    main()

